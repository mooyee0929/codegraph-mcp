"""Runtime spans as evidence for (or corrections to) the static ``http_calls`` edges.

A span says "service S answered METHOD url with status N", optionally naming the
function that made the call. Spans are stored in ``trace_spans`` and applied to the
graph on every index run, after the static ``http_calls`` pass has rebuilt its edges:

- a static edge from the caller that matches the span is *validated*: confidence 1.0
  and ``observed`` added to its detail;
- a call the static pass missed is *added* as a new ``http_calls`` edge with
  confidence 1.0 and detail ``observed``;
- without a caller, the span validates the only static edge to its route, if there is
  exactly one, and otherwise just records which route it hit.

404 and 405 responses mean the server had no such route, so they are stored but never
matched to one.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from codegraph.models import Edge
from codegraph.queries import resolve_target
from codegraph.services.linking import route_patterns
from codegraph.services.paths import (
    MIN_MATCH_SCORE,
    compile_path,
    match_route,
    normalize_method,
    normalize_path,
    route_qname,
    score_route,
    split_route_qname,
)
from codegraph.services.storage import TRACE_DDL, ensure_tables, mark_observed
from codegraph.store import GraphStore

SpanValue = str | int | float | bool | None
_UNROUTED = frozenset({404, 405})


@dataclass(frozen=True, slots=True)
class Span:
    service: str
    method: str
    url: str
    path: str
    status: int | None
    caller: str | None


@dataclass(frozen=True, slots=True)
class ApplyStats:
    spans: int
    validated: int
    added: int
    unmatched: int


def parse_spans(raw: Sequence[Mapping[str, SpanValue]]) -> tuple[list[Span], list[str]]:
    """Validate span dicts; returns the good spans and one message per bad entry."""
    spans: list[Span] = []
    errors: list[str] = []
    for index, item in enumerate(raw):
        problems: list[str] = []
        url_value = item.get("url") or item.get("path")
        url = url_value.strip() if isinstance(url_value, str) else ""
        if not url:
            problems.append("needs a non-empty 'url' or 'path'")
        method_value = item.get("method", "ANY")
        if not isinstance(method_value, str):
            problems.append("'method' must be a string")
        service_value = item.get("service", "")
        if not isinstance(service_value, str):
            problems.append("'service' must be a string")
        status_value = item.get("status")
        status = status_value if type(status_value) is int else None
        if status_value is not None and status is None:
            problems.append("'status' must be an integer")
        caller_value = item.get("caller")
        if caller_value is not None and not isinstance(caller_value, str):
            problems.append("'caller' must be a string")
        if problems:
            errors.append(f"span {index}: {'; '.join(problems)}")
            continue
        spans.append(
            Span(
                service=service_value if isinstance(service_value, str) else "",
                method=normalize_method(method_value if isinstance(method_value, str) else None),
                url=url,
                path=normalize_path(url),
                status=status,
                caller=caller_value.strip() or None if isinstance(caller_value, str) else None,
            )
        )
    return spans, errors


def store_spans(store: GraphStore, spans: list[Span]) -> int:
    ensure_tables(store, TRACE_DDL)
    now = time.time()
    store.connection.executemany(
        "INSERT INTO trace_spans (service, method, url, path, status, caller, ingested_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(s.service, s.method, s.url, s.path, s.status, s.caller, now) for s in spans],
    )
    return len(spans)


def _edge_matches(edge: Edge, method: str, path: str, route: str) -> bool:
    if edge.dst == route:
        return True
    parsed = split_route_qname(edge.dst)
    if parsed is None:
        return False
    score = score_route(compile_path(method, path), compile_path(*parsed))
    return score is not None and score >= MIN_MATCH_SCORE


def apply_spans(store: GraphStore) -> ApplyStats:
    """Re-apply every stored span to the current ``http_calls`` edges."""
    ensure_tables(store, TRACE_DDL)
    connection = store.connection
    rows = connection.execute("SELECT id, method, path, status, caller FROM trace_spans").fetchall()
    if not rows:
        return ApplyStats(0, 0, 0, 0)
    patterns = route_patterns(store.all_nodes(kinds=("route",)))
    groups: defaultdict[tuple[str, str, str | None, bool], list[int]] = defaultdict(list)
    for row in rows:
        routed = row["status"] not in _UNROUTED
        groups[(row["method"], row["path"], row["caller"], routed)].append(int(row["id"]))
    validated: set[int] = set()
    added = unmatched = 0
    for (method, path, caller, routed), ids in groups.items():
        match = match_route(method, path, patterns) if routed else None
        route = match.qname if match is not None else route_qname(method, path)
        placeholders = ",".join("?" * len(ids))
        connection.execute(
            f"UPDATE trace_spans SET route = ?, matched = ? WHERE id IN ({placeholders})",
            (route if routed else None, int(match is not None), *ids),
        )
        if not routed:
            unmatched += len(ids)
            continue
        if match is None:
            unmatched += len(ids)
        if caller is not None:
            node = resolve_target(store, caller)
            if node is None:
                continue
            hits = [
                (edge_id, edge)
                for edge_id, edge in _http_edges(store, src=node.qname)
                if _edge_matches(edge, method, path, route)
            ]
            if hits:
                validated.update(edge_id for edge_id, _ in hits)
                continue
            store.add_edges(
                [
                    Edge(
                        src=node.qname,
                        dst=route,
                        kind="http_calls",
                        path=node.path,
                        line=node.line_start,
                        resolved=match is not None,
                        confidence=1.0,
                        detail="observed",
                    )
                ]
            )
            added += 1
            continue
        to_route = _http_edges(store, dst=route)
        if len({edge.src for _, edge in to_route}) == 1:
            validated.update(edge_id for edge_id, _ in to_route)
    for edge_id in validated:
        mark_observed(store, edge_id)
    return ApplyStats(spans=len(rows), validated=len(validated), added=added, unmatched=unmatched)


def _http_edges(
    store: GraphStore, *, src: str | None = None, dst: str | None = None
) -> list[tuple[int, Edge]]:
    column, value = ("src", src) if src is not None else ("dst", dst)
    rows = store.connection.execute(
        f"SELECT id, src, dst, path, line, resolved, confidence, detail FROM edges "
        f"WHERE kind = 'http_calls' AND {column} = ?",
        (value,),
    ).fetchall()
    return [
        (
            int(r["id"]),
            Edge(
                src=r["src"],
                dst=r["dst"],
                kind="http_calls",
                path=r["path"],
                line=int(r["line"]),
                resolved=bool(r["resolved"]),
                confidence=float(r["confidence"]),
                detail=r["detail"],
            ),
        )
        for r in rows
    ]


def ingest(
    store: GraphStore, raw: Sequence[Mapping[str, SpanValue]]
) -> tuple[ApplyStats | None, list[str]]:
    """Validate, store and apply spans in one transaction; nothing is stored on bad input."""
    spans, errors = parse_spans(raw)
    if errors:
        return None, errors
    with store.transaction():
        store_spans(store, spans)
        stats = apply_spans(store)
    return stats, []


__all__ = ["ApplyStats", "Span", "SpanValue", "apply_spans", "ingest", "parse_spans", "store_spans"]
