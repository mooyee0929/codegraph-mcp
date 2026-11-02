"""Structured symbol search: filter by name, kind, place and connectivity, then page.

Keyword search answers "what is called like this"; this answers questions such as
"methods under ``api/`` with more than ten callers and no tests" or "the longest
functions in the TypeScript code", which need fan-in, fan-out, span and test coverage
as filters and sort keys. ``total`` counts every match so an agent can tell whether to
fetch the next page.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, get_args

from codegraph.cypher.vocab import resolve_label
from codegraph.graph import GraphView
from codegraph.models import NODE_KINDS, Node, NodeKind
from codegraph.store import GraphStore

SortKey = Literal["name", "fan_in", "fan_out", "span"]
SORT_KEYS: tuple[str, ...] = get_args(SortKey)
MAX_PAGE = 500


@dataclass(frozen=True, slots=True)
class GraphQuery:
    name_pattern: str | None = None
    """Regular expression searched (not anchored) in the bare name."""
    qname_pattern: str | None = None
    kinds: tuple[NodeKind, ...] = ()
    file_glob: str | None = None
    language: str | None = None
    min_fan_in: int | None = None
    max_fan_in: int | None = None
    min_fan_out: int | None = None
    max_fan_out: int | None = None
    has_tests: bool | None = None
    sort: SortKey = "name"
    limit: int = 50
    offset: int = 0
    case_sensitive: bool = False


@dataclass(frozen=True, slots=True)
class GraphHit:
    node: Node
    fan_in: int
    fan_out: int
    tests: int

    def as_dict(self) -> dict[str, object]:
        return {
            "qname": self.node.qname,
            "kind": self.node.kind,
            "path": self.node.path,
            "lines": [self.node.line_start, self.node.line_end],
            "language": self.node.language,
            "signature": self.node.signature,
            "fan_in": self.fan_in,
            "fan_out": self.fan_out,
            "span": self.node.span,
            "tests": self.tests,
        }


@dataclass(frozen=True, slots=True)
class GraphPage:
    total: int
    offset: int
    limit: int
    hits: tuple[GraphHit, ...]

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.hits) < self.total

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "offset": self.offset,
            "limit": self.limit,
            "has_more": self.has_more,
            "results": [hit.as_dict() for hit in self.hits],
        }


def parse_kinds(raw: str | Iterable[str] | None) -> tuple[NodeKind, ...]:
    """Accept node kinds or Cypher labels, comma-separated or as a list."""
    if raw is None:
        return ()
    parts = raw.split(",") if isinstance(raw, str) else list(raw)
    kinds: list[NodeKind] = []
    for part in (p.strip() for p in parts):
        if not part:
            continue
        matched = resolve_label(part)
        if matched is None:
            known = ", ".join(NODE_KINDS)
            raise ValueError(f"unknown kind {part!r}; expected one of {known} (or Callable)")
        kinds.extend(k for k in NODE_KINDS if k in matched and k not in kinds)
    return tuple(kinds)


def search_graph(store: GraphStore, query: GraphQuery) -> GraphPage:
    """Run ``query``; raises ValueError for a bad regex, sort key or page bounds."""
    if query.sort not in SORT_KEYS:
        raise ValueError(f"unknown sort {query.sort!r}; expected one of {', '.join(SORT_KEYS)}")
    if query.limit < 1 or query.offset < 0:
        raise ValueError("limit must be >= 1 and offset >= 0")
    flags = 0 if query.case_sensitive else re.IGNORECASE
    name_re = _compile(query.name_pattern, flags, "name_pattern")
    qname_re = _compile(query.qname_pattern, flags, "qname_pattern")
    view = GraphView.load(store, edge_kinds=("calls", "tests"))
    wanted = frozenset(query.kinds) if query.kinds else None
    hits: list[GraphHit] = []
    for node in view.nodes.values():
        if wanted is not None and node.kind not in wanted:
            continue
        if name_re is not None and not name_re.search(node.name):
            continue
        if qname_re is not None and not qname_re.search(node.qname):
            continue
        if query.file_glob is not None and not fnmatch.fnmatchcase(node.path, query.file_glob):
            continue
        if query.language is not None and node.language != query.language:
            continue
        hit = GraphHit(
            node=node,
            fan_in=view.in_degree(node.qname, ("calls",)),
            fan_out=view.out_degree(node.qname, ("calls",)),
            tests=view.in_degree(node.qname, ("tests",)),
        )
        if _keep(hit, query):
            hits.append(hit)
    hits.sort(key=lambda h: h.node.qname)
    if query.sort == "name":
        hits.sort(key=lambda h: h.node.name.lower())
    else:
        hits.sort(key=lambda h: _metric(h, query.sort), reverse=True)
    limit = min(query.limit, MAX_PAGE)
    page = hits[query.offset : query.offset + limit]
    return GraphPage(total=len(hits), offset=query.offset, limit=limit, hits=tuple(page))


def _keep(hit: GraphHit, query: GraphQuery) -> bool:
    bounds = (
        (query.min_fan_in, hit.fan_in, True),
        (query.max_fan_in, hit.fan_in, False),
        (query.min_fan_out, hit.fan_out, True),
        (query.max_fan_out, hit.fan_out, False),
    )
    for bound, value, is_min in bounds:
        if bound is None:
            continue
        if (is_min and value < bound) or (not is_min and value > bound):
            return False
    return query.has_tests is None or (hit.tests > 0) == query.has_tests


def _metric(hit: GraphHit, key: SortKey) -> int:
    if key == "fan_in":
        return hit.fan_in
    if key == "fan_out":
        return hit.fan_out
    return hit.node.span


def _compile(pattern: str | None, flags: int, what: str) -> re.Pattern[str] | None:
    if pattern is None or pattern == "":
        return None
    try:
        return re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"invalid {what} {pattern!r}: {exc}") from exc


__all__ = [
    "MAX_PAGE",
    "SORT_KEYS",
    "GraphHit",
    "GraphPage",
    "GraphQuery",
    "SortKey",
    "parse_kinds",
    "search_graph",
]
