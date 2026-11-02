"""CLI commands: ``routes``, ``channels``, ``clones`` and ``ingest-traces``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from codegraph.commands import AddCommand
from codegraph.services import reports
from codegraph.services.traces import SpanValue, ingest
from codegraph.store import GraphStore
from codegraph.tools import dump


def cmd_routes(args: argparse.Namespace) -> int:
    with GraphStore.open(args.root) as store:
        found = reports.list_routes(store, args.method, args.path, args.framework, args.limit)
    if args.json:
        print(dump(found))
        return 0 if found else 1
    for route in found:
        handlers = ", ".join(route.handlers) or "(no handler found)"
        extra = f"  callers={route.callers}" if route.callers else ""
        extra += f"  observed={route.observed}" if route.observed else ""
        print(
            f"{route.method:<8} {route.path}  [{', '.join(route.frameworks)}]  -> {handlers}{extra}"
        )
        print(f"         {route.file}:{route.line}")
    if not found:
        print("no routes found", file=sys.stderr)
        return 1
    return 0


def cmd_channels(args: argparse.Namespace) -> int:
    with GraphStore.open(args.root) as store:
        found = reports.list_channels(store, args.name, args.broker, args.limit)
    if args.json:
        print(dump(found))
        return 0 if found else 1
    for channel in found:
        print(
            f"{channel.name}  [{', '.join(channel.brokers)}]  "
            f"emitters={channel.emitters} listeners={channel.listeners}  "
            f"({channel.file}:{channel.line})"
        )
    if not found:
        print("no channels found", file=sys.stderr)
        return 1
    return 0


def cmd_clones(args: argparse.Namespace) -> int:
    if not 0.0 <= args.min_score <= 1.0:
        print("--min-score must be between 0 and 1", file=sys.stderr)
        return 2
    with GraphStore.open(args.root) as store:
        pairs = reports.clone_pairs(store, args.target, args.min_score, args.limit)
    if pairs is None:
        print(f"no node matches {args.target!r}", file=sys.stderr)
        return 2
    if args.json:
        print(dump(pairs))
        return 0 if pairs else 1
    for pair in pairs:
        print(f"{pair.score:.2f}  {pair.a}  ({pair.a_location})")
        print(f"      {pair.b}  ({pair.b_location})")
    if not pairs:
        print("no clones found", file=sys.stderr)
        return 1
    return 0


def _load_spans(path: Path) -> list[dict[str, SpanValue]] | str:
    """Spans from a JSON file holding a list, or an object with a ``spans`` list."""
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return f"cannot read {path}: {exc}"
    except json.JSONDecodeError as exc:
        return f"{path} is not valid JSON: {exc}"
    if isinstance(payload, dict):
        payload = payload.get("spans")
    if not isinstance(payload, list):
        return f'{path} must contain a list of spans or {{"spans": [...]}}'
    spans: list[dict[str, SpanValue]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            return f"span {index} is not an object"
        span: dict[str, SpanValue] = {}
        for key, value in item.items():
            if isinstance(value, str | int | float | bool) or value is None:
                span[str(key)] = value
        spans.append(span)
    return spans


def cmd_ingest_traces(args: argparse.Namespace) -> int:
    spans = _load_spans(args.file)
    if isinstance(spans, str):
        print(spans, file=sys.stderr)
        return 2
    with GraphStore.open(args.root) as store:
        stats, errors = ingest(store, spans)
    if stats is None:
        for problem in errors:
            print(problem, file=sys.stderr)
        return 2
    if args.json:
        print(dump(stats))
    else:
        print(
            f"stored spans; {stats.spans} total: validated {stats.validated} edges, "
            f"added {stats.added}, {stats.unmatched} spans matched no route"
        )
    return 0


def register(add: AddCommand) -> None:
    p = add("routes", cmd_routes, "HTTP / gRPC / GraphQL routes and their handlers")
    p.add_argument("--method", help="only this method (GET, POST, GRPC, GRAPHQL, ...)")
    p.add_argument("--path", help="only paths containing this text")
    p.add_argument("--framework", help="only routes from this framework (flask, express, ...)")
    p.add_argument("--limit", type=int, default=200)

    p = add("channels", cmd_channels, "message / event channels with emitters and listeners")
    p.add_argument("--name", help="only channels whose name contains this text")
    p.add_argument("--broker", help="only this broker (kafka, socket.io, celery, ...)")
    p.add_argument("--limit", type=int, default=200)

    p = add("clones", cmd_clones, "near-duplicate functions")
    p.add_argument("target", nargs="?", help="only clones of this symbol")
    p.add_argument("--min-score", type=float, default=0.8)
    p.add_argument("--limit", type=int, default=50)

    p = add("ingest-traces", cmd_ingest_traces, "record runtime spans from a JSON file")
    p.add_argument(
        "file", type=Path, help="JSON list of {service, method, url|path, status, caller}"
    )


__all__ = ["register"]
