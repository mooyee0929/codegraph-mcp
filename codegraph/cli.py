"""Command-line front end. Same operations as the MCP server, human-readable output."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from codegraph import __version__
from codegraph.indexer import index_repository
from codegraph.queries import (
    PATTERNS,
    answer_side,
    architecture_overview,
    dead_code,
    impact_radius,
    query,
    resolve_target,
)
from codegraph.review import GitError, detect_changes, review_context
from codegraph.store import GraphStore, StoreError, default_db_path


def _open(root: Path) -> GraphStore:
    return GraphStore(default_db_path(root))


def cmd_index(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        stats = index_repository(args.root, store, force=args.force)
    print(stats.summary())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        counts = store.counts()
        print(f"db: {store.db_path}")
    for key, value in counts.items():
        print(f"{key:>17}: {value}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        hits = store.search_nodes(args.query, limit=args.limit)
    if args.json:
        print(json.dumps([asdict(n) for n in hits], indent=2))
        return 0
    for node in hits:
        print(f"{node.kind:<8} {node.qname}  ({node.path}:{node.line_start})")
        if node.signature:
            print(f"         {node.signature}")
    if not hits:
        print("no matches", file=sys.stderr)
        return 1
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        edges = query(store, args.pattern, args.target)
    side = answer_side(args.pattern)
    if args.json:
        print(json.dumps([asdict(e) for e in edges], indent=2))
        return 0
    seen: set[str] = set()
    for edge in sorted(edges, key=lambda e: (e.path, e.line)):
        name = edge.src if side == "src" else edge.dst
        marker = "" if edge.resolved else "  (unresolved)"
        if name in seen:
            continue
        seen.add(name)
        print(f"{name}  ({edge.path}:{edge.line}){marker}")
    if not edges:
        print("no results", file=sys.stderr)
        return 1
    return 0


def cmd_impact(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        report = impact_radius(store, args.target, depth=args.depth)
    if report is None:
        print(f"no node matches {args.target!r}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({**asdict(report), "total_affected": report.total_affected}, indent=2))
        return 0
    print(f"target: {report.target}  (depth {report.depth})")
    _print_list("direct callers", report.direct_callers)
    _print_list("transitive callers", report.transitive_callers)
    _print_list("importing modules", report.importing_modules)
    _print_list("covering tests", report.covering_tests)
    print(f"total affected: {report.total_affected}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        try:
            changed = detect_changes(store, args.root, args.base)
        except GitError as exc:
            print(f"git: {exc}", file=sys.stderr)
            return 2
        if args.json:
            payload = [
                {
                    **asdict(c.node),
                    "changed_lines": list(c.changed_lines),
                    "fan_in": c.fan_in,
                    "fan_out": c.fan_out,
                    "covering_tests": list(c.covering_tests),
                    "risk": c.risk,
                    "reasons": list(c.reasons),
                }
                for c in changed
            ]
            print(json.dumps(payload, indent=2))
            return 0
        if not changed:
            print("no indexed symbols overlap the diff")
            return 0
        for item in changed:
            reasons = f"  [{', '.join(item.reasons)}]" if item.reasons else ""
            print(
                f"{item.risk:>5.1f}  {item.node.kind:<8} {item.node.qname}  "
                f"({item.node.path}:{item.node.line_start}){reasons}"
            )
        if args.context:
            print()
            for snippet in review_context(store, args.root, [c.node.qname for c in changed]):
                print(
                    f"--- {snippet.qname}  {snippet.path}:{snippet.line_start}-{snippet.line_end}"
                )
                print(snippet.source)
                if snippet.callers:
                    print(f"    callers: {', '.join(snippet.callers)}")
                if snippet.tests:
                    print(f"    tests:   {', '.join(snippet.tests)}")
                print()
    return 0


def cmd_arch(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        overview = architecture_overview(store, top=args.top)
    if args.json:
        print(json.dumps(asdict(overview), indent=2))
        return 0
    print("counts:   " + ", ".join(f"{k}={v}" for k, v in overview.counts.items()))
    print("languages: " + ", ".join(f"{k}={v}" for k, v in overview.languages.items()))
    print("packages: " + ", ".join(f"{k}({v})" for k, v in overview.top_level_packages.items()))
    print("hubs:")
    for hub in overview.hubs:
        print(f"  {hub.qname}  in={hub.fan_in} out={hub.fan_out}")
    _print_list("entry points", overview.entry_points)
    _print_list("orphans", overview.orphans)
    _print_list("external imports", tuple(f"{n} ({c})" for n, c in overview.external_imports))
    return 0


def cmd_dead(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        candidates = dead_code(store)
    for node in candidates[: args.limit]:
        print(f"{node.kind:<8} {node.qname}  ({node.path}:{node.line_start})")
    print(f"{len(candidates)} candidates", file=sys.stderr)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        node = resolve_target(store, args.target)
        if node is None:
            print(f"no node matches {args.target!r}", file=sys.stderr)
            return 1
        snippets = review_context(store, args.root, [node.qname], context_lines=args.context)
    for snippet in snippets:
        print(f"{snippet.qname}  {snippet.path}:{snippet.line_start}-{snippet.line_end}")
        print(snippet.source)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from codegraph.server import main as serve_main

    return serve_main(["--root", str(args.root), "--transport", args.transport])


def _print_list(label: str, items: tuple[str, ...]) -> None:
    print(f"{label} ({len(items)}):")
    for item in items:
        print(f"  {item}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codegraph", description=__doc__)
    parser.add_argument("--version", action="version", version=f"codegraph {__version__}")
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="repository root (default: cwd)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(
        name: str, handler: Callable[[argparse.Namespace], int], help_text: str
    ) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(handler=handler)
        p.add_argument("--json", action="store_true", help="machine-readable output")
        return p

    p = add("index", cmd_index, "build or refresh the graph")
    p.add_argument("--force", action="store_true", help="re-parse unchanged files too")

    add("status", cmd_status, "row counts and db location")

    p = add("search", cmd_search, "find symbols by keyword")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=20)

    p = add("query", cmd_query, "trace a relationship")
    p.add_argument("pattern", choices=PATTERNS)
    p.add_argument("target", help="qname, unique bare name, or path:line")

    p = add("impact", cmd_impact, "blast radius of changing a symbol")
    p.add_argument("target")
    p.add_argument("--depth", type=int, default=3)

    p = add("review", cmd_review, "risk-ranked symbols touched by the diff")
    p.add_argument("--base", help="git revision to diff HEAD against (default: working tree)")
    p.add_argument("--context", action="store_true", help="print source snippets too")

    p = add("arch", cmd_arch, "architecture overview")
    p.add_argument("--top", type=int, default=10)

    p = add("dead", cmd_dead, "callables nothing references")
    p.add_argument("--limit", type=int, default=50)

    p = add("show", cmd_show, "print a symbol's source")
    p.add_argument("target")
    p.add_argument("--context", type=int, default=2)

    p = add("serve", cmd_serve, "run the MCP server")
    p.add_argument("--transport", choices=("stdio", "sse", "streamable-http"), default="stdio")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.root = args.root.resolve()
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except StoreError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
