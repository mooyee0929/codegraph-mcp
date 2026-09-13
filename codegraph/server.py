"""MCP server exposing the graph over stdio.

Run with ``codegraph-mcp --root /path/to/repo`` (or let it default to the current
directory). Every tool takes and returns plain JSON so any MCP client can use it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Literal, cast

from mcp.server.mcpserver import MCPServer

from codegraph.indexer import index_repository
from codegraph.models import Node
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

QueryPattern = Literal[
    "callers_of",
    "callees_of",
    "imports_of",
    "imported_by",
    "tests_for",
    "subclasses_of",
    "defines",
    "defined_in",
]


def _dump(value: object) -> str:
    def default(obj: object) -> object:
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        if isinstance(obj, tuple | set | frozenset):
            return list(obj)
        raise TypeError(f"cannot serialise {type(obj).__name__}")

    return json.dumps(value, default=default, indent=2)


def _node_summary(node: Node) -> dict[str, object]:
    return {
        "qname": node.qname,
        "kind": node.kind,
        "path": node.path,
        "lines": [node.line_start, node.line_end],
        "signature": node.signature,
        "docstring": node.docstring,
    }


def build_server(root: Path) -> MCPServer[None]:
    root = root.resolve()
    server: MCPServer[None] = MCPServer(
        "codegraph",
        instructions=(
            "Structural knowledge of the repository at "
            f"{root}. Prefer these tools over grep for questions about callers, "
            "callees, imports, tests and blast radius. Call index_repository first if "
            "graph_status reports zero files."
        ),
    )

    def _store() -> GraphStore:
        return GraphStore(default_db_path(root))

    @server.tool()
    def index_repository_tool(force: bool = False) -> str:
        """Index or refresh the repository graph. Incremental unless force=true."""
        with _store() as store:
            stats = index_repository(root, store, force=force)
        return _dump({"summary": stats.summary(), **asdict(stats)})

    @server.tool()
    def graph_status() -> str:
        """Row counts and where the database lives."""
        try:
            with _store() as store:
                return _dump({"db": str(store.db_path), "root": str(root), **store.counts()})
        except StoreError as exc:
            return _dump({"error": str(exc)})

    @server.tool()
    def search_nodes(query_text: str, limit: int = 20) -> str:
        """Find functions, classes and modules by name, signature or docstring keyword."""
        with _store() as store:
            hits = store.search_nodes(query_text, limit=limit)
        return _dump([_node_summary(n) for n in hits])

    @server.tool()
    def get_node(target: str) -> str:
        """Look up one symbol by qname, unique bare name, or path:line."""
        with _store() as store:
            node = resolve_target(store, target)
            if node is None:
                return _dump({"error": f"no node matches {target!r}"})
            outgoing = [asdict(e) for e in store.edges_from(node.qname)]
            incoming = [asdict(e) for e in store.edges_to(node.qname)]
        return _dump({**_node_summary(node), "outgoing": outgoing, "incoming": incoming})

    @server.tool()
    def query_graph(pattern: QueryPattern, target: str) -> str:
        """Trace one relationship. Patterns: callers_of, callees_of, imports_of, imported_by,
        tests_for, subclasses_of, defines, defined_in."""
        with _store() as store:
            edges = query(store, pattern, target)
            side = answer_side(pattern)
            names = sorted({e.src if side == "src" else e.dst for e in edges})
            details = [asdict(e) for e in edges]
        return _dump({"pattern": pattern, "target": target, "results": names, "edges": details})

    @server.tool()
    def get_impact_radius(target: str, depth: int = 3) -> str:
        """Blast radius of changing a symbol: direct and transitive callers, importers, tests."""
        with _store() as store:
            report = impact_radius(store, target, depth=depth)
        if report is None:
            return _dump({"error": f"no node matches {target!r}"})
        return _dump({**asdict(report), "total_affected": report.total_affected})

    @server.tool()
    def detect_changes_tool(base: str | None = None) -> str:
        """Risk-ranked symbols touched by the current diff (working tree vs HEAD, or
        base..HEAD when base is given)."""
        with _store() as store:
            try:
                changed = detect_changes(store, root, base)
            except GitError as exc:
                return _dump({"error": str(exc)})
        return _dump(
            [
                {
                    **_node_summary(c.node),
                    "changed_lines": list(c.changed_lines),
                    "fan_in": c.fan_in,
                    "fan_out": c.fan_out,
                    "covering_tests": list(c.covering_tests),
                    "risk": c.risk,
                    "reasons": list(c.reasons),
                }
                for c in changed
            ]
        )

    @server.tool()
    def get_review_context(qnames: list[str], context_lines: int = 2) -> str:
        """Source snippets plus callers and tests for the given symbols."""
        with _store() as store:
            snippets = review_context(store, root, qnames, context_lines=context_lines)
        return _dump(snippets)

    @server.tool()
    def get_architecture_overview(top: int = 10) -> str:
        """Languages, top-level packages, hub functions, entry points and external deps."""
        with _store() as store:
            return _dump(architecture_overview(store, top=top))

    @server.tool()
    def find_dead_code(limit: int = 50) -> str:
        """Callables with no callers, tests or subclasses. Candidates, not verdicts."""
        with _store() as store:
            return _dump([_node_summary(n) for n in dead_code(store)[:limit]])

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="codegraph-mcp", description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository to serve")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
    )
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(f"not a directory: {args.root}", file=sys.stderr)
        return 2
    server = build_server(args.root)
    server.run(transport=cast(Literal["stdio", "sse", "streamable-http"], args.transport))
    return 0


__all__ = ["PATTERNS", "build_server", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
