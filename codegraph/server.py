"""MCP server exposing the graph over stdio.

Run with ``codegraph-mcp --root /path/to/repo`` (or let it default to the current
directory). Every tool takes and returns plain JSON so any MCP client can use it.

The core tools live here; feature tools are contributed by the modules in
:mod:`codegraph.tools`. ``--tools a,b,c`` (or ``CODEGRAPH_TOOLS``) exposes only the
named tools, which keeps the tool list short for clients that pay per tool schema.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Literal, cast

from mcp.server.mcpserver import MCPServer

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
from codegraph.store import GraphStore, StoreError
from codegraph.tools import ToolContext, register_all
from codegraph.tools import dump as _dump
from codegraph.tools import node_summary as _node_summary
from codegraph.watch import AUTO_INDEX_ENV, WATCH_ENV, env_flag, start_background_sync

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


def build_server(root: Path, tools: frozenset[str] | None = None) -> MCPServer[None]:
    """Build the server for ``root``. ``tools`` limits the exposed tools to those names."""
    root = root.resolve()
    ctx = ToolContext(root=root)
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
        return ctx.open_store()

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

    register_all(server, ctx)
    if tools is not None:
        _restrict_tools(server, tools)
    return server


def _registered_tool_names(server: MCPServer[None]) -> set[str]:
    # MCPServer.list_tools is async; build_server may run inside an event loop, so
    # read the synchronous tool manager instead.
    manager = getattr(server, "_tool_manager", None)
    list_tools = getattr(manager, "list_tools", None)
    if not callable(list_tools):
        raise RuntimeError("cannot enumerate tools on this mcp version; drop --tools")
    listed: list[object] = list_tools()
    return {str(getattr(tool, "name", "")) for tool in listed}


def _restrict_tools(server: MCPServer[None], allowed: frozenset[str]) -> None:
    registered = _registered_tool_names(server)
    unknown = allowed - registered
    if unknown:
        raise ValueError(f"unknown tool(s) in --tools: {', '.join(sorted(unknown))}")
    for name in registered - allowed:
        server.remove_tool(name)


def parse_tool_filter(raw: str | None) -> frozenset[str] | None:
    """``"a, b"`` -> ``{"a", "b"}``; empty or missing -> None (expose everything)."""
    if not raw:
        return None
    names = frozenset(part.strip() for part in raw.split(",") if part.strip())
    return names or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="codegraph-mcp", description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository to serve")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
    )
    parser.add_argument(
        "--tools",
        default=os.environ.get("CODEGRAPH_TOOLS"),
        help="comma-separated tool names to expose (default: all; env CODEGRAPH_TOOLS)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        default=env_flag(WATCH_ENV),
        help=f"re-index in the background when files change (env {WATCH_ENV}=1)",
    )
    parser.add_argument(
        "--auto-index",
        action="store_true",
        default=env_flag(AUTO_INDEX_ENV),
        help="index on startup when the graph is missing, empty or stale, restoring "
        f".codegraph/graph.db.gz first when there is no database (env {AUTO_INDEX_ENV}=1)",
    )
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(f"not a directory: {args.root}", file=sys.stderr)
        return 2
    try:
        server = build_server(args.root, tools=parse_tool_filter(args.tools))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sync = start_background_sync(args.root.resolve(), auto=args.auto_index, watch=args.watch)
    try:
        server.run(transport=cast(Literal["stdio", "sse", "streamable-http"], args.transport))
    finally:
        if sync is not None:
            sync.stop()
    return 0


__all__ = ["PATTERNS", "build_server", "main", "parse_tool_filter"]


if __name__ == "__main__":
    raise SystemExit(main())
