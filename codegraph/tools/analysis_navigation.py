"""MCP tools for walking the graph: traversal trees, call chains and minimal context."""

from __future__ import annotations

from typing import cast

from mcp.server.mcpserver import MCPServer

from codegraph.analysis.context import minimal_context
from codegraph.analysis.navigation import (
    DIRECTIONS,
    MAX_TRACE_DEPTH,
    MODES,
    TRACE_DIRECTIONS,
    Direction,
    Mode,
    TraceDirection,
    trace_paths,
    traverse,
)
from codegraph.analysis.present import parse_edge_kinds
from codegraph.graph import GraphView
from codegraph.queries import resolve_target
from codegraph.tools import ToolContext, dump

MAX_TRAVERSE_DEPTH = 10


def register(server: MCPServer[None], ctx: ToolContext) -> None:
    @server.tool()
    def traverse_graph(
        start: str,
        edge_kinds: list[str] | None = None,
        direction: str = "out",
        depth: int = 3,
        mode: str = "bfs",
        max_nodes: int = 200,
    ) -> str:
        """Spanning tree from a symbol over chosen edge kinds (default: all present).
        direction: out, in, both; mode: bfs or dfs."""
        if direction not in DIRECTIONS:
            return dump({"error": f"direction must be one of {', '.join(DIRECTIONS)}"})
        if mode not in MODES:
            return dump({"error": f"mode must be one of {', '.join(MODES)}"})
        if not 1 <= depth <= MAX_TRAVERSE_DEPTH:
            return dump({"error": f"depth must be between 1 and {MAX_TRAVERSE_DEPTH}"})
        try:
            kinds = parse_edge_kinds(edge_kinds)
        except ValueError as exc:
            return dump({"error": str(exc)})
        with ctx.open_store() as store:
            node = resolve_target(store, start)
            if node is None:
                return dump({"error": f"no node matches {start!r}"})
            view = GraphView.load(store)
        result = traverse(
            view,
            node.qname,
            edge_kinds=kinds,
            direction=cast(Direction, direction),
            depth=depth,
            mode=cast(Mode, mode),
            max_nodes=max_nodes,
        )
        return dump(result)

    @server.tool()
    def trace_path(
        target: str,
        to: str | None = None,
        direction: str = "both",
        depth: int = 3,
        max_paths: int = 50,
        include_tests: bool = False,
    ) -> str:
        """Inbound and/or outbound call chains of a symbol (depth 1-5), callers ranked
        by hop risk, and the shortest path to `to` when given."""
        if direction not in TRACE_DIRECTIONS:
            return dump({"error": f"direction must be one of {', '.join(TRACE_DIRECTIONS)}"})
        if not 1 <= depth <= MAX_TRACE_DEPTH:
            return dump({"error": f"depth must be between 1 and {MAX_TRACE_DEPTH}"})
        with ctx.open_store() as store:
            node = resolve_target(store, target)
            if node is None:
                return dump({"error": f"no node matches {target!r}"})
            other = resolve_target(store, to) if to is not None else None
            if to is not None and other is None:
                return dump({"error": f"no node matches {to!r}"})
            view = GraphView.load(store)
        result = trace_paths(
            view,
            node.qname,
            direction=cast(TraceDirection, direction),
            depth=depth,
            max_paths=max_paths,
            include_tests=include_tests,
            to=other.qname if other is not None else None,
        )
        return dump(result)

    @server.tool()
    def get_minimal_context(
        target: str | None = None, task: str | None = None, max_tokens: int = 800
    ) -> str:
        """Token-bounded starting bundle for a symbol or a free-text task: signature,
        callers, callees, tests, community, flows and the tools to call next."""
        if not target and not task:
            return dump({"error": "give a target, a task, or both"})
        if max_tokens < 50:
            return dump({"error": "max_tokens must be at least 50"})
        with ctx.open_store() as store:
            bundle = minimal_context(
                store, ctx.root, target=target, task=task, max_tokens=max_tokens
            )
        if bundle is None:
            return dump({"error": f"no node matches {target or task!r}"})
        return dump(bundle)
