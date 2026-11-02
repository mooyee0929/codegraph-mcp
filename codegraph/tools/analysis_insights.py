"""MCP tools for graph insights: hubs, bridges, gaps, surprises, questions, large code."""

from __future__ import annotations

from dataclasses import asdict

from mcp.server.mcpserver import MCPServer

from codegraph.analysis.centrality import DEFAULT_MAX_SOURCES, bridge_nodes, hub_nodes
from codegraph.analysis.common import SYMBOL_KINDS
from codegraph.analysis.insights import (
    knowledge_gaps,
    large_functions,
    suggested_questions,
    surprising_connections,
)
from codegraph.analysis.present import flows_of, parse_node_kinds, snapshot
from codegraph.tools import ToolContext, dump


def register(server: MCPServer[None], ctx: ToolContext) -> None:
    @server.tool()
    def get_hub_nodes(top: int = 10, kinds: list[str] | None = None) -> str:
        """Most connected production symbols (fan-in excludes tests). kinds defaults to
        class, function, method."""
        try:
            wanted = parse_node_kinds(kinds) or SYMBOL_KINDS
        except ValueError as exc:
            return dump({"error": str(exc)})
        with ctx.open_store() as store:
            snap = snapshot(store)
        return dump(hub_nodes(snap.view, top=top, kinds=wanted, membership=snap.report.membership))

    @server.tool()
    def get_bridge_nodes(top: int = 10, max_sources: int = DEFAULT_MAX_SOURCES) -> str:
        """Chokepoints by betweenness centrality; sampled deterministically above
        max_sources nodes."""
        if max_sources < 1:
            return dump({"error": "max_sources must be at least 1"})
        with ctx.open_store() as store:
            snap = snapshot(store)
        return dump(
            bridge_nodes(
                snap.view, top=top, max_sources=max_sources, membership=snap.report.membership
            )
        )

    @server.tool()
    def get_knowledge_gaps(limit: int = 20) -> str:
        """Untested hubs, isolated code, tiny communities, undocumented modules and files
        full of unresolved calls."""
        with ctx.open_store() as store:
            snap = snapshot(store)
        gaps = knowledge_gaps(snap.view, snap.report, limit=limit)
        return dump({"total": gaps.total, **asdict(gaps)})

    @server.tool()
    def get_surprising_connections(top: int = 15) -> str:
        """Edges between weakly coupled communities or distant packages, most
        unexpected first."""
        with ctx.open_store() as store:
            snap = snapshot(store)
        return dump(surprising_connections(snap.view, snap.report, top=top))

    @server.tool()
    def get_suggested_questions(limit: int = 15) -> str:
        """Review questions generated from bridges, untested hubs, critical flows,
        surprising couplings and knowledge gaps."""
        with ctx.open_store() as store:
            snap = snapshot(store)
            flows = flows_of(store, ctx.root, snap)
        membership = snap.report.membership
        questions = suggested_questions(
            snap.view,
            snap.report,
            hubs=hub_nodes(snap.view, top=5, membership=membership),
            bridges=bridge_nodes(snap.view, top=3, membership=membership),
            flows=flows,
            gaps=knowledge_gaps(snap.view, snap.report),
            surprises=surprising_connections(snap.view, snap.report, top=3),
            limit=limit,
        )
        return dump(questions)

    @server.tool()
    def find_large_functions(
        min_lines: int = 50,
        kinds: list[str] | None = None,
        path: str | None = None,
        language: str | None = None,
        include_tests: bool = False,
        limit: int = 30,
    ) -> str:
        """Definitions spanning at least min_lines lines. kinds defaults to function and
        method; path is a prefix or a glob."""
        if min_lines < 1:
            return dump({"error": "min_lines must be at least 1"})
        try:
            wanted = parse_node_kinds(kinds) or ("function", "method")
        except ValueError as exc:
            return dump({"error": str(exc)})
        with ctx.open_store() as store:
            nodes = store.all_nodes(kinds=wanted)
        return dump(
            large_functions(
                nodes,
                min_lines=min_lines,
                kinds=wanted,
                path=path,
                language=language,
                include_tests=include_tests,
                limit=limit,
            )
        )
