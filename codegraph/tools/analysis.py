"""MCP tools for communities, flows, architecture and change impact.

Insight tools (hubs, bridges, gaps, ...) live in :mod:`codegraph.tools.analysis_insights`
and navigation tools in :mod:`codegraph.tools.analysis_navigation`. All of them read the
tables the communities and flows passes wrote, and compute in memory when those passes
have not run yet.
"""

from __future__ import annotations

from dataclasses import asdict

from mcp.server.mcpserver import MCPServer

from codegraph.analysis.architecture import architecture, change_impact
from codegraph.analysis.communities import find_community
from codegraph.analysis.flows import affected_flows, changed_qnames, find_flow
from codegraph.analysis.present import (
    communities_payload,
    community_detail_payload,
    flow_detail,
    flow_summary,
    flows_of,
    snapshot,
    sort_flows,
)
from codegraph.review import GitError
from codegraph.tools import ToolContext, dump


def register(server: MCPServer[None], ctx: ToolContext) -> None:
    @server.tool()
    def list_communities(min_size: int = 2, limit: int = 30) -> str:
        """Code communities (Louvain over calls, imports, inheritance and derived edges):
        name, size, cohesion, top members, plus the heaviest cross-community boundaries."""
        with ctx.open_store() as store:
            report = snapshot(store).report
        return dump(communities_payload(report, min_size=min_size, limit=limit))

    @server.tool()
    def get_community(community: str, limit: int = 100) -> str:
        """One community by id, name, name fragment or member qname: members and the
        communities it is coupled to."""
        with ctx.open_store() as store:
            snap = snapshot(store)
        found = find_community(snap.report, community)
        if found is None:
            return dump({"error": f"no community matches {community!r}"})
        return dump(community_detail_payload(snap.view, snap.report, found, limit=limit))

    @server.tool()
    def list_flows(limit: int = 20, sort_by: str = "criticality") -> str:
        """Execution flows from entry points (routes, scripts, main, framework hooks,
        uncalled roots). sort_by: criticality, depth, node_count, file_count, name."""
        with ctx.open_store() as store:
            flows = flows_of(store, ctx.root, snapshot(store))
        try:
            ordered = sort_flows(flows, sort_by)
        except ValueError as exc:
            return dump({"error": str(exc)})
        return dump(
            {"total": len(flows), "flows": [flow_summary(f) for f in ordered[: max(limit, 0)]]}
        )

    @server.tool()
    def get_flow(flow: str) -> str:
        """One flow by id, entry-point qname or name, with every step in BFS order."""
        with ctx.open_store() as store:
            snap = snapshot(store)
            flows = flows_of(store, ctx.root, snap)
        found = find_flow(flows, flow)
        if found is None:
            return dump({"error": f"no flow matches {flow!r}"})
        return dump(flow_detail(snap.view, found))

    @server.tool()
    def get_affected_flows(base: str | None = None, limit: int = 20) -> str:
        """Flows that pass through code changed in the diff (working tree vs HEAD, or
        base...HEAD when base is given)."""
        with ctx.open_store() as store:
            try:
                changed = changed_qnames(store, ctx.root, base)
            except GitError as exc:
                return dump({"error": str(exc)})
            flows = flows_of(store, ctx.root, snapshot(store))
        hits = affected_flows(flows, changed)
        return dump(
            {
                "changed": changed,
                "total": len(hits),
                "flows": [
                    {**flow_summary(h.flow), "touched": list(h.touched)}
                    for h in hits[: max(limit, 0)]
                ],
            }
        )

    @server.tool()
    def get_architecture(top: int = 10) -> str:
        """Languages, packages with layers, entry points, routes, hubs, communities with
        their boundaries, and churn hotspots when git history has been indexed."""
        with ctx.open_store() as store:
            return dump(asdict(architecture(store, ctx.root, top=top)))

    @server.tool()
    def get_change_impact(base: str | None = None) -> str:
        """Changed symbols with risk, affected flows, communities touched, untested
        changes and an overall risk level for the diff."""
        with ctx.open_store() as store:
            try:
                impact = change_impact(store, ctx.root, base)
            except GitError as exc:
                return dump({"error": str(exc)})
        return dump(asdict(impact))
