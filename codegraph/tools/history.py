"""MCP tools over the git history pass: hotspots, co-change partners, per-file history."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from codegraph.services import reports
from codegraph.tools import ToolContext, dump


def register(server: MCPServer[None], ctx: ToolContext) -> None:
    @server.tool()
    def get_hotspots(limit: int = 20) -> str:
        """Files that change often, are large and are depended on: ranked by
        commits x log2(1 + lines) x log2(2 + fan-in). Needs git history."""
        if limit < 1:
            return dump({"error": "limit must be at least 1"})
        with ctx.open_store() as store:
            status = store.get_meta("git_history") or "git history pass has not run"
            return dump({"git_history": status, "hotspots": reports.hotspots(store, limit)})

    @server.tool()
    def get_co_changes(target: str, limit: int = 20) -> str:
        """Files that are usually committed together with target (a path or module qname),
        with how often and the coupling ratio."""
        with ctx.open_store() as store:
            partners = reports.co_changes_for(store, target, limit)
        if partners is None:
            return dump({"error": f"no file or module matches {target!r}"})
        return dump({"target": target, "partners": partners})

    @server.tool()
    def get_file_history(path: str, limit: int = 20) -> str:
        """Churn, recent commits and co-change partners for one file (or module qname)."""
        with ctx.open_store() as store:
            history = reports.file_history(store, path, limit)
        if history is None:
            return dump({"error": f"no history for {path!r}"})
        return dump(history)


__all__ = ["register"]
