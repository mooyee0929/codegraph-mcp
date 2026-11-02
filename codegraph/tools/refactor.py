"""MCP tools for graph-guided refactoring: rename preview/apply and refactoring suggestions."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from codegraph import refactor
from codegraph.indexer import index_repository
from codegraph.refactor import RefactorError, RenamePreview
from codegraph.tools import ToolContext, dump


def preview_dict(preview: RenamePreview) -> dict[str, object]:
    return {
        "preview_id": preview.preview_id,
        "target": preview.target,
        "old_name": preview.old_name,
        "new_name": preview.new_name,
        "files": [f.path for f in preview.files],
        "edits": preview.edit_count,
        "skipped": list(preview.skipped),
        "expires_at": preview.expires_at,
        "diff": preview.diff,
    }


def register(server: MCPServer[None], ctx: ToolContext) -> None:
    @server.tool()
    def refactor_preview(target: str, new_name: str) -> str:
        """Preview renaming a function, method or class: its definition plus every reference
        the graph knows (calls, imports, inherits, tests), each confirmed by a whole-word
        match, as a unified diff. Returns a preview_id for apply_refactor (valid 10 min)."""
        with ctx.open_store() as store:
            try:
                preview = refactor.preview_rename(ctx.root, store, target, new_name)
            except RefactorError as exc:
                return dump({"error": str(exc)})
        return dump(preview_dict(preview))

    @server.tool()
    def apply_refactor(preview_id: str, reindex: bool = True) -> str:
        """Write a previewed refactor to disk. Refuses if the preview expired or any file it
        touches changed since; re-indexes afterwards unless reindex=false."""
        with ctx.open_store() as store:
            try:
                applied = refactor.apply_refactor(ctx.root, store, preview_id)
            except RefactorError as exc:
                return dump({"error": str(exc)})
            summary = index_repository(ctx.root, store).summary() if reindex else None
        return dump({"applied": applied, "index": summary})

    @server.tool()
    def suggest_refactorings(limit: int = 10, large_lines: int = 80) -> str:
        """Refactoring candidates from graph shape: near-duplicates (similar_to edges), functions
        whose callers mostly live in another module, functions longer than large_lines, and
        dead code. Up to limit of each kind; candidates, not verdicts."""
        if limit < 1 or large_lines < 1:
            return dump({"error": "limit and large_lines must be positive"})
        with ctx.open_store() as store:
            found = refactor.suggest_refactorings(store, limit=limit, large_lines=large_lines)
        return dump(found)


__all__ = ["preview_dict", "register"]
