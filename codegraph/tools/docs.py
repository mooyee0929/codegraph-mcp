"""MCP tools that write documents: ADRs, notes, the wiki, the visualisation and exports."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from mcp.server.mcpserver import MCPServer

from codegraph import adr, exports, notes, visualize, wiki
from codegraph.store import DB_DIRNAME
from codegraph.tools import ToolContext, dump

AdrAction = Literal["create", "list", "get", "update", "supersede"]
EXPORTS_DIRNAME = "exports"
_PAGE_LIST_LIMIT = 50


def note_dict(note: notes.Note, root: Path) -> dict[str, object]:
    return {
        "path": note.path.relative_to(root).as_posix(),
        "question": note.question,
        "answer": note.answer,
        "qnames": list(note.qnames),
        "kind": note.kind,
        "created": note.created,
    }


def inside(root: Path, raw: str) -> Path | None:
    """``raw`` resolved against ``root``, or None when it escapes the repository."""
    path = (root / raw).resolve()
    return path if path.is_relative_to(root.resolve()) else None


def register(server: MCPServer[None], ctx: ToolContext) -> None:
    root = ctx.root

    @server.tool()
    def manage_adr(
        action: AdrAction,
        number: int = 0,
        title: str = "",
        status: str = "",
        context: str = "",
        decision: str = "",
        consequences: str = "",
        directory: str = "",
    ) -> str:
        """Architecture decision records as markdown in docs/adr/NNNN-slug.md (Nygard format).
        list | get(number) | create(title, context, decision, consequences, status?) |
        update(number, any field; empty means unchanged) | supersede(number, title, ...)
        which adds an accepted ADR and marks the old one superseded. Status is one of
        Proposed, Accepted, Rejected, Deprecated, Superseded. directory overrides docs/adr
        (or the path in .adr-dir)."""
        try:
            return dump(
                adr.manage_adr(
                    root,
                    action,
                    number,
                    title=title,
                    status=status,
                    context=context,
                    decision=decision,
                    consequences=consequences,
                    directory=directory,
                )
            )
        except adr.AdrError as exc:
            return dump({"error": str(exc)})

    @server.tool()
    def save_note(
        question: str, answer: str, qnames: list[str] | None = None, kind: str = "answer"
    ) -> str:
        """Remember an answer for later sessions: writes .codegraph/notes/<time>-<slug>.md with
        the question, the answer and the related qnames."""
        try:
            note = notes.save_note(root, question, answer, tuple(qnames or ()), kind=kind)
        except notes.NoteError as exc:
            return dump({"error": str(exc)})
        return dump({"saved": note_dict(note, root)})

    @server.tool()
    def search_notes(query: str, limit: int = 10) -> str:
        """Search saved notes by words in the question, answer and related qnames."""
        if not query.strip():
            return dump({"error": "query must not be empty"})
        hits = notes.search_notes(root, query, limit=limit)
        return dump([{"score": h.score, **note_dict(h.note, root)} for h in hits])

    @server.tool()
    def generate_wiki() -> str:
        """Write a markdown wiki of the graph to .codegraph/wiki/: an index page, one page per
        module and one per community (when communities were computed). No LLM involved."""
        with ctx.open_store() as store:
            result = wiki.generate_wiki(root, store)
        return dump(
            {
                "directory": result.directory.relative_to(root).as_posix(),
                "page_count": len(result.pages),
                "pages": list(result.pages[:_PAGE_LIST_LIMIT]),
                "removed": list(result.removed),
            }
        )

    @server.tool()
    def get_wiki_page(page: str = "index") -> str:
        """Read one wiki page: index, modules/<qname>, a module qname, or a community label."""
        try:
            text = wiki.read_wiki_page(root, page)
        except wiki.WikiError as exc:
            return dump({"error": str(exc)})
        if text is None:
            available = wiki.list_wiki_pages(root)
            hint = "run generate_wiki first" if not available else "see available"
            return dump(
                {
                    "error": f"no wiki page {page!r}; {hint}",
                    "available": available[:_PAGE_LIST_LIMIT],
                }
            )
        return dump({"page": page, "content": text})

    @server.tool()
    def generate_visualization(
        max_nodes: int = visualize.DEFAULT_MAX_NODES, color_by: str = "auto", output: str = ""
    ) -> str:
        """Write a self-contained interactive HTML graph (no network needed) to
        .codegraph/graph.html or output. Keeps the max_nodes highest-degree nodes;
        color_by is package, kind, community or auto."""
        target = inside(root, output) if output else None
        if output and target is None:
            return dump({"error": f"output {output!r} is outside the repository"})
        try:
            with ctx.open_store() as store:
                path, data = visualize.write_visualization(
                    root, store, out=target, max_nodes=max_nodes, color_by=color_by
                )
        except visualize.VisualizeError as exc:
            return dump({"error": str(exc)})
        return dump(
            {
                "path": path.relative_to(root).as_posix(),
                "nodes": len(data.nodes),
                "total_nodes": data.total_nodes,
                "truncated": data.truncated,
            }
        )

    @server.tool()
    def export_graph(format: str = "json", max_nodes: int = 0, inline: bool = False) -> str:
        """Export the graph as json, graphml, dot, mermaid, csv (nodes.csv + edges.csv) or
        cypher (Neo4j MERGE statements). Writes to .codegraph/exports/ unless inline=true.
        max_nodes keeps the highest-degree nodes (0 = all; mermaid defaults to 150)."""
        try:
            fmt = exports.normalise_format(format)
        except exports.ExportError as exc:
            return dump({"error": str(exc)})
        cap = max_nodes or (exports.MERMAID_DEFAULT_CAP if fmt == "mermaid" else 0)
        with ctx.open_store() as store:
            data = exports.load_graph(store, max_nodes=cap or None)
        files = exports.render(data, fmt)
        meta = {"format": fmt, "nodes": len(data.nodes), "total_nodes": data.total_nodes}
        if inline:
            return dump({**meta, "files": files})
        out_dir = root / DB_DIRNAME / EXPORTS_DIRNAME
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for name, content in files.items():
            (out_dir / name).write_text(content, encoding="utf-8")
            written.append((out_dir / name).relative_to(root).as_posix())
        return dump({**meta, "written": written})


__all__ = ["inside", "note_dict", "register"]
