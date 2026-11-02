"""MCP tools for operating on graphs: the multi-repo registry and team snapshots."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from codegraph import registry, snapshot
from codegraph.registry import RegistryError, RepoStatus
from codegraph.store import StoreError
from codegraph.tools import ToolContext, dump, node_summary


def status_dict(status: RepoStatus) -> dict[str, object]:
    when = status.last_indexed
    iso = datetime.fromtimestamp(when, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if when else None
    return {**asdict(status), "last_indexed_iso": iso}


def _entries() -> list[registry.RepoEntry]:
    return registry.load_registry(registry.registry_path())


def register(server: MCPServer[None], ctx: ToolContext) -> None:
    @server.tool()
    def list_repos() -> str:
        """Registered repositories with file/node/edge counts and last-indexed time."""
        try:
            entries = _entries()
        except RegistryError as exc:
            return dump({"error": str(exc)})
        return dump(
            {
                "registry": str(registry.registry_path()),
                "repos": [status_dict(registry.repo_status(e)) for e in entries],
            }
        )

    @server.tool()
    def register_repo(path: str = "", name: str = "") -> str:
        """Add a repository to the shared registry (default: the one this server serves).
        Re-registering a path renames it."""
        root = Path(path).expanduser() if path else ctx.root
        root = root if root.is_absolute() else ctx.root / root
        try:
            entry = registry.register_repo(registry.registry_path(), root, name or None)
        except RegistryError as exc:
            return dump({"error": str(exc)})
        return dump({"registered": entry, "status": status_dict(registry.repo_status(entry))})

    @server.tool()
    def cross_repo_search(query: str, limit: int = 20) -> str:
        """Full-text symbol search across every registered repository's graph."""
        try:
            hits = registry.cross_repo_search(_entries(), query, limit=limit)
        except RegistryError as exc:
            return dump({"error": str(exc)})
        return dump([{"repo": h.repo, "score": h.score, **node_summary(h.node)} for h in hits])

    @server.tool()
    def cross_repo_links(repo: str = "", min_confidence: float = 0.5) -> str:
        """Match each registered repo's unresolved outbound HTTP calls (http_calls edges to
        route qnames) to the routes another registered repo serves. Path parameters such
        as {id}, :id or <id> match any segment. repo limits the callers to one repo."""
        try:
            entries = _entries()
        except RegistryError as exc:
            return dump({"error": str(exc)})
        source = registry.find_repo(entries, repo) if repo else None
        if repo and source is None:
            return dump({"error": f"no registered repo {repo!r}"})
        links = registry.cross_repo_links(
            entries, source=source.name if source else None, min_confidence=min_confidence
        )
        return dump({"links": links, "count": len(links)})

    @server.tool()
    def delete_project(repo: str, confirm: bool = False) -> str:
        """Delete a registered repo's graph database (.codegraph/graph.db and its WAL files).
        Without confirm=true nothing is deleted and the files are only listed."""
        try:
            entry = registry.find_repo(_entries(), repo)
        except RegistryError as exc:
            return dump({"error": str(exc)})
        if entry is None:
            return dump({"error": f"no registered repo {repo!r}"})
        result = registry.delete_project_db(entry, confirm=confirm)
        if not confirm:
            return dump(
                {
                    "confirm_required": True,
                    "would_delete": result.files,
                    "hint": "call again with confirm=true to delete these files",
                }
            )
        return dump(result)

    @server.tool()
    def export_snapshot() -> str:
        """Write .codegraph/graph.db.gz, a compressed copy of the graph to commit for the team,
        and add its merge rule to .gitattributes on first export."""
        try:
            with ctx.open_store() as store:
                result = snapshot.export_snapshot(ctx.root, store)
        except (snapshot.SnapshotError, StoreError, sqlite3.Error) as exc:
            return dump({"error": str(exc)})
        return dump(result)

    @server.tool()
    def import_snapshot() -> str:
        """Restore the graph from .codegraph/graph.db.gz after an integrity and schema check,
        then index whatever changed since the snapshot."""
        try:
            result = snapshot.import_snapshot(ctx.root)
        except (snapshot.SnapshotError, StoreError, sqlite3.Error) as exc:
            return dump({"error": str(exc)})
        return dump({**asdict(result), "summary": result.index.summary()})


__all__ = ["register", "status_dict"]
