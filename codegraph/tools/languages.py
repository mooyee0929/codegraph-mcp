"""MCP tools for the language and infrastructure coverage of the index.

``list_languages`` tells an agent which files codegraph understands (and how many of
each the index holds) before it trusts an empty answer; ``list_resources`` exposes the
Dockerfile stages, compose services, Kubernetes objects and Terraform blocks with
their ``depends_on`` edges.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from codegraph.parsers import inventory
from codegraph.store import StoreError
from codegraph.tools import ToolContext, dump


def register(server: MCPServer[None], ctx: ToolContext) -> None:
    @server.tool()
    def list_languages(indexed_only: bool = False) -> str:
        """Supported languages with their file suffixes/names and how many files of each
        the index holds. indexed_only=true hides languages with no indexed files."""
        try:
            with ctx.open_store() as store:
                rows = inventory.language_inventory(store, indexed_only=indexed_only)
        except StoreError as exc:
            return dump({"error": str(exc)})
        return dump({"languages": rows, "total_files": sum(r.files for r in rows)})

    @server.tool()
    def list_resources(resource_type: str | None = None) -> str:
        """Infrastructure resources (Dockerfile stages, compose services, Kubernetes objects,
        Terraform blocks) with their depends_on edges. resource_type filters by type
        prefix, e.g. "k8s", "k8s/Deployment", "compose", "terraform/aws_s3_bucket"."""
        if resource_type is not None and not resource_type.strip():
            return dump({"error": "resource_type must not be blank"})
        try:
            with ctx.open_store() as store:
                rows = inventory.list_resources(store, resource_type)
        except StoreError as exc:
            return dump({"error": str(exc)})
        return dump({"resources": rows, "count": len(rows)})


__all__ = ["register"]
