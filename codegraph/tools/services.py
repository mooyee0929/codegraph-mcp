"""MCP tools for cross-service structure: routes, channels, clones and runtime traces."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from codegraph.services import reports
from codegraph.services.traces import ingest
from codegraph.tools import ToolContext, dump

SpanInput = dict[str, str | int | float | bool | None]


def register(server: MCPServer[None], ctx: ToolContext) -> None:
    @server.tool()
    def list_routes(
        method: str | None = None,
        path: str | None = None,
        framework: str | None = None,
        limit: int = 200,
    ) -> str:
        """HTTP, gRPC and GraphQL routes with their handlers and caller counts.
        Filters: method (GET, POST, GRPC, GRAPHQL, ...), path substring, framework
        (flask, express, spring, rails, grpc, ...)."""
        if limit < 1:
            return dump({"error": "limit must be at least 1"})
        with ctx.open_store() as store:
            return dump(reports.list_routes(store, method, path, framework, limit))

    @server.tool()
    def get_route(target: str) -> str:
        """One route: where it is declared, its handler functions, the call sites that
        reach it (static http_calls and observed spans) and the tests covering it.
        target: a route qname, 'GET /users/{id}', a concrete 'GET /users/42' or a path."""
        with ctx.open_store() as store:
            detail = reports.route_detail(store, target)
        if detail is None:
            return dump({"error": f"no route matches {target!r}"})
        return dump(detail)

    @server.tool()
    def list_channels(name: str | None = None, broker: str | None = None, limit: int = 200) -> str:
        """Event / message channels (Kafka topics, socket events, Celery tasks, signals,
        Temporal workflows ...) with emitter and listener counts."""
        if limit < 1:
            return dump({"error": "limit must be at least 1"})
        with ctx.open_store() as store:
            return dump(reports.list_channels(store, name, broker, limit))

    @server.tool()
    def get_channel(target: str) -> str:
        """Who emits to and who listens on one channel. target: 'channel:orders' or 'orders'."""
        with ctx.open_store() as store:
            detail = reports.channel_detail(store, target)
        if detail is None:
            return dump({"error": f"no channel matches {target!r}"})
        return dump(detail)

    @server.tool()
    def find_clones(target: str | None = None, min_score: float = 0.8, limit: int = 50) -> str:
        """Near-duplicate functions (exact Jaccard over normalised token shingles).
        With target, only clones of that symbol."""
        if not 0.0 <= min_score <= 1.0:
            return dump({"error": "min_score must be between 0 and 1"})
        with ctx.open_store() as store:
            pairs = reports.clone_pairs(store, target, min_score, limit)
        if pairs is None:
            return dump({"error": f"no node matches {target!r}"})
        return dump(pairs)

    @server.tool()
    def ingest_traces(spans: list[SpanInput]) -> str:
        """Record runtime spans and use them to confirm or add http_calls edges.
        Each span: {service, method, url or path, status?, caller? (function qname)}."""
        with ctx.open_store() as store:
            stats, errors = ingest(store, spans)
        if stats is None:
            return dump({"error": "invalid spans; nothing was stored", "problems": errors})
        return dump(stats)


__all__ = ["register"]
