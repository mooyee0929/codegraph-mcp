"""MCP tools for graph queries: ``run_cypher`` and ``get_graph_schema``.

The schema tool exists so an agent can look before it writes Cypher: which labels and
relationship types occur in *this* repository, which label pairs they connect, and
example queries built from real names.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from codegraph.cypher import CypherError, Limits, run_cypher
from codegraph.cypher.schema import graph_schema
from codegraph.cypher.vocab import EDGE_PROPERTIES, NODE_LABELS, NODE_PROPERTIES, REL_TYPES
from codegraph.tools import ToolContext, dump

MAX_ROWS = Limits().max_rows

CYPHER_HELP = (
    "Run a read-only Cypher query over the code graph; returns columns and rows. "
    "Supported: MATCH / OPTIONAL MATCH with comma-separated patterns and inline property "
    f"maps; labels {', '.join(sorted(NODE_LABELS))} (Callable = function or method); "
    f"relationship types {', '.join(REL_TYPES)} with | alternatives, ->, <- or undirected, "
    f"and variable length *1..3 (capped at {Limits().max_hops} hops, one shortest path per "
    "node). WHERE: = <> < <= > >= =~ CONTAINS, STARTS WITH, ENDS WITH, IN [...], "
    "IS [NOT] NULL, AND / OR / XOR / NOT. WITH and RETURN: DISTINCT, AS, count(*), "
    "count/collect/min/max/avg/sum with implicit grouping, ORDER BY ... ASC|DESC, SKIP, LIMIT. "
    f"Node properties: {', '.join(NODE_PROPERTIES)}. Relationship properties: "
    f"{', '.join(EDGE_PROPERTIES)}. Call get_graph_schema first to see what occurs here. "
    "Example: MATCH (c:Callable)-[:CALLS]->(f {name: 'charge'}) RETURN c.qname, c.path"
)


def register(server: MCPServer[None], ctx: ToolContext) -> None:
    @server.tool(name="run_cypher", description=CYPHER_HELP)
    def run_cypher_tool(query: str, max_rows: int = 200) -> str:
        if not 1 <= max_rows <= MAX_ROWS:
            return dump({"error": f"max_rows must be between 1 and {MAX_ROWS}"})
        try:
            with ctx.open_store() as store:
                result = run_cypher(store, query, limits=Limits(default_rows=max_rows))
        except CypherError as exc:
            return dump(exc.as_dict())
        return dump(result.as_dict())

    @server.tool()
    def get_graph_schema() -> str:
        """Node labels and relationship types with counts, the (Label)-[TYPE]->(Label)
        patterns that occur, the properties per label, and ready-to-run Cypher examples."""
        with ctx.open_store() as store:
            return dump(graph_schema(store).as_dict())


__all__ = ["register"]
