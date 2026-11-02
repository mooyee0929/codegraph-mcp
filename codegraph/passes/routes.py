"""Post-index pass: HTTP / gRPC / GraphQL route nodes and the ``handles`` edges to them.

Rebuilt from scratch on every run: all ``route`` nodes (with every edge touching them)
and ``passes_routes`` rows are dropped first. The ``http_calls`` and ``trace_spans``
passes run after this one and re-attach to the fresh route nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codegraph.services.linking import route_graph
from codegraph.services.resolve import NodeIndex
from codegraph.services.routes import detect_routes
from codegraph.services.sources import load_sources
from codegraph.services.storage import ROUTES_DDL, ensure_tables
from codegraph.store import GraphStore


@dataclass(frozen=True, slots=True)
class RoutesPass:
    name: str = "routes"
    order: int = 20

    def run(self, root: Path, store: GraphStore) -> int:
        ensure_tables(store, ROUTES_DDL)
        store.delete_nodes("route")
        store.delete_edges("handles")
        store.connection.execute("DELETE FROM passes_routes")
        definitions = detect_routes(load_sources(root))
        nodes, edges, rows = route_graph(definitions, NodeIndex.load(store))
        store.add_nodes(nodes)
        store.add_edges(edges)
        store.connection.executemany(
            "INSERT INTO passes_routes "
            "(route, method, path, framework, file, line, handler, handler_ref) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (r.route, r.method, r.path, r.framework, r.file, r.line, r.handler, r.handler_ref)
                for r in rows
            ],
        )
        return len(nodes) + len(edges)


PASS = RoutesPass()
