"""Post-index pass: ``http_calls`` edges from client call sites to the routes they hit.

Runs after :mod:`codegraph.passes.routes`. Calls that match no route in this
repository are kept as unresolved edges whose ``dst`` is the would-be route qname
(``route:GET /users/{id}``), so a cross-repository matcher can pair them later.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from codegraph.services.clients import (
    ClientCall,
    detect_client_calls,
    graphql_client_calls,
    grpc_client_calls,
)
from codegraph.services.linking import http_call_edges, route_patterns
from codegraph.services.paths import split_route_qname
from codegraph.services.resolve import NodeIndex
from codegraph.services.sources import load_sources
from codegraph.store import GraphStore


@dataclass(frozen=True, slots=True)
class HttpCallsPass:
    name: str = "http_calls"
    order: int = 22

    def run(self, root: Path, store: GraphStore) -> int:
        store.delete_edges("http_calls")
        route_nodes = store.all_nodes(kinds=("route",))
        services: defaultdict[str, list[str]] = defaultdict(list)
        for node in route_nodes:
            parsed = split_route_qname(node.qname)
            if parsed is not None and parsed[0] == "GRPC" and "/" in parsed[1]:
                service, _, method = parsed[1].partition("/")
                services[service].append(method)
        calls: list[ClientCall] = []
        for source in load_sources(root):
            calls.extend(detect_client_calls(source))
            calls.extend(grpc_client_calls(source, dict(services)))
            calls.extend(graphql_client_calls(source))
        edges = http_call_edges(
            calls,
            NodeIndex.load(store),
            route_patterns(route_nodes),
            {n.qname for n in route_nodes},
        )
        return store.add_edges(edges)


PASS = HttpCallsPass()
