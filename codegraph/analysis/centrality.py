"""Hubs and bridges: the nodes the rest of the graph leans on.

A *hub* has many direct dependants or dependencies (degree). A *bridge* sits on many
shortest paths between other nodes (betweenness centrality) even with a modest degree;
breaking it disconnects regions of the code from each other.

Betweenness uses Brandes' algorithm on the undirected dependency graph between
production symbols. Exact Brandes costs O(V·E); above ``max_sources`` nodes it runs
from an evenly spaced, sorted sample of sources and scales the result, so the answer is
approximate but identical from run to run.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from codegraph.analysis.common import SYMBOL_KINDS, is_test_node
from codegraph.graph import GraphView
from codegraph.models import EdgeKind, NodeKind

HUB_EDGE_KINDS: tuple[EdgeKind, ...] = (
    "calls",
    "imports",
    "inherits",
    "handles",
    "http_calls",
    "emits",
    "listens_on",
    "depends_on",
)
BRIDGE_EDGE_KINDS: tuple[EdgeKind, ...] = (
    "calls",
    "inherits",
    "handles",
    "http_calls",
    "emits",
    "listens_on",
)
BRIDGE_NODE_KINDS: tuple[NodeKind, ...] = (*SYMBOL_KINDS, "route", "channel")
DEFAULT_MAX_SOURCES = 400


@dataclass(frozen=True, slots=True)
class Hub:
    qname: str
    kind: str
    path: str
    fan_in: int
    fan_out: int
    degree: int
    community: int | None


@dataclass(frozen=True, slots=True)
class Bridge:
    qname: str
    kind: str
    path: str
    betweenness: float
    communities_linked: int
    """Distinct communities among the node's neighbours, its own included."""
    community: int | None


def hub_nodes(
    view: GraphView,
    *,
    top: int = 10,
    kinds: Iterable[NodeKind] = SYMBOL_KINDS,
    membership: Mapping[str, int] | None = None,
) -> list[Hub]:
    """Highest-degree production nodes; test callers do not count towards fan-in."""
    wanted = set(kinds)
    groups = membership or {}
    hubs: list[Hub] = []
    for node in view.nodes.values():
        if node.kind not in wanted or is_test_node(node):
            continue
        fan_in = sum(
            1
            for q in view.predecessors(node.qname, HUB_EDGE_KINDS)
            if q != node.qname and not is_test_node(view.nodes[q])
        )
        fan_out = sum(1 for q in view.successors(node.qname, HUB_EDGE_KINDS) if q != node.qname)
        if fan_in + fan_out == 0:
            continue
        hubs.append(
            Hub(
                qname=node.qname,
                kind=node.kind,
                path=node.path,
                fan_in=fan_in,
                fan_out=fan_out,
                degree=fan_in + fan_out,
                community=groups.get(node.qname),
            )
        )
    hubs.sort(key=lambda h: (-h.degree, -h.fan_in, h.qname))
    return hubs[: max(top, 0)]


def undirected_adjacency(
    view: GraphView,
    *,
    edge_kinds: Iterable[EdgeKind] = BRIDGE_EDGE_KINDS,
    node_kinds: Iterable[NodeKind] = BRIDGE_NODE_KINDS,
) -> dict[str, list[str]]:
    """Sorted neighbour lists between production nodes of ``node_kinds``."""
    wanted = set(node_kinds)
    keep = {q for q, n in view.nodes.items() if n.kind in wanted and not is_test_node(n)}
    neighbours: dict[str, set[str]] = {q: set() for q in keep}
    for edge in view.edges(edge_kinds):
        if edge.src in keep and edge.dst in keep and edge.src != edge.dst:
            neighbours[edge.src].add(edge.dst)
            neighbours[edge.dst].add(edge.src)
    return {q: sorted(adj) for q, adj in sorted(neighbours.items())}


def betweenness(
    adjacency: Mapping[str, list[str]], *, max_sources: int = DEFAULT_MAX_SOURCES
) -> dict[str, float]:
    """Normalised betweenness centrality of an undirected, unweighted graph.

    Exact when the graph has at most ``max_sources`` nodes, otherwise estimated from a
    deterministic sample of sources. Values are in ``0..1``.
    """
    nodes = sorted(adjacency)
    count = len(nodes)
    scores = dict.fromkeys(nodes, 0.0)
    if count < 3:
        return scores
    sample = max(1, min(max_sources, count))
    sources = nodes if sample == count else [nodes[i * count // sample] for i in range(sample)]
    for source in sources:
        _accumulate(adjacency, source, scores)
    scale = (count / len(sources)) / ((count - 1) * (count - 2))
    return {q: round(v * scale, 6) for q, v in scores.items()}


def _accumulate(adjacency: Mapping[str, list[str]], source: str, scores: dict[str, float]) -> None:
    """One Brandes single-source pass: BFS, then dependency back-propagation."""
    order: list[str] = []
    parents: dict[str, list[str]] = {source: []}
    paths: dict[str, int] = {source: 1}
    distance: dict[str, int] = {source: 0}
    queue: deque[str] = deque([source])
    while queue:
        current = queue.popleft()
        order.append(current)
        for neighbour in adjacency[current]:
            if neighbour not in distance:
                distance[neighbour] = distance[current] + 1
                parents[neighbour] = []
                paths[neighbour] = 0
                queue.append(neighbour)
            if distance[neighbour] == distance[current] + 1:
                paths[neighbour] += paths[current]
                parents[neighbour].append(current)
    dependency = dict.fromkeys(order, 0.0)
    for node in reversed(order):
        for parent in parents[node]:
            dependency[parent] += paths[parent] / paths[node] * (1.0 + dependency[node])
        if node != source:
            scores[node] += dependency[node]


def bridge_nodes(
    view: GraphView,
    *,
    top: int = 10,
    max_sources: int = DEFAULT_MAX_SOURCES,
    membership: Mapping[str, int] | None = None,
) -> list[Bridge]:
    """Production nodes with the highest non-zero betweenness."""
    adjacency = undirected_adjacency(view)
    scores = betweenness(adjacency, max_sources=max_sources)
    groups = membership or {}
    bridges: list[Bridge] = []
    for qname, score in scores.items():
        if score <= 0.0:
            continue
        node = view.nodes[qname]
        linked = {groups[q] for q in (qname, *adjacency[qname]) if q in groups}
        bridges.append(
            Bridge(
                qname=qname,
                kind=node.kind,
                path=node.path,
                betweenness=score,
                communities_linked=len(linked),
                community=groups.get(qname),
            )
        )
    bridges.sort(key=lambda b: (-b.betweenness, b.qname))
    return bridges[: max(top, 0)]


__all__ = [
    "BRIDGE_EDGE_KINDS",
    "BRIDGE_NODE_KINDS",
    "DEFAULT_MAX_SOURCES",
    "HUB_EDGE_KINDS",
    "Bridge",
    "Hub",
    "betweenness",
    "bridge_nodes",
    "hub_nodes",
    "undirected_adjacency",
]
