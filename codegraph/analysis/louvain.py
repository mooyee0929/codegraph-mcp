"""Deterministic multi-level Louvain community detection, in pure Python.

The algorithm alternates two phases until nothing moves: *local moving* puts each node
into the neighbouring community with the best modularity gain, then *aggregation*
collapses every community into a single node (its internal weight becomes a self-loop)
and repeats on the smaller graph.

Determinism matters more than the last percent of modularity here: community ids end up
in tool output and in persisted tables, so two runs over the same graph must agree.
Nodes are visited in index order, candidate communities in ascending label order, and a
node only leaves its community for a strictly better gain. Callers make indices
meaningful by numbering nodes in sorted qname order.
"""

from __future__ import annotations

from collections.abc import Mapping

_EPSILON = 1e-12


def louvain(
    size: int,
    edges: Mapping[tuple[int, int], float],
    *,
    resolution: float = 1.0,
    max_levels: int = 32,
    max_sweeps: int = 64,
) -> list[int]:
    """Community label for each node ``0..size-1``.

    ``edges`` maps an unordered node pair to its weight; ``(a, b)`` and ``(b, a)`` are
    summed, ``(a, a)`` is a self-loop and non-positive weights are ignored. Labels are
    dense (``0..k-1``) and numbered in order of each community's lowest node index.
    Isolated nodes keep a community of their own.
    """
    if size < 0:
        raise ValueError("size must be non-negative")
    adjacency, loops = _build(size, edges)
    total = _total_weight(adjacency, loops)
    if total <= 0.0:
        return list(range(size))

    assignment = list(range(size))
    for _ in range(max_levels):
        local = _move_nodes(adjacency, loops, total, resolution, max_sweeps)
        labels, count = _dense(local)
        assignment = [labels[a] for a in assignment]
        if count == len(adjacency):
            break
        adjacency, loops = _aggregate(adjacency, loops, labels, count)
    return _dense(assignment)[0]


def modularity(
    size: int,
    edges: Mapping[tuple[int, int], float],
    labels: list[int],
    *,
    resolution: float = 1.0,
) -> float:
    """Newman modularity of ``labels`` on the weighted undirected graph ``edges``."""
    if len(labels) != size:
        raise ValueError("one label per node is required")
    adjacency, loops = _build(size, edges)
    total = _total_weight(adjacency, loops)
    if total <= 0.0:
        return 0.0
    internal: dict[int, float] = {}
    degree: dict[int, float] = {}
    for node in range(size):
        label = labels[node]
        node_degree = sum(adjacency[node].values()) + 2.0 * loops[node]
        degree[label] = degree.get(label, 0.0) + node_degree
        inside = loops[node] + sum(
            w / 2.0 for other, w in adjacency[node].items() if labels[other] == label
        )
        internal[label] = internal.get(label, 0.0) + inside
    two_m = 2.0 * total
    return sum(
        internal.get(label, 0.0) / total - resolution * (degree[label] / two_m) ** 2
        for label in degree
    )


def _build(
    size: int, edges: Mapping[tuple[int, int], float]
) -> tuple[list[dict[int, float]], list[float]]:
    adjacency: list[dict[int, float]] = [{} for _ in range(size)]
    loops = [0.0] * size
    for (a, b), weight in sorted(edges.items()):
        if weight <= 0.0:
            continue
        if not (0 <= a < size and 0 <= b < size):
            raise ValueError(f"edge ({a}, {b}) refers to a node outside 0..{size - 1}")
        if a == b:
            loops[a] += weight
            continue
        adjacency[a][b] = adjacency[a].get(b, 0.0) + weight
        adjacency[b][a] = adjacency[b].get(a, 0.0) + weight
    return adjacency, loops


def _total_weight(adjacency: list[dict[int, float]], loops: list[float]) -> float:
    return sum(loops) + sum(sum(row.values()) for row in adjacency) / 2.0


def _move_nodes(
    adjacency: list[dict[int, float]],
    loops: list[float],
    total: float,
    resolution: float,
    max_sweeps: int,
) -> list[int]:
    count = len(adjacency)
    degree = [sum(row.values()) + 2.0 * loop for row, loop in zip(adjacency, loops, strict=True)]
    community = list(range(count))
    totals = list(degree)
    two_m = 2.0 * total
    for _ in range(max_sweeps):
        moved = False
        for node in range(count):
            current = community[node]
            node_degree = degree[node]
            links: dict[int, float] = {}
            for neighbour, weight in adjacency[node].items():
                label = community[neighbour]
                links[label] = links.get(label, 0.0) + weight
            totals[current] -= node_degree
            best = current
            best_gain = links.get(current, 0.0) - resolution * totals[current] * node_degree / two_m
            for label in sorted(links):
                gain = links[label] - resolution * totals[label] * node_degree / two_m
                if gain > best_gain + _EPSILON:
                    best, best_gain = label, gain
            totals[best] += node_degree
            if best != current:
                community[node] = best
                moved = True
        if not moved:
            break
    return community


def _aggregate(
    adjacency: list[dict[int, float]], loops: list[float], labels: list[int], count: int
) -> tuple[list[dict[int, float]], list[float]]:
    merged: list[dict[int, float]] = [{} for _ in range(count)]
    merged_loops = [0.0] * count
    for node, row in enumerate(adjacency):
        label = labels[node]
        merged_loops[label] += loops[node]
        for neighbour, weight in row.items():
            other = labels[neighbour]
            if other == label:
                merged_loops[label] += weight / 2.0  # each internal edge is seen from both ends
            else:
                merged[label][other] = merged[label].get(other, 0.0) + weight
    return merged, merged_loops


def _dense(labels: list[int]) -> tuple[list[int], int]:
    mapping: dict[int, int] = {}
    dense: list[int] = []
    for label in labels:
        if label not in mapping:
            mapping[label] = len(mapping)
        dense.append(mapping[label])
    return dense, len(mapping)


__all__ = ["louvain", "modularity"]
