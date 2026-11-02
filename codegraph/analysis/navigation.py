"""Walking the graph from a symbol: trees, call chains and shortest paths.

- :func:`traverse` returns a spanning tree of everything reachable from a node over
  chosen edge kinds and direction, breadth- or depth-first, with a node budget.
- :func:`trace_paths` lists the actual call chains into (inbound) or out of (outbound)
  a symbol, every maximal simple path up to the depth, so "how does X reach Y" can be
  read off directly. Inbound callers also get a risk label by hop distance: a direct
  caller breaks first.
- :func:`shortest_path` finds the shortest chain between two symbols, trying the call
  direction first, then the reverse, then ignoring direction.

Call chains follow ``calls`` and, when present, ``http_calls`` -> route -> ``handles``,
so a chain can cross from a client into the service that answers it.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from codegraph.analysis.common import is_test_node
from codegraph.graph import GraphView
from codegraph.models import EdgeKind

Direction = Literal["out", "in", "both"]
TraceDirection = Literal["inbound", "outbound", "both"]
Mode = Literal["bfs", "dfs"]

DIRECTIONS: tuple[str, ...] = ("out", "in", "both")
TRACE_DIRECTIONS: tuple[str, ...] = ("inbound", "outbound", "both")
MODES: tuple[str, ...] = ("bfs", "dfs")
CHAIN_KINDS: tuple[EdgeKind, ...] = ("calls", "http_calls", "handles")
MAX_TRACE_DEPTH = 5

_RISK_BY_HOP = {1: "critical", 2: "high", 3: "medium"}


@dataclass(frozen=True, slots=True)
class TreeNode:
    qname: str
    kind: str
    path: str
    line: int
    depth: int
    via: str
    """How the parent reached this node: ``calls``, ``<-calls`` (inbound) or ``root``."""
    children: tuple[TreeNode, ...]


@dataclass(frozen=True, slots=True)
class Traversal:
    root: TreeNode
    visited: int
    truncated: bool
    direction: str
    mode: str
    edge_kinds: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Reach:
    qname: str
    hop: int
    risk: str


@dataclass(frozen=True, slots=True)
class PathResult:
    nodes: tuple[str, ...]
    direction: str
    """``forward`` (src calls towards dst), ``backward`` or ``undirected``."""


@dataclass(frozen=True, slots=True)
class Trace:
    target: str
    depth: int
    outbound: tuple[tuple[str, ...], ...]
    """Chains starting at the target: ``(target, callee, callee's callee, ...)``."""
    inbound: tuple[tuple[str, ...], ...]
    """Chains ending at the target, in call order: ``(..., caller, target)``."""
    callers: tuple[Reach, ...]
    callees: tuple[Reach, ...]
    truncated: bool
    to: str | None
    path: PathResult | None


def _hops(
    view: GraphView, qname: str, kinds: tuple[EdgeKind, ...], direction: str
) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    if direction in {"out", "both"}:
        for edge in view.edges_out(qname, kinds):
            if edge.dst in view.nodes and edge.dst not in seen:
                seen.add(edge.dst)
                found.append((edge.dst, edge.kind))
    if direction in {"in", "both"}:
        for edge in view.edges_in(qname, kinds):
            if edge.src not in seen:
                seen.add(edge.src)
                found.append((edge.src, f"<-{edge.kind}"))
    return found


def traverse(
    view: GraphView,
    start: str,
    *,
    edge_kinds: Iterable[EdgeKind] | None = None,
    direction: Direction = "out",
    depth: int = 3,
    mode: Mode = "bfs",
    max_nodes: int = 200,
) -> Traversal:
    """Spanning tree from ``start``; each node appears once, at its first visit."""
    if start not in view.nodes:
        raise KeyError(start)
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {', '.join(DIRECTIONS)}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    kinds: tuple[EdgeKind, ...] = (
        tuple(edge_kinds) if edge_kinds is not None else tuple(_present_kinds(view))
    )
    walk = _bfs_tree if mode == "bfs" else _dfs_tree
    children, meta, truncated = walk(view, start, kinds, direction, depth, max_nodes)
    return Traversal(
        root=_tree(view, start, children, meta),
        visited=len(meta),
        truncated=truncated,
        direction=direction,
        mode=mode,
        edge_kinds=kinds,
    )


_Walk = tuple[dict[str, list[str]], dict[str, tuple[int, str]], bool]
"""``(children, (depth, via) per node, truncated)`` of a traversal."""


def _bfs_tree(
    view: GraphView,
    start: str,
    kinds: tuple[EdgeKind, ...],
    direction: str,
    depth: int,
    max_nodes: int,
) -> _Walk:
    children: dict[str, list[str]] = {start: []}
    meta: dict[str, tuple[int, str]] = {start: (0, "root")}
    queue: deque[str] = deque([start])
    while queue:
        current = queue.popleft()
        level = meta[current][0]
        if level >= depth:
            continue
        for neighbour, via in _hops(view, current, kinds, direction):
            if neighbour in meta:
                continue
            if len(meta) >= max_nodes:
                return children, meta, True
            meta[neighbour] = (level + 1, via)
            children[current].append(neighbour)
            children[neighbour] = []
            queue.append(neighbour)
    return children, meta, False


def _dfs_tree(
    view: GraphView,
    start: str,
    kinds: tuple[EdgeKind, ...],
    direction: str,
    depth: int,
    max_nodes: int,
) -> _Walk:
    children: dict[str, list[str]] = {}
    meta: dict[str, tuple[int, str]] = {}
    stack: list[tuple[str, str | None, int, str]] = [(start, None, 0, "root")]
    while stack:
        current, parent, level, via = stack.pop()
        if current in meta:
            continue
        if len(meta) >= max_nodes:
            return children, meta, True
        meta[current] = (level, via)
        children[current] = []
        if parent is not None:
            children[parent].append(current)
        if level >= depth:
            continue
        for neighbour, hop_via in reversed(_hops(view, current, kinds, direction)):
            if neighbour not in meta:
                stack.append((neighbour, current, level + 1, hop_via))
    return children, meta, False


def _present_kinds(view: GraphView) -> list[EdgeKind]:
    kinds: list[EdgeKind] = []
    for edges in view.in_edges.values():
        for edge in edges:
            if edge.kind not in kinds:
                kinds.append(edge.kind)
    return sorted(kinds)


def _tree(
    view: GraphView, qname: str, children: dict[str, list[str]], meta: dict[str, tuple[int, str]]
) -> TreeNode:
    node = view.nodes[qname]
    level, via = meta[qname]
    return TreeNode(
        qname=qname,
        kind=node.kind,
        path=node.path,
        line=node.line_start,
        depth=level,
        via=via,
        children=tuple(_tree(view, child, children, meta) for child in children[qname]),
    )


def trace_paths(
    view: GraphView,
    target: str,
    *,
    direction: TraceDirection = "both",
    depth: int = 3,
    max_paths: int = 50,
    include_tests: bool = False,
    to: str | None = None,
) -> Trace:
    """Inbound and/or outbound call chains of ``target`` up to ``depth`` (1-5) hops."""
    if target not in view.nodes:
        raise KeyError(target)
    if to is not None and to not in view.nodes:
        raise KeyError(to)
    if direction not in TRACE_DIRECTIONS:
        raise ValueError(f"direction must be one of {', '.join(TRACE_DIRECTIONS)}")
    if not 1 <= depth <= MAX_TRACE_DEPTH:
        raise ValueError(f"depth must be between 1 and {MAX_TRACE_DEPTH}")
    truncated = False
    outbound: list[tuple[str, ...]] = []
    inbound: list[tuple[str, ...]] = []
    if direction in {"outbound", "both"}:
        outbound, cut = _chains(view, target, "out", depth, max_paths, include_tests)
        truncated = truncated or cut
    if direction in {"inbound", "both"}:
        backwards, cut = _chains(view, target, "in", depth, max_paths, include_tests)
        inbound = [tuple(reversed(chain)) for chain in backwards]
        truncated = truncated or cut
    return Trace(
        target=target,
        depth=depth,
        outbound=tuple(outbound),
        inbound=tuple(inbound),
        callers=tuple(_reach(view, target, "in", depth, include_tests))
        if direction != "outbound"
        else (),
        callees=tuple(_reach(view, target, "out", depth, include_tests))
        if direction != "inbound"
        else (),
        truncated=truncated,
        to=to,
        path=shortest_path(view, target, to) if to is not None else None,
    )


def _chain_neighbours(
    view: GraphView, qname: str, direction: str, include_tests: bool
) -> list[str]:
    hops = _hops(view, qname, CHAIN_KINDS, direction)
    return [q for q, _ in hops if include_tests or not is_test_node(view.nodes[q])]


def _chains(
    view: GraphView, target: str, direction: str, depth: int, max_paths: int, include_tests: bool
) -> tuple[list[tuple[str, ...]], bool]:
    chains: list[tuple[str, ...]] = []
    stack: list[tuple[str, ...]] = [(target,)]
    while stack:
        chain = stack.pop()
        nexts = []
        if len(chain) <= depth:
            nexts = [
                q
                for q in _chain_neighbours(view, chain[-1], direction, include_tests)
                if q not in chain
            ]
        if not nexts:
            if len(chain) > 1:
                chains.append(chain)
                if len(chains) >= max_paths:
                    return chains, bool(stack)
            continue
        stack.extend((*chain, q) for q in reversed(nexts))
    return chains, False


def _reach(
    view: GraphView, target: str, direction: str, depth: int, include_tests: bool
) -> list[Reach]:
    hop_of: dict[str, int] = {target: 0}
    queue: deque[str] = deque([target])
    while queue:
        current = queue.popleft()
        if hop_of[current] >= depth:
            continue
        for neighbour in _chain_neighbours(view, current, direction, include_tests):
            if neighbour not in hop_of:
                hop_of[neighbour] = hop_of[current] + 1
                queue.append(neighbour)
    return [
        Reach(qname=q, hop=hop, risk=_RISK_BY_HOP.get(hop, "low"))
        for q, hop in sorted(hop_of.items(), key=lambda item: (item[1], item[0]))
        if hop > 0
    ]


def shortest_path(
    view: GraphView, src: str, dst: str, *, kinds: tuple[EdgeKind, ...] = CHAIN_KINDS
) -> PathResult | None:
    """Fewest-hop chain from ``src`` to ``dst``; falls back to reverse, then undirected."""
    for direction, label in (("out", "forward"), ("in", "backward"), ("both", "undirected")):
        found = _bfs_path(view, src, dst, kinds, direction)
        if found is not None:
            return PathResult(nodes=found, direction=label)
    return None


def _bfs_path(
    view: GraphView, src: str, dst: str, kinds: tuple[EdgeKind, ...], direction: str
) -> tuple[str, ...] | None:
    parents: dict[str, str | None] = {src: None}
    queue: deque[str] = deque([src])
    while queue:
        current = queue.popleft()
        if current == dst:
            chain: list[str] = []
            cursor: str | None = current
            while cursor is not None:
                chain.append(cursor)
                cursor = parents[cursor]
            return tuple(reversed(chain))
        for neighbour, _ in _hops(view, current, kinds, direction):
            if neighbour not in parents:
                parents[neighbour] = current
                queue.append(neighbour)
    return None


__all__ = [
    "CHAIN_KINDS",
    "DIRECTIONS",
    "MAX_TRACE_DEPTH",
    "MODES",
    "TRACE_DIRECTIONS",
    "Direction",
    "Mode",
    "PathResult",
    "Reach",
    "Trace",
    "TraceDirection",
    "Traversal",
    "TreeNode",
    "shortest_path",
    "trace_paths",
    "traverse",
]
