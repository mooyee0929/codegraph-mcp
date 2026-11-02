"""An in-memory snapshot of the graph for whole-graph algorithms.

:class:`GraphStore` answers point lookups well; community detection, flow tracing,
centrality and Cypher matching want adjacency lists they can walk thousands of times.
:meth:`GraphView.load` reads every node and edge once and indexes them both ways.

Edges whose ``dst`` is not a known node (unresolved names, external imports) are kept
in :attr:`GraphView.out_edges` so callers can still see them, but they never appear in
:attr:`GraphView.in_edges` and are skipped by :meth:`GraphView.successors`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from codegraph.models import Edge, EdgeKind, Node, NodeKind
from codegraph.store import GraphStore


@dataclass(slots=True)
class GraphView:
    nodes: dict[str, Node]
    out_edges: dict[str, list[Edge]] = field(default_factory=dict)
    in_edges: dict[str, list[Edge]] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        store: GraphStore,
        *,
        edge_kinds: Iterable[EdgeKind] | None = None,
        node_kinds: Iterable[NodeKind] | None = None,
    ) -> GraphView:
        nodes = {n.qname: n for n in store.all_nodes(kinds=node_kinds)}
        return cls.from_parts(nodes, store.all_edges(kinds=edge_kinds))

    @classmethod
    def from_parts(cls, nodes: dict[str, Node], edges: Iterable[Edge]) -> GraphView:
        out_edges: defaultdict[str, list[Edge]] = defaultdict(list)
        in_edges: defaultdict[str, list[Edge]] = defaultdict(list)
        for edge in edges:
            if edge.src not in nodes:
                continue
            out_edges[edge.src].append(edge)
            if edge.dst in nodes:
                in_edges[edge.dst].append(edge)
        return cls(nodes=nodes, out_edges=dict(out_edges), in_edges=dict(in_edges))

    # -- traversal --------------------------------------------------------

    def edges_out(self, qname: str, kinds: Iterable[EdgeKind] | None = None) -> list[Edge]:
        edges = self.out_edges.get(qname, [])
        if kinds is None:
            return edges
        wanted = set(kinds)
        return [e for e in edges if e.kind in wanted]

    def edges_in(self, qname: str, kinds: Iterable[EdgeKind] | None = None) -> list[Edge]:
        edges = self.in_edges.get(qname, [])
        if kinds is None:
            return edges
        wanted = set(kinds)
        return [e for e in edges if e.kind in wanted]

    def successors(self, qname: str, kinds: Iterable[EdgeKind] | None = None) -> list[str]:
        """Distinct known nodes ``qname`` points at, in first-seen order."""
        return _unique(e.dst for e in self.edges_out(qname, kinds) if e.dst in self.nodes)

    def predecessors(self, qname: str, kinds: Iterable[EdgeKind] | None = None) -> list[str]:
        return _unique(e.src for e in self.edges_in(qname, kinds))

    def neighbors(self, qname: str, kinds: Iterable[EdgeKind] | None = None) -> list[str]:
        """Successors and predecessors, treating the graph as undirected."""
        wanted = list(kinds) if kinds is not None else None
        return _unique([*self.successors(qname, wanted), *self.predecessors(qname, wanted)])

    def in_degree(self, qname: str, kinds: Iterable[EdgeKind] | None = None) -> int:
        return len(self.predecessors(qname, kinds))

    def out_degree(self, qname: str, kinds: Iterable[EdgeKind] | None = None) -> int:
        return len(self.successors(qname, kinds))

    # -- selection --------------------------------------------------------

    def of_kind(self, *kinds: NodeKind) -> Iterator[Node]:
        wanted = set(kinds)
        return (n for n in self.nodes.values() if n.kind in wanted)

    def edges(self, kinds: Iterable[EdgeKind] | None = None) -> Iterator[Edge]:
        """Every edge between two known nodes."""
        wanted = set(kinds) if kinds is not None else None
        for edge_list in self.in_edges.values():
            for edge in edge_list:
                if wanted is None or edge.kind in wanted:
                    yield edge

    def module_of(self, qname: str) -> str | None:
        """The module node that owns ``qname`` (the file it was parsed from)."""
        node = self.nodes.get(qname)
        if node is None:
            return None
        if node.kind == "module":
            return node.qname
        for candidate in self.edges_in(qname, ("defines",)):
            parent = self.nodes.get(candidate.src)
            if parent is not None:
                return self.module_of(parent.qname)
        return None


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


__all__ = ["GraphView"]
