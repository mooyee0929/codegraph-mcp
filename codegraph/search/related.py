"""Nearest neighbours of one symbol by vocabulary, and the pairs worth an edge.

:func:`find_related` answers "what else in this codebase looks like this?" for one
target, every candidate scored. :func:`related_pairs` is what the ``semantic_edges``
pass stores: pairs above a threshold, in different modules, with no edge of any kind
between them already, each symbol keeping only its best few. Both use
:mod:`codegraph.search.similarity`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from codegraph.graph import GraphView
from codegraph.models import CALLABLE_KINDS, Edge, Node
from codegraph.queries import resolve_target
from codegraph.search.similarity import (
    SimilarPair,
    comparison_group,
    neighbours,
    semantic_corpus,
    similar_pairs,
)
from codegraph.search.snippet import SymbolNotFound, candidates_for
from codegraph.store import GraphStore

DEFAULT_THRESHOLD = 0.8
DEFAULT_TOP_K = 3


@dataclass(frozen=True, slots=True)
class RelatedHit:
    node: Node
    score: float
    connected: bool
    """A structural edge (anything but ``semantically_related``) links the two symbols."""
    same_module: bool
    semantic_edge: bool = False
    """The semantic_edges pass already stored a ``semantically_related`` edge for the pair."""

    def as_dict(self) -> dict[str, object]:
        return {
            "qname": self.node.qname,
            "kind": self.node.kind,
            "path": self.node.path,
            "line": self.node.line_start,
            "score": round(self.score, 4),
            "connected": self.connected,
            "same_module": self.same_module,
            "semantic_edge": self.semantic_edge,
        }


@dataclass(frozen=True, slots=True)
class RelatedResult:
    target: Node
    hits: tuple[RelatedHit, ...]
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "target": self.target.qname,
            "results": [hit.as_dict() for hit in self.hits],
        }
        if self.note:
            payload["note"] = self.note
        return payload


def find_related(
    store: GraphStore, target: str, *, limit: int = 10, min_score: float = 0.0
) -> RelatedResult:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    node = resolve_target(store, target)
    if node is None:
        found = candidates_for(store, target)
        reason = "is ambiguous" if len(found) > 1 else "matches no symbol"
        raise SymbolNotFound(f"{target!r} {reason}", found)
    view = GraphView.load(store, node_kinds=sorted(comparison_group(node.kind)))
    corpus = semantic_corpus([*view.nodes.values()], min_terms=1)
    if not corpus.unit_vector(node.qname):
        return RelatedResult(target=node, hits=(), note="the symbol has no descriptive words")
    scores = neighbours(corpus, node.qname)
    structural = {p for p in _connected_pairs(view.edges()) if node.qname in p}
    semantic = {
        e.src if e.dst == node.qname else e.dst
        for e in (
            *view.edges_out(node.qname, ("semantically_related",)),
            *view.edges_in(node.qname, ("semantically_related",)),
        )
    }
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    hits = tuple(
        RelatedHit(
            node=view.nodes[qname],
            score=score,
            connected=(min(qname, node.qname), max(qname, node.qname)) in structural,
            same_module=view.nodes[qname].path == node.path,
            semantic_edge=qname in semantic,
        )
        for qname, score in ordered
        if score >= min_score
    )
    return RelatedResult(target=node, hits=hits[:limit])


def related_pairs(
    view: GraphView,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
) -> list[SimilarPair]:
    """Cross-module callable pairs with no existing edge, best first, ``top_k`` per node."""
    callables = [n for n in view.nodes.values() if n.kind in CALLABLE_KINDS]
    corpus = semantic_corpus(callables)
    connected = _connected_pairs(view.edges())
    eligible = [
        pair
        for pair in similar_pairs(corpus, threshold)
        if view.nodes[pair.a].path != view.nodes[pair.b].path and (pair.a, pair.b) not in connected
    ]
    eligible.sort(key=lambda p: (-p.score, p.a, p.b))
    kept: list[SimilarPair] = []
    per_node: dict[str, int] = {}
    for pair in eligible:
        if per_node.get(pair.a, 0) >= top_k or per_node.get(pair.b, 0) >= top_k:
            continue
        per_node[pair.a] = per_node.get(pair.a, 0) + 1
        per_node[pair.b] = per_node.get(pair.b, 0) + 1
        kept.append(pair)
    return kept


def _connected_pairs(edges: Iterable[Edge]) -> set[tuple[str, str]]:
    """Unordered endpoint pairs of every edge except earlier semantic ones."""
    return {
        (min(e.src, e.dst), max(e.src, e.dst)) for e in edges if e.kind != "semantically_related"
    }


__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_TOP_K",
    "RelatedHit",
    "RelatedResult",
    "find_related",
    "related_pairs",
]
