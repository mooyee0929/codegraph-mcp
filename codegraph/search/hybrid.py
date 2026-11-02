"""Hybrid symbol search: several rankers fused with reciprocal-rank fusion.

Each ranker is good at a different query: FTS5 at exact identifiers and prefixes, BM25
over split identifiers at "invoice total" finding ``Invoice.total``, TF-IDF cosine at
short queries against short documents, embeddings (when installed and computed) at
paraphrase. Their scores are not comparable, so they are fused by rank:
``score(d) = sum(1 / (k + rank_i(d)))`` with ``k = 60``, the constant from Cormack et
al. that code-review-graph also uses. The result reports which rankers answered.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from codegraph.models import Node, NodeKind
from codegraph.search import embeddings
from codegraph.search.index import bm25_scores, open_index, ranked, tfidf_scores
from codegraph.search.tokenize import terms
from codegraph.store import GraphStore

RRF_K = 60
CANDIDATES_PER_RANKER = 100


@dataclass(frozen=True, slots=True)
class SearchHit:
    node: Node
    score: float
    ranks: dict[str, int]
    """Ranker name -> 1-based rank of this node in that ranker's list."""


@dataclass(frozen=True, slots=True)
class HybridResult:
    query: str
    backend: str
    """Which rankers contributed, e.g. ``"bm25+tfidf+fts5"``."""
    hits: tuple[SearchHit, ...]
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "backend": self.backend,
            "notes": list(self.notes),
            "results": [
                {
                    "qname": hit.node.qname,
                    "kind": hit.node.kind,
                    "path": hit.node.path,
                    "lines": [hit.node.line_start, hit.node.line_end],
                    "signature": hit.node.signature,
                    "score": round(hit.score, 6),
                    "ranks": hit.ranks,
                }
                for hit in self.hits
            ],
        }


def rrf_fuse(
    rankings: Mapping[str, Sequence[str]], *, k: int = RRF_K
) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse ranked qname lists; ties break on qname so output is deterministic."""
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for ranker, qnames in rankings.items():
        for position, qname in enumerate(qnames, start=1):
            scores[qname] = scores.get(qname, 0.0) + 1.0 / (k + position)
            ranks.setdefault(qname, {})[ranker] = position
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [(qname, score, ranks[qname]) for qname, score in ordered]


def hybrid_search(
    store: GraphStore,
    query: str,
    *,
    limit: int = 20,
    kinds: Collection[NodeKind] | None = None,
    encoder: embeddings.Encoder | None = None,
) -> HybridResult:
    """Search nodes by ``query``; ``encoder`` adds an embeddings ranker when given."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    wanted = frozenset(kinds) if kinds else None
    notes: list[str] = []
    rankings: dict[str, list[str]] = {}
    nodes: dict[str, Node] = {}

    def keep(qnames: Sequence[str]) -> list[str]:
        kept: list[str] = []
        for qname in qnames:
            node = nodes.get(qname) or store.get_node(qname)
            if node is None or (wanted is not None and node.kind not in wanted):
                continue
            nodes[qname] = node
            kept.append(qname)
            if len(kept) >= CANDIDATES_PER_RANKER:
                break
        return kept

    words = terms(query)
    if words:
        index, persisted = open_index(store)
        if not persisted:
            notes.append(
                "search index missing or stale; built it in memory (run `codegraph index`)"
            )
        rankings["bm25"] = keep([q for q, _ in ranked(bm25_scores(index, words))])
        rankings["tfidf"] = keep([q for q, _ in ranked(tfidf_scores(index, words))])
    fts_hits = store.search_nodes(query, limit=CANDIDATES_PER_RANKER * 2) if query.strip() else []
    for node in fts_hits:
        nodes.setdefault(node.qname, node)
    rankings["fts5"] = keep([node.qname for node in fts_hits])
    if encoder is not None:
        try:
            scores = embeddings.embedding_scores(store, encoder, query, kinds=wanted)
        except embeddings.EmbeddingsUnavailable as exc:
            notes.append(f"embeddings skipped: {exc}")
            scores = {}
        else:
            if not scores:
                notes.append(f"no stored embeddings for model {encoder.model}; run embed_graph")
        if scores:
            rankings[f"embeddings:{encoder.model}"] = keep([q for q, _ in ranked(scores)])
    answered = [name for name, qnames in rankings.items() if qnames]
    fused = rrf_fuse({name: rankings[name] for name in answered})[:limit]
    hits = tuple(
        SearchHit(node=nodes[qname], score=score, ranks=ranks) for qname, score, ranks in fused
    )
    return HybridResult(
        query=query,
        backend="+".join(answered) if answered else "none",
        hits=hits,
        notes=tuple(notes),
    )


__all__ = ["RRF_K", "HybridResult", "SearchHit", "hybrid_search", "rrf_fuse"]
