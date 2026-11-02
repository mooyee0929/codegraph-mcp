"""Vocabulary similarity between symbols: nearest neighbours and above-threshold pairs.

Two functions in different modules whose names, signatures and docstrings use the same
rare words (``refund_invoice`` / ``InvoiceRefunder.apply``) are often duplicated logic
or two halves of one feature, even when no call connects them. Similarity is the
cosine of TF-IDF vectors built from :func:`codegraph.search.tokenize.semantic_terms`;
codebase-memory-mcp blends eleven signals for this, this module keeps the one that
needs nothing beyond the graph.

Pairs are found through the inverted index rather than by comparing every pair: only
documents that share a reasonably rare term are scored, which keeps the pass close to
linear on real repositories.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from codegraph.models import CALLABLE_KINDS, Node, NodeKind
from codegraph.search.index import Corpus
from codegraph.search.tokenize import semantic_terms

MIN_TERMS = 3
"""Symbols with fewer distinct terms (``run``, ``get_x``) match too much to be useful."""


@dataclass(frozen=True, slots=True)
class SimilarPair:
    a: str
    b: str
    score: float


def comparison_group(kind: NodeKind) -> frozenset[NodeKind]:
    """Kinds a symbol is compared with: callables with callables, otherwise same kind."""
    if kind in CALLABLE_KINDS:
        return frozenset(CALLABLE_KINDS)
    return frozenset({kind})


def semantic_corpus(nodes: Iterable[Node], *, min_terms: int = MIN_TERMS) -> Corpus:
    documents = []
    for node in nodes:
        words = semantic_terms(node)
        if len(set(words)) >= min_terms:
            documents.append((node.qname, node.kind, words))
    return Corpus.build(documents)


def neighbours(corpus: Corpus, qname: str) -> dict[str, float]:
    """Cosine of ``qname`` against every document sharing at least one term with it."""
    vector = corpus.unit_vector(qname)
    others = {other for term in vector for other in corpus.postings(term) if other != qname}
    return {other: cosine(vector, corpus.unit_vector(other)) for other in sorted(others)}


def similar_pairs(
    corpus: Corpus,
    threshold: float,
    *,
    max_doc_freq: int | None = None,
    max_candidates: int | None = None,
) -> Iterator[SimilarPair]:
    """Every unordered pair with cosine >= ``threshold``, each yielded once.

    Candidate pairs come from terms in at most ``max_doc_freq`` documents (default:
    twice the square root of the corpus size, at least 25), so a word every handler
    shares does not turn the pass quadratic. Terms are taken rarest first and
    collection stops at ``max_candidates`` pairs (default: 25 per document, at least
    10,000); the rare terms that remain are the ones that make two symbols alike.
    Exact cosine is then computed per candidate.
    """
    total = corpus.doc_count
    cap = max_doc_freq if max_doc_freq is not None else max(25, int(2 * math.sqrt(total)))
    budget = max_candidates if max_candidates is not None else max(10_000, 25 * total)
    candidates: set[tuple[str, str]] = set()
    by_rarity = sorted(corpus.postings_by_term.items(), key=lambda item: (len(item[1]), item[0]))
    for _, docs in by_rarity:
        if len(docs) < 2:
            continue
        if len(docs) > cap or len(candidates) >= budget:
            break
        members = sorted(docs)
        for i, first in enumerate(members):
            for second in members[i + 1 :]:
                candidates.add((first, second))
    vectors: dict[str, dict[str, float]] = {}
    for first, second in sorted(candidates):
        left = vectors.get(first)
        if left is None:
            left = vectors[first] = corpus.unit_vector(first)
        right = vectors.get(second)
        if right is None:
            right = vectors[second] = corpus.unit_vector(second)
        score = cosine(left, right)
        if score >= threshold:
            yield SimilarPair(first, second, score)


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    """Dot product of two unit vectors."""
    if len(right) < len(left):
        left, right = right, left
    return sum(weight * right.get(term, 0.0) for term, weight in left.items())


__all__ = [
    "MIN_TERMS",
    "SimilarPair",
    "comparison_group",
    "cosine",
    "neighbours",
    "semantic_corpus",
    "similar_pairs",
]
