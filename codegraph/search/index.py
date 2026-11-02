"""Term index over graph nodes: BM25 ranking and TF-IDF cosine from one set of postings.

FTS5 already ranks by BM25, but with SQLite's tokenizer: ``parseHttpRequest`` is one
token there, so "http request" misses it. This index stores codegraph's own terms
(:mod:`codegraph.search.tokenize`) as postings, ``term -> {qname: frequency}``, plus
each document's length and TF-IDF norm. Both rankers read only the postings of the
query terms, so a query touches a few hundred rows whether the repo has one thousand
nodes or one hundred thousand.

The ``search_index`` pass persists the index in ``search_*`` tables. When those are
missing or stale (the pass was skipped, or the tokenizer changed) :func:`open_index`
builds the same index in memory, so search still works, just slower to start.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from codegraph.models import Node
from codegraph.search.tokenize import TOKENIZER_VERSION, node_terms, unique
from codegraph.store import GraphStore

SEARCH_DDL = """
CREATE TABLE IF NOT EXISTS search_docs (
    qname  TEXT PRIMARY KEY,
    kind   TEXT NOT NULL,
    length INTEGER NOT NULL,
    norm   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS search_postings (
    term  TEXT NOT NULL,
    qname TEXT NOT NULL,
    tf    INTEGER NOT NULL,
    PRIMARY KEY (term, qname)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS search_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

BM25_K1 = 1.2
BM25_B = 0.75
_CHUNK = 500


@dataclass(frozen=True, slots=True)
class DocStats:
    length: int
    norm: float
    kind: str


class TermIndex(Protocol):
    @property
    def doc_count(self) -> int: ...

    @property
    def average_length(self) -> float: ...

    def postings(self, term: str) -> dict[str, int]: ...

    def stats(self, docs: Collection[str]) -> dict[str, DocStats]: ...


def idf(doc_count: int, doc_freq: int) -> float:
    """Okapi IDF, floored at zero-plus so a term in every document still counts a little."""
    return math.log(1.0 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))


def tf_weight(tf: int) -> float:
    return 1.0 + math.log(tf) if tf > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Corpus:
    """An in-memory index; also the builder for the persisted one."""

    postings_by_term: dict[str, dict[str, int]]
    terms_by_doc: dict[str, dict[str, int]]
    docs: dict[str, DocStats]

    @classmethod
    def build(cls, documents: Iterable[tuple[str, str, Sequence[str]]]) -> Corpus:
        """``documents`` yields ``(qname, kind, terms)``."""
        postings: dict[str, dict[str, int]] = {}
        forward: dict[str, dict[str, int]] = {}
        kinds: dict[str, str] = {}
        lengths: dict[str, int] = {}
        for qname, kind, words in documents:
            counts = Counter(words)
            forward[qname] = dict(counts)
            kinds[qname] = kind
            lengths[qname] = len(words)
            for term, tf in counts.items():
                postings.setdefault(term, {})[qname] = tf
        total = len(forward)
        squares: dict[str, float] = dict.fromkeys(forward, 0.0)
        for docs in postings.values():
            weight = idf(total, len(docs))
            for qname, tf in docs.items():
                squares[qname] += (tf_weight(tf) * weight) ** 2
        stats = {
            qname: DocStats(
                length=lengths[qname], norm=math.sqrt(squares[qname]), kind=kinds[qname]
            )
            for qname in forward
        }
        return cls(postings_by_term=postings, terms_by_doc=forward, docs=stats)

    @classmethod
    def from_nodes(cls, nodes: Iterable[Node]) -> Corpus:
        return cls.build((node.qname, node.kind, node_terms(node)) for node in nodes)

    @property
    def doc_count(self) -> int:
        return len(self.docs)

    @property
    def average_length(self) -> float:
        if not self.docs:
            return 0.0
        return sum(d.length for d in self.docs.values()) / len(self.docs)

    def postings(self, term: str) -> dict[str, int]:
        return self.postings_by_term.get(term, {})

    def stats(self, docs: Collection[str]) -> dict[str, DocStats]:
        return {qname: self.docs[qname] for qname in docs if qname in self.docs}

    def unit_vector(self, qname: str) -> dict[str, float]:
        """The document's TF-IDF vector scaled to length 1 (empty if it has no terms)."""
        stats = self.docs.get(qname)
        if stats is None or stats.norm == 0.0:
            return {}
        total = self.doc_count
        return {
            term: tf_weight(tf) * idf(total, len(self.postings_by_term[term])) / stats.norm
            for term, tf in self.terms_by_doc[qname].items()
        }


class PersistedIndex:
    """:class:`TermIndex` over the ``search_*`` tables written by the search_index pass."""

    def __init__(self, store: GraphStore) -> None:
        self._conn = store.connection
        meta = dict(self._conn.execute("SELECT key, value FROM search_meta").fetchall())
        self._doc_count = int(meta.get("doc_count", "0"))
        self._average = float(meta.get("average_length", "0"))

    @property
    def doc_count(self) -> int:
        return self._doc_count

    @property
    def average_length(self) -> float:
        return self._average

    def postings(self, term: str) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT qname, tf FROM search_postings WHERE term = ?", (term,)
        ).fetchall()
        return {str(row["qname"]): int(row["tf"]) for row in rows}

    def stats(self, docs: Collection[str]) -> dict[str, DocStats]:
        found: dict[str, DocStats] = {}
        ordered = list(docs)
        for start in range(0, len(ordered), _CHUNK):
            chunk = ordered[start : start + _CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                "SELECT qname, kind, length, norm FROM search_docs "
                f"WHERE qname IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                found[str(row["qname"])] = DocStats(
                    length=int(row["length"]), norm=float(row["norm"]), kind=str(row["kind"])
                )
        return found


def ensure_search_schema(store: GraphStore) -> None:
    """Create the ``search_*`` tables without committing a caller's open transaction.

    ``GraphStore.ensure_schema`` uses ``executescript``, which commits first; a pass
    runs inside the indexer's transaction, so there the statements run one by one.
    """
    if store.connection.in_transaction:
        for statement in SEARCH_DDL.split(";"):
            if statement.strip():
                store.connection.execute(statement)
    else:
        store.ensure_schema(SEARCH_DDL)


def source_digest(store: GraphStore) -> str:
    """Fingerprint of what the index was built from: file hashes plus the node count."""
    digest = hashlib.sha1()
    for row in store.connection.execute("SELECT path, sha FROM files ORDER BY path"):
        digest.update(f"{row['path']}\0{row['sha']}\n".encode())
    count = store.connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    digest.update(f"nodes={count}".encode())
    return digest.hexdigest()


def write_index(store: GraphStore, corpus: Corpus) -> int:
    """Replace the persisted index with ``corpus``; returns the documents written."""
    conn = store.connection
    conn.execute("DELETE FROM search_postings")
    conn.execute("DELETE FROM search_docs")
    conn.execute("DELETE FROM search_meta")
    conn.executemany(
        "INSERT INTO search_docs (qname, kind, length, norm) VALUES (?, ?, ?, ?)",
        ((q, s.kind, s.length, s.norm) for q, s in corpus.docs.items()),
    )
    conn.executemany(
        "INSERT INTO search_postings (term, qname, tf) VALUES (?, ?, ?)",
        (
            (term, qname, tf)
            for term, docs in corpus.postings_by_term.items()
            for qname, tf in docs.items()
        ),
    )
    meta = {
        "doc_count": str(corpus.doc_count),
        "average_length": repr(corpus.average_length),
        "tokenizer_version": TOKENIZER_VERSION,
        "source_digest": source_digest(store),
    }
    conn.executemany("INSERT INTO search_meta (key, value) VALUES (?, ?)", meta.items())
    return corpus.doc_count


def open_index(store: GraphStore) -> tuple[TermIndex, bool]:
    """The persisted index when it is current, else one built in memory.

    Returns ``(index, persisted)``.
    """
    ensure_search_schema(store)
    meta = dict(store.connection.execute("SELECT key, value FROM search_meta").fetchall())
    if meta.get("tokenizer_version") == TOKENIZER_VERSION and meta.get(
        "source_digest"
    ) == source_digest(store):
        return PersistedIndex(store), True
    return Corpus.from_nodes(store.all_nodes()), False


def bm25_scores(index: TermIndex, query: Sequence[str]) -> dict[str, float]:
    total = index.doc_count
    average = index.average_length or 1.0
    postings = {term: index.postings(term) for term in unique(query)}
    candidates = {qname for docs in postings.values() for qname in docs}
    stats = index.stats(candidates)
    scores: dict[str, float] = {}
    for docs in postings.values():
        weight = idf(total, len(docs))
        for qname, tf in docs.items():
            doc = stats.get(qname)
            if doc is None:
                continue
            norm = tf + BM25_K1 * (1 - BM25_B + BM25_B * doc.length / average)
            scores[qname] = scores.get(qname, 0.0) + weight * tf * (BM25_K1 + 1) / norm
    return scores


def tfidf_scores(index: TermIndex, query: Sequence[str]) -> dict[str, float]:
    """Cosine between the query's TF-IDF vector and each document's."""
    total = index.doc_count
    counts = Counter(query)
    postings = {term: index.postings(term) for term in counts}
    query_weights = {
        term: tf_weight(counts[term]) * idf(total, len(docs))
        for term, docs in postings.items()
        if docs
    }
    query_norm = math.sqrt(sum(w * w for w in query_weights.values()))
    if query_norm == 0.0:
        return {}
    dots: dict[str, float] = {}
    for term, weight in query_weights.items():
        term_idf = idf(total, len(postings[term]))
        for qname, tf in postings[term].items():
            dots[qname] = dots.get(qname, 0.0) + weight * tf_weight(tf) * term_idf
    stats = index.stats(dots)
    return {
        qname: dot / (query_norm * stats[qname].norm)
        for qname, dot in dots.items()
        if qname in stats and stats[qname].norm > 0.0
    }


def ranked(scores: dict[str, float], limit: int | None = None) -> list[tuple[str, float]]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered if limit is None else ordered[:limit]


__all__ = [
    "SEARCH_DDL",
    "Corpus",
    "DocStats",
    "PersistedIndex",
    "TermIndex",
    "bm25_scores",
    "ensure_search_schema",
    "idf",
    "open_index",
    "ranked",
    "source_digest",
    "tf_weight",
    "tfidf_scores",
    "write_index",
]
