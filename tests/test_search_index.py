from __future__ import annotations

import math
from pathlib import Path

import pytest

from codegraph.indexer import index_repository
from codegraph.models import Node
from codegraph.passes import GraphPass, discover_passes
from codegraph.passes.search_index import PASS as SEARCH_INDEX_PASS
from codegraph.search.index import (
    Corpus,
    PersistedIndex,
    bm25_scores,
    ensure_search_schema,
    idf,
    open_index,
    ranked,
    source_digest,
    tf_weight,
    tfidf_scores,
    write_index,
)
from codegraph.search.tokenize import (
    node_terms,
    semantic_terms,
    split_identifier,
    terms,
    unique,
)
from codegraph.store import GraphStore

# -- tokenizer -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("parseHTTPRequest", ["parse", "http", "request"]),
        ("parse_http_request", ["parse", "http", "request"]),
        ("APIRoute", ["api", "route"]),
        ("getV2Data", ["get", "v", "2", "data"]),
        ("pkg/http/request_handler.py", ["pkg", "http", "request", "handler", "py"]),
        ("pkg.billing.Invoice.total", ["pkg", "billing", "invoice", "total"]),
        ("kebab-case-name", ["kebab", "case", "name"]),
        ("XMLHttpRequest", ["xml", "http", "request"]),
        ("", []),
    ],
)
def test_split_identifier(text: str, words: list[str]) -> None:
    assert split_identifier(text) == words


def test_terms_expand_abbreviations_and_drop_stopwords() -> None:
    assert terms("def handle_req(self, ctx)") == ["handle", "req", "request", "ctx", "context"]
    assert terms("fmt", expand=False) == ["fmt"]
    assert terms("a b 7") == ["7"]
    assert terms("str", drop=frozenset({"str", "string"})) == []


def _node(name: str, **fields: str) -> Node:
    return Node(
        qname=fields.get("qname", f"pkg.mod.{name}"),
        kind="function",
        path=fields.get("path", "pkg/mod.py"),
        line_start=1,
        line_end=2,
        language="python",
        name=name,
        signature=fields.get("signature", ""),
        docstring=fields.get("docstring"),
    )


def test_node_terms_weight_the_name() -> None:
    bag = node_terms(_node("fetchUser", signature="def fetchUser(uid: int) -> User"))
    assert bag.count("fetch") > bag.count("uid")
    assert "function" in bag
    assert "pkg" in bag and "mod" in bag


def test_semantic_terms_ignore_location_and_types() -> None:
    node = _node(
        "refund",
        qname="billing.api.refund",
        path="billing/api.py",
        signature="def refund(amount: int) -> bool",
        docstring="Refund the payment.",
    )
    bag = semantic_terms(node)
    assert "billing" not in bag and "api" not in bag
    assert "int" not in bag and "bool" not in bag
    assert {"refund", "amount", "payment"} <= set(bag)


def test_unique_keeps_first_occurrence() -> None:
    assert unique(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


# -- corpus and rankers ----------------------------------------------------------


def _corpus() -> Corpus:
    return Corpus.build(
        [
            ("a", "function", ["invoice", "total", "invoice"]),
            ("b", "function", ["invoice", "rate"]),
            ("c", "class", ["user", "profile", "avatar", "upload"]),
        ]
    )


def test_corpus_statistics() -> None:
    corpus = _corpus()
    assert corpus.doc_count == 3
    assert corpus.average_length == 3.0
    assert corpus.postings("invoice") == {"a": 2, "b": 1}
    assert corpus.postings("missing") == {}
    assert corpus.stats(["a", "zzz"]).keys() == {"a"}
    vector = corpus.unit_vector("a")
    assert math.isclose(sum(w * w for w in vector.values()), 1.0)
    assert corpus.unit_vector("nope") == {}


def test_idf_and_tf_weight() -> None:
    assert idf(10, 1) > idf(10, 5) > 0
    assert idf(10, 10) > 0
    assert tf_weight(0) == 0.0
    assert tf_weight(1) == 1.0
    assert tf_weight(3) > tf_weight(2)


def test_bm25_prefers_more_matching_terms() -> None:
    corpus = _corpus()
    scores = bm25_scores(corpus, ["invoice", "total"])
    assert [q for q, _ in ranked(scores)] == ["a", "b"]
    assert bm25_scores(corpus, ["nothing"]) == {}
    assert ranked(scores, limit=1)[0][0] == "a"


def test_tfidf_cosine_is_bounded() -> None:
    corpus = _corpus()
    scores = tfidf_scores(corpus, ["user", "profile", "avatar", "upload"])
    assert math.isclose(scores["c"], 1.0)
    partial = tfidf_scores(corpus, ["invoice"])
    assert 0 < partial["b"] <= 1 and 0 < partial["a"] <= 1
    assert tfidf_scores(corpus, ["unknown"]) == {}


def test_ranked_breaks_ties_by_name() -> None:
    assert ranked({"b": 1.0, "a": 1.0, "c": 2.0}) == [("c", 2.0), ("a", 1.0), ("b", 1.0)]


# -- persistence -----------------------------------------------------------------


def test_search_index_pass_is_discovered_in_the_search_band() -> None:
    passes = {p.name: p for p in discover_passes()}
    assert isinstance(passes["search_index"], GraphPass)
    assert 90 <= passes["search_index"].order <= 99
    assert passes["semantic_edges"].order > passes["search_index"].order


def test_persisted_index_matches_memory(store: GraphStore) -> None:
    index, persisted = open_index(store)
    assert persisted and isinstance(index, PersistedIndex)
    memory = Corpus.from_nodes(store.all_nodes())
    assert index.doc_count == memory.doc_count == store.counts()["nodes"]
    assert math.isclose(index.average_length, memory.average_length)
    query = terms("invoice total")
    persisted_bm25 = bm25_scores(index, query)
    memory_bm25 = bm25_scores(memory, query)
    assert persisted_bm25.keys() == memory_bm25.keys()
    assert all(math.isclose(persisted_bm25[q], memory_bm25[q]) for q in memory_bm25)
    persisted_tfidf = ranked(tfidf_scores(index, query))
    memory_tfidf = ranked(tfidf_scores(memory, query))
    assert [q for q, _ in persisted_tfidf] == [q for q, _ in memory_tfidf]
    assert persisted_tfidf[0][0] == "pkg.billing.Invoice.total"


def test_stale_index_falls_back_to_memory(repo: Path, store: GraphStore) -> None:
    (repo / "pkg" / "extra.py").write_text("def brand_new_widget() -> None:\n    return None\n")
    with store.transaction():
        store.record_file("pkg/extra.py", "deadbeef", "python")
    index, persisted = open_index(store)
    assert not persisted and isinstance(index, Corpus)

    index_repository(repo, store)
    index, persisted = open_index(store)
    assert persisted
    assert "pkg.extra.brand_new_widget" in index.postings("widget")


def test_tokenizer_version_change_invalidates(store: GraphStore) -> None:
    with store.transaction() as conn:
        conn.execute("UPDATE search_meta SET value = 'old' WHERE key = 'tokenizer_version'")
    _, persisted = open_index(store)
    assert not persisted


def test_pass_is_idempotent(repo: Path, store: GraphStore) -> None:
    def rows() -> tuple[int, int]:
        conn = store.connection
        return (
            int(conn.execute("SELECT COUNT(*) FROM search_docs").fetchone()[0]),
            int(conn.execute("SELECT COUNT(*) FROM search_postings").fetchone()[0]),
        )

    first = rows()
    with store.transaction():
        written = SEARCH_INDEX_PASS.run(repo, store)
    assert written == first[0]
    assert rows() == first


def test_schema_creation_does_not_commit_open_transaction(tmp_path: Path) -> None:
    with GraphStore.open(tmp_path) as fresh:
        conn = fresh.connection
        conn.execute("INSERT INTO meta (key, value) VALUES ('probe', '1')")
        assert conn.in_transaction
        ensure_search_schema(fresh)
        assert conn.in_transaction
        conn.rollback()
        assert fresh.get_meta("probe") is None
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "search_docs" not in tables
        ensure_search_schema(fresh)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"search_docs", "search_postings", "search_meta"} <= tables


def test_write_index_records_digest(store: GraphStore) -> None:
    with store.transaction():
        count = write_index(store, Corpus.from_nodes(store.all_nodes()[:3]))
    assert count == 3
    meta = dict(store.connection.execute("SELECT key, value FROM search_meta").fetchall())
    assert meta["doc_count"] == "3"
    assert meta["source_digest"] == source_digest(store)
