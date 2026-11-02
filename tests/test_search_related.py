from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from codegraph.cypher import run_cypher
from codegraph.graph import GraphView
from codegraph.indexer import index_repository
from codegraph.passes import GraphPass, discover_passes
from codegraph.passes.semantic_edges import SemanticEdgesPass
from codegraph.search.index import Corpus
from codegraph.search.related import find_related, related_pairs
from codegraph.search.similarity import (
    comparison_group,
    cosine,
    neighbours,
    semantic_corpus,
    similar_pairs,
)
from codegraph.search.snippet import SymbolNotFound
from codegraph.store import GraphStore

FIXTURE = Path(__file__).parent / "fixtures" / "query_semantic"

REFUND = "billing.refunds.refund_invoice_payment"
PARTIAL = "billing.refunds.refund_invoice_payment_partially"
REVERSAL = "payments.reversal.issue_invoice_payment_refund"
DESK = "payments.desk.desk_refund_invoice_payment"


@pytest.fixture
def semantic_repo(tmp_path: Path) -> Path:
    target = tmp_path / "semantic"
    shutil.copytree(FIXTURE, target)
    return target


@pytest.fixture
def semantic_store(semantic_repo: Path) -> Iterator[GraphStore]:
    with GraphStore.open(semantic_repo) as graph_store:
        index_repository(semantic_repo, graph_store)
        yield graph_store


# -- similarity primitives ---------------------------------------------------------


def test_comparison_groups() -> None:
    assert comparison_group("function") == frozenset({"function", "method"})
    assert comparison_group("method") == frozenset({"function", "method"})
    assert comparison_group("route") == frozenset({"route"})


def test_cosine_of_unit_vectors() -> None:
    assert cosine({"a": 1.0}, {"a": 1.0}) == 1.0
    assert cosine({"a": 1.0}, {"b": 1.0}) == 0.0
    assert cosine({}, {"a": 1.0}) == 0.0


def test_similar_pairs_and_neighbours_from_a_corpus() -> None:
    corpus = Corpus.build(
        [
            ("x", "function", ["refund", "invoice", "payment", "card"]),
            ("y", "function", ["refund", "invoice", "payment", "card", "issue"]),
            ("z", "function", ["slug", "title", "lower"]),
            ("w", "function", ["tar", "gzip", "archive"]),
        ]
    )
    pairs = list(similar_pairs(corpus, 0.5))
    assert [(p.a, p.b) for p in pairs] == [("x", "y")]
    assert 0.5 <= pairs[0].score <= 1.0
    assert list(similar_pairs(corpus, 0.5, max_doc_freq=1)) == []
    near = neighbours(corpus, "x")
    assert set(near) == {"y"}
    assert neighbours(corpus, "missing") == {}


def test_semantic_corpus_skips_wordless_symbols(semantic_store: GraphStore) -> None:
    nodes = semantic_store.all_nodes(kinds=["function"])
    corpus = semantic_corpus(nodes)
    assert REFUND in corpus.docs
    loose = semantic_corpus(nodes, min_terms=1)
    assert loose.doc_count >= corpus.doc_count


# -- related pairs and the pass ------------------------------------------------------


def test_pass_links_cross_module_lookalikes(semantic_store: GraphStore) -> None:
    edges = semantic_store.all_edges(kinds=["semantically_related"])
    assert [(e.src, e.dst) for e in edges] == [(REFUND, REVERSAL)]
    [edge] = edges
    assert edge.confidence >= 0.8
    assert edge.detail == f"tfidf={edge.confidence:.2f}"
    assert edge.path == "billing/refunds.py"
    rows = run_cypher(
        semantic_store,
        "MATCH (a)-[r:SEMANTICALLY_RELATED]-(b {name: 'issue_invoice_payment_refund'}) "
        "RETURN a.qname, r.confidence",
    ).rows
    assert rows == ((REFUND, edge.confidence),)


def test_related_pairs_exclude_structure_and_same_module(semantic_store: GraphStore) -> None:
    view = GraphView.load(semantic_store)
    pairs = {(p.a, p.b) for p in related_pairs(view, threshold=0.5, top_k=10)}
    assert (REFUND, REVERSAL) in pairs
    assert (REFUND, PARTIAL) not in pairs
    assert (REFUND, DESK) not in pairs
    assert all(not ("text.slugs" in a or "text.slugs" in b) for a, b in pairs)
    assert all(view.nodes[a].path != view.nodes[b].path for a, b in pairs)


def test_related_pairs_respect_top_k(semantic_store: GraphStore) -> None:
    view = GraphView.load(semantic_store)
    loose = related_pairs(view, threshold=0.3, top_k=10)
    capped = related_pairs(view, threshold=0.3, top_k=1)
    assert len(capped) < len(loose)
    per_node: dict[str, int] = {}
    for pair in capped:
        per_node[pair.a] = per_node.get(pair.a, 0) + 1
        per_node[pair.b] = per_node.get(pair.b, 0) + 1
    assert max(per_node.values()) == 1
    scores = [p.score for p in loose]
    assert scores == sorted(scores, reverse=True)


def test_pass_is_idempotent_and_tunable(semantic_repo: Path, semantic_store: GraphStore) -> None:
    default = discover_passes()
    found = next(p for p in default if p.name == "semantic_edges")
    assert isinstance(found, GraphPass) and 90 <= found.order <= 99
    with semantic_store.transaction():
        assert SemanticEdgesPass().run(semantic_repo, semantic_store) == 1
        assert SemanticEdgesPass().run(semantic_repo, semantic_store) == 1
    assert len(semantic_store.all_edges(kinds=["semantically_related"])) == 1
    with semantic_store.transaction():
        loose = SemanticEdgesPass(threshold=0.5, top_k=10).run(semantic_repo, semantic_store)
    assert loose > 1
    assert len(semantic_store.all_edges(kinds=["semantically_related"])) == loose


def test_pass_can_be_skipped(semantic_repo: Path, tmp_path: Path) -> None:
    with GraphStore(tmp_path / "skip.db") as fresh:
        index_repository(semantic_repo, fresh, skip_passes=frozenset({"semantic_edges"}))
        assert fresh.all_edges(kinds=["semantically_related"]) == []


def test_sample_repo_gets_no_spurious_edges(store: GraphStore) -> None:
    assert store.all_edges(kinds=["semantically_related"]) == []


# -- find_related --------------------------------------------------------------------


def test_find_related_ranks_neighbours(semantic_store: GraphStore) -> None:
    result = find_related(semantic_store, REFUND, limit=3)
    assert result.target.qname == REFUND
    qnames = [hit.node.qname for hit in result.hits]
    assert qnames[0] == REVERSAL
    assert set(qnames) == {REVERSAL, PARTIAL, DESK}
    by_name = {hit.node.qname: hit for hit in result.hits}
    assert by_name[PARTIAL].same_module and by_name[PARTIAL].connected
    assert by_name[DESK].connected and not by_name[DESK].same_module
    assert not by_name[REVERSAL].connected and by_name[REVERSAL].semantic_edge
    assert not by_name[DESK].semantic_edge
    scores = [hit.score for hit in result.hits]
    assert scores == sorted(scores, reverse=True)
    payload = result.as_dict()
    assert payload["target"] == REFUND and len(payload["results"]) == 3  # type: ignore[arg-type]


def test_find_related_min_score_and_unrelated(semantic_store: GraphStore) -> None:
    strict = find_related(semantic_store, REFUND, min_score=0.8)
    assert [hit.node.qname for hit in strict.hits] == [REVERSAL]
    slug = find_related(semantic_store, "slugify_title")
    assert all(hit.score < 0.3 for hit in slug.hits)


def test_find_related_errors(semantic_store: GraphStore) -> None:
    with pytest.raises(SymbolNotFound):
        find_related(semantic_store, "no_such_symbol")
    with pytest.raises(ValueError, match="limit"):
        find_related(semantic_store, REFUND, limit=0)


def test_find_related_for_wordless_symbol(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text("def a():\n    return None\n\n\ndef parse_header():\n    pass\n")
    with GraphStore.open(tmp_path) as tiny:
        index_repository(tmp_path, tiny)
        result = find_related(tiny, "m.a")
    assert result.hits == ()
    assert "no descriptive words" in result.note
    assert result.as_dict()["note"] == result.note


def test_similar_pairs_candidate_budget_prefers_rare_terms() -> None:
    corpus = Corpus.build(
        [
            ("a", "function", ["common", "rare", "word"]),
            ("b", "function", ["common", "rare", "word"]),
            ("c", "function", ["common", "other", "thing"]),
            ("d", "function", ["common", "other", "thing"]),
        ]
    )
    everything = {(p.a, p.b) for p in similar_pairs(corpus, 0.9)}
    assert everything == {("a", "b"), ("c", "d")}
    budgeted = {(p.a, p.b) for p in similar_pairs(corpus, 0.9, max_candidates=1)}
    assert len(budgeted) == 1 and budgeted <= everything
