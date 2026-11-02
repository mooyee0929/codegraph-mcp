from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from codegraph.indexer import index_repository
from codegraph.models import Node
from codegraph.search import embeddings
from codegraph.search.hybrid import RRF_K, hybrid_search, rrf_fuse
from codegraph.store import GraphStore

_CONCEPTS = (
    {"invoice", "charge", "fmt", "money", "billing", "rate", "total", "payment"},
    {"handler", "request", "web", "app", "start", "helper", "compute", "run"},
    {"model", "base", "save", "validate", "persist", "store"},
)


class FakeEncoder:
    """Maps texts onto three concept axes so tests can reason about nearness."""

    def __init__(self, model: str = "fake-model") -> None:
        self._model = model
        self.encoded: list[str] = []

    @property
    def model(self) -> str:
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.encoded.extend(texts)
        vectors: list[list[float]] = []
        for text in texts:
            words = {w.strip(".,()|:").lower() for w in text.replace("_", " ").split()}
            vectors.append([float(len(words & concept)) + 0.01 for concept in _CONCEPTS])
        return vectors


class ShortEncoder(FakeEncoder):
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return super().encode(texts)[:-1]


# -- fusion --------------------------------------------------------------------


def test_rrf_fuse_rewards_agreement() -> None:
    fused = rrf_fuse({"one": ["a", "b", "c"], "two": ["b", "a"], "three": ["b"]})
    assert [q for q, _, _ in fused] == ["b", "a", "c"]
    qname, score, ranks = fused[0]
    assert ranks == {"one": 2, "two": 1, "three": 1}
    assert math.isclose(score, 1 / (RRF_K + 2) + 2 / (RRF_K + 1))
    assert rrf_fuse({}) == []


def test_rrf_ties_break_on_qname() -> None:
    assert [q for q, _, _ in rrf_fuse({"x": ["b"], "y": ["a"]})] == ["a", "b"]


# -- hybrid search without embeddings -----------------------------------------


def test_hybrid_search_uses_word_and_fts_rankers(store: GraphStore) -> None:
    result = hybrid_search(store, "invoice total", limit=3)
    assert result.backend == "bm25+tfidf+fts5"
    assert result.hits[0].node.qname == "pkg.billing.Invoice.total"
    assert set(result.hits[0].ranks) == {"bm25", "tfidf", "fts5"}
    assert len(result.hits) == 3
    assert result.notes == ()
    payload = result.as_dict()
    assert payload["backend"] == "bm25+tfidf+fts5"
    first = payload["results"]
    assert isinstance(first, list) and first[0]["qname"] == "pkg.billing.Invoice.total"


def test_split_identifiers_find_what_fts_misses(store: GraphStore) -> None:
    result = hybrid_search(store, "format", limit=5)
    assert result.hits[0].node.qname == "pkg.util.fmt"
    assert "fts5" not in result.hits[0].ranks


def test_kind_filter(store: GraphStore) -> None:
    result = hybrid_search(store, "invoice", kinds=["class"])
    assert [h.node.qname for h in result.hits] == ["pkg.billing.Invoice"]


def test_query_without_terms(store: GraphStore) -> None:
    result = hybrid_search(store, "??")
    assert result.backend == "none" and result.hits == ()
    with pytest.raises(ValueError, match="limit"):
        hybrid_search(store, "x", limit=0)


def test_missing_index_is_built_in_memory(store: GraphStore) -> None:
    with store.transaction() as conn:
        conn.execute("DELETE FROM search_meta")
    result = hybrid_search(store, "invoice total")
    assert result.hits[0].node.qname == "pkg.billing.Invoice.total"
    assert any("in memory" in note for note in result.notes)


# -- embeddings ----------------------------------------------------------------


def test_pack_round_trip_is_float32() -> None:
    blob = embeddings.pack([0.5, -1.25, 3.0])
    assert len(blob) == 12
    assert embeddings.unpack(blob) == [0.5, -1.25, 3.0]


def test_node_text_has_words_and_context() -> None:
    node = Node(
        qname="web.app.Handler.handleRequest",
        kind="method",
        path="web/app.ts",
        line_start=1,
        line_end=3,
        language="typescript",
        name="handleRequest",
        signature="handleRequest(req: Request)",
        docstring="Handles   a\nrequest.",
    )
    text = embeddings.node_text(node)
    assert "handleRequest" in text and "handle request" in text
    assert "in web.app.Handler" in text
    assert "Handles a request." in text
    assert text.endswith("web")


def test_embed_graph_refreshes_only_changes(repo: Path, store: GraphStore) -> None:
    encoder = FakeEncoder()
    first = embeddings.embed_graph(store, encoder, batch_size=4)
    expected = len(store.all_nodes(kinds=list(embeddings.EMBEDDED_KINDS)))
    assert (first.embedded, first.unchanged, first.removed) == (expected, 0, 0)
    assert first.total == expected
    assert embeddings.stored_models(store) == {"fake-model": expected}

    encoder.encoded.clear()
    second = embeddings.embed_graph(store, encoder)
    assert (second.embedded, second.unchanged) == (0, expected)
    assert encoder.encoded == []

    util = repo / "pkg" / "util.py"
    util.write_text('def fmt(value: int) -> str:\n    """Money format."""\n    return str(value)\n')
    index_repository(repo, store)
    third = embeddings.embed_graph(store, encoder)
    assert third.embedded == 1
    assert len(encoder.encoded) == 1 and "Money format." in encoder.encoded[0]

    util.unlink()
    index_repository(repo, store)
    fourth = embeddings.embed_graph(store, encoder)
    assert fourth.removed == 1 and fourth.embedded == 0

    other = embeddings.embed_graph(store, FakeEncoder("other-model"))
    assert other.embedded == expected - 1
    assert embeddings.stored_models(store) == {"other-model": expected - 1}


def test_embed_graph_validates(store: GraphStore) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        embeddings.embed_graph(store, FakeEncoder(), batch_size=0)
    with pytest.raises(embeddings.EmbeddingsUnavailable, match="vectors for"):
        embeddings.embed_graph(store, ShortEncoder())
    assert embeddings.stored_models(store) == {}


def test_embedding_scores_and_kind_filter(store: GraphStore) -> None:
    encoder = FakeEncoder()
    embeddings.embed_graph(store, encoder)
    scores = embeddings.embedding_scores(store, encoder, "money payment")
    best = max(scores, key=lambda q: scores[q])
    assert best.startswith("pkg.billing") or best == "pkg.util.fmt"
    classes = embeddings.embedding_scores(store, encoder, "money", kinds=["class"])
    assert classes and all(
        q in {"pkg.billing.Invoice", "pkg.models.Base", "web.app.Handler"} for q in classes
    )
    assert embeddings.embedding_scores(store, FakeEncoder("unknown"), "money") == {}


def test_hybrid_search_blends_embeddings(store: GraphStore) -> None:
    encoder = FakeEncoder()
    embeddings.embed_graph(store, encoder)
    result = hybrid_search(store, "money payment", encoder=encoder, limit=5)
    assert result.backend.endswith("embeddings:fake-model")
    assert any("embeddings:fake-model" in hit.ranks for hit in result.hits)
    assert result.hits[0].node.qname.startswith(("pkg.billing", "pkg.util"))


def test_hybrid_search_notes_missing_vectors(store: GraphStore) -> None:
    result = hybrid_search(store, "invoice", encoder=FakeEncoder())
    assert any("run embed_graph" in note for note in result.notes)
    assert "embeddings" not in result.backend


def test_search_encoder_selection(store: GraphStore, monkeypatch: pytest.MonkeyPatch) -> None:
    assert embeddings.search_encoder(store) == (None, "")
    embeddings.embed_graph(store, FakeEncoder())

    monkeypatch.setattr(embeddings, "embeddings_installed", lambda: False)
    encoder, note = embeddings.search_encoder(store)
    assert encoder is None and "codegraph-mcp[embeddings]" in note

    encoder, note = embeddings.search_encoder(store, "missing-model")
    assert encoder is None and "missing-model" in note

    fake = FakeEncoder()
    monkeypatch.setattr(embeddings, "embeddings_installed", lambda: True)
    monkeypatch.setattr(embeddings, "load_encoder", lambda model: fake)
    encoder, note = embeddings.search_encoder(store)
    assert encoder is fake and note == ""


def test_load_encoder_without_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embeddings, "embeddings_installed", lambda: False)
    embeddings.load_encoder.cache_clear()
    with pytest.raises(embeddings.EmbeddingsUnavailable, match=r"codegraph-mcp\[embeddings\]"):
        embeddings.load_encoder("all-MiniLM-L6-v2")
    lazy = embeddings.SentenceTransformerEncoder("all-MiniLM-L6-v2")
    assert lazy.model == "all-MiniLM-L6-v2"
    with pytest.raises(embeddings.EmbeddingsUnavailable, match="not installed"):
        lazy.encode(["text"])
    embeddings.load_encoder.cache_clear()


def test_default_model_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(embeddings.MODEL_ENV, raising=False)
    assert embeddings.default_model() == embeddings.DEFAULT_MODEL
    monkeypatch.setenv(embeddings.MODEL_ENV, "my-model")
    assert embeddings.default_model() == "my-model"


class BrokenEncoder(FakeEncoder):
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        raise embeddings.EmbeddingsUnavailable("model download failed")


def test_hybrid_search_falls_back_when_encoder_fails(store: GraphStore) -> None:
    embeddings.embed_graph(store, FakeEncoder())
    result = hybrid_search(store, "invoice total", encoder=BrokenEncoder())
    assert result.backend == "bm25+tfidf+fts5"
    assert result.hits[0].node.qname == "pkg.billing.Invoice.total"
    assert any("model download failed" in note for note in result.notes)
