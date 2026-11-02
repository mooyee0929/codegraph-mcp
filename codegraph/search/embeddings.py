"""Optional neural embeddings for :func:`codegraph.search.hybrid.hybrid_search`.

Everything here works without the ``[embeddings]`` extra installed: the encoder sits
behind the small :class:`Encoder` protocol, ``sentence_transformers`` is imported only
when an encoder is actually built, and callers learn it is missing from
:class:`EmbeddingsUnavailable` (with the install command) instead of an ImportError at
import time. Tests pass a fake encoder.

Vectors live in ``search_embeddings`` as float32 blobs keyed by qname, together with the
model name and a hash of the text that was embedded. :func:`embed_graph` re-encodes only
nodes whose text or model changed, and drops rows for nodes that no longer exist.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import math
import os
from array import array
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import PurePosixPath
from typing import Protocol

from codegraph.models import Node, NodeKind
from codegraph.search.tokenize import split_identifier
from codegraph.store import GraphStore

DEFAULT_MODEL = "all-MiniLM-L6-v2"
MODEL_ENV = "CODEGRAPH_EMBEDDING_MODEL"
INSTALL_HINT = "install the optional extra: uv pip install 'codegraph-mcp[embeddings]'"

EMBEDDINGS_DDL = """
CREATE TABLE IF NOT EXISTS search_embeddings (
    qname     TEXT PRIMARY KEY,
    model     TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL
);
"""

EMBEDDED_KINDS: tuple[NodeKind, ...] = ("class", "function", "method", "route", "channel")
"""Modules are left out: a whole file's text says little an embedding can use."""

_MAX_DOCSTRING_CHARS = 400


class Encoder(Protocol):
    @property
    def model(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class EmbeddingsUnavailable(RuntimeError):
    """The ``[embeddings]`` extra is not installed or the model could not be loaded."""


@dataclass(frozen=True, slots=True)
class EmbedStats:
    model: str
    embedded: int
    unchanged: int
    removed: int

    @property
    def total(self) -> int:
        return self.embedded + self.unchanged


def default_model() -> str:
    return os.environ.get(MODEL_ENV) or DEFAULT_MODEL


def embeddings_installed() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


class SentenceTransformerEncoder:
    """Wraps a ``sentence_transformers`` model, loaded on first use."""

    def __init__(self, model: str) -> None:
        self._model = model
        self._encode: Callable[[list[str]], list[list[float]]] | None = None

    @property
    def model(self) -> str:
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if self._encode is None:
            self._encode = _load_sentence_transformer(self._model)
        return self._encode(list(texts))


def _load_sentence_transformer(model: str) -> Callable[[list[str]], list[list[float]]]:
    if not embeddings_installed():
        raise EmbeddingsUnavailable(f"sentence-transformers is not installed; {INSTALL_HINT}")
    module = importlib.import_module("sentence_transformers")
    try:
        impl = module.SentenceTransformer(model)
    except (OSError, ValueError) as exc:
        raise EmbeddingsUnavailable(f"could not load embedding model {model!r}: {exc}") from exc

    def encode(texts: list[str]) -> list[list[float]]:
        matrix = impl.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in row] for row in matrix.tolist()]

    return encode


@cache
def load_encoder(model: str) -> Encoder:
    """A cached encoder for ``model``; raises :class:`EmbeddingsUnavailable` without the extra."""
    if not embeddings_installed():
        raise EmbeddingsUnavailable(f"sentence-transformers is not installed; {INSTALL_HINT}")
    return SentenceTransformerEncoder(model)


def ensure_embeddings_schema(store: GraphStore) -> None:
    if store.connection.in_transaction:
        store.connection.execute(EMBEDDINGS_DDL)
    else:
        store.ensure_schema(EMBEDDINGS_DDL)


def node_text(node: Node) -> str:
    """What gets embedded: names in both code and word form, signature, docstring, place."""
    parts = [node.name]
    words = " ".join(split_identifier(node.name))
    if words and words != node.name.lower():
        parts.append(words)
    parts.append(node.kind)
    owner = node.qname.rsplit(".", 1)[0] if "." in node.qname else ""
    if owner:
        parts.append(f"in {owner}")
    if node.signature:
        parts.append(node.signature)
    if node.docstring:
        parts.append(" ".join(node.docstring.split())[:_MAX_DOCSTRING_CHARS])
    directory = PurePosixPath(node.path).parent.name
    if directory:
        parts.append(directory)
    return " | ".join(parts)


def pack(vector: Sequence[float]) -> bytes:
    return array("f", vector).tobytes()


def unpack(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return values.tolist()


def embed_graph(
    store: GraphStore,
    encoder: Encoder,
    *,
    batch_size: int = 64,
    kinds: Collection[NodeKind] = EMBEDDED_KINDS,
) -> EmbedStats:
    """Bring ``search_embeddings`` up to date for ``encoder.model``; commits."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    ensure_embeddings_schema(store)
    existing = {
        str(row["qname"]): (str(row["model"]), str(row["text_hash"]))
        for row in store.connection.execute("SELECT qname, model, text_hash FROM search_embeddings")
    }
    nodes = store.all_nodes(kinds=list(kinds))
    pending: list[tuple[Node, str, str]] = []
    unchanged = 0
    for node in nodes:
        text = node_text(node)
        digest = hashlib.sha256(text.encode()).hexdigest()
        if existing.get(node.qname) == (encoder.model, digest):
            unchanged += 1
            continue
        pending.append((node, text, digest))
    live = {node.qname for node in nodes}
    stale = [qname for qname in existing if qname not in live]
    with store.transaction() as conn:
        conn.executemany("DELETE FROM search_embeddings WHERE qname = ?", ((q,) for q in stale))
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            vectors = encoder.encode([text for _, text, _ in batch])
            if len(vectors) != len(batch):
                raise EmbeddingsUnavailable(
                    f"encoder returned {len(vectors)} vectors for {len(batch)} texts"
                )
            conn.executemany(
                "INSERT INTO search_embeddings (qname, model, text_hash, dim, vector) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(qname) DO UPDATE SET model=excluded.model, "
                "text_hash=excluded.text_hash, dim=excluded.dim, vector=excluded.vector",
                (
                    (node.qname, encoder.model, digest, len(vector), pack(vector))
                    for (node, _, digest), vector in zip(batch, vectors, strict=True)
                ),
            )
    return EmbedStats(
        model=encoder.model, embedded=len(pending), unchanged=unchanged, removed=len(stale)
    )


def stored_models(store: GraphStore) -> dict[str, int]:
    """Model name -> stored vector count, most common first."""
    ensure_embeddings_schema(store)
    rows = store.connection.execute(
        "SELECT model, COUNT(*) AS n FROM search_embeddings GROUP BY model ORDER BY n DESC, model"
    ).fetchall()
    return {str(row["model"]): int(row["n"]) for row in rows}


def search_encoder(store: GraphStore, model: str | None = None) -> tuple[Encoder | None, str]:
    """The encoder to blend into search, or None plus the reason (empty when none stored)."""
    models = stored_models(store)
    if not models:
        return None, ""
    chosen = model or next(iter(models))
    if chosen not in models:
        return None, f"no embeddings stored for model {chosen}; run embed_graph"
    if not embeddings_installed():
        return None, f"embeddings are stored but sentence-transformers is missing; {INSTALL_HINT}"
    return load_encoder(chosen), ""


def embedding_scores(
    store: GraphStore, encoder: Encoder, query: str, *, kinds: Collection[str] | None = None
) -> dict[str, float]:
    """Cosine between the query's embedding and every stored vector of the same model."""
    ensure_embeddings_schema(store)
    [query_vector] = encoder.encode([query])
    query_norm = math.sqrt(sum(x * x for x in query_vector))
    if query_norm == 0.0:
        return {}
    sql = (
        "SELECT e.qname, e.vector FROM search_embeddings e JOIN nodes n ON n.qname = e.qname "
        "WHERE e.model = ? AND e.dim = ?"
    )
    params: list[str | int] = [encoder.model, len(query_vector)]
    if kinds:
        sql += f" AND n.kind IN ({','.join('?' * len(kinds))})"
        params.extend(kinds)
    scores: dict[str, float] = {}
    for row in store.connection.execute(sql, params):
        vector = unpack(row["vector"])
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0.0:
            dot = sum(a * b for a, b in zip(query_vector, vector, strict=True))
            scores[str(row["qname"])] = dot / (query_norm * norm)
    return scores


__all__ = [
    "DEFAULT_MODEL",
    "EMBEDDED_KINDS",
    "INSTALL_HINT",
    "EmbedStats",
    "EmbeddingsUnavailable",
    "Encoder",
    "SentenceTransformerEncoder",
    "default_model",
    "embed_graph",
    "embedding_scores",
    "embeddings_installed",
    "load_encoder",
    "node_text",
    "pack",
    "search_encoder",
    "stored_models",
    "unpack",
]
