from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from codegraph.cli import main
from codegraph.search import embeddings
from codegraph.server import build_server

SEARCH_TOOLS = {
    "search_graph",
    "search_code",
    "get_code_snippet",
    "semantic_search_nodes",
    "embed_graph",
    "find_related",
}


class WordEncoder:
    model = "word-model"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float("invoice" in t.lower()), float("handler" in t.lower()), 0.1] for t in texts]


def _text(result: object) -> str:
    blocks = getattr(result, "content", None)
    assert isinstance(blocks, list), f"unexpected tool result: {result!r}"
    return "".join(str(getattr(block, "text", "")) for block in blocks)


async def _call(repo: Path, name: str, args: dict[str, object]) -> dict[str, object]:
    server = build_server(repo)
    payload = json.loads(_text(await server.call_tool(name, args)))
    assert isinstance(payload, dict)
    return payload


@pytest.fixture
def indexed(repo: Path) -> Path:
    assert main(["--root", str(repo), "index"]) == 0
    return repo


# -- MCP tools -----------------------------------------------------------------------


async def test_search_tools_are_registered(repo: Path) -> None:
    names = {t.name for t in await build_server(repo).list_tools()}
    assert names >= SEARCH_TOOLS


async def test_search_graph_tool(indexed: Path) -> None:
    page = await _call(indexed, "search_graph", {"kind": "Callable", "sort": "fan_out", "limit": 2})
    results = page["results"]
    assert isinstance(results, list) and len(results) == 2
    assert results[0]["qname"] == "pkg.billing.charge"
    assert page["has_more"] is True
    tested = await _call(indexed, "search_graph", {"has_tests": True, "name_pattern": "charge"})
    assert [r["qname"] for r in tested["results"]] == ["pkg.billing.charge"]  # type: ignore[union-attr]
    bad = await _call(indexed, "search_graph", {"kind": "widget"})
    assert "unknown kind" in str(bad["error"])
    bad_regex = await _call(indexed, "search_graph", {"name_pattern": "("})
    assert "invalid name_pattern" in str(bad_regex["error"])


async def test_search_code_tool(indexed: Path) -> None:
    found = await _call(indexed, "search_code", {"pattern": "inv.total()", "context": 1})
    [match] = found["matches"]  # type: ignore[misc]
    assert match["symbol"] == "pkg.billing.charge"
    assert match["before"] == ["    inv.save()"]
    bad = await _call(indexed, "search_code", {"pattern": "(", "regex": True})
    assert "invalid regular expression" in str(bad["error"])


async def test_get_code_snippet_tool(indexed: Path) -> None:
    snippet = await _call(indexed, "get_code_snippet", {"target": "Invoice.total"})
    assert snippet["qname"] == "pkg.billing.Invoice.total"
    assert "return self.rate() * 2" in str(snippet["source"])
    missing = await _call(indexed, "get_code_snippet", {"target": "nothing_like_this"})
    assert "matches no symbol" in str(missing["error"]) and missing["candidates"] == []
    bad = await _call(indexed, "get_code_snippet", {"target": "charge", "context": -1})
    assert "context" in str(bad["error"])


async def test_semantic_search_tool(indexed: Path) -> None:
    result = await _call(indexed, "semantic_search_nodes", {"query": "invoice rate", "limit": 3})
    assert result["backend"] == "bm25+tfidf+fts5"
    hits = result["results"]
    assert isinstance(hits, list) and hits[0]["qname"] == "pkg.billing.Invoice.rate"
    only_classes = await _call(
        indexed, "semantic_search_nodes", {"query": "invoice", "kind": "class"}
    )
    assert [h["qname"] for h in only_classes["results"]] == ["pkg.billing.Invoice"]  # type: ignore[union-attr]
    bad = await _call(indexed, "semantic_search_nodes", {"query": "x", "limit": 0})
    assert "limit" in str(bad["error"])


async def test_embed_graph_tool_without_extra(
    indexed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(embeddings, "embeddings_installed", lambda: False)
    embeddings.load_encoder.cache_clear()
    result = await _call(indexed, "embed_graph", {})
    assert "codegraph-mcp[embeddings]" in str(result["error"])
    embeddings.load_encoder.cache_clear()


async def test_embed_graph_tool_with_fake_encoder(
    indexed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = WordEncoder()
    monkeypatch.setattr(embeddings, "embeddings_installed", lambda: True)
    monkeypatch.setattr(embeddings, "load_encoder", lambda model: encoder)
    stats = await _call(indexed, "embed_graph", {"model": "word-model", "batch_size": 5})
    assert stats["model"] == "word-model" and stats["embedded"] == stats["total"]
    again = await _call(indexed, "embed_graph", {})
    assert again["embedded"] == 0
    result = await _call(indexed, "semantic_search_nodes", {"query": "invoice"})
    assert str(result["backend"]).endswith("embeddings:word-model")


async def test_semantic_search_notes_missing_extra(
    indexed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(embeddings, "embeddings_installed", lambda: True)
    monkeypatch.setattr(embeddings, "load_encoder", lambda model: WordEncoder())
    await _call(indexed, "embed_graph", {})
    monkeypatch.setattr(embeddings, "embeddings_installed", lambda: False)
    result = await _call(indexed, "semantic_search_nodes", {"query": "invoice"})
    assert "embeddings" not in str(result["backend"])
    assert any("codegraph-mcp[embeddings]" in n for n in result["notes"])  # type: ignore[union-attr]


async def test_find_related_tool(indexed: Path) -> None:
    result = await _call(indexed, "find_related", {"target": "pkg.billing.charge"})
    assert result["target"] == "pkg.billing.charge"
    assert isinstance(result["results"], list)
    missing = await _call(indexed, "find_related", {"target": "nope_nope"})
    assert "error" in missing and missing["candidates"] == []
    bad = await _call(indexed, "find_related", {"target": "charge", "limit": 0})
    assert "limit" in str(bad["error"])


# -- CLI commands --------------------------------------------------------------------


def test_cli_find(indexed: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = ["--root", str(indexed)]
    assert main([*root, "find", "--kind", "function", "--untested", "--name", "helper"]) == 0
    out = capsys.readouterr().out
    assert "pkg.billing.unused_helper" in out and "web.util.helper" in out
    assert main([*root, "find", "--name", "^charge$", "--tested", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 1 and payload["results"][0]["fan_out"] == 3
    assert main([*root, "find", "--name", "zzz"]) == 1
    assert main([*root, "find", "--kind", "widget"]) == 2
    assert main([*root, "find", "--sort", "fan_in", "--limit", "2", "--offset", "1"]) == 0
    assert "2-3 of" in capsys.readouterr().err


def test_cli_grep(indexed: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = ["--root", str(indexed)]
    assert main([*root, "grep", "fmt(", "-C", "1"]) == 0
    out = capsys.readouterr().out
    assert "pkg/billing.py:19:12  [pkg.billing.charge]" in out
    assert "pkg/billing.py-18-" in out and "pkg/billing.py-20-" in out
    assert main([*root, "grep", "INVOICE", "-i", "--file", "pkg/*", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["match_count"] > 0
    assert all(m["path"].startswith("pkg/") for m in payload["matches"])
    assert main([*root, "grep", "no such text anywhere"]) == 1
    assert main([*root, "grep", "(", "--regex"]) == 2


def test_cli_semantic(indexed: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = ["--root", str(indexed)]
    assert main([*root, "semantic", "invoice total", "--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("backend: bm25+tfidf+fts5")
    assert "pkg.billing.Invoice.total" in out and "bm25#1" in out
    assert main([*root, "semantic", "invoice", "--kind", "class", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["qname"] for r in payload["results"]] == ["pkg.billing.Invoice"]
    assert main([*root, "semantic", "??"]) == 1
    assert main([*root, "semantic", "x", "--kind", "widget"]) == 2


def test_cli_embed(
    indexed: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = ["--root", str(indexed)]
    monkeypatch.setattr(embeddings, "embeddings_installed", lambda: False)
    embeddings.load_encoder.cache_clear()
    assert main([*root, "embed"]) == 2
    assert "codegraph-mcp[embeddings]" in capsys.readouterr().err
    assert main([*root, "embed", "--batch-size", "0"]) == 2

    monkeypatch.setattr(embeddings, "embeddings_installed", lambda: True)
    monkeypatch.setattr(embeddings, "load_encoder", lambda model: WordEncoder())
    assert main([*root, "embed"]) == 0
    assert "word-model: embedded" in capsys.readouterr().out
    assert main([*root, "embed", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["embedded"] == 0
    assert main([*root, "semantic", "invoice"]) == 0
    assert "embeddings:word-model" in capsys.readouterr().out


def test_cli_related(indexed: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = ["--root", str(indexed)]
    assert main([*root, "related", "pkg.billing.charge"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("target: pkg.billing.charge")
    assert "tests.test_billing.test_charge" in out and "[connected]" in out
    assert main([*root, "related", "charge", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["target"] == "pkg.billing.charge"
    assert main([*root, "related", "charge", "--min-score", "0.99"]) == 1
    assert main([*root, "related", "nope_nope"]) == 1
    assert main([*root, "related", "charge", "--limit", "0"]) == 2
