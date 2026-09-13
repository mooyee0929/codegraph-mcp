from __future__ import annotations

import json
from pathlib import Path

import pytest

from codegraph.cli import main
from codegraph.server import build_server


def test_cli_end_to_end(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--root", str(repo), "index"]) == 0
    assert "parsed 8" in capsys.readouterr().out

    assert main(["--root", str(repo), "query", "callers_of", "pkg.billing.charge"]) == 0
    assert "tests.test_billing.test_charge" in capsys.readouterr().out

    assert main(["--root", str(repo), "impact", "Invoice.rate", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "pkg.billing.Invoice.rate"

    assert main(["--root", str(repo), "search", "zzz_nothing"]) == 1
    assert main(["--root", str(repo), "review"]) == 0
    assert "no indexed symbols" in capsys.readouterr().out
    assert main(["--root", str(repo), "arch"]) == 0
    assert main(["--root", str(repo), "show", "charge"]) == 0
    assert "def charge" in capsys.readouterr().out


async def test_mcp_tools(repo: Path) -> None:
    server = build_server(repo)
    names = {t.name for t in await server.list_tools()}
    assert {
        "index_repository_tool",
        "graph_status",
        "search_nodes",
        "get_node",
        "query_graph",
        "get_impact_radius",
        "detect_changes_tool",
        "get_review_context",
        "get_architecture_overview",
        "find_dead_code",
    } <= names

    indexed = await server.call_tool("index_repository_tool", {})
    assert "parsed" in _text(indexed)

    status = json.loads(_text(await server.call_tool("graph_status", {})))
    assert status["files"] == 8

    result = json.loads(
        _text(await server.call_tool("query_graph", {"pattern": "callers_of", "target": "charge"}))
    )
    assert result["results"] == ["tests.test_billing.test_charge"]

    impact = json.loads(
        _text(await server.call_tool("get_impact_radius", {"target": "Invoice.rate"}))
    )
    assert "pkg.billing.charge" in impact["transitive_callers"]

    missing = json.loads(_text(await server.call_tool("get_node", {"target": "nope"})))
    assert "error" in missing


def _text(result: object) -> str:
    """Concatenate the text blocks of a CallToolResult."""
    blocks = getattr(result, "content", None)
    assert isinstance(blocks, list), f"unexpected tool result: {result!r}"
    return "".join(str(getattr(block, "text", "")) for block in blocks)
