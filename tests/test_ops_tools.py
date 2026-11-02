from __future__ import annotations

import json
from pathlib import Path

import pytest

from codegraph import prompts
from codegraph.server import build_server

OPS_TOOLS = {
    "list_repos",
    "register_repo",
    "cross_repo_search",
    "cross_repo_links",
    "delete_project",
    "export_snapshot",
    "import_snapshot",
    "manage_adr",
    "save_note",
    "search_notes",
    "refactor_preview",
    "apply_refactor",
    "suggest_refactorings",
    "generate_wiki",
    "get_wiki_page",
    "generate_visualization",
    "export_graph",
}


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEGRAPH_HOME", str(tmp_path / "cg-home"))
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))


def _text(result: object) -> str:
    blocks = getattr(result, "content", None)
    assert isinstance(blocks, list), f"unexpected tool result: {result!r}"
    return "".join(str(getattr(block, "text", "")) for block in blocks)


async def _call(repo: Path, name: str, arguments: dict[str, object]) -> object:
    server = build_server(repo)
    payload: object = json.loads(_text(await server.call_tool(name, arguments)))
    return payload


async def test_tools_and_prompts_are_registered(repo: Path) -> None:
    server = build_server(repo)
    names = {t.name for t in await server.list_tools()}
    assert names >= OPS_TOOLS
    listed = {p.name for p in await server.list_prompts()}
    assert listed >= set(prompts.PROMPT_NAMES)
    rendered = await server.get_prompt("pre_merge_check", {"base": "develop"})
    text = str(getattr(rendered.messages[0].content, "text", ""))
    assert "base='develop'" in text and "get_change_impact" in text


def test_prompt_texts_mention_their_tools() -> None:
    assert "detect_changes_tool" in prompts.review_changes()
    assert "base='main'" in prompts.review_changes("main")
    assert "list_communities" in prompts.architecture_map()
    assert "flaky login" in prompts.debug_issue("flaky login")
    assert "semantic_search_nodes" in prompts.debug_issue()
    assert "run_cypher" in prompts.onboard_developer()
    assert "GO or NO-GO" in prompts.pre_merge_check()


async def test_registry_tools(repo: Path) -> None:
    server = build_server(repo)
    await server.call_tool("index_repository_tool", {})

    registered = json.loads(_text(await server.call_tool("register_repo", {"name": "sample"})))
    assert registered["registered"]["name"] == "sample"
    listed = json.loads(_text(await server.call_tool("list_repos", {})))
    [status] = listed["repos"]
    assert status["indexed"] and status["files"] == 8 and status["last_indexed_iso"]

    hits = json.loads(_text(await server.call_tool("cross_repo_search", {"query": "charge"})))
    assert hits[0]["repo"] == "sample" and hits[0]["qname"] == "pkg.billing.charge"
    links = json.loads(_text(await server.call_tool("cross_repo_links", {})))
    assert links == {"links": [], "count": 0}
    missing = json.loads(_text(await server.call_tool("cross_repo_links", {"repo": "x"})))
    assert "error" in missing

    dry = json.loads(_text(await server.call_tool("delete_project", {"repo": "sample"})))
    assert dry["confirm_required"] and (repo / ".codegraph" / "graph.db").exists()
    done = json.loads(
        _text(await server.call_tool("delete_project", {"repo": "sample", "confirm": True}))
    )
    assert done["deleted"] and not (repo / ".codegraph" / "graph.db").exists()


async def test_snapshot_tools(repo: Path) -> None:
    server = build_server(repo)
    await server.call_tool("index_repository_tool", {})
    exported = json.loads(_text(await server.call_tool("export_snapshot", {})))
    assert exported["gitattributes_updated"] and exported["nodes"] > 0
    imported = json.loads(_text(await server.call_tool("import_snapshot", {})))
    assert imported["nodes"] == exported["nodes"] and "scanned 8" in imported["summary"]
    (repo / ".codegraph" / "graph.db.gz").write_bytes(b"junk")
    assert "error" in json.loads(_text(await server.call_tool("import_snapshot", {})))


async def test_document_tools(repo: Path) -> None:
    server = build_server(repo)
    await server.call_tool("index_repository_tool", {})

    created = json.loads(
        _text(
            await server.call_tool(
                "manage_adr",
                {"action": "create", "title": "Use SQLite", "decision": "One file."},
            )
        )
    )
    assert created["created"]["path"] == "docs/adr/0001-use-sqlite.md"
    got = json.loads(_text(await server.call_tool("manage_adr", {"action": "get", "number": 1})))
    assert got["decision"] == "One file." and got["status"] == "Proposed"
    bad = json.loads(_text(await server.call_tool("manage_adr", {"action": "get"})))
    assert "error" in bad

    saved = json.loads(
        _text(
            await server.call_tool(
                "save_note",
                {"question": "who calls charge", "answer": "test_charge", "qnames": ["charge"]},
            )
        )
    )
    assert saved["saved"]["path"].startswith(".codegraph/notes/")
    found = json.loads(_text(await server.call_tool("search_notes", {"query": "charge"})))
    assert found[0]["question"] == "who calls charge"

    wiki = json.loads(_text(await server.call_tool("generate_wiki", {})))
    assert wiki["page_count"] >= 9
    page = json.loads(_text(await server.call_tool("get_wiki_page", {"page": "pkg.billing"})))
    assert "pkg.billing.charge" in page["content"]
    nope = json.loads(_text(await server.call_tool("get_wiki_page", {"page": "nope"})))
    assert "error" in nope and "index.md" in nope["available"]

    visual = json.loads(_text(await server.call_tool("generate_visualization", {"max_nodes": 4})))
    assert visual["path"] == ".codegraph/graph.html"
    assert visual["nodes"] == 4 and visual["truncated"]
    escape = json.loads(
        _text(await server.call_tool("generate_visualization", {"output": "../x.html"}))
    )
    assert "error" in escape

    inline = json.loads(
        _text(await server.call_tool("export_graph", {"format": "dot", "inline": True}))
    )
    assert inline["files"]["graph.dot"].startswith("digraph")
    written = json.loads(_text(await server.call_tool("export_graph", {"format": "csv"})))
    assert written["written"] == [".codegraph/exports/nodes.csv", ".codegraph/exports/edges.csv"]
    assert "error" in json.loads(_text(await server.call_tool("export_graph", {"format": "x"})))


async def test_refactor_tools(repo: Path) -> None:
    server = build_server(repo)
    await server.call_tool("index_repository_tool", {})
    preview = json.loads(
        _text(
            await server.call_tool(
                "refactor_preview", {"target": "pkg.billing.charge", "new_name": "bill"}
            )
        )
    )
    assert preview["edits"] == 3 and "+def bill" in preview["diff"]
    applied = json.loads(
        _text(await server.call_tool("apply_refactor", {"preview_id": preview["preview_id"]}))
    )
    assert applied["applied"]["files"] == ["pkg/billing.py", "tests/test_billing.py"]
    assert "parsed 2" in applied["index"]
    node = json.loads(_text(await server.call_tool("get_node", {"target": "pkg.billing.bill"})))
    assert node["qname"] == "pkg.billing.bill"

    again = json.loads(
        _text(await server.call_tool("apply_refactor", {"preview_id": preview["preview_id"]}))
    )
    assert "error" in again
    suggestions = json.loads(_text(await server.call_tool("suggest_refactorings", {})))
    assert any(s["kind"] == "remove" for s in suggestions)
    assert "error" in json.loads(
        _text(await server.call_tool("suggest_refactorings", {"limit": 0}))
    )
