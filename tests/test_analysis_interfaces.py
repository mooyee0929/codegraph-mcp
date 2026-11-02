"""The analysis MCP tools and CLI commands, end to end on the fixture repository."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from codegraph.cli import main
from codegraph.indexer import index_repository
from codegraph.server import build_server
from codegraph.store import GraphStore

FIXTURE = Path(__file__).parent / "fixtures" / "analysis_repo"

TOOLS = {
    "list_communities",
    "get_community",
    "list_flows",
    "get_flow",
    "get_affected_flows",
    "get_hub_nodes",
    "get_bridge_nodes",
    "get_knowledge_gaps",
    "get_surprising_connections",
    "get_suggested_questions",
    "find_large_functions",
    "get_minimal_context",
    "traverse_graph",
    "trace_path",
    "get_architecture",
    "get_change_impact",
}


@pytest.fixture
def shop(tmp_path: Path) -> Path:
    target = tmp_path / "shop"
    shutil.copytree(FIXTURE, target)
    for args in (
        ["init", "-q"],
        ["config", "user.email", "ci@example.com"],
        ["config", "user.name", "ci"],
        ["add", "-A"],
        ["commit", "-q", "-m", "init"],
    ):
        subprocess.run(["git", *args], cwd=target, check=True, capture_output=True)
    with GraphStore.open(target) as store:
        index_repository(target, store)
    return target


def _text(result: object) -> str:
    blocks = getattr(result, "content", None)
    assert isinstance(blocks, list), f"unexpected tool result: {result!r}"
    return "".join(str(getattr(block, "text", "")) for block in blocks)


async def _call(shop: Path, name: str, args: dict[str, object]) -> object:
    server = build_server(shop, tools=frozenset({name}))
    payload: object = json.loads(_text(await server.call_tool(name, args)))
    return payload


async def test_tools_are_registered(shop: Path) -> None:
    names = {t.name for t in await build_server(shop).list_tools()}
    assert names >= TOOLS
    assert {"get_architecture_overview", "detect_changes_tool", "find_dead_code"} <= names


async def test_community_and_flow_tools(shop: Path) -> None:
    listing = await _call(shop, "list_communities", {})
    assert isinstance(listing, dict) and listing["source"] == "stored"
    first = listing["communities"][0]
    assert {"id", "name", "size", "cohesion", "top_members"} <= set(first)
    assert listing["boundaries"][0]["src_name"]

    detail = await _call(shop, "get_community", {"community": "shop.orders.place_order"})
    assert isinstance(detail, dict)
    assert "shop.orders.place_order" in {m["qname"] for m in detail["members"]}
    missing = await _call(shop, "get_community", {"community": "nothing-like-this"})
    assert isinstance(missing, dict) and "error" in missing

    flows = await _call(shop, "list_flows", {"sort_by": "depth"})
    assert isinstance(flows, dict) and flows["total"] >= 2
    assert "steps" not in flows["flows"][0]
    bad_sort = await _call(shop, "list_flows", {"sort_by": "colour"})
    assert isinstance(bad_sort, dict) and "error" in bad_sort

    flow = await _call(shop, "get_flow", {"flow": "run_cli"})
    assert isinstance(flow, dict) and flow["steps"][0]["qname"] == "shop.cli.run_cli"
    assert flow["steps"][1]["path"]
    assert "error" in json.dumps(await _call(shop, "get_flow", {"flow": "999"}))


async def test_change_tools(shop: Path) -> None:
    clean = await _call(shop, "get_affected_flows", {})
    assert clean == {"changed": [], "total": 0, "flows": []}
    cli = shop / "shop" / "cli.py"
    cli.write_text(cli.read_text().replace('"widget", 3', '"widget", 4'))
    affected = await _call(shop, "get_affected_flows", {})
    assert isinstance(affected, dict) and affected["changed"] == ["shop.cli.run_cli"]
    assert affected["flows"][0]["touched"] == ["shop.cli.run_cli"]

    impact = await _call(shop, "get_change_impact", {})
    assert isinstance(impact, dict) and impact["risk_level"] in {"low", "medium", "high"}
    assert impact["changed"][0]["qname"] == "shop.cli.run_cli"
    bad = await _call(shop, "get_change_impact", {"base": "no-such-rev"})
    assert isinstance(bad, dict) and "error" in bad
    assert "error" in json.dumps(await _call(shop, "get_affected_flows", {"base": "nope"}))


async def test_insight_tools(shop: Path) -> None:
    hubs = await _call(shop, "get_hub_nodes", {"top": 3})
    assert isinstance(hubs, list) and len(hubs) == 3 and hubs[0]["degree"] >= hubs[2]["degree"]
    classes = await _call(shop, "get_hub_nodes", {"kinds": ["class"]})
    assert isinstance(classes, list) and {h["kind"] for h in classes} == {"class"}
    assert "error" in json.dumps(await _call(shop, "get_hub_nodes", {"kinds": ["blob"]}))

    bridges = await _call(shop, "get_bridge_nodes", {"top": 2, "max_sources": 5})
    assert isinstance(bridges, list) and bridges and bridges[0]["betweenness"] > 0
    assert "error" in json.dumps(await _call(shop, "get_bridge_nodes", {"max_sources": 0}))

    gaps = await _call(shop, "get_knowledge_gaps", {})
    assert isinstance(gaps, dict) and gaps["total"] > 0
    assert gaps["undocumented_modules"][0]["target"] == "shop.report"

    surprises = await _call(shop, "get_surprising_connections", {"top": 2})
    assert isinstance(surprises, list) and len(surprises) <= 2

    questions = await _call(shop, "get_suggested_questions", {"limit": 4})
    assert isinstance(questions, list) and 0 < len(questions) <= 4
    assert questions[0]["priority"] == "high"

    large = await _call(shop, "find_large_functions", {"min_lines": 40})
    assert isinstance(large, list) and large[0]["qname"] == "shop.report.build_table"
    none = await _call(shop, "find_large_functions", {"min_lines": 40, "path": "shop/util"})
    assert none == []
    assert "error" in json.dumps(await _call(shop, "find_large_functions", {"min_lines": 0}))


async def test_navigation_tools(shop: Path) -> None:
    tree = await _call(
        shop, "traverse_graph", {"start": "run_cli", "edge_kinds": ["calls"], "depth": 1}
    )
    assert isinstance(tree, dict)
    assert {c["qname"] for c in tree["root"]["children"]} == {
        "shop.orders.place_order",
        "shop.report.render",
    }
    for bad in (
        {"start": "run_cli", "direction": "up"},
        {"start": "run_cli", "mode": "random"},
        {"start": "run_cli", "depth": 0},
        {"start": "run_cli", "edge_kinds": ["nope"]},
        {"start": "nope_nothing"},
    ):
        assert "error" in json.dumps(await _call(shop, "traverse_graph", bad)), bad

    trace = await _call(
        shop, "trace_path", {"target": "validate", "direction": "inbound", "to": "run_cli"}
    )
    assert isinstance(trace, dict) and trace["inbound"]
    assert trace["path"]["nodes"][-1] == "shop.cli.run_cli"
    for bad in (
        {"target": "validate", "depth": 6},
        {"target": "validate", "direction": "sideways"},
        {"target": "nope_nothing"},
        {"target": "validate", "to": "nope_nothing"},
    ):
        assert "error" in json.dumps(await _call(shop, "trace_path", bad)), bad

    context = await _call(shop, "get_minimal_context", {"target": "place_order"})
    assert isinstance(context, dict) and context["target"]["qname"] == "shop.orders.place_order"
    assert "error" in json.dumps(await _call(shop, "get_minimal_context", {}))
    assert "error" in json.dumps(
        await _call(shop, "get_minimal_context", {"target": "x", "max_tokens": 1})
    )


async def test_architecture_tool(shop: Path) -> None:
    arch = await _call(shop, "get_architecture", {"top": 3})
    assert isinstance(arch, dict)
    assert arch["languages"] == {"python": 11}
    assert len(arch["hubs"]) == 3
    assert arch["entry_points"][0]["qname"] == "shop.cli.run_cli"
    assert arch["hotspots"] == []


def test_cli_commands(shop: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def run(*args: str) -> tuple[int, str]:
        code = main(["--root", str(shop), *args])
        return code, capsys.readouterr().out

    code, out = run("communities")
    assert code == 0 and "modularity" in out
    code, out = run("communities", "--json")
    assert code == 0 and json.loads(out)["communities"]
    assert run("community", "shop.cli.run_cli")[0] == 0
    assert run("community", "nothing-like-this")[0] == 1

    code, out = run("flows")
    assert code == 0 and "run_cli" in out
    code, out = run("flow", "run_cli", "--json")
    assert code == 0 and json.loads(out)["entry_point"] == "shop.cli.run_cli"
    assert run("flow", "nope")[0] == 1
    assert run("flows", "--affected")[0] == 1

    for command in (
        ["hubs"],
        ["hubs", "--kind", "class", "--json"],
        ["bridges", "--top", "3"],
        ["gaps"],
        ["surprises"],
        ["questions"],
        ["large", "--min-lines", "40"],
        ["context", "place_order"],
        ["context", "--task", "charge the card", "--json"],
        ["traverse", "run_cli", "--edge-kind", "calls", "--depth", "2"],
        ["trace", "validate", "--direction", "inbound", "--to", "run_cli"],
        ["architecture"],
        ["architecture", "--json"],
        ["change-impact"],
    ):
        code, _ = run(*command)
        assert code == 0, command

    code, out = run("trace", "validate", "--direction", "inbound")
    assert "[critical] shop.models.Base.save" in out
    assert run("large", "--min-lines", "500")[0] == 1
    assert run("large", "--min-lines", "0")[0] == 2
    assert run("bridges", "--max-sources", "0")[0] == 2
    assert run("trace", "validate", "--depth", "9")[0] == 2
    assert run("traverse", "run_cli", "--depth", "0")[0] == 2
    assert run("trace", "nope_nothing")[0] == 1
    assert run("context")[0] == 2
    assert run("context", "nope_nothing")[0] == 1
    assert run("change-impact", "--base", "no-such-rev")[0] == 2

    cli = shop / "shop" / "cli.py"
    cli.write_text(cli.read_text().replace('"widget", 3', '"gadget", 3'))
    code, out = run("flows", "--affected", "--json")
    assert code == 0 and json.loads(out)["changed"] == ["shop.cli.run_cli"]
    code, out = run("change-impact", "--json")
    assert code == 0 and json.loads(out)["changed"]
