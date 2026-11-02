from __future__ import annotations

import json
from pathlib import Path

import pytest

from codegraph.cli import main
from codegraph.commands.query import format_table
from codegraph.cypher import CypherError, CypherResult, run_cypher
from codegraph.cypher.schema import graph_schema
from codegraph.server import build_server
from codegraph.store import GraphStore


def test_run_cypher_on_indexed_repo(store: GraphStore) -> None:
    result = run_cypher(
        store,
        "MATCH (c)-[:CALLS]->(f:Function {name: 'charge'}) RETURN c.qname AS caller",
    )
    assert result.columns == ("caller",)
    assert result.rows == (("tests.test_billing.test_charge",),)


def test_run_cypher_rejects_bad_query_before_loading(store: GraphStore) -> None:
    with pytest.raises(CypherError, match="read-only"):
        run_cypher(store, "MATCH (n) DELETE n")


def test_schema_counts_patterns_and_properties(store: GraphStore) -> None:
    schema = graph_schema(store)
    assert schema.node_labels == {"Class": 3, "Function": 7, "Method": 5, "Module": 8}
    assert schema.relationship_types["CALLS"] > 0
    texts = {p.text: p.count for p in schema.patterns}
    assert texts["(Class)-[:INHERITS]->(Class)"] == 1
    assert texts["(Class)-[:DEFINES]->(Method)"] == 5
    assert sum(texts.values()) <= sum(schema.relationship_types.values())
    assert "docstring" in schema.node_properties["Class"]
    assert "docstring" not in schema.node_properties["Function"]
    assert "fan_in" in schema.node_properties["Function"]
    assert "confidence" in schema.edge_properties
    assert schema.label_kinds["Callable"] == ("function", "method")


def test_schema_examples_all_run(store: GraphStore) -> None:
    schema = graph_schema(store)
    assert len(schema.examples) >= 5
    assert any("INHERITS" in example for example in schema.examples)
    for example in schema.examples:
        run_cypher(store, example)


def test_schema_of_empty_graph(tmp_path: Path) -> None:
    with GraphStore.open(tmp_path) as empty:
        schema = graph_schema(empty)
    assert schema.node_labels == {}
    assert schema.patterns == ()
    payload = schema.as_dict()
    assert payload["examples"]


def test_format_table() -> None:
    result = CypherResult(columns=("name", "n"), rows=(("charge", 1), ("x" * 80, None)))
    lines = format_table(result).splitlines()
    assert lines[0].split() == ["name", "n"]
    assert set(lines[1].replace(" ", "")) == {"-"}
    assert lines[2].split() == ["charge", "1"]
    assert lines[3].endswith("...  null")
    assert lines[-1] == "(2 rows)"


def test_cli_cypher_and_schema(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--root", str(repo), "index"]) == 0
    capsys.readouterr()

    assert (
        main(["--root", str(repo), "cypher", "MATCH (c:Class) RETURN c.name ORDER BY c.name"]) == 0
    )
    out = capsys.readouterr().out
    assert "Base" in out and "Handler" in out and "(3 rows)" in out

    query = "MATCH (f:Function) RETURN f.qname ORDER BY f.qname"
    assert main(["--root", str(repo), "cypher", query, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["columns"] == ["f.qname"]
    assert ["pkg.billing.charge"] in payload["rows"]

    assert main(["--root", str(repo), "cypher", "MATCH (f:Function {name: 'zzz'}) RETURN f"]) == 1
    assert main(["--root", str(repo), "cypher", "MATCH (f:Functoin) RETURN f"]) == 2
    err = capsys.readouterr().err
    assert "unknown label 'Functoin'" in err and "^" in err
    assert main(["--root", str(repo), "cypher", "MATCH (n) RETURN n", "--max-rows", "0"]) == 2

    assert main(["--root", str(repo), "cypher", "MATCH (n) RETURN n.qname", "--max-rows", "2"]) == 0
    captured = capsys.readouterr()
    assert "(2 rows)" in captured.out and "first 2 rows" in captured.err

    assert main(["--root", str(repo), "schema"]) == 0
    out = capsys.readouterr().out
    assert "(Class)-[:INHERITS]->(Class)" in out and "examples:" in out
    assert main(["--root", str(repo), "schema", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["node_labels"]["Module"] == 8


def test_cli_schema_on_empty_repo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--root", str(tmp_path), "schema"]) == 1
    assert "codegraph index" in capsys.readouterr().err


async def test_mcp_cypher_tools(repo: Path) -> None:
    server = build_server(repo)
    names = {t.name for t in await server.list_tools()}
    assert {"run_cypher", "get_graph_schema"} <= names
    await server.call_tool("index_repository_tool", {})

    result = json.loads(
        _text(
            await server.call_tool(
                "run_cypher",
                {"query": "MATCH (m:Module) RETURN count(*) AS modules"},
            )
        )
    )
    assert result["rows"] == [[8]]

    error = json.loads(
        _text(await server.call_tool("run_cypher", {"query": "MATCH (n) SET n.x = 1 RETURN n"}))
    )
    assert "read-only" in error["error"]
    assert error["position"] == len("MATCH (n) ")
    assert error["column"] == 11

    capped = json.loads(
        _text(await server.call_tool("run_cypher", {"query": "MATCH (n) RETURN n", "max_rows": 3}))
    )
    assert capped["row_count"] == 3 and capped["truncated"] is True
    bad = json.loads(
        _text(await server.call_tool("run_cypher", {"query": "MATCH (n) RETURN n", "max_rows": 0}))
    )
    assert "max_rows" in bad["error"]

    schema = json.loads(_text(await server.call_tool("get_graph_schema", {})))
    assert schema["node_labels"]["Function"] == 7
    assert any(p["pattern"] == "(Class)-[:INHERITS]->(Class)" for p in schema["patterns"])


def _text(result: object) -> str:
    blocks = getattr(result, "content", None)
    assert isinstance(blocks, list), f"unexpected tool result: {result!r}"
    return "".join(str(getattr(block, "text", "")) for block in blocks)
