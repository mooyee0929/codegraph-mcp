from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from codegraph.graph import GraphView
from codegraph.indexer import (
    DEFAULT_IGNORES,
    index_repository,
    iter_source_files,
    load_ignore_file,
)
from codegraph.linker import link
from codegraph.models import Edge
from codegraph.parsers import language_of
from codegraph.passes import GraphPass, discover_passes
from codegraph.server import build_server, parse_tool_filter
from codegraph.store import SCHEMA_VERSION, GraphStore, StoreError


def test_old_schema_is_rebuilt(tmp_path: Path) -> None:
    db = tmp_path / "graph.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '1')")
    conn.execute("CREATE TABLE stale (x INTEGER)")
    conn.commit()
    conn.close()

    with GraphStore(db) as store:
        assert store.rebuilt
        assert store.get_meta("schema_version") == str(SCHEMA_VERSION)
        tables = {
            r[0] for r in store.connection.execute("SELECT name FROM sqlite_master").fetchall()
        }
        assert "stale" not in tables
        assert "edges" in tables


def test_non_database_file_is_reported(tmp_path: Path) -> None:
    db = tmp_path / "graph.db"
    db.write_bytes(b"not sqlite at all" * 100)
    with pytest.raises(StoreError):
        GraphStore(db)


def test_edge_confidence_round_trips(store: GraphStore) -> None:
    with store.transaction():
        store.add_edges(
            [
                Edge(
                    src="pkg.billing.charge",
                    dst="pkg.util.money",
                    kind="similar_to",
                    path="pkg/billing.py",
                    line=1,
                    confidence=0.75,
                    detail="jaccard=0.75",
                )
            ]
        )
    [edge] = store.all_edges(kinds=["similar_to"])
    assert edge.confidence == 0.75
    assert edge.detail == "jaccard=0.75"
    with store.transaction():
        assert store.delete_edges("similar_to") == 1
    assert store.all_edges(kinds=["similar_to"]) == []


def test_ensure_schema_keeps_the_open_transaction(store: GraphStore) -> None:
    with pytest.raises(RuntimeError, match="abort"), store.transaction():
        store.set_meta("probe", "written")
        store.ensure_schema(
            "CREATE TABLE IF NOT EXISTS t_probe (x INTEGER);\n"
            "CREATE INDEX IF NOT EXISTS idx_t_probe ON t_probe (x);\n"
        )
        raise RuntimeError("abort")
    assert store.get_meta("probe") is None
    with pytest.raises(StoreError, match="incomplete"):
        store.ensure_schema("CREATE TABLE IF NOT EXISTS t_broken (x INTEGER")


def test_fts_rows_follow_nodes_through_reindex_delete_and_vacuum(repo: Path) -> None:
    def orphans(store: GraphStore) -> int:
        row = store.connection.execute(
            "SELECT COUNT(*) FROM nodes_fts WHERE rowid NOT IN (SELECT id FROM nodes)"
        ).fetchone()
        return int(row[0])

    def fts_rows(store: GraphStore) -> int:
        return int(store.connection.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0])

    with GraphStore.open(repo) as store:
        index_repository(repo, store)
        (repo / "pkg" / "util.py").write_text("def renamed_helper() -> None:\n    return None\n")
        index_repository(repo, store)
        assert orphans(store) == 0 and fts_rows(store) == store.counts()["nodes"]
        hits = [n.qname for n in store.search_nodes("renamed_helper")]
        assert hits == ["pkg.util.renamed_helper"]

        with store.transaction():
            store.delete_nodes("class")
        assert orphans(store) == 0 and fts_rows(store) == store.counts()["nodes"]

        store.connection.execute("VACUUM")
        assert orphans(store) == 0
        assert [n.qname for n in store.search_nodes("renamed_helper")] == hits


def test_counts_by_kind(store: GraphStore) -> None:
    grouped = store.counts_by_kind()
    assert grouped["languages"] == {"python": 5, "typescript": 3}
    assert grouped["nodes"]["module"] == 8
    assert grouped["edges"]["calls"] > 0


def test_graph_view_adjacency(store: GraphStore) -> None:
    view = GraphView.load(store)
    assert "pkg.billing.Invoice.total" in view.successors("pkg.billing.charge", ["calls"])
    assert "pkg.billing.charge" in view.predecessors("pkg.billing.Invoice.total", ["calls"])
    assert view.module_of("pkg.billing.Invoice.rate") == "pkg.billing"
    assert all(e.dst in view.nodes for e in view.edges(["calls"]))
    assert {n.qname for n in view.of_kind("module")} >= {"pkg.billing", "web.app"}


def test_delete_nodes_drops_touching_edges(store: GraphStore) -> None:
    with store.transaction():
        removed = store.delete_nodes("class")
    assert removed > 0
    assert store.get_node("pkg.billing.Invoice") is None
    assert store.edges_to("pkg.billing.Invoice") == []


def test_passes_are_discovered_and_well_formed() -> None:
    passes = discover_passes()
    assert all(isinstance(p, GraphPass) for p in passes)
    assert [p.order for p in passes] == sorted(p.order for p in passes)
    assert len({p.name for p in passes}) == len(passes)


def test_ignore_file(repo: Path) -> None:
    (repo / ".codegraphignore").write_text("# comment\nweb/\npkg/util.py\n")
    ignores = (*DEFAULT_IGNORES, *load_ignore_file(repo))
    found = {p.relative_to(repo).as_posix() for p in iter_source_files(repo, ignores)}
    assert "pkg/billing.py" in found
    assert "pkg/util.py" not in found
    assert not any(p.startswith("web/") for p in found)

    with GraphStore.open(repo) as store:
        index_repository(repo, store)
        assert store.get_node("web.app") is None


def test_language_of_uses_suffix_case_insensitively() -> None:
    assert language_of("a/b/Thing.PY") == "python"
    assert language_of("README") is None
    assert language_of("web/index.tsx") == "typescript"


def test_parse_tool_filter() -> None:
    assert parse_tool_filter(None) is None
    assert parse_tool_filter(" , ") is None
    assert parse_tool_filter("graph_status, get_node") == frozenset({"graph_status", "get_node"})


async def test_tool_filter_restricts_server(repo: Path) -> None:
    server = build_server(repo, tools=frozenset({"graph_status", "get_node"}))
    assert {t.name for t in await server.list_tools()} == {"graph_status", "get_node"}
    with pytest.raises(ValueError, match="nope"):
        build_server(repo, tools=frozenset({"nope"}))


def test_db_api_method_calls_are_not_linked_to_a_unique_definition(tmp_path: Path) -> None:
    (tmp_path / "query.py").write_text("def execute(text: str) -> str:\n    return text\n")
    (tmp_path / "db.py").write_text(
        "import sqlite3\n\n\ndef count(conn: sqlite3.Connection) -> int:\n"
        "    return conn.execute('SELECT 1').fetchone()[0]\n"
    )
    with GraphStore.open(tmp_path) as store:
        index_repository(tmp_path, store)
        callees = {e.dst for e in store.edges_from("db.count", "calls")}
    assert "query.execute" not in callees
    assert "*.execute" in callees


def test_linker_leaves_non_code_targets_alone(tmp_path: Path) -> None:
    (tmp_path / "fmt.py").write_text("def json() -> str:\n    return '{}'\n")
    (tmp_path / "client.py").write_text("def fetch() -> None:\n    return None\n")
    with GraphStore.open(tmp_path) as store:
        index_repository(tmp_path, store)
        with store.transaction():
            store.add_edges(
                [
                    Edge(
                        src="client.fetch",
                        dst="route:GET /api/v1/users.json",
                        kind="http_calls",
                        path="client.py",
                        line=1,
                        resolved=False,
                    )
                ]
            )
            link(store)
        [edge] = store.all_edges(kinds=["http_calls"])
    assert edge.dst == "route:GET /api/v1/users.json"
    assert not edge.resolved
