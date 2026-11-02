"""Analysis over an indexed fixture repository, including the persisted passes."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from codegraph.analysis.architecture import (
    HISTORY_TABLE,
    architecture,
    change_impact,
    classify_layer,
    package_of,
    risk_level,
    risk_score,
)
from codegraph.analysis.common import ensure_tables, table_exists
from codegraph.analysis.communities import (
    DDL as COMMUNITY_DDL,
)
from codegraph.analysis.communities import (
    community_report,
    detect_communities,
    load_communities,
)
from codegraph.analysis.context import minimal_context, suggest_tools
from codegraph.analysis.entrypoints import cli_scripts, detect_entry_points
from codegraph.analysis.flows import (
    affected_flows,
    changed_qnames,
    detect_flows,
    flow_report,
    load_flows,
)
from codegraph.graph import GraphView
from codegraph.indexer import index_repository
from codegraph.models import Edge, Node
from codegraph.passes import discover_passes
from codegraph.queries import dead_code
from codegraph.review import GitError
from codegraph.store import GraphStore

FIXTURE = Path(__file__).parent / "fixtures" / "analysis_repo"


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def shop(tmp_path: Path) -> Path:
    target = tmp_path / "shop"
    shutil.copytree(FIXTURE, target)
    _git(target, "init", "-q")
    _git(target, "config", "user.email", "ci@example.com")
    _git(target, "config", "user.name", "ci")
    _git(target, "add", "-A")
    _git(target, "commit", "-q", "-m", "init")
    return target


@pytest.fixture
def shop_store(shop: Path) -> Iterator[GraphStore]:
    with GraphStore.open(shop) as store:
        index_repository(shop, store)
        yield store


def test_passes_are_registered_in_the_analysis_band() -> None:
    orders = {p.name: p.order for p in discover_passes()}
    assert 50 <= orders["communities"] < orders["flows"] <= 89


def test_passes_persist_and_rerun_idempotently(shop: Path, shop_store: GraphStore) -> None:
    stored = load_communities(shop_store)
    assert stored is not None and stored.source == "stored"
    computed = detect_communities(GraphView.load(shop_store))
    assert stored.communities == computed.communities
    assert stored.membership == computed.membership
    assert stored.links == computed.links
    assert stored.modularity == computed.modularity

    flows = load_flows(shop_store)
    assert flows is not None
    assert flows == detect_flows(
        GraphView.load(shop_store), root=shop, membership=stored.membership
    )

    rows = shop_store.connection.execute("SELECT COUNT(*) FROM analysis_flow_steps").fetchone()
    index_repository(shop, shop_store, force=True)
    again = shop_store.connection.execute("SELECT COUNT(*) FROM analysis_flow_steps").fetchone()
    assert rows[0] == again[0]
    assert load_communities(shop_store) == stored


def test_readers_fall_back_to_computing(shop: Path) -> None:
    with GraphStore.open(shop) as store:
        index_repository(shop, store, skip_passes=frozenset({"communities", "flows"}))
        assert not table_exists(store, "analysis_communities")
        assert load_communities(store) is None and load_flows(store) is None
        assert community_report(store).source == "computed"
        assert flow_report(store, shop)


def test_ensure_tables_does_not_commit(shop_store: GraphStore) -> None:
    conn = shop_store.connection
    conn.execute("DELETE FROM analysis_flows")
    assert conn.in_transaction
    ensure_tables(shop_store, COMMUNITY_DDL)
    assert conn.in_transaction
    conn.rollback()
    assert load_flows(shop_store)


def test_entry_points_and_flows(shop: Path, shop_store: GraphStore) -> None:
    view = GraphView.load(shop_store)
    entries = {e.qname: e for e in detect_entry_points(view, cli_scripts(shop))}
    assert entries["shop.cli.run_cli"].kind == "cli"
    assert entries["shop.api.list_orders"].kind == "route"
    assert entries["shop.api.list_orders"].reason == "serves GET /orders"
    assert not any(q.startswith("tests.") for q in entries)

    flows = load_flows(shop_store)
    assert flows is not None
    top = flows[0]
    assert top.entry_point == "shop.cli.run_cli"
    steps = {s.qname: s for s in top.steps}
    assert steps["shop.orders.Order.__init__"].via == "constructs"
    assert {"shop.models.validate", "shop.util.money.to_dollars"} <= set(steps)
    assert top.file_count >= 5 and top.community_count >= 2
    assert 0.0 < top.criticality <= 1.0


def test_dead_code_is_smarter(shop_store: GraphStore) -> None:
    dead = {n.qname for n in dead_code(shop_store)}
    assert {"shop.orders._unused_private", "shop.report.orphan_tool"} <= dead
    for alive in (
        "shop.api.list_orders",
        "shop.cli.run_cli",
        "shop.models.Base.__repr__",
        "shop.models.Repository.all",
        "shop.models.MemoryRepository.all",
        "shop.payments.StripeGateway.charge",
        "shop.payments.Receipt",
        "shop.orders.Order.total",
        "shop.orders.Order",
        "tests.test_orders.test_place_order",
    ):
        assert alive not in dead, alive


def test_dead_code_honours_js_exports(tmp_path: Path) -> None:
    (tmp_path / "lib.ts").write_text(
        "export function api(): number {\n  return 1;\n}\n"
        "function hidden(): number {\n  return 2;\n}\n"
        "export class Widget {\n  render(): void {}\n  _secret(): void {}\n}\n"
    )
    with GraphStore.open(tmp_path) as store:
        index_repository(tmp_path, store)
        dead = {n.qname for n in dead_code(store)}
    assert "lib.hidden" in dead
    assert "lib.Widget._secret" in dead
    assert not {"lib.api", "lib.Widget", "lib.Widget.render"} & dead


def test_dead_code_keeps_callbacks_and_registries(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text(
        "def on_start() -> None:\n    return None\n\n\n"
        "def lonely() -> None:\n    return None\n\n\n"
        "HOOKS = {'start': on_start}\n"
    )
    with GraphStore.open(tmp_path) as store:
        index_repository(tmp_path, store)
        dead = {n.qname for n in dead_code(store)}
    assert dead == {"app.lonely"}


def test_architecture_and_history_hotspots(shop: Path, shop_store: GraphStore) -> None:
    arch = architecture(shop_store, shop, top=5)
    assert arch.languages == {"python": 11}
    layers = {p.name: p.layer for p in arch.packages}
    assert layers["shop.util"] == "core"
    assert [e.qname for e in arch.entry_points][:2] == ["shop.api.list_orders", "shop.cli.run_cli"]
    assert arch.hubs and arch.communities and arch.community_boundaries
    assert arch.hotspots and all(s.table == HISTORY_TABLE for s in arch.hotspots)
    assert any(s.path in shop_store.indexed_paths() for s in arch.hotspots)

    shop_store.connection.execute(f"DELETE FROM {HISTORY_TABLE}")
    shop_store.connection.executemany(
        f"INSERT INTO {HISTORY_TABLE} (path, commits, authors, first_ts, last_ts, added, removed) "
        "VALUES (?, ?, 1, 0, 0, 0, 0)",
        [("shop/orders.py", 7), ("shop/cli.py", 1)],
    )
    spots = architecture(shop_store, shop).hotspots
    assert [(s.path, s.changes) for s in spots] == [("shop/orders.py", 7), ("shop/cli.py", 1)]
    assert spots[0].symbols >= 5

    shop_store.connection.execute(f"DROP TABLE {HISTORY_TABLE}")
    assert architecture(shop_store, shop).hotspots == ()


def test_architecture_lists_routes(shop: Path, shop_store: GraphStore) -> None:
    with shop_store.transaction():
        shop_store.add_nodes(
            [Node("route:GET /orders", "route", "shop/api.py", 8, 11, "python", "GET /orders")]
        )
        shop_store.add_edges(
            [Edge("route:GET /orders", "shop.api.list_orders", "handles", "shop/api.py", 8)]
        )
    arch = architecture(shop_store, shop)
    assert [(r.route, r.handlers) for r in arch.routes] == [
        ("GET /orders", ("shop.api.list_orders",))
    ]
    view = GraphView.load(shop_store)
    kinds = {e.qname: e.kind for e in detect_entry_points(view)}
    assert kinds["shop.api.list_orders"] == "route"


def test_layer_rules_and_risk() -> None:
    assert classify_layer(0, 3, has_routes=False, has_entry_points=True)[0] == "entry"
    assert classify_layer(5, 5, has_routes=True, has_entry_points=False)[0] == "api"
    assert classify_layer(5, 1, has_routes=False, has_entry_points=False)[0] == "core"
    assert classify_layer(1, 0, has_routes=False, has_entry_points=False)[0] == "leaf"
    assert classify_layer(1, 1, has_routes=False, has_entry_points=False)[0] == "internal"
    assert package_of("setup.py") == "(root)"
    assert package_of("a/b/c.py") == "a.b"
    assert risk_level(0.9, has_changes=False) == "none"
    assert risk_level(0.7, has_changes=True) == "high"
    assert risk_level(0.4, has_changes=True) == "medium"
    assert risk_level(0.1, has_changes=True) == "low"
    assert (
        risk_score(max_symbol_risk=99, test_gaps=99, max_flow_criticality=1, communities=9) == 1.0
    )


def test_affected_flows_and_change_impact(shop: Path, shop_store: GraphStore) -> None:
    assert changed_qnames(shop_store, shop) == []
    clean = change_impact(shop_store, shop)
    assert clean.risk_level == "none" and clean.changed == ()

    money = shop / "shop" / "util" / "money.py"
    money.write_text(money.read_text().replace("cents / 100", "cents / 100.0"))
    changed = changed_qnames(shop_store, shop)
    assert changed == ["shop.util.money.to_dollars"]
    flows = load_flows(shop_store)
    assert flows is not None
    hits = affected_flows(flows, changed)
    assert "shop.cli.run_cli" in {h.flow.entry_point for h in hits}

    impact = change_impact(shop_store, shop)
    assert [c.qname for c in impact.changed] == ["shop.util.money.to_dollars"]
    assert impact.test_gaps == ("shop.util.money.to_dollars",)
    assert impact.affected_flows and impact.communities
    assert impact.risk_level in {"low", "medium", "high"}
    assert "1 changed symbol" in impact.summary

    with pytest.raises(GitError):
        change_impact(shop_store, shop, base="no-such-branch")


def test_minimal_context(shop: Path, shop_store: GraphStore) -> None:
    bundle = minimal_context(shop_store, shop, target="place_order")
    assert bundle is not None and bundle.target is not None
    assert bundle.target.qname == "shop.orders.place_order"
    assert "shop.cli.run_cli" in bundle.target.callers
    assert bundle.target.tests == ("tests.test_orders.test_place_order",)
    assert bundle.target.community and bundle.target.flows
    assert 0 < bundle.tokens <= 800

    by_task = minimal_context(shop_store, shop, task="how does the gateway charge work")
    assert by_task is not None and by_task.target is not None
    assert "charge" in by_task.target.qname.lower() or by_task.candidates
    assert minimal_context(shop_store, shop, target="nope_nothing") is None
    tight = minimal_context(shop_store, shop, target="place_order", max_tokens=60)
    assert tight is not None and tight.truncated
    assert suggest_tools("review this PR")[0] == "get_change_impact"
    assert suggest_tools(None) == suggest_tools("something unrelated")
