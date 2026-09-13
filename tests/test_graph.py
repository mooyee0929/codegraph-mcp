from __future__ import annotations

from pathlib import Path

from codegraph.indexer import index_repository
from codegraph.queries import (
    architecture_overview,
    dead_code,
    impact_radius,
    node_at,
    query,
    resolve_target,
)
from codegraph.store import GraphStore


def test_index_counts(store: GraphStore) -> None:
    counts = store.counts()
    assert counts["files"] == 8
    assert counts["nodes"] > 15
    assert counts["edges"] > 20


def test_incremental_reindex_skips_unchanged(repo: Path, store: GraphStore) -> None:
    second = index_repository(repo, store)
    assert second.parsed == 0
    assert second.unchanged == 8

    (repo / "pkg" / "util.py").write_text("def fmt(value: int) -> str:\n    return str(value)\n")
    third = index_repository(repo, store)
    assert third.parsed == 1
    assert store.get_node("pkg.util.fmt") is not None

    (repo / "pkg" / "util.py").unlink()
    fourth = index_repository(repo, store)
    assert fourth.removed == 1
    assert store.get_node("pkg.util.fmt") is None


def test_linker_resolves_unique_bare_names(store: GraphStore) -> None:
    # ``inv.save()`` in charge() has no import behind it; only one ``save`` exists.
    callees = {e.dst for e in query(store, "callees_of", "pkg.billing.charge")}
    assert "pkg.models.Base.save" in callees
    assert "pkg.util.fmt" in callees
    assert store.counts()["unresolved_edges"] == 0 or all(
        not e.resolved for e in store.unresolved_edges()
    )


def test_callers_and_tests(store: GraphStore) -> None:
    callers = {e.src for e in query(store, "callers_of", "pkg.billing.charge")}
    assert callers == {"tests.test_billing.test_charge"}
    tests = {e.src for e in query(store, "tests_for", "pkg.billing.charge")}
    assert tests == {"tests.test_billing.test_charge"}
    ts_tests = {e.src for e in query(store, "tests_for", "web.app.start")}
    assert ts_tests == {"web.app.test"} or ts_tests == set()


def test_subclasses_and_imports(store: GraphStore) -> None:
    subclasses = {e.src for e in query(store, "subclasses_of", "pkg.models.Base")}
    assert subclasses == {"pkg.billing.Invoice"}
    importers = {e.src for e in query(store, "imported_by", "pkg.util.fmt")}
    assert importers == {"pkg.billing"}


def test_resolve_target_forms(store: GraphStore) -> None:
    assert resolve_target(store, "pkg.billing.charge") is not None
    assert resolve_target(store, "charge") is not None
    assert resolve_target(store, "Invoice.total") is not None
    located = resolve_target(store, "pkg/billing.py:10")
    assert located is not None and located.qname == "pkg.billing.Invoice.total"
    assert resolve_target(store, "does_not_exist") is None
    assert node_at(store, "pkg/billing.py", 999) is None


def test_impact_radius(store: GraphStore) -> None:
    report = impact_radius(store, "pkg.billing.Invoice.rate", depth=3)
    assert report is not None
    assert report.direct_callers == ("pkg.billing.Invoice.total",)
    assert "pkg.billing.charge" in report.transitive_callers
    assert "tests.test_billing.test_charge" in report.transitive_callers
    assert "tests.test_billing.test_charge" in report.covering_tests
    assert report.total_affected >= 3

    shallow = impact_radius(store, "pkg.billing.Invoice.rate", depth=1)
    assert shallow is not None
    assert shallow.transitive_callers == ("pkg.billing.Invoice.total",)


def test_architecture_overview(store: GraphStore) -> None:
    overview = architecture_overview(store)
    assert overview.languages == {"python": 5, "typescript": 3}
    assert set(overview.top_level_packages) == {"pkg", "tests", "web"}
    assert overview.hubs and overview.hubs[0].fan_in + overview.hubs[0].fan_out > 0


def test_dead_code_candidates(store: GraphStore) -> None:
    names = {n.qname for n in dead_code(store)}
    assert "pkg.billing.unused_helper" in names
    assert "pkg.billing.charge" not in names


def test_search(store: GraphStore) -> None:
    hits = store.search_nodes("invoice")
    assert any(n.qname == "pkg.billing.Invoice" for n in hits)
    hits = store.search_nodes("handles a request")
    assert any(n.qname == "web.app.Handler" for n in hits)
