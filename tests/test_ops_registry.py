from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from codegraph.indexer import index_repository
from codegraph.models import Edge, Node
from codegraph.registry import (
    RegistryError,
    RouteKey,
    codegraph_home,
    cross_repo_links,
    cross_repo_search,
    delete_project_db,
    find_repo,
    load_registry,
    match_routes,
    parse_route_qname,
    register_repo,
    registry_path,
    repo_status,
    unregister_repo,
)
from codegraph.store import GraphStore, default_db_path


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "cg-home"
    monkeypatch.setenv("CODEGRAPH_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))
    return home


def _indexed_copy(repo: Path, tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    shutil.copytree(repo, target, ignore=shutil.ignore_patterns(".git", ".codegraph"))
    with GraphStore.open(target) as store:
        index_repository(target, store)
    return target


def test_home_honours_env(isolated_home: Path) -> None:
    assert codegraph_home() == isolated_home
    assert codegraph_home({}) == Path.home() / ".codegraph"
    assert registry_path() == isolated_home / "registry.json"


def test_register_rename_unregister(repo: Path, tmp_path: Path) -> None:
    path = registry_path()
    entry = register_repo(path, repo, clock=lambda: 100.0)
    assert entry.name == "repo" and entry.root == str(repo.resolve())
    renamed = register_repo(path, repo, "billing", clock=lambda: 200.0)
    assert renamed.name == "billing" and renamed.added_at == 100.0
    assert [e.name for e in load_registry(path)] == ["billing"]

    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(RegistryError, match="already used"):
        register_repo(path, other, "billing")
    with pytest.raises(RegistryError, match="not a directory"):
        register_repo(path, tmp_path / "missing")

    entries = load_registry(path)
    assert find_repo(entries, "billing") == renamed
    assert find_repo(entries, str(repo)) == renamed
    assert unregister_repo(path, "billing") == renamed
    assert unregister_repo(path, "billing") is None
    assert load_registry(path) == []


def test_corrupt_registry_is_reported(isolated_home: Path) -> None:
    isolated_home.mkdir(parents=True)
    registry_path().write_text("{not json")
    with pytest.raises(RegistryError, match="not valid JSON"):
        load_registry(registry_path())
    registry_path().write_text('{"repos": [{"name": 1}]}')
    with pytest.raises(RegistryError, match="without name/root"):
        load_registry(registry_path())


def test_repo_status_counts_and_problems(repo: Path, tmp_path: Path) -> None:
    entry = register_repo(registry_path(), repo)
    unindexed = repo_status(entry)
    assert not unindexed.indexed and unindexed.problem == "not indexed"

    with GraphStore.open(repo) as store:
        index_repository(repo, store)
        counts = store.counts()
    status = repo_status(entry)
    assert status.indexed and status.files == 8
    assert status.nodes == counts["nodes"] and status.edges == counts["edges"]
    assert status.last_indexed is not None

    with GraphStore.open(repo) as store:
        store.set_meta("schema_version", "1")
        store.connection.commit()
    stale = repo_status(entry)
    assert not stale.indexed and "re-index" in (stale.problem or "")


def test_cross_repo_search_spans_repos(repo: Path, tmp_path: Path) -> None:
    first = _indexed_copy(repo, tmp_path, "alpha")
    second = _indexed_copy(repo, tmp_path, "beta")
    path = registry_path()
    entries = [register_repo(path, first), register_repo(path, second)]
    hits = cross_repo_search(entries, "charge", limit=10)
    assert {h.repo for h in hits} == {"alpha", "beta"}
    assert hits[0].node.name == "charge" and hits[0].score == 3.0
    with pytest.raises(RegistryError):
        cross_repo_search(entries, "  ")


@pytest.mark.parametrize(
    ("qname", "expected"),
    [
        ("route:GET /users/{id}", RouteKey("GET", ("users", "{id}"))),
        ("route:post https://api.example.com/v1/orders?x=1", RouteKey("POST", ("v1", "orders"))),
        ("route:/health", RouteKey("ANY", ("health",))),
        ("pkg.mod.func", None),
    ],
)
def test_parse_route_qname(qname: str, expected: RouteKey | None) -> None:
    assert parse_route_qname(qname) == expected


def test_match_routes_with_parameters() -> None:
    def key(qname: str) -> RouteKey:
        parsed = parse_route_qname(qname)
        assert parsed is not None
        return parsed

    assert match_routes(key("route:GET /users/1"), key("route:GET /users/1")) == 1.0
    assert match_routes(key("route:GET /users/42"), key("route:GET /users/:id")) == 0.9
    assert match_routes(key("route:GET /users/${id}"), key("route:GET /users/<int:id>")) == 0.9
    assert match_routes(key("route:GET /users/1"), key("route:POST /users/1")) is None
    assert match_routes(key("route:GET /users/1/x"), key("route:GET /users/{id}")) is None
    assert match_routes(key("route:GET /files/a/b"), key("route:ANY /files/**")) is not None
    assert match_routes(key("route:GET /files"), key("route:GET /files/**")) is None


def test_cross_repo_links_match_calls_to_other_repos(repo: Path, tmp_path: Path) -> None:
    client = _indexed_copy(repo, tmp_path, "client")
    service = _indexed_copy(repo, tmp_path, "service")
    with GraphStore.open(client) as store, store.transaction():
        store.add_edges(
            [
                Edge(
                    src="pkg.billing.charge",
                    dst="route:GET /invoices/42",
                    kind="http_calls",
                    path="pkg/billing.py",
                    line=18,
                    resolved=False,
                    detail="GET",
                ),
                Edge(
                    src="pkg.billing.charge",
                    dst="route:DELETE /nowhere",
                    kind="http_calls",
                    path="pkg/billing.py",
                    line=19,
                    resolved=False,
                ),
            ]
        )
    with GraphStore.open(service) as store, store.transaction():
        route = Node(
            qname="route:GET /invoices/{invoice_id}",
            kind="route",
            path="pkg/billing.py",
            line_start=17,
            line_end=17,
            language="python",
            name="GET /invoices/{invoice_id}",
        )
        store.add_nodes([route])
        store.add_edges([Edge(route.qname, "pkg.billing.charge", "handles", "pkg/billing.py", 17)])
    path = registry_path()
    entries = [register_repo(path, client), register_repo(path, service)]

    links = cross_repo_links(entries)
    assert len(links) == 1
    [link] = links
    assert (link.caller_repo, link.target_repo) == ("client", "service")
    assert link.route == "route:GET /invoices/{invoice_id}"
    assert link.handlers == ("pkg.billing.charge",)
    assert link.confidence == 0.9
    assert cross_repo_links(entries, source="service") == []
    assert cross_repo_links(entries, min_confidence=0.95) == []


def test_delete_project_needs_confirm(repo: Path) -> None:
    with GraphStore.open(repo) as store:
        index_repository(repo, store)
    (repo / ".codegraph" / "notes").mkdir()
    entry = register_repo(registry_path(), repo)
    db = default_db_path(repo)

    preview = delete_project_db(entry, confirm=False)
    assert str(db) in preview.files and not preview.deleted
    assert db.exists()

    done = delete_project_db(entry, confirm=True)
    assert done.deleted and not db.exists()
    assert (repo / ".codegraph" / "notes").is_dir()
