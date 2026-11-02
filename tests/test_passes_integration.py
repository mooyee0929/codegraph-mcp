from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from codegraph.cli import main
from codegraph.indexer import index_repository
from codegraph.passes import discover_passes
from codegraph.server import build_server
from codegraph.services.traces import ingest
from codegraph.store import GraphStore

FIXTURE = Path(__file__).parent / "fixtures" / "passes_services"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _copy(tmp_path: Path) -> Path:
    target = tmp_path / "svc"
    shutil.copytree(FIXTURE, target)
    return target


@pytest.fixture
def services_repo(tmp_path: Path) -> Path:
    """The fixture as a git repo where orders/api.py and web/client.ts change together."""
    target = _copy(tmp_path)
    _git(target, "init", "-q")
    _git(target, "config", "user.email", "ci@example.com")
    _git(target, "config", "user.name", "ci")
    _git(target, "add", "-A")
    _git(target, "commit", "-q", "-m", "init")
    for revision in range(3):
        for rel, comment in (("orders/api.py", "#"), ("web/client.ts", "//")):
            path = target / rel
            path.write_text(path.read_text() + f"{comment} revision {revision}\n")
        _git(target, "add", "-A")
        _git(target, "commit", "-q", "-m", f"revision {revision}")
    return target


@pytest.fixture
def indexed(services_repo: Path) -> Path:
    with GraphStore.open(services_repo) as store:
        index_repository(services_repo, store)
    return services_repo


def _is_error(payload: object) -> bool:
    return isinstance(payload, dict) and "error" in payload


def _snapshot(store: GraphStore) -> dict[str, dict[str, int]]:
    grouped = store.counts_by_kind()
    return {"nodes": grouped["nodes"], "edges": grouped["edges"]}


def test_passes_are_registered_in_band() -> None:
    mine = {p.name: p.order for p in discover_passes()}
    for name in ("routes", "http_calls", "channels", "clones", "git_history", "trace_spans"):
        assert 10 <= mine[name] <= 49
    assert mine["routes"] < mine["http_calls"] < mine["trace_spans"]


def test_routes_handlers_and_http_calls(indexed: Path) -> None:
    with GraphStore.open(indexed) as store:
        routes = {n.qname for n in store.all_nodes(kinds=("route",))}
        assert routes == {
            "route:GET /api/orders/{order_id}",
            "route:POST /api/orders",
            "route:GET /health",
            "route:POST /hooks/{source}",
        }
        [handles] = store.edges_from("route:GET /api/orders/{order_id}", "handles")
        assert handles.dst == "orders.api.get_order" and handles.confidence == 1.0
        [health] = store.edges_from("route:GET /health", "handles")
        assert health.dst == "web.server.health"

        calls = {e.src: e for e in store.all_edges(kinds=("http_calls",))}
        assert calls["web.client.loadOrder"].dst == "route:GET /api/orders/{order_id}"
        assert calls["web.client.loadOrder"].resolved
        assert calls["web.client.createOrder"].dst == "route:POST /api/orders"
        unmatched = calls["web.client.loadInvoices"]
        assert (unmatched.dst, unmatched.resolved) == ("route:GET /billing/invoices", False)
        assert calls["tests.test_orders.test_get_order"].dst == "route:GET /api/orders/{order_id}"


def test_channels_clones_and_git_history(indexed: Path) -> None:
    with GraphStore.open(indexed) as store:
        [emit] = store.edges_to("channel:orders.created", "emits")
        [listen] = store.edges_to("channel:orders.created", "listens_on")
        assert (emit.src, listen.src) == ("orders.api.publish_created", "orders.worker.consume")
        celery = store.edges_to("channel:celery:orders.worker.send_receipt")
        assert {(e.kind, e.src) for e in celery} == {
            ("emits", "orders.worker.consume"),
            ("listens_on", "orders.worker.send_receipt"),
        }
        assert {e.src for e in store.edges_to("channel:order:updated")} == {
            "web.server.notify",
            "web.server.onUpdate",
        }

        [clone] = store.all_edges(kinds=("similar_to",))
        assert (clone.src, clone.dst) == (
            "billing.totals.sum_invoice",
            "orders.pricing.compute_total",
        )
        assert clone.detail == "jaccard=1.00"

        [coupled] = store.all_edges(kinds=("co_changes",))
        assert {coupled.src, coupled.dst} == {"orders.api", "web.client"}
        assert coupled.detail == "count=4 ratio=1.00"
        churn = store.connection.execute(
            "SELECT commits FROM git_file_churn WHERE path = 'orders/api.py'"
        ).fetchone()
        assert churn["commits"] == 4
        assert store.get_meta("git_history") == "ok: 4 commits"


def test_passes_are_idempotent(indexed: Path) -> None:
    with GraphStore.open(indexed) as store:
        before = _snapshot(store)
        routes_rows = store.connection.execute("SELECT COUNT(*) FROM passes_routes").fetchone()[0]
        index_repository(indexed, store, force=True)
        assert _snapshot(store) == before
        assert (
            store.connection.execute("SELECT COUNT(*) FROM passes_routes").fetchone()[0]
            == routes_rows
        )


def test_trace_ingestion_validates_adds_and_survives_reindex(indexed: Path) -> None:
    spans: list[dict[str, str | int | None]] = [
        {
            "service": "orders",
            "method": "GET",
            "url": "http://orders/api/orders/7",
            "status": 200,
            "caller": "web.client.loadOrder",
        },
        {
            "service": "orders",
            "method": "GET",
            "path": "/api/orders/3",
            "status": 200,
            "caller": "orders.api.create_order",
        },
        {
            "service": "billing",
            "method": "GET",
            "url": "/billing/invoices",
            "status": 200,
            "caller": "web.client.loadInvoices",
        },
        {"service": "orders", "method": "GET", "url": "/nope", "status": 404},
    ]
    with GraphStore.open(indexed) as store:
        stats, errors = ingest(store, spans)
        assert errors == [] and stats is not None
        assert (stats.validated, stats.added, stats.unmatched) == (2, 1, 2)
        observed = {
            e.src: e for e in store.all_edges(kinds=("http_calls",)) if "observed" in e.detail
        }
        assert observed["web.client.loadOrder"].confidence == 1.0
        assert observed["orders.api.create_order"].detail == "observed"
        assert not observed["web.client.loadInvoices"].resolved

        index_repository(indexed, store, force=True)
        again = {e.src for e in store.all_edges(kinds=("http_calls",)) if "observed" in e.detail}
        assert again == set(observed)

        bad, problems = ingest(store, [{"method": 5}])
        assert bad is None and problems == [
            "span 0: needs a non-empty 'url' or 'path'; 'method' must be a string"
        ]


def test_repository_without_git_is_a_noop(tmp_path: Path) -> None:
    repo = _copy(tmp_path)
    with GraphStore.open(repo) as store:
        index_repository(repo, store)
        assert store.all_edges(kinds=("co_changes",)) == []
        status = store.get_meta("git_history") or ""
        assert status.startswith("no history") or status.startswith("ok")
        assert store.all_nodes(kinds=("route",))


def test_git_repository_without_commits_is_a_noop(tmp_path: Path) -> None:
    repo = _copy(tmp_path)
    _git(repo, "init", "-q")
    with GraphStore.open(repo) as store:
        index_repository(repo, store)
        assert store.connection.execute("SELECT COUNT(*) FROM git_commits").fetchone()[0] == 0
        assert (store.get_meta("git_history") or "").startswith("no history")


async def test_mcp_tools(indexed: Path) -> None:
    server = build_server(indexed)
    names = {t.name for t in await server.list_tools()}
    assert {
        "list_routes",
        "get_route",
        "list_channels",
        "get_channel",
        "find_clones",
        "get_hotspots",
        "get_co_changes",
        "get_file_history",
        "ingest_traces",
    } <= names

    async def call(name: str, args: dict[str, object]) -> object:
        result = await server.call_tool(name, args)
        blocks = getattr(result, "content", None)
        assert isinstance(blocks, list)
        payload: object = json.loads("".join(str(getattr(b, "text", "")) for b in blocks))
        return payload

    routes = await call("list_routes", {"framework": "fastapi"})
    assert isinstance(routes, list) and {r["path"] for r in routes} == {
        "/api/orders/{order_id}",
        "/api/orders",
    }
    detail = await call("get_route", {"target": "GET /api/orders/42"})
    assert isinstance(detail, dict)
    assert detail["route"]["qname"] == "route:GET /api/orders/{order_id}"
    assert [h["qname"] for h in detail["handlers"]] == ["orders.api.get_order"]
    assert "tests.test_orders.test_get_order" in detail["tests"]
    assert {c["src"] for c in detail["callers"]} >= {"web.client.loadOrder"}
    assert _is_error(await call("get_route", {"target": "DELETE /nothing"}))

    channels = await call("list_channels", {"broker": "kafka"})
    assert isinstance(channels, list) and [c["name"] for c in channels] == ["orders.created"]
    channel = await call("get_channel", {"target": "orders.created"})
    assert isinstance(channel, dict) and channel["listeners"][0]["node"] == "orders.worker.consume"

    clones = await call("find_clones", {"target": "compute_total"})
    assert isinstance(clones, list) and clones[0]["score"] == 1.0
    assert _is_error(await call("find_clones", {"min_score": 3}))

    hot = await call("get_hotspots", {"limit": 3})
    assert isinstance(hot, dict) and hot["hotspots"][0]["path"] in {
        "orders/api.py",
        "web/client.ts",
    }
    partners = await call("get_co_changes", {"target": "orders.api"})
    assert isinstance(partners, dict) and partners["partners"][0]["path"] == "web/client.ts"
    history = await call("get_file_history", {"path": "web/client.ts"})
    assert isinstance(history, dict) and len(history["commits"]) == 4
    assert _is_error(await call("get_file_history", {"path": "missing.py"}))

    ingested = await call(
        "ingest_traces",
        {"spans": [{"service": "web", "method": "GET", "url": "/health", "status": 200}]},
    )
    assert isinstance(ingested, dict) and ingested["spans"] == 1
    rejected = await call("ingest_traces", {"spans": [{"service": "web"}]})
    assert isinstance(rejected, dict) and "error" in rejected


def test_cli_commands(indexed: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = ["--root", str(indexed)]
    assert main([*root, "routes", "--method", "POST"]) == 0
    out = capsys.readouterr().out
    assert "/api/orders" in out and "/hooks/{source}" in out
    assert main([*root, "routes", "--path", "zzz"]) == 1
    assert main([*root, "routes", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 4

    assert main([*root, "channels", "--broker", "socket.io"]) == 0
    assert "order:updated" in capsys.readouterr().out
    assert main([*root, "clones"]) == 0
    assert "sum_invoice" in capsys.readouterr().out
    assert main([*root, "clones", "nothing_named_this"]) == 2
    assert main([*root, "clones", "--min-score", "2"]) == 2

    assert main([*root, "hotspots", "--limit", "2"]) == 0
    assert "orders/api.py" in capsys.readouterr().out
    assert main([*root, "co-changes", "orders/api.py", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["path"] == "web/client.ts"
    assert main([*root, "co-changes", "nope.py"]) == 2

    spans = tmp_path / "spans.json"
    spans.write_text(
        json.dumps({"spans": [{"method": "GET", "url": "/health", "caller": "web.server.notify"}]})
    )
    assert main([*root, "ingest-traces", str(spans)]) == 0
    assert "added 1" in capsys.readouterr().out
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert main([*root, "ingest-traces", str(broken)]) == 2
    assert main([*root, "ingest-traces", str(tmp_path / "missing.json")]) == 2
