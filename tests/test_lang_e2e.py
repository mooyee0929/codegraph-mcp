"""End to end: a mixed Go/Java/Rust/Docker/Kubernetes repo through the indexer, the
resolution pass, the MCP tools and the CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codegraph.cli import main
from codegraph.indexer import index_repository
from codegraph.parsers.inventory import language_inventory, list_resources
from codegraph.parsers.resolve import PASS
from codegraph.server import build_server
from codegraph.store import GraphStore

FILES = {
    "go.mod": "module example.com/shop\n\ngo 1.22\n",
    "cmd/api/main.go": (
        "package main\n\n"
        'import "example.com/shop/internal/billing"\n\n'
        "func main() {\n\tbilling.Charge(10)\n}\n"
    ),
    "internal/billing/billing.go": (
        "package billing\n\n"
        "// Charge bills a customer.\n"
        "func Charge(cents int) int {\n\treturn applyTax(cents)\n}\n"
    ),
    "internal/billing/tax.go": "package billing\n\nfunc applyTax(c int) int { return c }\n",
    "internal/billing/billing_test.go": (
        'package billing\n\nimport "testing"\n\nfunc TestCharge(t *testing.T) {\n\tCharge(1)\n}\n'
    ),
    "jvm/src/main/java/com/shop/app/Checkout.java": (
        "package com.shop.app;\n\n"
        "import com.shop.pricing.PriceList;\n\n"
        "public class Checkout {\n"
        '    public int total() { return PriceList.lookup("sku") + discount(); }\n'
        "    int discount() { return 0; }\n"
        "}\n"
    ),
    "jvm/src/main/java/com/shop/pricing/PriceList.java": (
        "package com.shop.pricing;\n\n"
        "public class PriceList {\n"
        "    public static int lookup(String sku) { return 1; }\n"
        "}\n"
    ),
    "engine/src/main.rs": "mod rules;\n\nfn main() {\n    rules::evaluate();\n}\n",
    "engine/src/rules.rs": "pub fn evaluate() -> bool {\n    true\n}\n",
    "Dockerfile": (
        "FROM golang:1.22 AS build\nRUN go build ./cmd/api\n"
        "FROM gcr.io/distroless/base\nCOPY --from=build /app /app\n"
    ),
    "deploy/api.yaml": (
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: api\n"
        "spec:\n  template:\n    metadata:\n      labels: {app: api}\n"
        "    spec:\n      containers:\n        - name: api\n          image: shop/api:1\n"
        "          envFrom: [{configMapRef: {name: api-config}}]\n"
        "---\napiVersion: v1\nkind: ConfigMap\nmetadata: {name: api-config}\ndata: {A: '1'}\n"
    ),
    "deploy/service.yaml": (
        "apiVersion: v1\nkind: Service\nmetadata: {name: api}\nspec:\n  selector: {app: api}\n"
    ),
    ".github/workflows/ci.yml": "on: [push]\njobs:\n  test:\n    runs-on: ubuntu-latest\n",
}


@pytest.fixture
def shop(tmp_path: Path) -> Path:
    root = tmp_path / "shop"
    for rel, text in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _index(root: Path, store: GraphStore) -> None:
    stats = index_repository(root, store)
    if PASS.name not in {p.name for p in stats.passes}:
        with store.transaction():
            PASS.run(root, store)


def _calls(store: GraphStore, src: str) -> set[str]:
    return {e.dst for e in store.edges_from(src, kind="calls") if e.resolved}


def test_mixed_repo_cross_file_calls(shop: Path) -> None:
    with GraphStore.open(shop) as store:
        stats = index_repository(shop, store)
        assert stats.parsed == len(FILES) - 1
        # Cross-file calls resolve through the indexer and linker alone ...
        assert "internal.billing.billing.Charge" in _calls(store, "cmd.api.main.main")
        assert "internal.billing.tax.applyTax" in _calls(store, "internal.billing.billing.Charge")
        lookup = "jvm.src.main.java.com.shop.pricing.PriceList.PriceList.lookup"
        assert lookup in _calls(store, "jvm.src.main.java.com.shop.app.Checkout.Checkout.total")
        assert "engine.src.rules.evaluate" in _calls(store, "engine.src.main.main")
        tests = {e.src for e in store.edges_to("internal.billing.billing.Charge", kind="tests")}
        assert tests == {"internal.billing.billing_test.TestCharge"}

        # ... and the resolution pass maps imports onto modules via go.mod.
        _index(shop, store)
        go_imports = {e.dst for e in store.edges_from("cmd.api.main", kind="imports")}
        assert go_imports == {"internal.billing.billing", "internal.billing.tax"}
        service = {e.dst for e in store.edges_from("resource:k8s/Service/api", "depends_on")}
        assert service == {"resource:k8s/Deployment/api"}

        assert store.get_node("resource:docker/stage/build") is not None
        assert store.get_node(".github.workflows.ci") is None
        languages = store.counts_by_kind()["languages"]
        assert languages["go"] == 4 and languages["java"] == 2 and languages["rust"] == 2


def test_inventory_on_mixed_repo(shop: Path) -> None:
    with GraphStore.open(shop) as store:
        _index(shop, store)
        rows = {r.language: r for r in language_inventory(store, indexed_only=True)}
        assert rows["go"].files == 4
        assert ".go" in rows["go"].suffixes
        assert rows["dockerfile"].files == 1
        assert "python" not in rows
        everything = list_resources(store)
        assert {r.type for r in everything} >= {"docker/stage", "k8s/Deployment", "k8s/Service"}
        deployments = list_resources(store, "k8s/deploy")
        assert [r.qname for r in deployments] == ["resource:k8s/Deployment/api"]
        [deployment] = deployments
        targets = {d.qname for d in deployment.depends_on}
        assert targets == {"resource:docker/image/shop/api:1", "resource:k8s/ConfigMap/api-config"}
        assert deployment.required_by == ("resource:k8s/Service/api",)


def _text(result: object) -> str:
    blocks = getattr(result, "content", None)
    assert isinstance(blocks, list), f"unexpected tool result: {result!r}"
    return "".join(str(getattr(block, "text", "")) for block in blocks)


async def test_mcp_tools(shop: Path) -> None:
    with GraphStore.open(shop) as store:
        _index(shop, store)
    server = build_server(shop)
    names = {t.name for t in await server.list_tools()}
    assert {"list_languages", "list_resources"} <= names

    languages = json.loads(_text(await server.call_tool("list_languages", {})))
    by_name = {row["language"]: row for row in languages["languages"]}
    assert by_name["go"]["files"] == 4
    assert by_name["haskell"]["files"] == 0
    assert languages["total_files"] == len(FILES) - 1

    indexed = json.loads(_text(await server.call_tool("list_languages", {"indexed_only": True})))
    assert all(row["files"] > 0 for row in indexed["languages"])

    found = json.loads(
        _text(await server.call_tool("list_resources", {"resource_type": "k8s/Service"}))
    )
    assert found["count"] == 1
    [service] = found["resources"]
    assert service["depends_on"][0]["qname"] == "resource:k8s/Deployment/api"

    blank = json.loads(_text(await server.call_tool("list_resources", {"resource_type": " "})))
    assert "error" in blank


def test_cli_commands(shop: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--root", str(shop), "index"]) == 0
    capsys.readouterr()

    assert main(["--root", str(shop), "languages", "--indexed"]) == 0
    out = capsys.readouterr().out
    assert "go" in out and "dockerfile" in out and "haskell" not in out

    assert main(["--root", str(shop), "languages", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["language"] for r in rows} >= {"go", "rust", "terraform"}

    assert main(["--root", str(shop), "resources", "--type", "docker"]) == 0
    out = capsys.readouterr().out
    assert "resource:docker/stage/build" in out
    assert "-> resource:docker/image/golang:1.22" in out

    assert main(["--root", str(shop), "resources", "--type", "terraform"]) == 1
    assert main(["--root", str(shop), "resources", "--type", " "]) == 2
    assert main(["--root", str(shop), "resources", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert any(r["qname"] == "resource:k8s/ConfigMap/api-config" for r in payload)
