"""Cross-file resolution: manifests (go.mod, Cargo.toml, package.json), guesses, selectors."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from codegraph.indexer import index_repository
from codegraph.models import Edge
from codegraph.parsers.manifests import candidate_dirs, package_files, read_manifests
from codegraph.parsers.resolve import PASS, resolve_repository
from codegraph.passes import GraphPass
from codegraph.store import GraphStore


def write(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def indexed(tmp_path: Path) -> Iterator[tuple[Path, GraphStore]]:
    root = tmp_path / "repo"
    root.mkdir()
    with GraphStore.open(root) as store:
        yield root, store


def index_and_resolve(root: Path, store: GraphStore) -> int:
    index_repository(root, store)
    with store.transaction():
        return resolve_repository(root, store)


def edges_from(store: GraphStore, src: str, kind: str = "calls") -> set[tuple[str, bool]]:
    return {(e.dst, e.resolved) for e in store.edges_from(src) if e.kind == kind}


def test_pass_object_satisfies_protocol() -> None:
    assert isinstance(PASS, GraphPass)
    assert PASS.name == "resolve_languages"
    assert 10 <= PASS.order < 50


def test_manifest_discovery(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "go.mod": "module example.com/app\n\ngo 1.22\n",
            "crates/core/Cargo.toml": '[package]\nname = "app-core"\n',
            "packages/ui/package.json": '{"name": "@acme/ui", "main": "dist/index.js"}',
            "node_modules/x/package.json": '{"name": "x"}',
        },
    )
    paths = ["main.go", "crates/core/src/lib.rs", "packages/ui/src/index.ts"]
    assert "crates/core" in candidate_dirs(paths)
    manifests = read_manifests(tmp_path, paths)
    assert manifests.go_module_for("example.com/app/internal/util") is not None
    assert manifests.go_module_for("github.com/other/lib") is None
    crate = manifests.crate("app_core")
    assert crate is not None and crate.source_dir == "crates/core/src"
    found = manifests.package_for("@acme/ui/button")
    assert found is not None and found[1] == "button"
    assert manifests.package_for("x") is None
    package = found[0]
    assert "packages/ui/src/index.ts" in package_files(package, "")
    assert "packages/ui/src/button.tsx" in package_files(package, "button")


def test_bad_manifests_are_ignored(tmp_path: Path) -> None:
    write(tmp_path, {"Cargo.toml": "[package\n", "package.json": "{nope", "go.mod": "go 1.22\n"})
    manifests = read_manifests(tmp_path, ["a.rs"])
    assert manifests.crates == () and manifests.packages == () and manifests.go == ()


GO_FILES = {
    "go.mod": "module example.com/app\n",
    "cmd/server/main.go": (
        "package main\n\n"
        'import (\n\t"example.com/app/internal/util"\n\t"github.com/pkg/errors"\n)\n\n'
        'func main() {\n\tutil.Pad("x")\n\terrors.Wrap(nil, "x")\n\thelper()\n}\n'
    ),
    "cmd/server/helper.go": "package main\n\nfunc helper() {}\n",
    "internal/util/util.go": "package util\n\nfunc Pad(s string) string { return s }\n",
    "internal/util/extra.go": "package util\n\nfunc Wrap() {}\n",
    "internal/other/other.go": "package other\n\nfunc Pad() {}\n",
}


def test_go_module_imports_and_calls(indexed: tuple[Path, GraphStore]) -> None:
    root, store = indexed
    write(root, GO_FILES)
    assert index_and_resolve(root, store) > 0
    imports = edges_from(store, "cmd.server.main", "imports")
    assert ("internal.util.util", True) in imports
    assert ("internal.util.extra", True) in imports
    assert ("github.com/pkg/errors", True) in imports
    calls = edges_from(store, "cmd.server.main.main")
    assert ("internal.util.util.Pad", True) in calls
    # The linker matched the unique local ``Wrap``; go.mod says the import is external.
    assert ("github.com/pkg/errors.Wrap", True) in calls
    assert ("cmd.server.helper.helper", True) in calls


def test_resolution_is_idempotent(indexed: tuple[Path, GraphStore]) -> None:
    root, store = indexed
    write(root, GO_FILES)
    index_and_resolve(root, store)
    before = sorted(store.all_edges(), key=_key)
    with store.transaction():
        assert resolve_repository(root, store) == 0
    assert sorted(store.all_edges(), key=_key) == before


def _key(edge: Edge) -> tuple[str, str, str, int]:
    return (edge.src, edge.dst, edge.kind, edge.line)


def test_rust_workspace_crates(indexed: tuple[Path, GraphStore]) -> None:
    root, store = indexed
    write(
        root,
        {
            "Cargo.toml": '[workspace]\nmembers = ["crates/*"]\n',
            "crates/net/Cargo.toml": '[package]\nname = "my-net"\n',
            "crates/net/src/lib.rs": "pub mod client;\npub fn version() -> u32 { 1 }\n",
            "crates/net/src/client.rs": "pub fn connect() {}\npub fn close() {}\n",
            "crates/app/Cargo.toml": '[package]\nname = "app"\n',
            "crates/app/src/main.rs": (
                "use my_net::client::connect;\nuse my_net::version;\n"
                "fn main() { connect(); version(); }\n"
            ),
            "other/src/client.rs": "pub fn connect() {}\n",
        },
    )
    index_and_resolve(root, store)
    calls = edges_from(store, "crates.app.src.main.main")
    assert ("crates.net.src.client.connect", True) in calls
    assert ("crates.net.src.lib.version", True) in calls
    imports = edges_from(store, "crates.app.src.main", "imports")
    assert ("crates.net.src.client", True) in imports


def test_js_workspace_packages(indexed: tuple[Path, GraphStore]) -> None:
    root, store = indexed
    write(
        root,
        {
            "package.json": '{"name": "root", "workspaces": ["packages/*"]}',
            "packages/utils/package.json": '{"name": "@acme/utils", "main": "dist/index.js"}',
            "packages/utils/src/index.ts": "export function slugify(s: string) { return s; }\n",
            "packages/utils/src/dates.ts": "export function today() { return 1; }\n",
            "apps/web/src/app.ts": (
                'import { slugify } from "@acme/utils";\n'
                'import { today } from "@acme/utils/dates";\n'
                'import React from "react";\n'
                "export function run() { slugify('x'); today(); }\n"
            ),
        },
    )
    index_and_resolve(root, store)
    imports = edges_from(store, "apps.web.src.app", "imports")
    assert ("packages.utils.src", True) in imports
    assert ("packages.utils.src.dates", True) in imports
    assert ("react", True) in imports
    calls = edges_from(store, "apps.web.src.app.run")
    assert ("packages.utils.src.slugify", True) in calls
    assert ("packages.utils.src.dates.today", True) in calls


def test_jvm_guesses_resolve_across_source_roots(indexed: tuple[Path, GraphStore]) -> None:
    root, store = indexed
    write(
        root,
        {
            "app/src/main/java/com/acme/app/Main.java": (
                "package com.acme.app;\nimport com.acme.core.Strings;\n"
                "class Main { void run() { Strings.pad(); Local.go(); } }\n"
            ),
            "app/src/main/java/com/acme/app/Local.java": (
                "package com.acme.app;\nclass Local { static void go() {} }\n"
            ),
            "core/src/main/java/com/acme/core/Strings.java": (
                "package com.acme.core;\npublic class Strings { static void pad() {} }\n"
            ),
            "other/Strings.java": "class Strings { static void pad() {} }\n",
            "other/Local.java": "class Local { static void go() {} }\n",
        },
    )
    index_and_resolve(root, store)
    calls = edges_from(store, "app.src.main.java.com.acme.app.Main.Main.run")
    assert ("core.src.main.java.com.acme.core.Strings.Strings.pad", True) in calls
    assert ("app.src.main.java.com.acme.app.Local.Local.go", True) in calls
    imports = edges_from(store, "app.src.main.java.com.acme.app.Main", "imports")
    assert ("core.src.main.java.com.acme.core.Strings", True) in imports


def test_includes_resolve_by_header_name(indexed: tuple[Path, GraphStore]) -> None:
    root, store = indexed
    write(
        root,
        {
            "src/main.c": '#include "util.h"\nint main(void) { return twice(1); }\n',
            "include/util.h": "int twice(int x);\n",
            "lib/util.c": "int twice(int x) { return 2 * x; }\n",
        },
    )
    index_and_resolve(root, store)
    assert ("include.util_h", True) in edges_from(store, "src.main", "imports")
    assert ("lib.util.twice", True) in edges_from(store, "src.main.main")


def test_kubernetes_selectors_across_files(indexed: tuple[Path, GraphStore]) -> None:
    root, store = indexed
    write(
        root,
        {
            "k8s/deploy.yaml": (
                "apiVersion: apps/v1\nkind: Deployment\nmetadata: {name: web}\n"
                "spec:\n  template:\n    metadata:\n      labels: {app: web}\n"
                "    spec:\n      containers: [{name: web, image: nginx}]\n"
            ),
            "k8s/service.yaml": (
                "apiVersion: v1\nkind: Service\nmetadata: {name: web}\n"
                "spec:\n  selector: {app: web}\n"
            ),
        },
    )
    index_and_resolve(root, store)
    found = edges_from(store, "resource:k8s/Service/web", "depends_on")
    assert found == {("resource:k8s/Deployment/web", True)}


def test_ambiguous_guesses_stay_unresolved(indexed: tuple[Path, GraphStore]) -> None:
    root, store = indexed
    write(
        root,
        {
            "a/main.rb": "def run\n  Util.pad\nend\n",
            "b/util.rb": "module Util\n  def self.pad; end\nend\n",
            "c/util.rb": "module Util\n  def self.pad; end\nend\n",
        },
    )
    index_and_resolve(root, store)
    assert edges_from(store, "a.main.run") == {("*.pad", False)}
