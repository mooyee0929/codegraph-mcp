"""Cross-file resolution for the languages beyond Python, run after the linker.

Parsers see one file at a time, so they record what they can and leave hints:
unresolved edges carry the parser's qualified guess in ``Edge.detail`` (the linker
overwrites ``dst`` but not ``detail``), and import edges carry the specifier as
written. With the whole index and the package manifests in hand, this pass finishes
the job, in order:

1. **Workspace packages** (``package.json`` names): JS/TS edges to ``@org/pkg`` or
   ``@org/pkg.helper`` are pointed at that package's entry module.
2. **Rust crates** (``Cargo.toml`` names): ``use mylib::net::Client`` becomes the
   workspace crate's module path.
3. **Go modules** (``go.mod``): import paths under the module become the package
   directory's files; calls through them resolve inside that directory, and calls the
   linker guessed through an *external* import are reverted to the external name.
4. **Dangling guesses** for every tree-sitter language: the guessed qname itself, the
   longest unique qname suffix (``Strings.pad``), the guessed package, or a unique
   definition in the caller's own directory (Go/Java/Kotlin/C# same-package calls).
   Import edges whose target is not a module are pointed at the longest module prefix
   or a unique module with the same trailing path.
5. **Kubernetes selectors** left unresolved by the manifest parser are matched against
   the pod labels of every workload in the repository.

Everything is a rewrite of an existing edge, or an added twin of one, so the pass is
idempotent: an edge pointing at a real node is never touched again.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import cast

from codegraph.models import Edge, EdgeKind, Node
from codegraph.parsers.base import module_qname_for
from codegraph.parsers.kubernetes import SELECTOR_PREFIX, parse_selector, workloads
from codegraph.parsers.manifests import Manifests, package_files, read_manifests
from codegraph.parsers.ts_languages import SPECS
from codegraph.store import GraphStore

TREE_SITTER_LANGUAGES = frozenset(SPECS)
JS_LANGUAGES = frozenset({"javascript", "typescript", "vue", "svelte"})
SAME_PACKAGE_LANGUAGES = frozenset({"go", "java", "kotlin", "scala", "groovy", "csharp", "swift"})
_FILE_SPEC_LANGUAGES = frozenset({"c", "cpp", "objc", "ruby", "lua", "bash", "zig", "r", "julia"})


@dataclass(frozen=True, slots=True)
class _Row:
    id: int
    edge: Edge


@dataclass(slots=True)
class _Index:
    nodes: dict[str, Node]
    languages: dict[str, str]
    by_name: dict[str, list[str]] = field(default_factory=dict)
    modules: dict[str, str] = field(default_factory=dict)
    """module qname -> path"""
    modules_in_dir: dict[str, list[str]] = field(default_factory=dict)
    module_of_path: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, store: GraphStore) -> _Index:
        nodes = {n.qname: n for n in store.all_nodes()}
        index = cls(nodes=nodes, languages=store.file_languages())
        by_name: defaultdict[str, list[str]] = defaultdict(list)
        in_dir: defaultdict[str, list[str]] = defaultdict(list)
        for node in sorted(nodes.values(), key=lambda n: (n.path, n.line_start)):
            if node.kind == "module":
                index.modules[node.qname] = node.path
                index.module_of_path[node.path] = node.qname
                in_dir[_dir(node.path)].append(node.qname)
            elif node.kind != "resource":
                by_name[node.name].append(node.qname)
        index.by_name = dict(by_name)
        index.modules_in_dir = dict(in_dir)
        return index

    def is_node(self, qname: str) -> bool:
        return qname in self.nodes

    def language(self, path: str) -> str:
        return self.languages.get(path, "")


def _dir(path: str) -> str:
    parent = PurePosixPath(path).parent.as_posix()
    return "" if parent == "." else parent


def _rows(store: GraphStore, where: str, params: Iterable[object] = ()) -> list[_Row]:
    rows = store.connection.execute(f"SELECT * FROM edges WHERE {where}", tuple(params))
    return [
        _Row(
            int(r["id"]),
            Edge(
                src=str(r["src"]),
                dst=str(r["dst"]),
                kind=cast(EdgeKind, r["kind"]),
                path=str(r["path"]),
                line=int(r["line"]),
                resolved=bool(r["resolved"]),
                confidence=float(r["confidence"]),
                detail=str(r["detail"]),
            ),
        )
        for r in rows.fetchall()
    ]


def _retarget(store: GraphStore, row: _Row, dst: str) -> int:
    """Point an edge at ``dst`` (marking it resolved); drop it if that edge already exists."""
    if row.edge.dst == dst and row.edge.resolved:
        return 0
    try:
        store.resolve_edge(row.id, dst)
    except sqlite3.IntegrityError:
        store.connection.execute("DELETE FROM edges WHERE id = ?", (row.id,))
    return 1


def _twins(store: GraphStore, row: _Row, targets: Iterable[str]) -> int:
    edges = [
        Edge(
            src=row.edge.src,
            dst=target,
            kind=row.edge.kind,
            path=row.edge.path,
            line=row.edge.line,
            detail=row.edge.detail,
        )
        for target in targets
    ]
    store.add_edges(edges)
    return len(edges)


# -- 1. JS workspace packages -----------------------------------------------------------


def _package_module(index: _Index, manifests: Manifests, specifier: str) -> str | None:
    found = manifests.package_for(specifier)
    if found is None:
        return None
    package, subpath = found
    for path in package_files(package, subpath):
        module = index.module_of_path.get(path)
        if module is not None:
            return module
    return None


def _resolve_packages(store: GraphStore, index: _Index, manifests: Manifests) -> int:
    if not manifests.packages:
        return 0
    changed = 0
    for row in _rows(store, "kind IN ('imports', 'calls', 'inherits')"):
        edge = row.edge
        if index.language(edge.path) not in JS_LANGUAGES or index.is_node(edge.dst):
            continue
        if edge.kind == "imports":
            module = _package_module(index, manifests, edge.dst)
            if module is not None:
                changed += _retarget(store, row, module)
            continue
        specifier, _, member = edge.dst.rpartition(".")
        if not specifier or not member:
            continue
        module = _package_module(index, manifests, specifier)
        if module is None:
            continue
        target = f"{module}.{member}"
        if not index.is_node(target):
            package_dir = _dir(index.modules.get(module, ""))
            under = [
                q
                for q in index.by_name.get(member, [])
                if index.nodes[q].path.startswith(package_dir + "/" if package_dir else "")
            ]
            if len(under) != 1:
                continue
            target = under[0]
        changed += _retarget(store, row, target)
    return changed


# -- 2. Rust crates -----------------------------------------------------------------------


def _resolve_crates(store: GraphStore, index: _Index, manifests: Manifests) -> int:
    if not manifests.crates:
        return 0
    changed = 0
    for row in _rows(store, "kind IN ('imports', 'calls', 'inherits')"):
        edge = row.edge
        if index.language(edge.path) != "rust" or index.is_node(edge.dst):
            continue
        hint = edge.detail if edge.kind != "imports" and edge.detail else edge.dst
        head, _, rest = hint.partition(".")
        crate = manifests.crate(head)
        if crate is None:
            continue
        prefix = module_qname_for(f"{crate.source_dir}/lib.rs").rsplit(".", 1)[0]
        options = [f"{prefix}.{rest}" if rest else f"{prefix}.lib"]
        if rest:
            options.append(f"{prefix}.lib.{rest}")
        target = next((o for o in options if index.is_node(o)), None)
        if edge.kind == "imports" and (target is None or target not in index.modules):
            target = _module_prefix(index, target or options[0])
        if target is not None:
            changed += _retarget(store, row, target)
    return changed


# -- 3. Go modules --------------------------------------------------------------------------


def _go_package_modules(index: _Index, directory: str) -> list[str]:
    modules = [
        m
        for m in index.modules_in_dir.get(directory, [])
        if index.modules[m].endswith(".go") and not index.modules[m].endswith("_test.go")
    ]
    base = PurePosixPath(directory).name
    return sorted(modules, key=lambda m: (PurePosixPath(index.modules[m]).stem != base, m))


def _resolve_go(store: GraphStore, index: _Index, manifests: Manifests) -> int:
    if not manifests.go:
        return 0
    changed = 0
    for row in _rows(store, "kind IN ('imports', 'calls', 'inherits')"):
        edge = row.edge
        if index.language(edge.path) != "go":
            continue
        if edge.kind == "imports":
            found = manifests.go_module_for(edge.dst)
            if found is None or index.is_node(edge.dst):
                continue
            module, inner = found
            directory = "/".join(p for p in (module.directory, inner) if p)
            package = _go_package_modules(index, directory)
            if package:
                changed += _retarget(store, row, package[0])
                changed += _twins(store, row, package[1:])
            continue
        if "/" not in edge.detail:
            continue
        import_path, _, member = edge.detail.rpartition(".")
        found = manifests.go_module_for(import_path)
        if found is None:
            changed += _retarget(store, row, edge.detail)
            continue
        module, inner = found
        directory = "/".join(p for p in (module.directory, inner) if p)
        candidates = [
            q
            for q in index.by_name.get(member, [])
            if _dir(index.nodes[q].path) == directory and index.nodes[q].kind != "method"
        ]
        if len(candidates) == 1:
            changed += _retarget(store, row, candidates[0])
    return changed


# -- 4. dangling guesses -------------------------------------------------------------------


def _module_prefix(index: _Index, qname: str) -> str | None:
    cursor = qname
    while "." in cursor:
        cursor = cursor.rsplit(".", 1)[0]
        if cursor in index.modules:
            return cursor
    return None


def _prefer_local(index: _Index, candidates: list[str], path: str) -> list[str]:
    if len(candidates) <= 1:
        return candidates
    near = [q for q in candidates if _dir(index.nodes[q].path) == _dir(path)]
    return near if near else candidates


def _by_suffix(index: _Index, parts: list[str], pool: list[str], path: str) -> str | None:
    for width in range(len(parts), 1, -1):
        suffix = ".".join(parts[-width:])
        matches = [q for q in pool if q == suffix or q.endswith("." + suffix)]
        if matches:
            matches = _prefer_local(index, matches, path)
            return matches[0] if len(matches) == 1 else None
    return None


def _by_package(parts: list[str], pool: list[str]) -> str | None:
    for width in range(len(parts) - 1, 1, -1):
        prefix = ".".join(parts[:width]) + "."
        matches = [q for q in pool if q.startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return None
    return None


def _resolve_hint(index: _Index, hint: str, path: str, *, dynamic: bool) -> str | None:
    """Best repo-local target for a parser's qualified guess, or None if ambiguous.

    ``dynamic`` marks a method call on an expression the parser could not name; only
    exact and suffix matches are trusted for it, never a same-directory guess.
    """
    cleaned = hint.removeprefix("*.")
    if index.is_node(cleaned) and index.nodes[cleaned].kind != "module":
        return cleaned
    parts = [p for p in cleaned.split(".") if p]
    if not parts:
        return None
    pool = index.by_name.get(parts[-1], [])
    if not pool:
        return None
    if len(parts) >= 2:
        found = _by_suffix(index, parts, pool, path) or _by_package(parts, pool)
        if found is not None:
            return found
    if not dynamic and index.language(path) in SAME_PACKAGE_LANGUAGES:
        local = [q for q in pool if _dir(index.nodes[q].path) == _dir(path)]
        if len(local) == 1:
            return local[0]
    return None


def _resolve_import(index: _Index, edge: Edge) -> str | None:
    found = _module_prefix(index, edge.dst)
    if found is not None:
        return found
    parts = [p for p in edge.dst.split(".") if p]
    looks_like_file = "/" in edge.detail or PurePosixPath(edge.detail).suffix != ""
    minimum = 1 if looks_like_file and index.language(edge.path) in _FILE_SPEC_LANGUAGES else 2
    if len(parts) < minimum:
        return None
    for width in range(len(parts), minimum - 1, -1):
        suffix = ".".join(parts[-width:])
        matches = [m for m in index.modules if m == suffix or m.endswith("." + suffix)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return None
    return None


def _resolve_dangling(store: GraphStore, index: _Index) -> int:
    changed = 0
    for row in _rows(store, "kind IN ('imports', 'calls', 'inherits')"):
        edge = row.edge
        if index.language(edge.path) not in TREE_SITTER_LANGUAGES or index.is_node(edge.dst):
            continue
        if edge.kind == "imports":
            target = _resolve_import(index, edge)
        elif not edge.resolved:
            dynamic = edge.dst.startswith("*.") and not edge.detail
            target = _resolve_hint(index, edge.detail or edge.dst, edge.path, dynamic=dynamic)
        else:
            continue
        if target is not None and target != edge.src:
            changed += _retarget(store, row, target)
    return changed


# -- 5. Kubernetes selectors ----------------------------------------------------------------


def _resolve_selectors(root: Path, store: GraphStore, index: _Index) -> int:
    rows = _rows(
        store,
        "kind = 'depends_on' AND resolved = 0 AND dst LIKE ?",
        (SELECTOR_PREFIX + "%",),
    )
    if not rows:
        return 0
    labelled: list[tuple[str, dict[str, str]]] = []
    for path, language in sorted(index.languages.items()):
        if language not in {"yaml", "kustomize"}:
            continue
        try:
            source = (root / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        labelled.extend((w.qname, w.labels) for w in workloads(source))
    changed = 0
    for row in rows:
        selector = parse_selector(row.edge.dst)
        matches = sorted(
            q for q, labels in labelled if all(labels.get(k) == v for k, v in selector.items())
        )
        if matches:
            changed += _retarget(store, row, matches[0])
            changed += _twins(store, row, matches[1:])
    return changed


def resolve_repository(root: Path, store: GraphStore) -> int:
    """Run every resolution step; returns how many edges were rewritten or added."""
    index = _Index.build(store)
    manifests = read_manifests(root, index.languages)
    changed = _resolve_packages(store, index, manifests)
    changed += _resolve_crates(store, index, manifests)
    changed += _resolve_go(store, index, manifests)
    changed += _resolve_dangling(store, index)
    changed += _resolve_selectors(root, store, index)
    return changed


class LanguageResolutionPass:
    """:class:`codegraph.passes.GraphPass` adapter for :func:`resolve_repository`."""

    name = "resolve_languages"
    order = 12

    def run(self, root: Path, store: GraphStore) -> int:
        return resolve_repository(root, store)


PASS = LanguageResolutionPass()

__all__ = ["PASS", "LanguageResolutionPass", "resolve_repository"]
