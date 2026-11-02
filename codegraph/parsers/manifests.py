"""Package manifests that turn bare import specifiers into repo-local paths.

A Go import path, a Rust crate name or an npm package name says nothing about where
the code lives until the manifest that declares it is read: ``go.mod``'s ``module``
line, ``Cargo.toml``'s ``[package] name``, ``package.json``'s ``name`` (workspace
packages included, since each has its own ``package.json``). Manifests are looked up
only in directories that contain indexed files and their ancestors, so ignored trees
(``node_modules``, ``vendor``) are never read.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

_GO_MODULE = re.compile(r"^\s*module\s+(\S+)", re.M)
_ENTRY_CANDIDATES = (
    "src/index.ts",
    "src/index.tsx",
    "src/index.js",
    "src/index.mjs",
    "index.ts",
    "index.tsx",
    "index.js",
    "index.mjs",
    "lib/index.js",
    "src/main.ts",
    "src/main.js",
)
_SOURCE_SUFFIXES = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte")


@dataclass(frozen=True, slots=True)
class GoModule:
    path: str
    directory: str


@dataclass(frozen=True, slots=True)
class Crate:
    name: str
    """The crate's identifier as written in ``use`` paths (``my-lib`` -> ``my_lib``)."""
    source_dir: str


@dataclass(frozen=True, slots=True)
class Package:
    name: str
    directory: str
    entries: tuple[str, ...]
    """Entry files the manifest declares (``source``, ``module``, ``main``, ``exports``)."""


@dataclass(frozen=True, slots=True)
class Manifests:
    go: tuple[GoModule, ...] = ()
    crates: tuple[Crate, ...] = ()
    packages: tuple[Package, ...] = field(default=())

    def go_module_for(self, import_path: str) -> tuple[GoModule, str] | None:
        """The local module an import path belongs to, plus the path inside it."""
        for module in sorted(self.go, key=lambda m: len(m.path), reverse=True):
            if import_path == module.path:
                return module, ""
            if import_path.startswith(module.path + "/"):
                return module, import_path[len(module.path) + 1 :]
        return None

    def crate(self, name: str) -> Crate | None:
        return next((c for c in self.crates if c.name == name), None)

    def package_for(self, specifier: str) -> tuple[Package, str] | None:
        """The workspace package a bare specifier names, plus the subpath after it."""
        for package in sorted(self.packages, key=lambda p: len(p.name), reverse=True):
            if specifier == package.name:
                return package, ""
            if specifier.startswith(package.name + "/"):
                return package, specifier[len(package.name) + 1 :]
        return None


def candidate_dirs(paths: Iterable[str]) -> list[str]:
    """Every directory holding an indexed file, plus all their ancestors ("" is the root)."""
    found: set[str] = {""}
    for path in paths:
        parent = PurePosixPath(path).parent
        while parent.as_posix() not in {".", ""}:
            found.add(parent.as_posix())
            parent = parent.parent
    return sorted(found)


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _join(directory: str, name: str) -> str:
    return f"{directory}/{name}" if directory else name


def _package_entries(manifest: dict[str, object]) -> tuple[str, ...]:
    entries: list[str] = []
    for key in ("source", "module", "main", "types", "typings"):
        value = manifest.get(key)
        if isinstance(value, str):
            entries.append(value)
    exports = manifest.get("exports")
    if isinstance(exports, str):
        entries.append(exports)
    elif isinstance(exports, dict):
        dot = exports.get(".", exports)
        if isinstance(dot, str):
            entries.append(dot)
        elif isinstance(dot, dict):
            entries.extend(v for k, v in dot.items() if isinstance(v, str) and k != "types")
    return tuple(e.removeprefix("./") for e in entries)


def read_manifests(root: Path, paths: Iterable[str]) -> Manifests:
    go: list[GoModule] = []
    crates: list[Crate] = []
    packages: list[Package] = []
    for directory in candidate_dirs(paths):
        base = root / directory if directory else root
        go_mod = _read(base / "go.mod")
        if go_mod is not None:
            match = _GO_MODULE.search(go_mod)
            if match:
                go.append(GoModule(match.group(1), directory))
        cargo = _read(base / "Cargo.toml")
        if cargo is not None:
            crate = _crate(cargo, directory)
            if crate is not None:
                crates.append(crate)
        package_json = _read(base / "package.json")
        if package_json is not None:
            package = _package(package_json, directory)
            if package is not None:
                packages.append(package)
    return Manifests(go=tuple(go), crates=tuple(crates), packages=tuple(packages))


def _crate(text: str, directory: str) -> Crate | None:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None
    package = data.get("package")
    library = data.get("lib")
    name = None
    if isinstance(library, dict) and isinstance(library.get("name"), str):
        name = library["name"]
    elif isinstance(package, dict) and isinstance(package.get("name"), str):
        name = package["name"]
    if not isinstance(name, str):
        return None
    return Crate(name.replace("-", "_"), _join(directory, "src"))


def _package(text: str, directory: str) -> Package | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        return None
    return Package(str(data["name"]), directory, _package_entries(data))


def package_files(package: Package, subpath: str) -> list[str]:
    """Repo-relative files that might implement ``package`` or ``package/subpath``."""
    if subpath:
        stems = [subpath, f"src/{subpath}", f"lib/{subpath}"]
        candidates = [f"{s}{suffix}" for s in stems for suffix in ("", *_SOURCE_SUFFIXES)]
        candidates += [f"{s}/index{suffix}" for s in stems for suffix in _SOURCE_SUFFIXES]
    else:
        declared = list(package.entries)
        declared += [
            str(PurePosixPath(e).with_suffix(s))
            for e in package.entries
            for s in _SOURCE_SUFFIXES
            if PurePosixPath(e).suffix
        ]
        candidates = declared + list(_ENTRY_CANDIDATES)
    return [_join(package.directory, c) for c in candidates]


__all__ = [
    "Crate",
    "GoModule",
    "Manifests",
    "Package",
    "candidate_dirs",
    "package_files",
    "read_manifests",
]
