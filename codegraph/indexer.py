"""Walk a repository, parse what changed, and write the result to the store."""

from __future__ import annotations

import fnmatch
import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from codegraph.linker import link
from codegraph.parsers import language_of, module_qname_for, parser_for
from codegraph.store import GraphStore

DEFAULT_IGNORES: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    ".codegraph",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".next",
    ".turbo",
    "coverage",
    "*.min.js",
    "*.bundle.js",
)

MAX_FILE_BYTES = 2_000_000


@dataclass(frozen=True, slots=True)
class IndexStats:
    scanned: int
    parsed: int
    unchanged: int
    removed: int
    nodes: int
    edges: int
    linked: int

    def summary(self) -> str:
        return (
            f"scanned {self.scanned} files: parsed {self.parsed}, unchanged {self.unchanged}, "
            f"removed {self.removed}; wrote {self.nodes} nodes, {self.edges} edges, "
            f"linked {self.linked} edges"
        )


def _ignored(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def iter_source_files(root: Path, ignores: tuple[str, ...] = DEFAULT_IGNORES) -> Iterator[Path]:
    """Yield every parseable source file under ``root``, honouring ``ignores``."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not _ignored(d, ignores))
        for filename in sorted(filenames):
            if _ignored(filename, ignores):
                continue
            path = Path(dirpath) / filename
            if language_of(filename) is None:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield path


def _sha(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def index_repository(
    root: Path,
    store: GraphStore,
    *,
    ignores: tuple[str, ...] = DEFAULT_IGNORES,
    force: bool = False,
) -> IndexStats:
    """Bring ``store`` up to date with the files under ``root``.

    Unchanged files (same content hash) are skipped unless ``force`` is set. Files that
    disappeared since the last run are removed from the graph. After every file is
    written, the linker pass tries to resolve dangling call edges.
    """
    root = root.resolve()
    known = store.file_shas()
    seen: set[str] = set()
    scanned = parsed = unchanged = 0
    nodes_written = edges_written = 0

    with store.transaction():
        for path in iter_source_files(root, ignores):
            rel = path.relative_to(root).as_posix()
            seen.add(rel)
            scanned += 1
            try:
                data = path.read_bytes()
            except OSError:
                continue
            digest = _sha(data)
            if not force and known.get(rel) == digest:
                unchanged += 1
                continue
            language = language_of(rel)
            if language is None:
                continue
            source = data.decode("utf-8", errors="replace")
            result = parser_for(language).parse(rel, source, module_qname_for(rel))
            store.forget_file(rel)
            nodes_written += store.add_nodes(result.nodes)
            edges_written += store.add_edges(result.edges)
            store.record_file(rel, digest, language)
            parsed += 1

        removed = 0
        for rel in known:
            if rel not in seen:
                store.forget_file(rel)
                removed += 1

        linked = link(store) if parsed or removed else 0
        store.set_meta("root", str(root))

    return IndexStats(
        scanned=scanned,
        parsed=parsed,
        unchanged=unchanged,
        removed=removed,
        nodes=nodes_written,
        edges=edges_written,
        linked=linked,
    )
