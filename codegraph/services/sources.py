"""Which files the service passes read, and how each one is labelled.

The indexer only walks files some parser understands. Routes and channels are also
declared in files no parser reads yet (Go, Java, ``.proto``, ``routes.rb``), so the
passes walk the tree themselves with the indexer's ignore rules plus ``vendor/``.
"""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from codegraph.indexer import DEFAULT_IGNORES, MAX_FILE_BYTES, load_ignore_file
from codegraph.linker import is_test_path
from codegraph.services.text import CommentMask, Family, line_at, line_starts

LANGUAGE_BY_SUFFIX: dict[str, tuple[str, Family]] = {
    ".py": ("python", "python"),
    ".js": ("javascript", "js"),
    ".jsx": ("javascript", "js"),
    ".mjs": ("javascript", "js"),
    ".cjs": ("javascript", "js"),
    ".ts": ("typescript", "js"),
    ".tsx": ("typescript", "js"),
    ".mts": ("typescript", "js"),
    ".cts": ("typescript", "js"),
    ".go": ("go", "go"),
    ".java": ("java", "jvm"),
    ".kt": ("kotlin", "jvm"),
    ".kts": ("kotlin", "jvm"),
    ".cs": ("csharp", "csharp"),
    ".rb": ("ruby", "ruby"),
    ".php": ("php", "php"),
    ".rs": ("rust", "rust"),
    ".proto": ("protobuf", "proto"),
    ".graphql": ("graphql", "graphql"),
    ".graphqls": ("graphql", "graphql"),
    ".gql": ("graphql", "graphql"),
}

EXTRA_IGNORES: tuple[str, ...] = ("vendor", "third_party", "bower_components", "target")

_EXTRA_TEST_PATH = re.compile(
    r"_test\.go$|(^|/)src/test/|Tests?\.(java|kt|cs)$|_spec\.rb$|(^|/)spec/|Test\.php$"
)


def is_test_file(rel_path: str) -> bool:
    """The linker's test-path rule extended to Go, JVM, .NET, Ruby and PHP layouts."""
    return is_test_path(rel_path) or bool(_EXTRA_TEST_PATH.search(rel_path))


def language_of(rel_path: str) -> tuple[str, Family] | None:
    suffix = PurePosixPath(rel_path).suffix.lower()
    return LANGUAGE_BY_SUFFIX.get(suffix)


@dataclass(slots=True)
class SourceFile:
    """One file's text plus lazily built line and comment lookups."""

    path: str
    language: str
    family: Family
    text: str
    _starts: list[int] | None = field(default=None, repr=False)
    _comments: CommentMask | None = field(default=None, repr=False)

    def line(self, offset: int) -> int:
        if self._starts is None:
            self._starts = line_starts(self.text)
        return line_at(self._starts, offset)

    def in_comment(self, offset: int) -> bool:
        if self._comments is None:
            self._comments = CommentMask(self.text, self.family)
        return offset in self._comments

    @property
    def is_test(self) -> bool:
        return is_test_file(self.path)


def _ignored(name: str, rel: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(rel, p) for p in patterns)


def iter_source_files(root: Path) -> Iterator[SourceFile]:
    """Every readable file under ``root`` with a suffix in :data:`LANGUAGE_BY_SUFFIX`."""
    ignores = (*DEFAULT_IGNORES, *EXTRA_IGNORES, *load_ignore_file(root))
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        prefix = "" if rel_dir == "." else f"{rel_dir}/"
        dirnames[:] = sorted(d for d in dirnames if not _ignored(d, prefix + d, ignores))
        for filename in sorted(filenames):
            rel = prefix + filename
            labels = language_of(filename)
            if labels is None or _ignored(filename, rel, ignores):
                continue
            path = Path(dirpath) / filename
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            language, family = labels
            yield SourceFile(
                path=rel,
                language=language,
                family=family,
                text=data.decode("utf-8", errors="replace"),
            )


def load_sources(root: Path) -> list[SourceFile]:
    return list(iter_source_files(root))


__all__ = ["SourceFile", "is_test_file", "iter_source_files", "language_of", "load_sources"]
