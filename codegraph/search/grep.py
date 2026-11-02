"""Grep restricted to indexed files, with every hit placed in its enclosing symbol.

Plain grep over a repository wades through vendored code, build output and lockfiles,
and reports ``line 812`` where an agent wants ``inside Invoice.total``. Searching only
the files the indexer accepted (ignores and ``.codegraphignore`` already applied) and
annotating each hit with :func:`codegraph.queries.node_at` turns a text match into a
graph entry point: the qname can go straight to ``get_impact_radius`` or Cypher.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path

from codegraph.queries import node_at
from codegraph.store import GraphStore

MAX_CONTEXT = 10
MAX_LINE_CHARS = 300


@dataclass(frozen=True, slots=True)
class GrepMatch:
    path: str
    line: int
    column: int
    text: str
    symbol: str | None
    symbol_kind: str | None
    before: tuple[str, ...] = ()
    after: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "path": self.path,
            "line": self.line,
            "column": self.column,
            "text": self.text,
            "symbol": self.symbol,
            "symbol_kind": self.symbol_kind,
        }
        if self.before or self.after:
            payload["before"] = list(self.before)
            payload["after"] = list(self.after)
        return payload


@dataclass(frozen=True, slots=True)
class GrepResult:
    pattern: str
    matches: tuple[GrepMatch, ...]
    files_searched: int
    files_matched: int
    truncated: bool
    """True when a per-file or total cap stopped the search early."""
    unreadable: tuple[str, ...] = ()
    """Indexed files that could not be read (deleted since indexing, permissions)."""

    def as_dict(self) -> dict[str, object]:
        return {
            "pattern": self.pattern,
            "files_searched": self.files_searched,
            "files_matched": self.files_matched,
            "match_count": len(self.matches),
            "truncated": self.truncated,
            "unreadable": list(self.unreadable),
            "matches": [m.as_dict() for m in self.matches],
        }


def compile_pattern(pattern: str, *, regex: bool, case_sensitive: bool) -> re.Pattern[str]:
    if not pattern:
        raise ValueError("pattern must not be empty")
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        return re.compile(pattern if regex else re.escape(pattern), flags)
    except re.error as exc:
        raise ValueError(f"invalid regular expression {pattern!r}: {exc}") from exc


def grep_code(
    store: GraphStore,
    root: Path,
    pattern: str,
    *,
    regex: bool = False,
    case_sensitive: bool = True,
    file_glob: str | None = None,
    context: int = 0,
    max_per_file: int = 20,
    max_total: int = 200,
) -> GrepResult:
    """Search indexed files for ``pattern`` (literal unless ``regex``)."""
    if context < 0 or max_per_file < 1 or max_total < 1:
        raise ValueError("context must be >= 0 and the match caps >= 1")
    compiled = compile_pattern(pattern, regex=regex, case_sensitive=case_sensitive)
    context = min(context, MAX_CONTEXT)
    base = root.resolve()
    matches: list[GrepMatch] = []
    unreadable: list[str] = []
    searched = matched_files = 0
    truncated = False
    for rel in store.indexed_paths():
        if file_glob is not None and not fnmatch.fnmatchcase(rel, file_glob):
            continue
        path = (base / rel).resolve()
        if not path.is_relative_to(base):
            unreadable.append(rel)
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            unreadable.append(rel)
            continue
        searched += 1
        in_file = 0
        for number, line in enumerate(lines, start=1):
            found = compiled.search(line)
            if found is None:
                continue
            if in_file >= max_per_file or len(matches) >= max_total:
                truncated = True
                break
            in_file += 1
            symbol = node_at(store, rel, number)
            matches.append(
                GrepMatch(
                    path=rel,
                    line=number,
                    column=found.start() + 1,
                    text=_clip(line),
                    symbol=symbol.qname if symbol else None,
                    symbol_kind=symbol.kind if symbol else None,
                    before=tuple(
                        _clip(t) for t in lines[max(0, number - 1 - context) : number - 1]
                    ),
                    after=tuple(_clip(t) for t in lines[number : number + context]),
                )
            )
        if in_file:
            matched_files += 1
        if len(matches) >= max_total and truncated:
            break
    return GrepResult(
        pattern=pattern,
        matches=tuple(matches),
        files_searched=searched,
        files_matched=matched_files,
        truncated=truncated,
        unreadable=tuple(unreadable),
    )


def _clip(line: str) -> str:
    return line if len(line) <= MAX_LINE_CHARS else line[:MAX_LINE_CHARS] + "..."


__all__ = ["GrepMatch", "GrepResult", "compile_pattern", "grep_code"]
