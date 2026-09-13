"""Parser protocol and the path <-> qname convention shared by every language."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

from codegraph.models import Language, ParseResult


@runtime_checkable
class Parser(Protocol):
    """Extracts nodes and edges from a single file's source."""

    language: Language

    def parse(self, rel_path: str, source: str, module_qname: str) -> ParseResult:
        """Return everything found in ``source``.

        ``rel_path`` is repository-relative and POSIX-separated; ``module_qname`` is the
        dotted name the file itself gets, as produced by :func:`module_qname_for`.
        Implementations must not raise on malformed input — return what was recovered.
        """
        ...


_INDEX_STEMS = frozenset({"__init__", "index", "mod"})


def module_qname_for(rel_path: str) -> str:
    """Map a repository-relative path to the dotted qname of its module.

    ``pkg/billing.py`` -> ``pkg.billing``; ``pkg/__init__.py`` -> ``pkg``;
    ``web/routes/index.ts`` -> ``web.routes``. Path segments that are not valid
    identifiers (``my-lib``) keep their spelling, so qnames stay reversible.
    """
    parts = PurePosixPath(rel_path)
    stem = parts.stem
    if stem.endswith(".d"):  # foo.d.ts
        stem = stem[:-2]
    segments = list(parts.parts[:-1])
    if stem not in _INDEX_STEMS:
        segments.append(stem)
    return ".".join(segments) if segments else stem


def child_qname(parent: str, name: str) -> str:
    return f"{parent}.{name}" if parent else name


def clip(text: str, limit: int = 400) -> str:
    """Trim a signature or docstring down to something worth storing."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


__all__ = ["Parser", "ParseResult", "child_qname", "clip", "module_qname_for"]
