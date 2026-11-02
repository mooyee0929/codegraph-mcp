"""File name -> language identifier.

Suffix lookup covers almost everything; :data:`LANGUAGE_BY_FILENAME` catches files whose
name *is* the signal (``Dockerfile``, ``Makefile``). Matching is case-insensitive on the
suffix and exact on the file name.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from codegraph.models import Language

LANGUAGE_BY_SUFFIX: dict[str, Language] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}

LANGUAGE_BY_FILENAME: dict[str, Language] = {}


def language_of(rel_path: str) -> Language | None:
    """Language identifier for ``rel_path``, or None when codegraph does not index it."""
    name = PurePosixPath(rel_path).name
    by_name = LANGUAGE_BY_FILENAME.get(name)
    if by_name is not None:
        return by_name
    dot = name.rfind(".")
    if dot == -1:
        return None
    return LANGUAGE_BY_SUFFIX.get(name[dot:].lower())


__all__ = ["LANGUAGE_BY_FILENAME", "LANGUAGE_BY_SUFFIX", "language_of"]
