"""File name -> language identifier.

Suffix lookup covers almost everything; :data:`LANGUAGE_BY_FILENAME` catches files whose
name *is* the signal (``Dockerfile``, ``Makefile``), and :data:`LANGUAGE_BY_PATTERN`
catches the families of names that only differ by an environment tag
(``Dockerfile.prod``, ``docker-compose.override.yml``). Matching is case-insensitive on
the suffix and exact (or regex) on the file name.

Plain ``.yaml`` files map to ``yaml``; the YAML parser decides from the content whether
a file is a Kubernetes manifest, a compose file or noise (CI configs), and returns an
empty result for noise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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
    ".mts": "typescript",
    ".cts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sc": "scala",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".c++": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".h++": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".rake": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".dart": "dart",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ex": "elixir",
    ".exs": "elixir",
    ".sol": "solidity",
    ".m": "objc",
    ".mm": "objc",
    ".zig": "zig",
    ".jl": "julia",
    ".r": "r",
    ".pl": "perl",
    ".pm": "perl",
    ".hs": "haskell",
    ".groovy": "groovy",
    ".gradle": "groovy",
    ".erl": "erlang",
    ".hrl": "erlang",
    ".vue": "vue",
    ".svelte": "svelte",
    ".ipynb": "notebook",
    ".dockerfile": "dockerfile",
    ".tf": "terraform",
    ".hcl": "hcl",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".mk": "make",
}

LANGUAGE_BY_FILENAME: dict[str, Language] = {
    "Dockerfile": "dockerfile",
    "Containerfile": "dockerfile",
    "docker-compose.yml": "compose",
    "docker-compose.yaml": "compose",
    "compose.yml": "compose",
    "compose.yaml": "compose",
    "kustomization.yaml": "kustomize",
    "kustomization.yml": "kustomize",
    "Kustomization": "kustomize",
    "Makefile": "make",
    "makefile": "make",
    "GNUmakefile": "make",
    "Rakefile": "ruby",
    "Gemfile": "ruby",
    "Jenkinsfile": "groovy",
}

LANGUAGE_BY_PATTERN: tuple[tuple[re.Pattern[str], Language], ...] = (
    (re.compile(r"^(Dockerfile|Containerfile)\.[\w.-]+$"), "dockerfile"),
    (re.compile(r"^(docker-)?compose[.-][\w.-]+\.ya?ml$"), "compose"),
)


@dataclass(frozen=True, slots=True)
class LanguageInfo:
    """One supported language: what it is called and which files belong to it."""

    language: Language
    family: str
    suffixes: tuple[str, ...]
    filenames: tuple[str, ...]


_FAMILY: dict[Language, str] = {
    "python": "code",
    "javascript": "code",
    "typescript": "code",
    "vue": "component",
    "svelte": "component",
    "notebook": "notebook",
    "dockerfile": "infrastructure",
    "compose": "infrastructure",
    "kustomize": "infrastructure",
    "yaml": "infrastructure",
    "terraform": "infrastructure",
    "hcl": "infrastructure",
    "make": "build",
}


def supported_languages() -> tuple[LanguageInfo, ...]:
    """Every language codegraph indexes, with the suffixes and file names that select it."""
    suffixes: dict[Language, list[str]] = {}
    filenames: dict[Language, list[str]] = {}
    for suffix, language in LANGUAGE_BY_SUFFIX.items():
        suffixes.setdefault(language, []).append(suffix)
    for name, language in LANGUAGE_BY_FILENAME.items():
        filenames.setdefault(language, []).append(name)
    for pattern, language in LANGUAGE_BY_PATTERN:
        filenames.setdefault(language, []).append(pattern.pattern)
    return tuple(
        LanguageInfo(
            language=language,
            family=_FAMILY.get(language, "code"),
            suffixes=tuple(suffixes.get(language, ())),
            filenames=tuple(filenames.get(language, ())),
        )
        for language in sorted(set(suffixes) | set(filenames))
    )


def language_of(rel_path: str) -> Language | None:
    """Language identifier for ``rel_path``, or None when codegraph does not index it."""
    name = PurePosixPath(rel_path).name
    by_name = LANGUAGE_BY_FILENAME.get(name)
    if by_name is not None:
        return by_name
    for pattern, language in LANGUAGE_BY_PATTERN:
        if pattern.match(name):
            return language
    dot = name.rfind(".")
    if dot == -1:
        return None
    return LANGUAGE_BY_SUFFIX.get(name[dot:].lower())


__all__ = [
    "LANGUAGE_BY_FILENAME",
    "LANGUAGE_BY_PATTERN",
    "LANGUAGE_BY_SUFFIX",
    "LanguageInfo",
    "language_of",
    "supported_languages",
]
