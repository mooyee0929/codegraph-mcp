"""Path arithmetic for import resolution that needs nothing but the importing file's path.

``#include "../util.h"``, ``require_relative "lib/x"`` and ``source ./common.sh`` name a
file relative to the importer; Java, Kotlin, Scala and Haskell files declare a package
whose segments mirror the directories below a source root. Both can be turned into the
importee's module qname without looking at the rest of the repository.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from codegraph.parsers.base import module_qname_for


def normalize(path: str) -> str | None:
    """Collapse ``.`` and ``..``; None when the path climbs out of the repository."""
    parts: list[str] = []
    for part in PurePosixPath(path).parts:
        if part in {".", ""}:
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def relative_path(rel_path: str, spec: str) -> str | None:
    """The repo-relative path ``spec`` names when written inside ``rel_path``."""
    if spec.startswith("/"):
        return None
    here = PurePosixPath(rel_path).parent.as_posix()
    return normalize(f"{here}/{spec}" if here != "." else spec)


def relative_module(rel_path: str, spec: str, default_suffix: str = "") -> str | None:
    """Module qname for a relative file reference, adding ``default_suffix`` if it has none."""
    target = relative_path(rel_path, spec)
    if not target:
        return None
    if default_suffix and not PurePosixPath(target).suffix:
        target += default_suffix
    return module_qname_for(target)


def package_root(rel_path: str, package: str, separator: str = ".") -> str | None:
    """Dotted directory prefix under which ``package`` lives, inferred from the file path.

    ``src/main/java/com/acme/app/Main.java`` declaring ``com.acme.app`` -> ``src.main.java``.
    None when the directories do not end with the package segments (the file is not laid
    out by package, which is legal in Kotlin and Scala).
    """
    segments = [s for s in package.split(separator) if s]
    dirs = list(PurePosixPath(rel_path).parent.parts)
    if not segments:
        return ".".join(dirs)
    if len(dirs) < len(segments) or dirs[len(dirs) - len(segments) :] != segments:
        return None
    return ".".join(dirs[: len(dirs) - len(segments)])


def under(prefix: str | None, dotted: str) -> str:
    return f"{prefix}.{dotted}" if prefix else dotted


__all__ = ["normalize", "package_root", "relative_module", "relative_path", "under"]
