"""Architecture decision records as version-controlled markdown.

ADRs live next to the code they describe, one file per decision at
``docs/adr/NNNN-slug.md`` in Michael Nygard's format (the one ``adr-tools`` writes):
a numbered title, a date, then Status, Context, Decision and Consequences sections.
Keeping them as plain files rather than database rows means they are reviewed in pull
requests, survive a re-index, and stay readable without codegraph.

The directory is, in order of precedence: an explicit argument, the path written in
``<repo>/.adr-dir`` (the ``adr-tools`` convention), or ``docs/adr``. It must resolve
inside the repository.

Superseding never edits history away: the new record links back with ``Supersedes``
and the old one keeps its text, changes status to ``Superseded`` and gains a
``Superseded by`` link.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

DEFAULT_ADR_DIR = "docs/adr"
ADR_DIR_FILE = ".adr-dir"
STATUSES: tuple[str, ...] = ("Proposed", "Accepted", "Rejected", "Deprecated", "Superseded")

_FILENAME = re.compile(r"^(\d{4})-[^/]+\.md$")
_TITLE = re.compile(r"^#\s+(\d+)\.\s+(.+?)\s*$")
_DATE = re.compile(r"^Date:\s*(.+?)\s*$")
_SECTION = re.compile(r"^##\s+(.+?)\s*$")


class AdrError(ValueError):
    """Bad input, or a file that is not an ADR."""


@dataclass(frozen=True, slots=True)
class Adr:
    number: int
    title: str
    date: str
    status: str
    links: tuple[str, ...]
    context: str
    decision: str
    consequences: str
    path: Path

    @property
    def filename(self) -> str:
        return self.path.name


def resolve_adr_dir(root: Path, directory: str | None = None) -> Path:
    root = root.resolve()
    raw = directory
    marker = root / ADR_DIR_FILE
    if not raw and marker.is_file():
        raw = marker.read_text(encoding="utf-8").strip()
    path = (root / (raw or DEFAULT_ADR_DIR)).resolve()
    if not path.is_relative_to(root):
        raise AdrError(f"ADR directory {raw!r} is outside the repository")
    return path


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:60].rstrip("-") or "decision"


def normalise_status(status: str) -> str:
    for known in STATUSES:
        if status.strip().lower() == known.lower():
            return known
    raise AdrError(f"unknown status {status!r}; expected one of {', '.join(STATUSES)}")


def render_adr(adr: Adr) -> str:
    status_block = "\n\n".join([adr.status, *adr.links])
    return (
        f"# {adr.number}. {adr.title}\n\n"
        f"Date: {adr.date}\n\n"
        f"## Status\n\n{status_block}\n\n"
        f"## Context\n\n{adr.context.strip()}\n\n"
        f"## Decision\n\n{adr.decision.strip()}\n\n"
        f"## Consequences\n\n{adr.consequences.strip()}\n"
    )


def parse_adr(text: str, path: Path) -> Adr:
    number: int | None = None
    title = record_date = ""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if number is None:
            match = _TITLE.match(line)
            if match:
                number, title = int(match.group(1)), match.group(2)
            continue
        section = _SECTION.match(line)
        if section:
            current = section.group(1).strip().lower()
            sections[current] = []
            continue
        if current is None:
            date_match = _DATE.match(line)
            if date_match:
                record_date = date_match.group(1)
            continue
        sections[current].append(line)
    if number is None:
        raise AdrError(f"{path} has no '# N. Title' heading")
    status_lines = [ln.strip() for ln in sections.get("status", []) if ln.strip()]
    return Adr(
        number=number,
        title=title,
        date=record_date,
        status=status_lines[0] if status_lines else "",
        links=tuple(status_lines[1:]),
        context=_body(sections, "context"),
        decision=_body(sections, "decision"),
        consequences=_body(sections, "consequences"),
        path=path,
    )


def _body(sections: dict[str, list[str]], name: str) -> str:
    return "\n".join(sections.get(name, [])).strip()


def list_adrs(directory: Path) -> list[Adr]:
    if not directory.is_dir():
        return []
    found = [
        parse_adr(path.read_text(encoding="utf-8"), path)
        for path in sorted(directory.iterdir())
        if _FILENAME.match(path.name)
    ]
    return sorted(found, key=lambda a: a.number)


def get_adr(directory: Path, number: int) -> Adr | None:
    return next((a for a in list_adrs(directory) if a.number == number), None)


def _require(directory: Path, number: int) -> Adr:
    adr = get_adr(directory, number)
    if adr is None:
        raise AdrError(f"no ADR number {number} in {directory}")
    return adr


def create_adr(
    directory: Path,
    title: str,
    *,
    context: str = "",
    decision: str = "",
    consequences: str = "",
    status: str = "Proposed",
    links: tuple[str, ...] = (),
    today: date | None = None,
) -> Adr:
    clean_title = " ".join(title.split())
    if not clean_title:
        raise AdrError("an ADR needs a title")
    directory.mkdir(parents=True, exist_ok=True)
    existing = list_adrs(directory)
    number = (existing[-1].number if existing else 0) + 1
    path = directory / f"{number:04d}-{slugify(clean_title)}.md"
    adr = Adr(
        number=number,
        title=clean_title,
        date=(today or date.today()).isoformat(),
        status=normalise_status(status),
        links=links,
        context=context,
        decision=decision,
        consequences=consequences,
        path=path,
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(render_adr(adr))
    return adr


def update_adr(
    directory: Path,
    number: int,
    *,
    title: str | None = None,
    status: str | None = None,
    context: str | None = None,
    decision: str | None = None,
    consequences: str | None = None,
) -> Adr:
    """Rewrite the given fields in place; the file name stays as first created."""
    adr = _require(directory, number)
    updated = replace(
        adr,
        title=" ".join(title.split()) if title else adr.title,
        status=normalise_status(status) if status else adr.status,
        context=adr.context if context is None else context,
        decision=adr.decision if decision is None else decision,
        consequences=adr.consequences if consequences is None else consequences,
    )
    updated.path.write_text(render_adr(updated), encoding="utf-8")
    return updated


def supersede_adr(
    directory: Path,
    number: int,
    title: str,
    *,
    context: str = "",
    decision: str = "",
    consequences: str = "",
    today: date | None = None,
) -> tuple[Adr, Adr]:
    """Record a new, accepted decision replacing ``number``; return ``(new, old)``."""
    old = _require(directory, number)
    if old.status.startswith("Superseded"):
        raise AdrError(f"ADR {number} is already superseded")
    new = create_adr(
        directory,
        title,
        context=context,
        decision=decision,
        consequences=consequences,
        status="Accepted",
        links=(f"Supersedes [{old.number}. {old.title}]({old.filename})",),
        today=today,
    )
    retired = replace(
        old,
        status="Superseded",
        links=(*old.links, f"Superseded by [{new.number}. {new.title}]({new.filename})"),
    )
    retired.path.write_text(render_adr(retired), encoding="utf-8")
    return new, retired


ACTIONS: tuple[str, ...] = ("list", "get", "create", "update", "supersede")


def adr_summary(record: Adr, root: Path, *, full: bool = True) -> dict[str, object]:
    summary: dict[str, object] = {
        "number": record.number,
        "title": record.title,
        "date": record.date,
        "status": record.status,
        "links": list(record.links),
        "path": record.path.relative_to(root.resolve()).as_posix(),
    }
    if full:
        summary |= {
            "context": record.context,
            "decision": record.decision,
            "consequences": record.consequences,
        }
    return summary


def manage_adr(
    root: Path,
    action: str,
    number: int = 0,
    *,
    title: str = "",
    status: str = "",
    context: str = "",
    decision: str = "",
    consequences: str = "",
    directory: str = "",
) -> dict[str, object]:
    """One entry point for the MCP tool and the CLI; empty strings mean "not given"."""
    if action not in ACTIONS:
        raise AdrError(f"unknown action {action!r}; expected one of {', '.join(ACTIONS)}")
    folder = resolve_adr_dir(root, directory or None)
    if action == "list":
        return {
            "directory": folder.relative_to(root.resolve()).as_posix(),
            "adrs": [adr_summary(a, root, full=False) for a in list_adrs(folder)],
        }
    if action == "create":
        created = create_adr(
            folder,
            title,
            context=context,
            decision=decision,
            consequences=consequences,
            status=status or "Proposed",
        )
        return {"created": adr_summary(created, root)}
    if number < 1:
        raise AdrError(f"action {action!r} needs the ADR number")
    if action == "get":
        return adr_summary(_require(folder, number), root)
    if action == "update":
        updated = update_adr(
            folder,
            number,
            title=title or None,
            status=status or None,
            context=context or None,
            decision=decision or None,
            consequences=consequences or None,
        )
        return {"updated": adr_summary(updated, root)}
    new, old = supersede_adr(
        folder, number, title, context=context, decision=decision, consequences=consequences
    )
    return {"created": adr_summary(new, root), "superseded": adr_summary(old, root, full=False)}


__all__ = [
    "ACTIONS",
    "DEFAULT_ADR_DIR",
    "STATUSES",
    "Adr",
    "AdrError",
    "adr_summary",
    "create_adr",
    "get_adr",
    "list_adrs",
    "manage_adr",
    "normalise_status",
    "parse_adr",
    "render_adr",
    "resolve_adr_dir",
    "slugify",
    "supersede_adr",
    "update_adr",
]
