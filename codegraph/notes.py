"""Saved answers: a memory loop so the next session does not re-derive the same answer.

``save_note`` writes one markdown file per answer under ``.codegraph/notes/`` with a
small front matter (kind, creation time, related qnames), so a note reads well in an
editor and can be committed if a team wants to share them. ``search_notes`` ranks
notes by query-term overlap, weighting the question over the answer, with a boost
when the query names one of a note's qnames. It needs no index: a repository collects
dozens of notes, not millions, and reading them all is instant.

Parsing is lenient on purpose — a hand-edited note without front matter is still
found, using its first heading as the question.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from codegraph.store import DB_DIRNAME

NOTES_DIRNAME = "notes"

_TOKEN = re.compile(r"[a-z0-9_]+")


class NoteError(ValueError):
    """A note without a question or an answer."""


@dataclass(frozen=True, slots=True)
class Note:
    path: Path
    question: str
    answer: str
    qnames: tuple[str, ...]
    kind: str
    created: str


@dataclass(frozen=True, slots=True)
class NoteHit:
    note: Note
    score: float


def notes_dir(root: Path) -> Path:
    return root / DB_DIRNAME / NOTES_DIRNAME


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50].rstrip("-") or "note"


def render_note(note: Note) -> str:
    lines = ["---", f"kind: {note.kind}", f"created: {note.created}"]
    if note.qnames:
        lines.append("qnames:")
        lines.extend(f"  - {q}" for q in note.qnames)
    lines += ["---", "", f"# {note.question}", "", note.answer.strip(), ""]
    return "\n".join(lines)


def save_note(
    root: Path,
    question: str,
    answer: str,
    qnames: tuple[str, ...] = (),
    *,
    kind: str = "answer",
    now: float | None = None,
) -> Note:
    clean_question = " ".join(question.split())
    if not clean_question:
        raise NoteError("a note needs a question")
    if not answer.strip():
        raise NoteError("a note needs an answer")
    clean_kind = re.sub(r"[^a-z0-9_-]+", "-", kind.lower()).strip("-") or "answer"
    moment = datetime.fromtimestamp(time.time() if now is None else now, tz=UTC)
    directory = notes_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{moment:%Y%m%d-%H%M%S}-{_slug(clean_question)}"
    path = directory / f"{stem}.md"
    suffix = 2
    while path.exists():
        path = directory / f"{stem}-{suffix}.md"
        suffix += 1
    note = Note(
        path=path,
        question=clean_question,
        answer=answer,
        qnames=tuple(dict.fromkeys(q.strip() for q in qnames if q.strip())),
        kind=clean_kind,
        created=moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    path.write_text(render_note(note), encoding="utf-8")
    return note


def parse_note(path: Path, text: str) -> Note:
    meta: dict[str, str] = {}
    qnames: list[str] = []
    body = text
    if text.startswith("---\n"):
        header, sep, rest = text[4:].partition("\n---")
        if sep:
            body = rest.split("\n", 1)[1] if "\n" in rest else ""
            in_list = False
            for line in header.splitlines():
                if in_list and line.startswith("  - "):
                    qnames.append(line[4:].strip())
                    continue
                key, _, value = line.partition(":")
                in_list = key.strip() == "qnames" and not value.strip()
                if not in_list:
                    meta[key.strip()] = value.strip()
    question = path.stem
    answer_lines: list[str] = []
    found_heading = False
    for line in body.splitlines():
        if not found_heading and line.startswith("# "):
            question, found_heading = line[2:].strip(), True
            continue
        answer_lines.append(line)
    return Note(
        path=path,
        question=question,
        answer="\n".join(answer_lines).strip(),
        qnames=tuple(qnames),
        kind=meta.get("kind", "answer"),
        created=meta.get("created", ""),
    )


def list_notes(root: Path) -> list[Note]:
    directory = notes_dir(root)
    if not directory.is_dir():
        return []
    notes = [
        parse_note(path, path.read_text(encoding="utf-8"))
        for path in directory.glob("*.md")
        if path.is_file()
    ]
    return sorted(notes, key=lambda n: (n.created, n.path.name), reverse=True)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def score_note(note: Note, query: str) -> float:
    terms = set(_tokens(query))
    if not terms:
        return 0.0
    question = Counter(_tokens(note.question))
    answer = Counter(_tokens(note.answer))
    qname_terms = Counter(t for q in note.qnames for t in _tokens(q))
    score = 0.0
    for term in terms:
        score += 2.0 * min(question[term], 2) + 1.0 * min(answer[term], 3)
        score += 1.5 * min(qname_terms[term], 2)
    lowered = query.lower()
    score += sum(3.0 for q in note.qnames if q.lower() in lowered)
    return score


def search_notes(root: Path, query: str, limit: int = 10) -> list[NoteHit]:
    hits = [NoteHit(note, score_note(note, query)) for note in list_notes(root)]
    ranked = sorted((h for h in hits if h.score > 0), key=lambda h: -h.score)
    return ranked[:limit]


__all__ = [
    "Note",
    "NoteError",
    "NoteHit",
    "list_notes",
    "notes_dir",
    "parse_note",
    "render_note",
    "save_note",
    "score_note",
    "search_notes",
]
