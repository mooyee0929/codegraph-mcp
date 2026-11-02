"""Graph-guided refactoring: preview a rename as a diff, apply it only if nothing moved.

A rename touches the definition plus every place the graph says references it —
``calls``, ``imports``, ``inherits`` and ``tests`` edges into the symbol. The graph
gives the candidate lines; a word-boundary match of the old name on each line confirms
them, so a stale or heuristic edge never produces an edit on a line that does not
mention the name. Edges the linker could not resolve are ambiguous by definition and
are listed as skipped instead of guessed.

The preview is stored in ``ops_refactor_previews`` with the SHA-1 of every file it
would change and an expiry, so ``apply_refactor`` can run from another process (the
CLI after the MCP server, say) and refuses outright when any file changed since the
preview. Text is matched per line, so a rename also rewrites the name inside strings
or comments on a referencing line; the diff shows exactly what will change.

``suggest_refactorings`` turns graph shape into candidates: functions whose callers
mostly live in one other module (move), very long callables (split), ``similar_to``
pairs when a clone pass produced them (dedupe), and :func:`codegraph.queries.dead_code`
(remove). They are prompts for a human, not verdicts.
"""

from __future__ import annotations

import difflib
import hashlib
import io
import json
import keyword
import os
import re
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from codegraph.graph import GraphView
from codegraph.linker import is_test_path
from codegraph.models import CALLABLE_KINDS, EdgeKind
from codegraph.queries import dead_code, resolve_target
from codegraph.store import GraphStore

PREVIEW_TTL_SECONDS = 600.0
RENAMEABLE_KINDS = frozenset({"function", "method", "class"})
REFERENCE_KINDS: tuple[EdgeKind, ...] = ("calls", "imports", "inherits", "tests")

_IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")

_DDL = """
CREATE TABLE IF NOT EXISTS ops_refactor_previews (
    id         TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    payload    TEXT NOT NULL
);
"""

SuggestionKind = Literal["dedupe", "move", "split", "remove"]


class RefactorError(ValueError):
    """Bad input, an expired or unknown preview, or files that changed underneath it."""


@dataclass(frozen=True, slots=True)
class LineEdit:
    line: int
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class FileEdit:
    path: str
    sha: str
    lines: tuple[LineEdit, ...]


@dataclass(frozen=True, slots=True)
class RenamePreview:
    preview_id: str
    target: str
    old_name: str
    new_name: str
    files: tuple[FileEdit, ...]
    skipped: tuple[str, ...]
    diff: str
    created_at: float
    expires_at: float

    @property
    def edit_count(self) -> int:
        return sum(len(f.lines) for f in self.files)


@dataclass(frozen=True, slots=True)
class AppliedRefactor:
    preview_id: str
    files: tuple[str, ...]
    lines_changed: int


@dataclass(frozen=True, slots=True)
class Suggestion:
    kind: SuggestionKind
    qname: str
    path: str
    line: int
    score: float
    reason: str
    related: tuple[str, ...] = ()


# -- rename preview ----------------------------------------------------------


def _sha(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _lines(text: str) -> list[str]:
    """Split like the parsers count lines (\\n, \\r\\n, \\r), keeping the endings."""
    return io.StringIO(text, newline="").readlines()


def _safe_path(root: Path, rel: str) -> Path:
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RefactorError(f"{rel} is outside the repository")
    return path


def _read_text(path: Path, rel: str) -> tuple[bytes, str]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise RefactorError(f"cannot read {rel}: {exc}") from exc
    try:
        return data, data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RefactorError(f"{rel} is not UTF-8; refusing to rewrite it") from exc


def validate_new_name(new_name: str, language: str) -> None:
    if not _IDENTIFIER.match(new_name):
        raise RefactorError(f"{new_name!r} is not a valid identifier")
    if language == "python" and ("$" in new_name or keyword.iskeyword(new_name)):
        raise RefactorError(f"{new_name!r} is not a valid Python identifier")


def preview_rename(
    root: Path,
    store: GraphStore,
    target: str,
    new_name: str,
    *,
    clock: Callable[[], float] = time.time,
    ttl: float = PREVIEW_TTL_SECONDS,
) -> RenamePreview:
    node = resolve_target(store, target)
    if node is None:
        raise RefactorError(f"no node matches {target!r}")
    if node.kind not in RENAMEABLE_KINDS:
        raise RefactorError(f"cannot rename a {node.kind}; only {sorted(RENAMEABLE_KINDS)}")
    validate_new_name(new_name, node.language)
    if new_name == node.name:
        raise RefactorError(f"{node.qname} is already called {new_name}")

    word = re.compile(rf"(?<![\w$]){re.escape(node.name)}(?![\w$])")
    sites: dict[str, set[int]] = {node.path: set()}
    skipped: list[str] = []
    for edge in store.edges_to(node.qname):
        if edge.kind in REFERENCE_KINDS:
            sites.setdefault(edge.path, set()).add(edge.line)
    for edge in store.unresolved_edges():
        if edge.kind == "calls" and edge.dst in {node.name, f"*.{node.name}"}:
            skipped.append(f"{edge.path}:{edge.line} (unresolved call to {edge.dst})")

    files: list[FileEdit] = []
    diffs: list[str] = []
    for rel in sorted(sites):
        data, text = _read_text(_safe_path(root, rel), rel)
        lines = _lines(text)
        wanted = set(sites[rel])
        if rel == node.path:
            wanted.add(_definition_line(node.line_start, node.line_end, lines, word))
        edits: list[LineEdit] = []
        for number in sorted(wanted):
            if not 1 <= number <= len(lines):
                skipped.append(f"{rel}:{number} (line no longer exists)")
                continue
            before = lines[number - 1]
            after = word.sub(new_name, before)
            if after == before:
                skipped.append(f"{rel}:{number} (name not on that line)")
                continue
            edits.append(LineEdit(number, before, after))
        if not edits:
            continue
        files.append(FileEdit(path=rel, sha=_sha(data), lines=tuple(edits)))
        diffs.append(_unified(rel, lines, edits))

    if not files:
        raise RefactorError(f"found no text to rename for {node.qname}")
    created = clock()
    preview = RenamePreview(
        preview_id=uuid.uuid4().hex[:12],
        target=node.qname,
        old_name=node.name,
        new_name=new_name,
        files=tuple(files),
        skipped=tuple(skipped),
        diff="".join(diffs),
        created_at=created,
        expires_at=created + ttl,
    )
    _save_preview(store, preview)
    return preview


def _definition_line(
    line_start: int, line_end: int, lines: list[str], word: re.Pattern[str]
) -> int:
    """The first line of the span that names the symbol.

    Decorators sit inside the span, so that is not necessarily ``line_start``.
    """
    for number in range(line_start, min(line_end, len(lines)) + 1):
        if word.search(lines[number - 1]):
            return number
    return line_start


def _unified(rel: str, lines: list[str], edits: list[LineEdit]) -> str:
    updated = list(lines)
    for edit in edits:
        updated[edit.line - 1] = edit.after
    diff = difflib.unified_diff(lines, updated, fromfile=f"a/{rel}", tofile=f"b/{rel}")
    return "".join(line if line.endswith("\n") else f"{line}\n" for line in diff)


# -- preview storage ------------------------------------------------------------


def _save_preview(store: GraphStore, preview: RenamePreview) -> None:
    store.ensure_schema(_DDL)
    with store.transaction() as conn:
        conn.execute(
            "DELETE FROM ops_refactor_previews WHERE expires_at < ?", (preview.created_at,)
        )
        conn.execute(
            "INSERT INTO ops_refactor_previews (id, created_at, expires_at, payload) "
            "VALUES (?, ?, ?, ?)",
            (
                preview.preview_id,
                preview.created_at,
                preview.expires_at,
                json.dumps(asdict(preview)),
            ),
        )


def load_preview(store: GraphStore, preview_id: str) -> RenamePreview | None:
    store.ensure_schema(_DDL)
    row = store.connection.execute(
        "SELECT payload FROM ops_refactor_previews WHERE id = ?", (preview_id,)
    ).fetchone()
    if row is None:
        return None
    return _preview_from_json(str(row["payload"]))


def _preview_from_json(payload: str) -> RenamePreview:
    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise RefactorError("corrupt refactor preview")
    files = tuple(
        FileEdit(
            path=str(f["path"]),
            sha=str(f["sha"]),
            lines=tuple(
                LineEdit(int(e["line"]), str(e["before"]), str(e["after"])) for e in f["lines"]
            ),
        )
        for f in raw["files"]
    )
    return RenamePreview(
        preview_id=str(raw["preview_id"]),
        target=str(raw["target"]),
        old_name=str(raw["old_name"]),
        new_name=str(raw["new_name"]),
        files=files,
        skipped=tuple(str(s) for s in raw["skipped"]),
        diff=str(raw["diff"]),
        created_at=float(raw["created_at"]),
        expires_at=float(raw["expires_at"]),
    )


def _drop_preview(store: GraphStore, preview_id: str) -> None:
    with store.transaction() as conn:
        conn.execute("DELETE FROM ops_refactor_previews WHERE id = ?", (preview_id,))


# -- apply ------------------------------------------------------------------------


def apply_refactor(
    root: Path,
    store: GraphStore,
    preview_id: str,
    *,
    clock: Callable[[], float] = time.time,
) -> AppliedRefactor:
    """Write a stored preview to disk, or refuse if it expired or any file changed."""
    preview = load_preview(store, preview_id)
    if preview is None:
        raise RefactorError(f"no pending preview {preview_id!r} (already applied or never made)")
    if clock() > preview.expires_at:
        _drop_preview(store, preview_id)
        raise RefactorError(f"preview {preview_id} expired; run the preview again")

    staged: list[tuple[Path, str]] = []
    changed: list[str] = []
    for file_edit in preview.files:
        path = _safe_path(root, file_edit.path)
        data, text = _read_text(path, file_edit.path)
        if _sha(data) != file_edit.sha:
            changed.append(file_edit.path)
            continue
        lines = _lines(text)
        for edit in file_edit.lines:
            lines[edit.line - 1] = edit.after
        staged.append((path, "".join(lines)))
    if changed:
        raise RefactorError(
            f"refusing to apply {preview_id}: changed since the preview: {', '.join(changed)}"
        )
    for path, text in staged:
        _atomic_write(path, text)
    _drop_preview(store, preview_id)
    return AppliedRefactor(
        preview_id=preview_id,
        files=tuple(f.path for f in preview.files),
        lines_changed=preview.edit_count,
    )


def _atomic_write(path: Path, text: str) -> None:
    mode = path.stat().st_mode
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# -- suggestions --------------------------------------------------------------------


def suggest_refactorings(
    store: GraphStore,
    *,
    limit: int = 10,
    large_lines: int = 80,
    move_ratio: float = 0.7,
    min_callers: int = 3,
) -> list[Suggestion]:
    """Up to ``limit`` suggestions of each kind, strongest first within a kind."""
    view = GraphView.load(store)
    groups: list[list[Suggestion]] = [
        _dedupe_candidates(view),
        _move_candidates(view, move_ratio, min_callers),
        _split_candidates(view, large_lines),
        _remove_candidates(store),
    ]
    suggestions: list[Suggestion] = []
    for group in groups:
        suggestions.extend(sorted(group, key=lambda s: (-s.score, s.path, s.line))[:limit])
    return suggestions


def _dedupe_candidates(view: GraphView) -> list[Suggestion]:
    seen: set[frozenset[str]] = set()
    found: list[Suggestion] = []
    for edge in view.edges(["similar_to"]):
        pair = frozenset({edge.src, edge.dst})
        if pair in seen or len(pair) < 2:
            continue
        seen.add(pair)
        node = view.nodes[edge.src]
        detail = f" ({edge.detail})" if edge.detail else ""
        found.append(
            Suggestion(
                kind="dedupe",
                qname=edge.src,
                path=node.path,
                line=node.line_start,
                score=round(edge.confidence, 3),
                reason=f"near-duplicate of {edge.dst}{detail}; extract the shared part",
                related=(edge.dst,),
            )
        )
    return found


def _move_candidates(view: GraphView, move_ratio: float, min_callers: int) -> list[Suggestion]:
    found: list[Suggestion] = []
    for node in view.of_kind(*CALLABLE_KINDS, "class"):
        if is_test_path(node.path) or node.kind == "method":
            continue
        home = view.module_of(node.qname)
        callers = [c for c in view.predecessors(node.qname, ["calls"]) if c in view.nodes]
        callers = [c for c in callers if not is_test_path(view.nodes[c].path)]
        if len(callers) < min_callers:
            continue
        modules = Counter(view.module_of(c) for c in callers)
        top, count = modules.most_common(1)[0]
        ratio = count / len(callers)
        if top is None or top == home or ratio < move_ratio:
            continue
        found.append(
            Suggestion(
                kind="move",
                qname=node.qname,
                path=node.path,
                line=node.line_start,
                score=round(ratio, 3),
                reason=f"{count} of {len(callers)} callers live in {top}; consider moving it there",
                related=(top,),
            )
        )
    return found


def _split_candidates(view: GraphView, large_lines: int) -> list[Suggestion]:
    return [
        Suggestion(
            kind="split",
            qname=node.qname,
            path=node.path,
            line=node.line_start,
            score=round(node.span / large_lines, 3),
            reason=f"{node.span} lines long; split it into smaller functions",
        )
        for node in view.of_kind(*CALLABLE_KINDS)
        if node.span >= large_lines
    ]


def _remove_candidates(store: GraphStore) -> list[Suggestion]:
    return [
        Suggestion(
            kind="remove",
            qname=node.qname,
            path=node.path,
            line=node.line_start,
            score=0.3,
            reason="nothing calls, tests or subclasses it; delete it if it is not an entry point",
        )
        for node in dead_code(store)
        if not is_test_path(node.path)
    ]


__all__ = [
    "AppliedRefactor",
    "FileEdit",
    "LineEdit",
    "RefactorError",
    "RenamePreview",
    "Suggestion",
    "apply_refactor",
    "load_preview",
    "preview_rename",
    "suggest_refactorings",
    "validate_new_name",
]
