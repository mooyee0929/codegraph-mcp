"""Bounded git history: per-file churn and files that change together.

One ``git log --numstat`` call reads at most :data:`MAX_COMMITS` commits, and commits
older than :data:`MAX_DAYS` before the newest one are dropped (measured from the newest
commit rather than from today, so an archived repository still has history). Merges are
skipped, and so are commits touching more than :data:`MAX_FILES` files: bulk renames,
formatting sweeps and vendoring say nothing about which files belong together.

Coupling follows codebase-memory's definition: ``co_changes / min(changes_a,
changes_b)``. A pair is reported when it changed together at least
:data:`MIN_CO_CHANGES` times with coupling at least :data:`MIN_COUPLING`.

A directory that is not a git work tree, a repository with no commits, a missing
``git`` binary or a timeout all yield an empty history; :class:`History` carries the
reason in ``status`` so callers can show it.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path, PurePosixPath

from codegraph.review import GitError

MAX_COMMITS = 1000
MAX_DAYS = 365
MAX_FILES = 50
MIN_CO_CHANGES = 3
MIN_COUPLING = 0.3
GIT_TIMEOUT = 60.0

_RECORD = "\x1e"
_FIELD = "\x1f"
_LOG_FORMAT = f"--format={_RECORD}%H{_FIELD}%at{_FIELD}%aN{_FIELD}%s"

_UNTRACKABLE_DIRS = ("node_modules/", "vendor/", "__pycache__/", ".git/", "dist/", "build/")
_UNTRACKABLE_NAMES = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "Cargo.lock",
        "poetry.lock",
        "composer.lock",
        "Gemfile.lock",
        "Pipfile.lock",
        "uv.lock",
        "go.sum",
    }
)
_UNTRACKABLE_SUFFIX = re.compile(
    r"\.(?:lock|sum|min\.js|min\.css|map|wasm|png|jpe?g|gif|ico|svg|pdf|zip|gz|jar|woff2?|ttf)$",
    re.I,
)


@dataclass(frozen=True, slots=True)
class FileChange:
    path: str
    added: int
    removed: int


@dataclass(frozen=True, slots=True)
class Commit:
    sha: str
    timestamp: int
    author: str
    subject: str
    files: tuple[FileChange, ...]


@dataclass(frozen=True, slots=True)
class FileChurn:
    path: str
    commits: int
    authors: int
    first_change: int
    last_change: int
    added: int
    removed: int


@dataclass(frozen=True, slots=True)
class CoChange:
    a: str
    b: str
    count: int
    ratio: float
    last_change: int


@dataclass(frozen=True, slots=True)
class History:
    commits: tuple[Commit, ...] = field(default=())
    status: str = "ok"


def is_trackable(path: str) -> bool:
    """Lock files, binaries and generated output do not count as changes."""
    if path.startswith(_UNTRACKABLE_DIRS) or any(f"/{d}" in path for d in _UNTRACKABLE_DIRS):
        return False
    if PurePosixPath(path).name in _UNTRACKABLE_NAMES:
        return False
    return not _UNTRACKABLE_SUFFIX.search(path)


def _run_git(root: Path, args: list[str], timeout: float) -> str:
    try:
        completed = subprocess.run(
            ["git", "-c", "core.quotepath=off", *args],
            cwd=root,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise GitError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {args[0]} timed out after {timeout:.0f}s") from exc
    if completed.returncode != 0:
        raise GitError(completed.stderr.strip() or f"git {args[0]} failed")
    return completed.stdout


def parse_log(output: str) -> list[Commit]:
    """Parse ``git log --numstat`` output written with :data:`_LOG_FORMAT`."""
    commits: list[Commit] = []
    for record in output.split(_RECORD):
        lines = record.strip("\n").split("\n")
        if not lines or not lines[0]:
            continue
        header = lines[0].split(_FIELD)
        if len(header) < 4 or not header[1].isdigit():
            continue
        files: list[FileChange] = []
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) != 3 or not parts[2]:
                continue
            added = int(parts[0]) if parts[0].isdigit() else 0
            removed = int(parts[1]) if parts[1].isdigit() else 0
            files.append(FileChange(path=parts[2], added=added, removed=removed))
        commits.append(
            Commit(
                sha=header[0],
                timestamp=int(header[1]),
                author=header[2],
                subject=_FIELD.join(header[3:]),
                files=tuple(files),
            )
        )
    return commits


def read_history(
    root: Path,
    *,
    max_commits: int = MAX_COMMITS,
    max_days: int = MAX_DAYS,
    timeout: float = GIT_TIMEOUT,
) -> History:
    """Recent non-merge commits under ``root`` with per-file line counts."""
    args = [
        "log",
        "--no-merges",
        "--no-renames",
        "--relative",
        "--numstat",
        _LOG_FORMAT,
        "-n",
        str(max_commits),
    ]
    try:
        output = _run_git(root, args, timeout)
    except GitError as exc:
        return History(status=f"no history: {exc}")
    commits = parse_log(output)
    if not commits:
        return History(status="no history: no commits")
    newest = max(c.timestamp for c in commits)
    cutoff = newest - max_days * 86400
    kept = tuple(c for c in commits if c.timestamp >= cutoff)
    return History(commits=kept, status=f"ok: {len(kept)} commits")


def _relevant(commit: Commit, max_files: int) -> list[FileChange]:
    files = [f for f in commit.files if is_trackable(f.path)]
    return files if 0 < len(files) <= max_files else []


def churn(commits: tuple[Commit, ...], max_files: int = MAX_FILES) -> dict[str, FileChurn]:
    counts: Counter[str] = Counter()
    authors: defaultdict[str, set[str]] = defaultdict(set)
    first: dict[str, int] = {}
    last: dict[str, int] = {}
    added: Counter[str] = Counter()
    removed: Counter[str] = Counter()
    for commit in commits:
        for change in _relevant(commit, max_files):
            counts[change.path] += 1
            authors[change.path].add(commit.author)
            first[change.path] = min(first.get(change.path, commit.timestamp), commit.timestamp)
            last[change.path] = max(last.get(change.path, commit.timestamp), commit.timestamp)
            added[change.path] += change.added
            removed[change.path] += change.removed
    return {
        path: FileChurn(
            path=path,
            commits=counts[path],
            authors=len(authors[path]),
            first_change=first[path],
            last_change=last[path],
            added=added[path],
            removed=removed[path],
        )
        for path in counts
    }


def co_changes(
    commits: tuple[Commit, ...],
    *,
    min_count: int = MIN_CO_CHANGES,
    min_ratio: float = MIN_COUPLING,
    max_files: int = MAX_FILES,
) -> list[CoChange]:
    totals: Counter[str] = Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    latest: dict[tuple[str, str], int] = {}
    for commit in commits:
        paths = sorted({f.path for f in _relevant(commit, max_files)})
        totals.update(paths)
        for pair in combinations(paths, 2):
            pairs[pair] += 1
            latest[pair] = max(latest.get(pair, 0), commit.timestamp)
    found: list[CoChange] = []
    for (a, b), count in pairs.items():
        if count < min_count:
            continue
        ratio = count / min(totals[a], totals[b])
        if ratio >= min_ratio:
            found.append(
                CoChange(a=a, b=b, count=count, ratio=round(ratio, 3), last_change=latest[(a, b)])
            )
    return sorted(found, key=lambda c: (-c.count, -c.ratio, c.a, c.b))


__all__ = [
    "CoChange",
    "Commit",
    "FileChange",
    "FileChurn",
    "History",
    "co_changes",
    "churn",
    "is_trackable",
    "parse_log",
    "read_history",
]
