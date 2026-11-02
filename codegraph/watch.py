"""Keep the graph in step with the working tree, without a filesystem-events dependency.

The watcher polls. Each scan stats every file the indexer would read (same default
ignores plus ``.codegraphignore``, re-read every scan) and compares ``(mtime_ns, size)``
with the previous scan. Changes accumulate until none arrive for ``debounce`` seconds,
then one incremental index runs for the whole batch: an editor saving ten files in a
burst costs one re-index, not ten. Polling is portable and cheap enough for
repositories of tens of thousands of files; the indexer's content hashes make a
spurious batch (a ``touch`` with no edit) nearly free.

The clock and the sleep function are injected so tests can drive the loop without
waiting, and the server reuses the same loop on a daemon thread for ``--watch``.

:func:`auto_index` is the server's ``--auto-index``: it indexes on startup when the
database is missing, empty or older than the files, restoring a committed snapshot
first when there is no database at all.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from codegraph.indexer import (
    DEFAULT_IGNORES,
    IndexStats,
    index_repository,
    iter_source_files,
    load_ignore_file,
)
from codegraph.snapshot import SnapshotError, import_snapshot, snapshot_path
from codegraph.store import GraphStore, StoreError, default_db_path

WATCH_ENV = "CODEGRAPH_WATCH"
AUTO_INDEX_ENV = "CODEGRAPH_AUTO_INDEX"

Snapshot = dict[str, tuple[int, int]]
"""Repo-relative path -> ``(mtime_ns, size)``."""

IndexState = Literal["missing", "empty", "stale", "fresh"]
Report = Callable[[str], None]


def env_flag(name: str, environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _stderr(message: str) -> None:
    print(f"codegraph: {message}", file=sys.stderr, flush=True)


def scan(root: Path) -> Snapshot:
    ignores = (*DEFAULT_IGNORES, *load_ignore_file(root))
    found: Snapshot = {}
    for path in iter_source_files(root, ignores):
        try:
            stat = path.stat()
        except FileNotFoundError:
            # Deleted between the directory walk and the stat: absent from this scan,
            # so the diff against the previous scan reports it as removed.
            continue
        found[path.relative_to(root).as_posix()] = (stat.st_mtime_ns, stat.st_size)
    return found


def changed_paths(before: Snapshot, after: Snapshot) -> frozenset[str]:
    return frozenset(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))


def reindex(root: Path) -> IndexStats:
    with GraphStore.open(root) as store:
        return index_repository(root, store)


def make_reindexer(root: Path, report: Report = _stderr) -> Callable[[frozenset[str]], None]:
    """A batch handler that re-indexes and reports, surviving a locked or broken database."""

    def on_batch(batch: frozenset[str]) -> None:
        try:
            stats = reindex(root)
        except (StoreError, sqlite3.Error, OSError) as exc:
            report(f"re-index after {len(batch)} change(s) failed: {exc}")
            return
        report(f"{len(batch)} change(s): {stats.summary()}")

    return on_batch


@dataclass(slots=True)
class Watcher:
    root: Path
    on_batch: Callable[[frozenset[str]], None]
    interval: float = 1.0
    debounce: float = 0.5
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], object] = time.sleep
    scanner: Callable[[Path], Snapshot] = scan
    _baseline: Snapshot | None = field(default=None, init=False)
    _pending: set[str] = field(default_factory=set, init=False)
    _last_change: float = field(default=0.0, init=False)

    def poll(self) -> frozenset[str]:
        """Scan once; return the batch handed to ``on_batch``, or an empty set.

        The first poll only records the baseline.
        """
        current = self.scanner(self.root)
        if self._baseline is None:
            self._baseline = current
            return frozenset()
        changed = changed_paths(self._baseline, current)
        self._baseline = current
        now = self.clock()
        if changed:
            self._pending |= changed
            self._last_change = now
        if not self._pending or now - self._last_change < self.debounce:
            return frozenset()
        batch = frozenset(self._pending)
        self._pending.clear()
        self.on_batch(batch)
        return batch

    def run(self, should_stop: Callable[[], bool], max_polls: int | None = None) -> int:
        """Poll until ``should_stop()`` (or ``max_polls``); return how many batches ran."""
        batches = polls = 0
        self.poll()
        while not should_stop() and (max_polls is None or polls < max_polls):
            self.sleep(self.interval)
            if should_stop():
                break
            if self.poll():
                batches += 1
            polls += 1
        return batches


# -- auto-index -------------------------------------------------------------


def index_state(root: Path) -> IndexState:
    """``missing`` / ``empty`` database, ``stale`` files, or ``fresh``."""
    db = default_db_path(root)
    if not db.is_file():
        return "missing"
    with GraphStore(db) as store:
        rows = store.connection.execute("SELECT path, indexed_at FROM files").fetchall()
    if not rows:
        return "empty"
    indexed = {str(r["path"]): float(r["indexed_at"]) for r in rows}
    current = scan(root)
    if current.keys() != indexed.keys():
        return "stale"
    for path, (mtime_ns, _size) in current.items():
        if mtime_ns / 1e9 > indexed[path]:
            return "stale"
    return "fresh"


@dataclass(frozen=True, slots=True)
class AutoIndexResult:
    state: IndexState
    restored_snapshot: bool
    stats: IndexStats | None
    message: str


def auto_index(root: Path) -> AutoIndexResult:
    state = index_state(root)
    if state == "fresh":
        return AutoIndexResult(state, False, None, "graph is up to date")
    if state == "missing" and snapshot_path(root).is_file():
        try:
            imported = import_snapshot(root)
        except SnapshotError as exc:
            stats = reindex(root)
            return AutoIndexResult(
                state, False, stats, f"snapshot rejected ({exc}); indexed: {stats.summary()}"
            )
        return AutoIndexResult(
            state,
            True,
            imported.index,
            f"restored snapshot ({imported.nodes} nodes); {imported.index.summary()}",
        )
    stats = reindex(root)
    return AutoIndexResult(state, False, stats, f"{state} graph; {stats.summary()}")


# -- background sync for the server -------------------------------------------


@dataclass(frozen=True, slots=True)
class BackgroundSync:
    thread: threading.Thread
    stop_event: threading.Event

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        self.thread.join(timeout)


def start_background_sync(
    root: Path,
    *,
    auto: bool,
    watch: bool,
    interval: float = 2.0,
    debounce: float = 1.0,
    report: Report = _stderr,
) -> BackgroundSync | None:
    """Run ``auto_index`` and/or the watcher on a daemon thread; None when neither is on.

    Indexing happens off the request path so an MCP client's startup handshake never
    waits on a full index; the indexer's single transaction means tools keep reading the
    previous graph until it commits.
    """
    if not (auto or watch):
        return None
    stop = threading.Event()

    def work() -> None:
        if auto:
            try:
                report(auto_index(root).message)
            except (StoreError, sqlite3.Error, OSError) as exc:
                report(f"auto-index failed: {exc}")
        if watch:
            watcher = Watcher(
                root=root,
                on_batch=make_reindexer(root, report),
                interval=interval,
                debounce=debounce,
                sleep=stop.wait,
            )
            watcher.run(should_stop=stop.is_set)

    thread = threading.Thread(target=work, name="codegraph-sync", daemon=True)
    thread.start()
    return BackgroundSync(thread=thread, stop_event=stop)


__all__ = [
    "AUTO_INDEX_ENV",
    "WATCH_ENV",
    "AutoIndexResult",
    "BackgroundSync",
    "Watcher",
    "auto_index",
    "changed_paths",
    "env_flag",
    "index_state",
    "make_reindexer",
    "reindex",
    "scan",
    "start_background_sync",
]
