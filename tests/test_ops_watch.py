from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from codegraph.indexer import IndexStats, index_repository
from codegraph.server import main as server_main
from codegraph.snapshot import export_snapshot
from codegraph.store import GraphStore, default_db_path
from codegraph.watch import (
    Snapshot,
    Watcher,
    auto_index,
    changed_paths,
    env_flag,
    index_state,
    make_reindexer,
    scan,
    start_background_sync,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_changed_paths() -> None:
    before: Snapshot = {"a.py": (1, 10), "b.py": (1, 10)}
    after: Snapshot = {"a.py": (2, 10), "c.py": (1, 1), "b.py": (1, 10)}
    assert changed_paths(before, after) == frozenset({"a.py", "c.py"})
    assert changed_paths(after, {}) == frozenset(after)


def test_scan_honours_ignore_file(repo: Path) -> None:
    assert "pkg/billing.py" in scan(repo)
    (repo / ".codegraphignore").write_text("web/\n")
    assert not any(p.startswith("web/") for p in scan(repo))


def test_watcher_debounces_a_burst_into_one_batch() -> None:
    clock = FakeClock()
    states: list[Snapshot] = [
        {"a.py": (1, 1)},
        {"a.py": (2, 1)},
        {"a.py": (2, 1), "b.py": (1, 1)},
        {"a.py": (2, 1), "b.py": (1, 1)},
        {"a.py": (2, 1), "b.py": (1, 1)},
    ]
    batches: list[frozenset[str]] = []
    watcher = Watcher(
        root=Path("."),
        on_batch=batches.append,
        interval=0.2,
        debounce=0.5,
        clock=clock,
        sleep=clock.sleep,
        scanner=lambda _root: states.pop(0) if len(states) > 1 else states[0],
    )
    assert watcher.run(should_stop=lambda: False, max_polls=6) == 1
    assert batches == [frozenset({"a.py", "b.py"})]
    assert clock.now == pytest.approx(1.2)


def test_watcher_reindexes_real_changes(repo: Path) -> None:
    with GraphStore.open(repo) as store:
        index_repository(repo, store)
    messages: list[str] = []
    clock = FakeClock()
    watcher = Watcher(
        root=repo,
        on_batch=make_reindexer(repo, messages.append),
        debounce=0.0,
        clock=clock,
        sleep=clock.sleep,
    )
    assert watcher.poll() == frozenset()
    (repo / "pkg" / "new_mod.py").write_text("def fresh() -> None:\n    return None\n")
    assert watcher.poll() == frozenset({"pkg/new_mod.py"})
    assert "parsed 1" in messages[-1]
    with GraphStore.open(repo) as store:
        assert store.get_node("pkg.new_mod.fresh") is not None


def test_reindexer_reports_failures(tmp_path: Path) -> None:
    root = tmp_path / "broken"
    (root / ".codegraph").mkdir(parents=True)
    (root / ".codegraph" / "graph.db").write_bytes(b"garbage" * 500)
    messages: list[str] = []
    make_reindexer(root, messages.append)(frozenset({"x.py"}))
    assert messages and "failed" in messages[0]


def test_index_state_transitions(repo: Path) -> None:
    assert index_state(repo) == "missing"
    with GraphStore.open(repo):
        pass
    assert index_state(repo) == "empty"
    with GraphStore.open(repo) as store:
        index_repository(repo, store)
    assert index_state(repo) == "fresh"
    future = time.time() + 60
    os.utime(repo / "pkg" / "util.py", (future, future))
    assert index_state(repo) == "stale"
    (repo / "pkg" / "util.py").unlink()
    assert index_state(repo) == "stale"


def test_auto_index_restores_snapshot_then_stays_fresh(repo: Path) -> None:
    with GraphStore.open(repo) as store:
        index_repository(repo, store)
        export_snapshot(repo, store)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{default_db_path(repo)}{suffix}").unlink(missing_ok=True)

    result = auto_index(repo)
    assert result.state == "missing" and result.restored_snapshot
    assert isinstance(result.stats, IndexStats)
    assert auto_index(repo).state == "fresh"


def test_auto_index_falls_back_when_snapshot_is_bad(repo: Path) -> None:
    (repo / ".codegraph").mkdir()
    (repo / ".codegraph" / "graph.db.gz").write_bytes(b"junk")
    result = auto_index(repo)
    assert not result.restored_snapshot and "snapshot rejected" in result.message
    assert result.stats is not None and result.stats.parsed == 8


def test_env_flag() -> None:
    assert env_flag("X", {"X": "1"}) and env_flag("X", {"X": "Yes"})
    assert not env_flag("X", {"X": "0"}) and not env_flag("X", {})


def test_background_sync_auto_index(repo: Path) -> None:
    messages: list[str] = []
    assert start_background_sync(repo, auto=False, watch=False) is None
    sync = start_background_sync(repo, auto=True, watch=False, report=messages.append)
    assert sync is not None
    sync.thread.join(30)
    assert not sync.thread.is_alive()
    assert messages and "missing graph" in messages[0]


def test_background_watch_stops_promptly(repo: Path) -> None:
    sync = start_background_sync(repo, auto=False, watch=True, interval=0.01, report=print)
    assert sync is not None
    sync.stop(timeout=10)
    assert not sync.thread.is_alive()


def test_server_main_accepts_sync_flags(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    assert server_main(["--root", str(missing), "--watch", "--auto-index"]) == 2
