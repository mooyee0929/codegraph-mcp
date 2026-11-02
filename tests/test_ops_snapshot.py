from __future__ import annotations

import gzip
import sqlite3
from pathlib import Path

import pytest

from codegraph.indexer import index_repository
from codegraph.snapshot import (
    GITATTRIBUTES_LINE,
    SnapshotError,
    ensure_gitattributes,
    export_snapshot,
    import_snapshot,
    snapshot_path,
    verify_database,
)
from codegraph.store import GraphStore, default_db_path


def _index(repo: Path) -> None:
    with GraphStore.open(repo) as store:
        index_repository(repo, store)


def test_export_then_import_round_trips(repo: Path) -> None:
    _index(repo)
    with GraphStore.open(repo) as store:
        store.ensure_schema("CREATE TABLE IF NOT EXISTS ops_refactor_previews (id TEXT);")
        store.connection.commit()
        exported = export_snapshot(repo, store)
    assert exported.path == snapshot_path(repo)
    assert exported.gitattributes_updated
    assert 0 < exported.compressed_bytes < exported.database_bytes
    assert exported.nodes > 0
    assert (repo / ".gitattributes").read_text() == f"{GITATTRIBUTES_LINE}\n"
    assert any("merge.ours.driver" in hint for hint in exported.hints)

    with gzip.open(exported.path) as handle, open(repo / "copy.db", "wb") as out:
        out.write(handle.read())
    conn = sqlite3.connect(repo / "copy.db")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "ops_refactor_previews" not in tables
    (repo / "copy.db").unlink()

    with GraphStore.open(repo) as store:
        again = export_snapshot(repo, store)
    assert not again.gitattributes_updated

    for suffix in ("", "-wal", "-shm"):
        Path(f"{default_db_path(repo)}{suffix}").unlink(missing_ok=True)
    (repo / "pkg" / "extra.py").write_text("def added() -> int:\n    return 1\n")
    imported = import_snapshot(repo)
    assert imported.nodes == exported.nodes
    assert imported.index.parsed == 1
    with GraphStore.open(repo) as store:
        assert store.get_node("pkg.billing.charge") is not None
        assert store.get_node("pkg.extra.added") is not None


def test_gitattributes_append_and_detect(tmp_path: Path) -> None:
    attributes = tmp_path / ".gitattributes"
    attributes.write_text("*.png binary")
    assert ensure_gitattributes(tmp_path)
    assert attributes.read_text() == f"*.png binary\n{GITATTRIBUTES_LINE}\n"
    assert not ensure_gitattributes(tmp_path)


def test_gitignore_hint(repo: Path) -> None:
    (repo / ".gitignore").write_text(".codegraph/\n")
    _index(repo)
    with GraphStore.open(repo) as store:
        exported = export_snapshot(repo, store)
    assert any("!.codegraph/graph.db.gz" in hint for hint in exported.hints)


def test_import_rejects_bad_snapshots(repo: Path) -> None:
    with pytest.raises(SnapshotError, match="no snapshot"):
        import_snapshot(repo)

    target = snapshot_path(repo)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"definitely not gzip")
    with pytest.raises(SnapshotError, match="decompressed"):
        import_snapshot(repo)

    target.write_bytes(gzip.compress(b"not a database" * 200))
    with pytest.raises(SnapshotError, match="not a codegraph database"):
        import_snapshot(repo)

    target.write_bytes(gzip.compress(b"\0" * 5000))
    with pytest.raises(SnapshotError):
        import_snapshot(repo, max_bytes=1000)


def test_verify_rejects_other_schema_version(tmp_path: Path) -> None:
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '1')")
    conn.commit()
    conn.close()
    with pytest.raises(SnapshotError, match="schema version 1"):
        verify_database(db)
