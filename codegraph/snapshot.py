"""Team-shared graph snapshots: commit a compressed database, skip the first full index.

Indexing a large repository from scratch takes minutes; restoring a snapshot takes
seconds. :func:`export_snapshot` writes ``.codegraph/graph.db.gz``: a ``VACUUM INTO``
copy (compact, and consistent even while another process holds the WAL) gzipped at
level 9 with a zeroed header timestamp, so an unchanged graph re-exports byte-for-byte.
Local-only state (pending refactor previews) is dropped from the copy.

:func:`import_snapshot` refuses anything that fails ``PRAGMA integrity_check`` or
carries another schema version, swaps it in, and then runs an incremental index so
the graph matches the working tree rather than the commit the snapshot was taken at.

The first export adds ``.codegraph/graph.db.gz binary merge=ours`` to
``.gitattributes`` (``binary`` first: the macro unsets ``merge`` if it comes last), so
two branches that both re-exported never produce a conflict in a binary file; the
re-index after a merge reconciles the content. ``merge=ours`` only takes effect once
``git config merge.ours.driver true`` is set, which the export result reminds about.
"""

from __future__ import annotations

import gzip
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from codegraph.indexer import IndexStats, index_repository
from codegraph.store import DB_DIRNAME, SCHEMA_VERSION, GraphStore, default_db_path

SNAPSHOT_FILENAME = "graph.db.gz"
SNAPSHOT_RELPATH = f"{DB_DIRNAME}/{SNAPSHOT_FILENAME}"
GITATTRIBUTES_LINE = f"{SNAPSHOT_RELPATH} binary merge=ours"
LOCAL_ONLY_TABLES: tuple[str, ...] = ("ops_refactor_previews",)
MAX_SNAPSHOT_BYTES = 16 * 1024**3
_CHUNK = 1 << 20


class SnapshotError(RuntimeError):
    """The snapshot is missing, corrupt, or from another schema version."""


@dataclass(frozen=True, slots=True)
class SnapshotExport:
    path: Path
    compressed_bytes: int
    database_bytes: int
    nodes: int
    edges: int
    gitattributes_updated: bool
    hints: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SnapshotImport:
    path: Path
    database_bytes: int
    nodes: int
    edges: int
    index: IndexStats


def snapshot_path(root: Path) -> Path:
    return root / DB_DIRNAME / SNAPSHOT_FILENAME


def export_snapshot(root: Path, store: GraphStore) -> SnapshotExport:
    """Write ``.codegraph/graph.db.gz`` from ``store`` and register it in .gitattributes."""
    target = snapshot_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if store.connection.in_transaction:
        raise SnapshotError("cannot snapshot inside an open transaction")
    counts = store.counts()
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".snapshot-") as tmp:
        raw = Path(tmp) / "graph.db"
        store.connection.execute("VACUUM INTO ?", (str(raw),))
        _drop_local_tables(raw)
        staged = Path(tmp) / SNAPSHOT_FILENAME
        with (
            raw.open("rb") as src,
            staged.open("wb") as out,
            gzip.GzipFile(filename="", mode="wb", fileobj=out, compresslevel=9, mtime=0) as gz,
        ):
            shutil.copyfileobj(src, gz, _CHUNK)
        database_bytes = raw.stat().st_size
        os.replace(staged, target)
    updated = ensure_gitattributes(root)
    return SnapshotExport(
        path=target,
        compressed_bytes=target.stat().st_size,
        database_bytes=database_bytes,
        nodes=counts["nodes"],
        edges=counts["edges"],
        gitattributes_updated=updated,
        hints=_commit_hints(root),
    )


def _drop_local_tables(db: Path) -> None:
    conn = sqlite3.connect(db)
    try:
        for table in LOCAL_ONLY_TABLES:
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()


def _commit_hints(root: Path) -> tuple[str, ...]:
    hints = ["run `git config merge.ours.driver true` once per clone so merge=ours takes effect"]
    gitignore = root / ".gitignore"
    if gitignore.is_file():
        lines = {line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()}
        if lines & {".codegraph", ".codegraph/", "/.codegraph", "/.codegraph/"}:
            hints.append(
                "your .gitignore ignores the whole .codegraph/ directory; replace that line "
                f"with `.codegraph/*` and `!{SNAPSHOT_RELPATH}` to commit the snapshot"
            )
    return tuple(hints)


def ensure_gitattributes(root: Path) -> bool:
    """Add the snapshot's merge rule unless some line already covers the path."""
    path = root / ".gitattributes"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    for line in existing.splitlines():
        fields = line.split()
        if fields and fields[0] in {SNAPSHOT_RELPATH, f"/{SNAPSHOT_RELPATH}"}:
            return False
    separator = "" if not existing or existing.endswith("\n") else "\n"
    path.write_text(f"{existing}{separator}{GITATTRIBUTES_LINE}\n", encoding="utf-8")
    return True


def import_snapshot(
    root: Path, *, max_bytes: int = MAX_SNAPSHOT_BYTES, skip_passes: frozenset[str] = frozenset()
) -> SnapshotImport:
    """Replace the repo's database with the verified snapshot, then index what changed."""
    source = snapshot_path(root)
    if not source.is_file():
        raise SnapshotError(f"no snapshot at {source}")
    db = default_db_path(root)
    db.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=db.parent, prefix=".snapshot-") as tmp:
        staged = Path(tmp) / "graph.db"
        database_bytes = _decompress(source, staged, max_bytes)
        nodes, edges = verify_database(staged)
        for suffix in ("-wal", "-shm", ""):
            Path(f"{db}{suffix}").unlink(missing_ok=True)
        os.replace(staged, db)
    with GraphStore(db) as store:
        stats = index_repository(root, store, skip_passes=skip_passes)
    return SnapshotImport(
        path=source, database_bytes=database_bytes, nodes=nodes, edges=edges, index=stats
    )


def _decompress(source: Path, target: Path, max_bytes: int) -> int:
    written = 0
    try:
        with gzip.open(source, "rb") as src, target.open("wb") as out:
            while chunk := src.read(_CHUNK):
                written += len(chunk)
                if written > max_bytes:
                    raise SnapshotError(f"{source} expands past {max_bytes} bytes; refusing it")
                out.write(chunk)
    except (gzip.BadGzipFile, EOFError, OSError) as exc:
        raise SnapshotError(f"{source} could not be decompressed: {exc}") from exc
    return written


def verify_database(db: Path) -> tuple[int, int]:
    """Check integrity and schema version; return ``(nodes, edges)``."""
    conn = sqlite3.connect(db)
    try:
        problems = [str(r[0]) for r in conn.execute("PRAGMA integrity_check").fetchall()]
        if problems != ["ok"]:
            raise SnapshotError(f"integrity check failed: {'; '.join(problems[:5])}")
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None or str(row[0]) != str(SCHEMA_VERSION):
            found = "none" if row is None else str(row[0])
            raise SnapshotError(
                f"snapshot has schema version {found}, expected {SCHEMA_VERSION}; "
                "re-export it with this codegraph version"
            )
        nodes = int(conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])
        edges = int(conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
    except sqlite3.DatabaseError as exc:
        raise SnapshotError(f"snapshot is not a codegraph database: {exc}") from exc
    finally:
        conn.close()
    return nodes, edges


__all__ = [
    "GITATTRIBUTES_LINE",
    "SNAPSHOT_FILENAME",
    "SnapshotError",
    "SnapshotExport",
    "SnapshotImport",
    "ensure_gitattributes",
    "export_snapshot",
    "import_snapshot",
    "snapshot_path",
    "verify_database",
]
