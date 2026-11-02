"""SQLite persistence for the graph.

One database file per indexed repository, kept at ``<repo>/.codegraph/graph.db``.
The store owns schema, transactions and the row<->dataclass mapping; it deliberately
holds no graph algorithms (those live in :mod:`codegraph.queries`).

Feature modules that need their own tables (flows, communities, embeddings, ...) call
:meth:`GraphStore.ensure_schema` with ``CREATE TABLE IF NOT EXISTS`` DDL. The whole
database is a derived cache of the source tree: when the core schema version changes
the file is rebuilt from scratch rather than migrated.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from codegraph.models import Edge, EdgeKind, Language, Node, NodeKind

SCHEMA_VERSION = 2

DB_DIRNAME = ".codegraph"
DB_FILENAME = "graph.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    path       TEXT PRIMARY KEY,
    sha        TEXT NOT NULL,
    language   TEXT NOT NULL,
    indexed_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    qname      TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    path       TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end   INTEGER NOT NULL,
    language   TEXT NOT NULL,
    name       TEXT NOT NULL,
    signature  TEXT NOT NULL DEFAULT '',
    docstring  TEXT
);

CREATE TABLE IF NOT EXISTS edges (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    src      TEXT NOT NULL,
    dst      TEXT NOT NULL,
    kind     TEXT NOT NULL,
    path     TEXT NOT NULL,
    line     INTEGER NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 1,
    confidence REAL NOT NULL DEFAULT 1.0,
    detail   TEXT NOT NULL DEFAULT '',
    UNIQUE (src, dst, kind, path, line)
);

CREATE INDEX IF NOT EXISTS idx_nodes_path ON nodes (path);
CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes (name);
CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes (kind);
CREATE INDEX IF NOT EXISTS idx_edges_src  ON edges (src, kind);
CREATE INDEX IF NOT EXISTS idx_edges_dst  ON edges (dst, kind);
CREATE INDEX IF NOT EXISTS idx_edges_path ON edges (path);
CREATE INDEX IF NOT EXISTS idx_edges_kind ON edges (kind);

CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5 (
    qname, name, signature, docstring
);
"""


class StoreError(RuntimeError):
    """Raised when the database is unusable or from an incompatible version."""


def default_db_path(repo_root: Path) -> Path:
    return repo_root / DB_DIRNAME / DB_FILENAME


class GraphStore:
    """A connection to one repository's graph database."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._connect()
        self.rebuilt = False
        """True when this open discarded a database written by another schema version."""
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
        except sqlite3.DatabaseError as exc:
            conn.close()
            raise StoreError(f"{self.db_path} is not a codegraph database: {exc}") from exc
        return conn

    @classmethod
    def open(cls, repo_root: Path) -> GraphStore:
        return cls(default_db_path(repo_root))

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> GraphStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- schema -----------------------------------------------------------

    def _migrate(self) -> None:
        found = self._stored_version()
        if found is not None and found != SCHEMA_VERSION:
            self._rebuild()
        self._conn.executescript(_SCHEMA)
        if self._stored_version() is None:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()

    def _stored_version(self) -> int | None:
        try:
            row = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        except sqlite3.OperationalError:
            return None  # fresh file, no meta table yet
        except sqlite3.DatabaseError as exc:
            raise StoreError(f"{self.db_path} is not a codegraph database: {exc}") from exc
        if row is None:
            return None
        try:
            return int(row["value"])
        except ValueError as exc:
            raise StoreError(f"{self.db_path} has a corrupt schema_version") from exc

    def _rebuild(self) -> None:
        """Throw away a database from another schema version; it is only a cache."""
        self._conn.close()
        for suffix in ("", "-wal", "-shm"):
            Path(f"{self.db_path}{suffix}").unlink(missing_ok=True)
        self._conn = self._connect()
        self.rebuilt = True

    @property
    def connection(self) -> sqlite3.Connection:
        """The raw connection, for feature modules that own extension tables."""
        return self._conn

    def ensure_schema(self, ddl: str) -> None:
        """Run idempotent ``CREATE ... IF NOT EXISTS`` statements for an extension table.

        Statements run one at a time rather than through ``executescript``, which would
        commit whatever transaction the caller (usually the indexer) has open.
        """
        pieces = ddl.split(";")
        pending = ""
        for index, piece in enumerate(pieces):
            pending += piece if index == len(pieces) - 1 else piece + ";"
            if sqlite3.complete_statement(pending):
                if pending.strip().strip(";").strip():
                    self._conn.execute(pending)
                pending = ""
        if pending.strip():
            raise StoreError(f"incomplete DDL statement: {pending.strip()[:80]!r}")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
        except Exception:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()

    # -- files ------------------------------------------------------------

    def file_shas(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT path, sha FROM files").fetchall()
        return {r["path"]: r["sha"] for r in rows}

    def record_file(self, path: str, sha: str, language: Language) -> None:
        self._conn.execute(
            "INSERT INTO files (path, sha, language, indexed_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET sha=excluded.sha, language=excluded.language, "
            "indexed_at=excluded.indexed_at",
            (path, sha, language, time.time()),
        )

    def forget_file(self, path: str) -> None:
        """Drop a file and everything parsed out of it."""
        qnames = [
            r["qname"]
            for r in self._conn.execute("SELECT qname FROM nodes WHERE path = ?", (path,))
        ]
        for qname in qnames:
            self._conn.execute("DELETE FROM nodes_fts WHERE qname = ?", (qname,))
        self._conn.execute("DELETE FROM nodes WHERE path = ?", (path,))
        self._conn.execute("DELETE FROM edges WHERE path = ?", (path,))
        self._conn.execute("DELETE FROM files WHERE path = ?", (path,))

    def indexed_paths(self) -> list[str]:
        return [r["path"] for r in self._conn.execute("SELECT path FROM files ORDER BY path")]

    # -- nodes ------------------------------------------------------------

    def add_nodes(self, nodes: Iterable[Node]) -> int:
        count = 0
        for node in nodes:
            self._conn.execute(
                "INSERT INTO nodes (qname, kind, path, line_start, line_end, language, name, "
                "signature, docstring) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(qname) DO UPDATE SET kind=excluded.kind, path=excluded.path, "
                "line_start=excluded.line_start, line_end=excluded.line_end, "
                "language=excluded.language, name=excluded.name, signature=excluded.signature, "
                "docstring=excluded.docstring",
                (
                    node.qname,
                    node.kind,
                    node.path,
                    node.line_start,
                    node.line_end,
                    node.language,
                    node.name,
                    node.signature,
                    node.docstring,
                ),
            )
            self._conn.execute("DELETE FROM nodes_fts WHERE qname = ?", (node.qname,))
            self._conn.execute(
                "INSERT INTO nodes_fts (qname, name, signature, docstring) VALUES (?, ?, ?, ?)",
                (node.qname, node.name, node.signature, node.docstring or ""),
            )
            count += 1
        return count

    def add_edges(self, edges: Iterable[Edge]) -> int:
        count = 0
        for edge in edges:
            self._conn.execute(
                "INSERT OR IGNORE INTO edges "
                "(src, dst, kind, path, line, resolved, confidence, detail) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    edge.src,
                    edge.dst,
                    edge.kind,
                    edge.path,
                    edge.line,
                    int(edge.resolved),
                    edge.confidence,
                    edge.detail,
                ),
            )
            count += 1
        return count

    def get_node(self, qname: str) -> Node | None:
        row = self._conn.execute("SELECT * FROM nodes WHERE qname = ?", (qname,)).fetchone()
        return _row_to_node(row) if row else None

    def nodes_in_file(self, path: str) -> list[Node]:
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE path = ? ORDER BY line_start", (path,)
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def nodes_named(self, name: str) -> list[Node]:
        rows = self._conn.execute("SELECT * FROM nodes WHERE name = ?", (name,)).fetchall()
        return [_row_to_node(r) for r in rows]

    def all_nodes(self, kinds: Iterable[NodeKind] | None = None) -> list[Node]:
        if kinds is None:
            rows = self._conn.execute("SELECT * FROM nodes").fetchall()
            return [_row_to_node(r) for r in rows]
        wanted = list(kinds)
        placeholders = ",".join("?" * len(wanted))
        rows = self._conn.execute(
            f"SELECT * FROM nodes WHERE kind IN ({placeholders})", wanted
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def search_nodes(self, query: str, limit: int = 25) -> list[Node]:
        """Full-text search over qname/name/signature/docstring, then substring fallback."""
        hits: list[Node] = []
        seen: set[str] = set()
        try:
            rows = self._conn.execute(
                "SELECT qname FROM nodes_fts WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?",
                (_fts_query(query), limit),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        for row in rows:
            node = self.get_node(row["qname"])
            if node and node.qname not in seen:
                seen.add(node.qname)
                hits.append(node)
        if len(hits) < limit:
            like = f"%{query}%"
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE qname LIKE ? OR name LIKE ? LIMIT ?",
                (like, like, limit),
            ).fetchall()
            for row in rows:
                node = _row_to_node(row)
                if node.qname not in seen:
                    seen.add(node.qname)
                    hits.append(node)
        return hits[:limit]

    # -- edges ------------------------------------------------------------

    def edges_from(self, src: str, kind: EdgeKind | None = None) -> list[Edge]:
        if kind is None:
            rows = self._conn.execute("SELECT * FROM edges WHERE src = ?", (src,)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE src = ? AND kind = ?", (src, kind)
            ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def edges_to(self, dst: str, kind: EdgeKind | None = None) -> list[Edge]:
        if kind is None:
            rows = self._conn.execute("SELECT * FROM edges WHERE dst = ?", (dst,)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE dst = ? AND kind = ?", (dst, kind)
            ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def all_edges(self, kinds: Iterable[EdgeKind] | None = None) -> list[Edge]:
        if kinds is None:
            rows = self._conn.execute("SELECT * FROM edges").fetchall()
        else:
            wanted = list(kinds)
            placeholders = ",".join("?" * len(wanted))
            rows = self._conn.execute(
                f"SELECT * FROM edges WHERE kind IN ({placeholders})", wanted
            ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def delete_edges(self, kind: EdgeKind) -> int:
        """Drop every edge of one kind — passes call this before recomputing theirs."""
        return self._conn.execute("DELETE FROM edges WHERE kind = ?", (kind,)).rowcount

    def delete_nodes(self, kind: NodeKind) -> int:
        """Drop every node of one kind plus the edges that touch them."""
        qnames = [
            r["qname"]
            for r in self._conn.execute("SELECT qname FROM nodes WHERE kind = ?", (kind,))
        ]
        for qname in qnames:
            self._conn.execute("DELETE FROM nodes_fts WHERE qname = ?", (qname,))
            self._conn.execute("DELETE FROM edges WHERE src = ? OR dst = ?", (qname, qname))
        self._conn.execute("DELETE FROM nodes WHERE kind = ?", (kind,))
        return len(qnames)

    def file_languages(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT path, language FROM files").fetchall()
        return {r["path"]: r["language"] for r in rows}

    def unresolved_edges(self) -> list[Edge]:
        rows = self._conn.execute("SELECT * FROM edges WHERE resolved = 0").fetchall()
        return [_row_to_edge(r) for r in rows]

    def resolve_edge(self, edge_id: int, dst: str) -> None:
        self._conn.execute("UPDATE edges SET dst = ?, resolved = 1 WHERE id = ?", (dst, edge_id))

    def edge_ids_unresolved(self) -> list[tuple[int, Edge]]:
        rows = self._conn.execute("SELECT * FROM edges WHERE resolved = 0").fetchall()
        return [(int(r["id"]), _row_to_edge(r)) for r in rows]

    def drop_unresolved_duplicates(self) -> int:
        """Remove unresolved edges whose resolved twin already exists."""
        cur = self._conn.execute(
            "DELETE FROM edges WHERE resolved = 0 AND EXISTS ("
            "  SELECT 1 FROM edges e2 WHERE e2.resolved = 1 AND e2.src = edges.src "
            "  AND e2.kind = edges.kind AND e2.line = edges.line AND e2.path = edges.path)"
        )
        return cur.rowcount

    # -- stats ------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        def scalar(sql: str) -> int:
            row = self._conn.execute(sql).fetchone()
            return int(row[0]) if row else 0

        return {
            "files": scalar("SELECT COUNT(*) FROM files"),
            "nodes": scalar("SELECT COUNT(*) FROM nodes"),
            "edges": scalar("SELECT COUNT(*) FROM edges"),
            "unresolved_edges": scalar("SELECT COUNT(*) FROM edges WHERE resolved = 0"),
        }

    def counts_by_kind(self) -> dict[str, dict[str, int]]:
        """``{"nodes": {kind: n}, "edges": {kind: n}, "languages": {lang: files}}``."""

        def grouped(sql: str) -> dict[str, int]:
            return {str(r[0]): int(r[1]) for r in self._conn.execute(sql).fetchall()}

        return {
            "nodes": grouped("SELECT kind, COUNT(*) FROM nodes GROUP BY kind ORDER BY kind"),
            "edges": grouped("SELECT kind, COUNT(*) FROM edges GROUP BY kind ORDER BY kind"),
            "languages": grouped(
                "SELECT language, COUNT(*) FROM files GROUP BY language ORDER BY language"
            ),
        }

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None


def _fts_query(raw: str) -> str:
    """Turn a human query into an FTS5 prefix query, quoting each token."""
    tokens = [t for t in "".join(c if c.isalnum() or c == "_" else " " for c in raw).split() if t]
    if not tokens:
        raise sqlite3.OperationalError("empty query")
    return " OR ".join(f'"{t}"*' for t in tokens)


def _row_to_node(row: sqlite3.Row) -> Node:
    return Node(
        qname=row["qname"],
        kind=cast(NodeKind, row["kind"]),
        path=row["path"],
        line_start=int(row["line_start"]),
        line_end=int(row["line_end"]),
        language=cast(Language, row["language"]),
        name=row["name"],
        signature=row["signature"],
        docstring=row["docstring"],
    )


def _row_to_edge(row: sqlite3.Row) -> Edge:
    return Edge(
        src=row["src"],
        dst=row["dst"],
        kind=cast(EdgeKind, row["kind"]),
        path=row["path"],
        line=int(row["line"]),
        resolved=bool(row["resolved"]),
        confidence=float(row["confidence"]),
        detail=row["detail"],
    )
