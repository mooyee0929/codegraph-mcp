"""Extension tables owned by the service passes, and the few raw SQL writes they need.

Tables are created statement by statement through ``connection.execute`` rather than
:meth:`GraphStore.ensure_schema`: ``executescript`` commits any open transaction first,
and passes run inside the indexer's transaction.

- ``passes_routes``: every place a route is declared (a route node can have several).
- ``passes_channels``: every emit / listen site, including its broker.
- ``git_commits``, ``git_commit_files``, ``git_file_churn``, ``git_co_changes``: the
  bounded history read by the git pass.
- ``trace_spans``: runtime spans handed to ``ingest_traces``; they survive re-indexing
  and are re-applied to the fresh ``http_calls`` edges each time.
"""

from __future__ import annotations

from codegraph.store import GraphStore

ROUTES_DDL: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS passes_routes ("
    " route TEXT NOT NULL, method TEXT NOT NULL, path TEXT NOT NULL, framework TEXT NOT NULL,"
    " file TEXT NOT NULL, line INTEGER NOT NULL, handler TEXT, handler_ref TEXT)",
    "CREATE INDEX IF NOT EXISTS idx_passes_routes_route ON passes_routes (route)",
)

CHANNELS_DDL: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS passes_channels ("
    " channel TEXT NOT NULL, name TEXT NOT NULL, broker TEXT NOT NULL, direction TEXT NOT NULL,"
    " node TEXT NOT NULL, file TEXT NOT NULL, line INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS idx_passes_channels_channel ON passes_channels (channel)",
)

GIT_DDL: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS git_commits ("
    " sha TEXT PRIMARY KEY, ts INTEGER NOT NULL, author TEXT NOT NULL, subject TEXT NOT NULL,"
    " files INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS git_commit_files ("
    " sha TEXT NOT NULL, path TEXT NOT NULL, added INTEGER NOT NULL, removed INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS idx_git_commit_files_path ON git_commit_files (path)",
    "CREATE TABLE IF NOT EXISTS git_file_churn ("
    " path TEXT PRIMARY KEY, commits INTEGER NOT NULL, authors INTEGER NOT NULL,"
    " first_ts INTEGER NOT NULL, last_ts INTEGER NOT NULL, added INTEGER NOT NULL,"
    " removed INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS git_co_changes ("
    " a TEXT NOT NULL, b TEXT NOT NULL, count INTEGER NOT NULL, ratio REAL NOT NULL,"
    " last_ts INTEGER NOT NULL, PRIMARY KEY (a, b))",
    "CREATE INDEX IF NOT EXISTS idx_git_co_changes_b ON git_co_changes (b)",
)

TRACE_DDL: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS trace_spans ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, service TEXT NOT NULL, method TEXT NOT NULL,"
    " url TEXT NOT NULL, path TEXT NOT NULL, status INTEGER, caller TEXT, route TEXT,"
    " matched INTEGER NOT NULL DEFAULT 0, ingested_at REAL NOT NULL)",
    "CREATE INDEX IF NOT EXISTS idx_trace_spans_route ON trace_spans (route)",
)

ALL_DDL: tuple[str, ...] = (*ROUTES_DDL, *CHANNELS_DDL, *GIT_DDL, *TRACE_DDL)


def ensure_tables(store: GraphStore, statements: tuple[str, ...] = ALL_DDL) -> None:
    connection = store.connection
    for statement in statements:
        connection.execute(statement)


def mark_observed(store: GraphStore, edge_id: int) -> None:
    """Raise one ``http_calls`` edge to confidence 1.0 and tag its detail ``observed``."""
    store.connection.execute(
        "UPDATE edges SET confidence = 1.0, detail = CASE WHEN detail LIKE '%observed%' "
        "THEN detail ELSE trim(detail || ' observed') END WHERE id = ?",
        (edge_id,),
    )


__all__ = [
    "ALL_DDL",
    "CHANNELS_DDL",
    "GIT_DDL",
    "ROUTES_DDL",
    "TRACE_DDL",
    "ensure_tables",
    "mark_observed",
]
