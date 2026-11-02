"""Post-index pass: git churn tables and ``co_changes`` edges between modules.

Reads one bounded ``git log`` (see :mod:`codegraph.services.history`). Churn and
co-change rows are kept for every tracked file, indexed or not; ``co_changes`` edges
are only drawn between files that have a module node. Without git or without commits
the tables are simply emptied, and ``meta.git_history`` says why.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codegraph.models import Edge
from codegraph.services.history import churn, co_changes, read_history
from codegraph.services.storage import GIT_DDL, ensure_tables
from codegraph.store import GraphStore


@dataclass(frozen=True, slots=True)
class GitHistoryPass:
    name: str = "git_history"
    order: int = 35

    def run(self, root: Path, store: GraphStore) -> int:
        ensure_tables(store, GIT_DDL)
        connection = store.connection
        for table in ("git_commits", "git_commit_files", "git_file_churn", "git_co_changes"):
            connection.execute(f"DELETE FROM {table}")
        store.delete_edges("co_changes")
        history = read_history(root)
        store.set_meta("git_history", history.status)
        if not history.commits:
            return 0
        connection.executemany(
            "INSERT OR IGNORE INTO git_commits (sha, ts, author, subject, files) "
            "VALUES (?, ?, ?, ?, ?)",
            [(c.sha, c.timestamp, c.author, c.subject, len(c.files)) for c in history.commits],
        )
        connection.executemany(
            "INSERT INTO git_commit_files (sha, path, added, removed) VALUES (?, ?, ?, ?)",
            [(c.sha, f.path, f.added, f.removed) for c in history.commits for f in c.files],
        )
        per_file = churn(history.commits)
        connection.executemany(
            "INSERT INTO git_file_churn "
            "(path, commits, authors, first_ts, last_ts, added, removed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (c.path, c.commits, c.authors, c.first_change, c.last_change, c.added, c.removed)
                for c in per_file.values()
            ],
        )
        pairs = co_changes(history.commits)
        connection.executemany(
            "INSERT OR REPLACE INTO git_co_changes (a, b, count, ratio, last_ts) "
            "VALUES (?, ?, ?, ?, ?)",
            [(p.a, p.b, p.count, p.ratio, p.last_change) for p in pairs],
        )
        modules = {n.path: n.qname for n in store.all_nodes(kinds=("module",))}
        edges = [
            Edge(
                src=modules[p.a],
                dst=modules[p.b],
                kind="co_changes",
                path=p.a,
                line=1,
                confidence=p.ratio,
                detail=f"count={p.count} ratio={p.ratio:.2f}",
            )
            for p in pairs
            if p.a in modules and p.b in modules
        ]
        store.add_edges(edges)
        return len(history.commits) + len(edges)


PASS = GitHistoryPass()
