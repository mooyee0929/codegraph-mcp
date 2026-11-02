"""Small helpers shared by the analysis modules.

They sit here rather than in :mod:`codegraph.graph` because they encode analysis
policy (what counts as test code, how identifiers split into words, how extension
tables are created inside a pass) rather than graph mechanics.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from codegraph.linker import is_test_path
from codegraph.models import EdgeKind, Node, NodeKind
from codegraph.parsers.base import module_qname_for
from codegraph.store import GraphStore

SYMBOL_KINDS: tuple[NodeKind, ...] = ("class", "function", "method")
CALLABLE_KINDS: frozenset[NodeKind] = frozenset({"function", "method"})
SERVICE_KINDS: frozenset[NodeKind] = frozenset({"route", "channel", "resource"})

DEPENDENCY_KINDS: tuple[EdgeKind, ...] = (
    "calls",
    "imports",
    "inherits",
    "handles",
    "http_calls",
    "emits",
    "listens_on",
    "depends_on",
    "co_changes",
)
"""Edges that mean "one side needs the other"; ``defines`` and ``tests`` are excluded."""

_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|\b|_)|[A-Z]?[a-z]+|[A-Z]+|\d+")


def is_test_node(node: Node) -> bool:
    return is_test_path(node.path)


def module_of_path(path: str) -> str:
    """The module qname a file path maps to (``pkg/a.py`` -> ``pkg.a``)."""
    return module_qname_for(path) if path else ""


def top_package(path: str) -> str:
    module = module_of_path(path)
    return module.split(".", 1)[0] if module else ""


def split_identifier(name: str) -> list[str]:
    """``parseHTTPRequest_fast`` -> ``["parse", "http", "request", "fast"]``."""
    return [w.lower() for w in _WORD.findall(name)]


def common_dotted_prefix(names: Iterable[str]) -> str:
    """Longest shared leading run of dotted segments, ``""`` when there is none."""
    iterator = iter(names)
    first = next(iterator, None)
    if first is None:
        return ""
    prefix = first.split(".")
    for name in iterator:
        parts = name.split(".")
        shared = 0
        for a, b in zip(prefix, parts, strict=False):
            if a != b:
                break
            shared += 1
        prefix = prefix[:shared]
        if not prefix:
            return ""
    return ".".join(prefix)


def table_exists(store: GraphStore, name: str) -> bool:
    row = store.connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def ensure_tables(store: GraphStore, ddl: str) -> None:
    """Create extension tables without committing.

    :meth:`GraphStore.ensure_schema` uses ``executescript``, which commits any pending
    transaction first. Passes run inside the indexer's transaction and must not commit,
    so the statements are executed one at a time instead.
    """
    for statement in ddl.split(";"):
        if statement.strip():
            store.connection.execute(statement)


def table_has_rows(store: GraphStore, name: str) -> bool:
    if not table_exists(store, name):
        return False
    row = store.connection.execute(f'SELECT 1 FROM "{name}" LIMIT 1').fetchone()
    return row is not None


def estimate_tokens(text: str) -> int:
    """Rough token count (four characters per token) for budgeting tool output."""
    return (len(text) + 3) // 4


__all__ = [
    "CALLABLE_KINDS",
    "DEPENDENCY_KINDS",
    "SERVICE_KINDS",
    "SYMBOL_KINDS",
    "common_dotted_prefix",
    "ensure_tables",
    "estimate_tokens",
    "is_test_node",
    "module_of_path",
    "split_identifier",
    "table_exists",
    "table_has_rows",
    "top_package",
]
