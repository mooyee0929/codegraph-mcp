"""Post-index passes: whole-repository work that runs after parsing and linking.

A pass is any module in this package that defines a module-level ``PASS`` object
satisfying :class:`GraphPass`. :func:`discover_passes` imports them all and orders them
by ``order``; the indexer runs them after :func:`codegraph.linker.link` whenever the
index changed.

Order bands, so passes that build on each other run in the right sequence:

- ``10-49`` mutate the graph: new nodes (routes, channels, resources) and derived
  edges (``handles``, ``http_calls``, ``emits``, ``similar_to``, ``co_changes``).
- ``50-89`` persist analysis results computed from the finished graph (flows,
  communities) into their own tables.
- ``90-99`` build search indexes over the final node set.

A pass must be idempotent: delete what it produced last time (``store.delete_edges``,
``store.delete_nodes`` or its own table) and recompute. It runs inside the indexer's
transaction, so it must not commit.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Protocol, runtime_checkable

from codegraph.store import GraphStore


@runtime_checkable
class GraphPass(Protocol):
    name: str
    order: int

    def run(self, root: Path, store: GraphStore) -> int:
        """Do the work; return how many rows (nodes, edges, records) it wrote."""
        ...


@dataclass(frozen=True, slots=True)
class PassResult:
    name: str
    written: int


@cache
def discover_passes() -> tuple[GraphPass, ...]:
    found: list[GraphPass] = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        candidate = getattr(module, "PASS", None)
        if candidate is None:
            continue
        if not isinstance(candidate, GraphPass):
            raise TypeError(f"{module.__name__}.PASS does not implement GraphPass")
        found.append(candidate)
    return tuple(sorted(found, key=lambda p: (p.order, p.name)))


def run_passes(
    root: Path, store: GraphStore, *, skip: frozenset[str] = frozenset()
) -> tuple[PassResult, ...]:
    results: list[PassResult] = []
    for graph_pass in discover_passes():
        if graph_pass.name in skip:
            continue
        results.append(PassResult(name=graph_pass.name, written=graph_pass.run(root, store)))
    return tuple(results)


__all__ = ["GraphPass", "PassResult", "discover_passes", "run_passes"]
