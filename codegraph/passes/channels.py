"""Post-index pass: ``channel`` nodes with ``emits`` / ``listens_on`` edges.

Rebuilt from scratch on every run: channel nodes (and every edge touching them) and
``passes_channels`` rows are dropped first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codegraph.services.channels import detect_all_channels
from codegraph.services.linking import channel_graph
from codegraph.services.resolve import NodeIndex
from codegraph.services.sources import load_sources
from codegraph.services.storage import CHANNELS_DDL, ensure_tables
from codegraph.store import GraphStore


@dataclass(frozen=True, slots=True)
class ChannelsPass:
    name: str = "channels"
    order: int = 24

    def run(self, root: Path, store: GraphStore) -> int:
        ensure_tables(store, CHANNELS_DDL)
        store.delete_nodes("channel")
        store.delete_edges("emits")
        store.delete_edges("listens_on")
        store.connection.execute("DELETE FROM passes_channels")
        sites = detect_all_channels(load_sources(root))
        nodes, edges, rows = channel_graph(sites, NodeIndex.load(store))
        store.add_nodes(nodes)
        store.add_edges(edges)
        store.connection.executemany(
            "INSERT INTO passes_channels (channel, name, broker, direction, node, file, line) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(r.channel, r.name, r.broker, r.direction, r.node, r.file, r.line) for r in rows],
        )
        return len(nodes) + len(edges)


PASS = ChannelsPass()
