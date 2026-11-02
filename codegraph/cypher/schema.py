"""What is in this graph, described in Cypher terms.

An agent about to write a query needs three facts it cannot guess: which labels and
relationship types actually occur (a Go repo has no ``INHERITS``), which label pairs
each relationship connects, and which properties carry data for each label. The
examples are built from the observed patterns and real names, so they run as-is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from codegraph.cypher.vocab import (
    EDGE_PROPERTIES,
    NODE_LABELS,
    NODE_PROPERTIES,
    label_for,
    rel_type_for,
)
from codegraph.models import EdgeKind, NodeKind
from codegraph.store import GraphStore

_OPTIONAL_TEXT_PROPERTIES = ("signature", "docstring")


@dataclass(frozen=True, slots=True)
class RelationshipPattern:
    source: str
    type: str
    target: str
    count: int

    @property
    def text(self) -> str:
        return f"({self.source})-[:{self.type}]->({self.target})"


@dataclass(frozen=True, slots=True)
class GraphSchema:
    node_labels: dict[str, int]
    """Label -> node count, for labels that occur."""
    relationship_types: dict[str, int]
    """Type -> edge count, including edges whose target is not a known node."""
    patterns: tuple[RelationshipPattern, ...]
    """Label pairs each type connects (edges between known nodes only), most common first."""
    node_properties: dict[str, tuple[str, ...]]
    edge_properties: tuple[str, ...]
    label_kinds: dict[str, tuple[str, ...]]
    examples: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "node_labels": self.node_labels,
            "relationship_types": self.relationship_types,
            "patterns": [{"pattern": p.text, "count": p.count} for p in self.patterns],
            "node_properties": {k: list(v) for k, v in self.node_properties.items()},
            "edge_properties": list(self.edge_properties),
            "label_kinds": {k: list(v) for k, v in self.label_kinds.items()},
            "examples": list(self.examples),
        }


def graph_schema(store: GraphStore) -> GraphSchema:
    grouped = store.counts_by_kind()
    node_counts = {label_for(cast(NodeKind, k)): n for k, n in grouped["nodes"].items()}
    edge_counts = {rel_type_for(cast(EdgeKind, k)): n for k, n in grouped["edges"].items()}
    patterns = _patterns(store)
    return GraphSchema(
        node_labels=node_counts,
        relationship_types=edge_counts,
        patterns=patterns,
        node_properties=_node_properties(store),
        edge_properties=EDGE_PROPERTIES,
        label_kinds={label: tuple(sorted(kinds)) for label, kinds in sorted(NODE_LABELS.items())},
        examples=_examples(store, patterns),
    )


def _patterns(store: GraphStore) -> tuple[RelationshipPattern, ...]:
    rows = store.connection.execute(
        "SELECT ns.kind AS src_kind, e.kind AS edge_kind, nd.kind AS dst_kind, COUNT(*) AS n "
        "FROM edges e JOIN nodes ns ON ns.qname = e.src JOIN nodes nd ON nd.qname = e.dst "
        "GROUP BY ns.kind, e.kind, nd.kind ORDER BY n DESC, e.kind, ns.kind, nd.kind"
    ).fetchall()
    return tuple(
        RelationshipPattern(
            source=label_for(cast(NodeKind, row["src_kind"])),
            type=rel_type_for(cast(EdgeKind, row["edge_kind"])),
            target=label_for(cast(NodeKind, row["dst_kind"])),
            count=int(row["n"]),
        )
        for row in rows
    )


def _node_properties(store: GraphStore) -> dict[str, tuple[str, ...]]:
    """Every property is readable on every label; only list text ones that carry data."""
    rows = store.connection.execute(
        "SELECT kind, SUM(signature != '') AS signatures, "
        "SUM(docstring IS NOT NULL AND docstring != '') AS docstrings "
        "FROM nodes GROUP BY kind ORDER BY kind"
    ).fetchall()
    found: dict[str, tuple[str, ...]] = {}
    for row in rows:
        present = {"signature": int(row["signatures"]), "docstring": int(row["docstrings"])}
        found[label_for(cast(NodeKind, row["kind"]))] = tuple(
            prop
            for prop in NODE_PROPERTIES
            if prop not in _OPTIONAL_TEXT_PROPERTIES or present[prop] > 0
        )
    return found


def _examples(store: GraphStore, patterns: tuple[RelationshipPattern, ...]) -> tuple[str, ...]:
    examples = [
        "MATCH (f:Callable) RETURN f.kind, count(*) AS n ORDER BY n DESC",
        "MATCH (f:Function) WHERE f.name =~ '.*parse.*' RETURN f.qname, f.path, f.line_start",
        "MATCH (f:Callable) WHERE f.fan_in >= 5 RETURN f.qname, f.fan_in ORDER BY f.fan_in DESC",
    ]
    hub = store.connection.execute(
        "SELECT dst, COUNT(DISTINCT src) AS n FROM edges WHERE kind = 'calls' "
        "AND dst IN (SELECT qname FROM nodes) GROUP BY dst ORDER BY n DESC, dst LIMIT 1"
    ).fetchone()
    if hub is not None:
        target = str(hub["dst"]).replace("'", "\\'")
        examples.append(f"MATCH (c)-[:CALLS]->(f {{qname: '{target}'}}) RETURN c.qname, c.path")
        examples.append(
            f"MATCH (c)-[:CALLS*1..3]->(f {{qname: '{target}'}}) RETURN count(c) AS callers"
        )
    examples.append(
        "MATCH (f:Callable) OPTIONAL MATCH (t)-[:TESTS]->(f) WITH f, count(t) AS tests "
        "WHERE tests = 0 RETURN f.qname ORDER BY f.qname LIMIT 20"
    )
    seen_types = {"CALLS", "TESTS"}
    for pattern in patterns:
        if pattern.type in seen_types:
            continue
        seen_types.add(pattern.type)
        examples.append(
            f"MATCH (a:{pattern.source})-[r:{pattern.type}]->(b:{pattern.target}) "
            "RETURN a.qname, b.qname LIMIT 10"
        )
    return tuple(examples)


__all__ = ["GraphSchema", "RelationshipPattern", "graph_schema"]
