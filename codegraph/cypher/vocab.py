"""How Cypher names map onto codegraph's node kinds, edge kinds and properties.

Labels and relationship types are derived from :data:`codegraph.models.NODE_KINDS` and
:data:`codegraph.models.EDGE_KINDS` rather than listed by hand, so a kind added to the
models (by a new language or pass) is queryable without touching this package:
node kind ``http_route`` becomes label ``HttpRoute``, edge kind ``http_calls`` becomes
relationship type ``HTTP_CALLS``.
"""

from __future__ import annotations

from codegraph.models import CALLABLE_KINDS, EDGE_KINDS, NODE_KINDS, EdgeKind, NodeKind

NODE_PROPERTIES: tuple[str, ...] = (
    "qname",
    "name",
    "kind",
    "label",
    "path",
    "line_start",
    "line_end",
    "language",
    "signature",
    "docstring",
    "span",
    "in_degree",
    "out_degree",
    "fan_in",
    "fan_out",
)
"""Readable on every node. ``in_degree`` / ``out_degree`` count distinct neighbours over
every edge kind; ``fan_in`` / ``fan_out`` count distinct callers / callees (``CALLS``)."""

EDGE_PROPERTIES: tuple[str, ...] = (
    "kind",
    "type",
    "src",
    "dst",
    "path",
    "line",
    "resolved",
    "confidence",
    "detail",
)

PROPERTY_ALIASES: dict[str, str] = {
    "qualified_name": "qname",
    "file_path": "path",
    "file": "path",
}
"""Spellings other code-graph tools use, accepted so a query written for them still runs."""


def label_for(kind: NodeKind) -> str:
    """``function`` -> ``Function``; ``http_route`` -> ``HttpRoute``."""
    return "".join(part.capitalize() for part in kind.split("_"))


def rel_type_for(kind: EdgeKind) -> str:
    return kind.upper()


NODE_LABELS: dict[str, frozenset[NodeKind]] = {
    **{label_for(kind): frozenset({kind}) for kind in NODE_KINDS},
    "Callable": frozenset(CALLABLE_KINDS),
}

REL_TYPES: dict[str, EdgeKind] = {rel_type_for(kind): kind for kind in EDGE_KINDS}

_LABELS_FOLDED = {label.lower(): kinds for label, kinds in NODE_LABELS.items()}
_REL_TYPES_FOLDED = {name.lower(): kind for name, kind in REL_TYPES.items()}


def resolve_label(name: str) -> frozenset[NodeKind] | None:
    """Kinds a label matches, case-insensitively; None for an unknown label."""
    return _LABELS_FOLDED.get(name.lower())


def resolve_rel_type(name: str) -> EdgeKind | None:
    return _REL_TYPES_FOLDED.get(name.lower())


def canonical_property(name: str) -> str:
    return PROPERTY_ALIASES.get(name, name)


def is_node_property(name: str) -> bool:
    return canonical_property(name) in NODE_PROPERTIES


def is_edge_property(name: str) -> bool:
    return name in EDGE_PROPERTIES or canonical_property(name) == "path"


__all__ = [
    "EDGE_PROPERTIES",
    "NODE_LABELS",
    "NODE_PROPERTIES",
    "PROPERTY_ALIASES",
    "REL_TYPES",
    "canonical_property",
    "is_edge_property",
    "is_node_property",
    "label_for",
    "rel_type_for",
    "resolve_label",
    "resolve_rel_type",
]
