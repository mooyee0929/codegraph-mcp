"""Export the graph to the formats other tools read.

:func:`load_graph` is also the shared loader for the wiki and the visualisation: it
reads every node and edge, optionally keeps only the highest-degree nodes, and picks
up community membership when an analysis pass has stored it. Community data is read
straight from SQLite (``analysis_communities`` must exist; the member table and its
columns are discovered with ``PRAGMA table_info``), so this module works with or
without that pass installed.

Every format renders every node and edge kind generically — kinds are data, not code
paths. Formats that need both ends of an edge to be declared nodes (GraphML, DOT,
Mermaid, Cypher) drop edges to unresolved or capped-away targets; JSON and CSV keep
them, flagged with ``resolved`` and ``dst_known``.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from xml.sax.saxutils import escape, quoteattr

from codegraph.models import Edge, EdgeKind, Node, NodeKind
from codegraph.store import GraphStore

FORMATS: tuple[str, ...] = ("json", "graphml", "dot", "mermaid", "csv", "cypher")
FORMAT_ALIASES = {"neo4j": "cypher", "gv": "dot", "md": "mermaid"}
MERMAID_DEFAULT_CAP = 150

COMMUNITY_TABLE = "analysis_communities"
_MEMBER_COLUMNS = ("qname", "node_qname", "node", "member", "member_qname")
_LABEL_COLUMNS = (
    "community_name",
    "name",
    "label",
    "title",
    "community_id",
    "community",
    "cluster",
    "id",
)


class ExportError(ValueError):
    """Unknown export format."""


@dataclass(frozen=True, slots=True)
class GraphData:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    communities: dict[str, str] = field(default_factory=dict)
    total_nodes: int = 0

    @property
    def truncated(self) -> bool:
        return self.total_nodes > len(self.nodes)


# -- loading ---------------------------------------------------------------------


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(r[1]) for r in conn.execute(f"PRAGMA table_info({_quote(table)})").fetchall()]


def community_membership(store: GraphStore) -> dict[str, str]:
    """``qname -> community label`` when an analysis pass stored communities, else ``{}``."""
    conn = store.connection
    tables = [
        str(r[0])
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'analysis_communit%' "
            "ORDER BY name = ? DESC, name",
            (COMMUNITY_TABLE,),
        ).fetchall()
    ]
    if COMMUNITY_TABLE not in tables:
        return {}
    names = _community_names(conn)
    for table in tables:
        columns = _columns(conn, table)
        member = next((c for c in _MEMBER_COLUMNS if c in columns), None)
        label = next((c for c in _LABEL_COLUMNS if c in columns and c != member), None)
        if member is None or label is None:
            continue
        rows = conn.execute(f"SELECT {_quote(member)}, {_quote(label)} FROM {_quote(table)}")
        return {
            str(qname): names.get(str(value), str(value))
            for qname, value in rows.fetchall()
            if qname is not None and value is not None
        }
    return {}


def _community_names(conn: sqlite3.Connection) -> dict[str, str]:
    """``id -> name`` from the communities table when membership lives elsewhere."""
    columns = _columns(conn, COMMUNITY_TABLE)
    key = next((c for c in ("id", "community_id") if c in columns), None)
    name = next((c for c in ("name", "label", "title") if c in columns), None)
    if key is None or name is None or any(c in columns for c in _MEMBER_COLUMNS):
        return {}
    rows = conn.execute(f"SELECT {_quote(key)}, {_quote(name)} FROM {_quote(COMMUNITY_TABLE)}")
    return {str(k): str(v) for k, v in rows.fetchall() if k is not None and v is not None}


def degrees(nodes: Iterable[str], edges: Iterable[Edge]) -> Counter[str]:
    known = set(nodes)
    counts: Counter[str] = Counter()
    for edge in edges:
        if edge.src in known and edge.dst in known and edge.src != edge.dst:
            counts[edge.src] += 1
            counts[edge.dst] += 1
    return counts


def load_graph(
    store: GraphStore,
    *,
    max_nodes: int | None = None,
    node_kinds: Iterable[NodeKind] | None = None,
    edge_kinds: Iterable[EdgeKind] | None = None,
) -> GraphData:
    """The whole graph, or its ``max_nodes`` highest-degree nodes and their edges."""
    nodes = store.all_nodes(kinds=node_kinds)
    edges = store.all_edges(kinds=edge_kinds)
    total = len(nodes)
    if max_nodes is not None and 0 < max_nodes < total:
        degree = degrees((n.qname for n in nodes), edges)
        nodes = sorted(nodes, key=lambda n: (-degree[n.qname], n.qname))[:max_nodes]
    kept = {n.qname for n in nodes}
    return GraphData(
        nodes=tuple(sorted(nodes, key=lambda n: n.qname)),
        edges=tuple(e for e in edges if e.src in kept),
        communities=community_membership(store),
        total_nodes=total,
    )


def normalise_format(fmt: str) -> str:
    name = fmt.strip().lower()
    name = FORMAT_ALIASES.get(name, name)
    if name not in FORMATS:
        raise ExportError(f"unknown format {fmt!r}; expected one of {', '.join(FORMATS)}")
    return name


# -- renderers -------------------------------------------------------------------


def _linked(data: GraphData) -> list[Edge]:
    known = {n.qname for n in data.nodes}
    return [e for e in data.edges if e.dst in known]


def render_json(data: GraphData) -> str:
    known = {n.qname for n in data.nodes}
    payload = {
        "nodes": [
            {
                "qname": n.qname,
                "kind": n.kind,
                "name": n.name,
                "path": n.path,
                "line_start": n.line_start,
                "line_end": n.line_end,
                "language": n.language,
                "signature": n.signature,
                "docstring": n.docstring,
                "community": data.communities.get(n.qname),
            }
            for n in data.nodes
        ],
        "edges": [
            {
                "src": e.src,
                "dst": e.dst,
                "kind": e.kind,
                "path": e.path,
                "line": e.line,
                "resolved": e.resolved,
                "confidence": e.confidence,
                "detail": e.detail,
                "dst_known": e.dst in known,
            }
            for e in data.edges
        ],
        "total_nodes": data.total_nodes,
        "truncated": data.truncated,
    }
    return json.dumps(payload, indent=2) + "\n"


_GRAPHML_NODE_KEYS = ("kind", "name", "path", "line_start", "line_end", "language", "signature")
_GRAPHML_EDGE_KEYS = ("kind", "path", "line", "confidence", "detail")


def render_graphml(data: GraphData) -> str:
    out = io.StringIO()
    out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    out.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
    for key in (*_GRAPHML_NODE_KEYS, "community"):
        out.write(f'  <key id="n_{key}" for="node" attr.name="{key}" attr.type="string"/>\n')
    for key in _GRAPHML_EDGE_KEYS:
        out.write(f'  <key id="e_{key}" for="edge" attr.name="{key}" attr.type="string"/>\n')
    out.write('  <graph id="codegraph" edgedefault="directed">\n')
    for node in data.nodes:
        out.write(f"    <node id={quoteattr(node.qname)}>\n")
        values = {
            "kind": node.kind,
            "name": node.name,
            "path": node.path,
            "line_start": str(node.line_start),
            "line_end": str(node.line_end),
            "language": node.language,
            "signature": node.signature,
            "community": data.communities.get(node.qname, ""),
        }
        for key, value in values.items():
            if value:
                out.write(f'      <data key="n_{key}">{escape(value)}</data>\n')
        out.write("    </node>\n")
    for index, edge in enumerate(_linked(data)):
        out.write(
            f'    <edge id="e{index}" source={quoteattr(edge.src)} target={quoteattr(edge.dst)}>\n'
        )
        edge_values = {
            "kind": edge.kind,
            "path": edge.path,
            "line": str(edge.line),
            "confidence": f"{edge.confidence:g}",
            "detail": edge.detail,
        }
        for key, value in edge_values.items():
            if value:
                out.write(f'      <data key="e_{key}">{escape(value)}</data>\n')
        out.write("    </edge>\n")
    out.write("  </graph>\n</graphml>\n")
    return out.getvalue()


_DOT_SHAPES: dict[str, str] = {
    "module": "folder",
    "class": "box",
    "function": "ellipse",
    "method": "ellipse",
    "route": "hexagon",
    "channel": "cds",
    "resource": "cylinder",
}


def _dot_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def render_dot(data: GraphData) -> str:
    lines = ["digraph codegraph {", "  rankdir=LR;", "  node [fontsize=10];"]
    for node in data.nodes:
        shape = _DOT_SHAPES.get(node.kind, "ellipse")
        lines.append(
            f"  {_dot_string(node.qname)} [label={_dot_string(node.name)}, shape={shape}, "
            f"tooltip={_dot_string(f'{node.kind} {node.path}:{node.line_start}')}];"
        )
    for edge in _linked(data):
        style = "" if edge.resolved and edge.confidence >= 1.0 else ", style=dashed"
        lines.append(
            f"  {_dot_string(edge.src)} -> {_dot_string(edge.dst)} "
            f"[label={_dot_string(edge.kind)}{style}];"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _mermaid_label(text: str) -> str:
    return text.replace('"', "#quot;").replace("<", "#lt;").replace(">", "#gt;")


_MERMAID_SHAPES: dict[str, tuple[str, str]] = {
    "module": ("[/", "/]"),
    "class": ("[", "]"),
    "route": ("{{", "}}"),
    "channel": ("([", "])"),
    "resource": ("[(", ")]"),
}


def render_mermaid(data: GraphData) -> str:
    ids = {node.qname: f"n{index}" for index, node in enumerate(data.nodes)}
    lines = ["flowchart LR"]
    for node in data.nodes:
        left, right = _MERMAID_SHAPES.get(node.kind, ("(", ")"))
        lines.append(f'  {ids[node.qname]}{left}"{_mermaid_label(node.qname)}"{right}')
    for edge in _linked(data):
        arrow = "-->" if edge.resolved else "-.->"
        lines.append(f"  {ids[edge.src]} {arrow}|{edge.kind}| {ids[edge.dst]}")
    if data.truncated:
        lines.append(f"  %% showing {len(data.nodes)} of {data.total_nodes} nodes")
    return "\n".join(lines) + "\n"


def _csv(rows: list[list[str]]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerows(rows)
    return out.getvalue()


def render_csv(data: GraphData) -> dict[str, str]:
    known = {n.qname for n in data.nodes}
    nodes = [["qname", "kind", "name", "path", "line_start", "line_end", "language", "community"]]
    nodes += [
        [
            n.qname,
            n.kind,
            n.name,
            n.path,
            str(n.line_start),
            str(n.line_end),
            n.language,
            data.communities.get(n.qname, ""),
        ]
        for n in data.nodes
    ]
    edges = [
        ["src", "dst", "kind", "path", "line", "resolved", "confidence", "detail", "dst_known"]
    ]
    edges += [
        [
            e.src,
            e.dst,
            e.kind,
            e.path,
            str(e.line),
            str(e.resolved).lower(),
            f"{e.confidence:g}",
            e.detail,
            str(e.dst in known).lower(),
        ]
        for e in data.edges
    ]
    return {"nodes.csv": _csv(nodes), "edges.csv": _csv(edges)}


def _cypher_string(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n") + "'"


def _cypher_label(kind: str) -> str:
    return "".join(part.capitalize() for part in kind.split("_")) or "Node"


def render_cypher(data: GraphData) -> str:
    """Idempotent ``MERGE`` statements; every node also carries the ``CodeNode`` label."""
    lines = [
        "CREATE CONSTRAINT codenode_qname IF NOT EXISTS FOR (n:CodeNode) REQUIRE n.qname IS UNIQUE;"
    ]
    for node in data.nodes:
        props = {
            "kind": node.kind,
            "name": node.name,
            "path": node.path,
            "language": node.language,
            "signature": node.signature,
        }
        sets = ", ".join(f"n.{k} = {_cypher_string(v)}" for k, v in props.items())
        community = data.communities.get(node.qname)
        if community is not None:
            sets += f", n.community = {_cypher_string(community)}"
        lines.append(
            f"MERGE (n:CodeNode {{qname: {_cypher_string(node.qname)}}}) "
            f"SET n:{_cypher_label(node.kind)}, {sets}, "
            f"n.line_start = {node.line_start}, n.line_end = {node.line_end};"
        )
    for edge in _linked(data):
        lines.append(
            f"MATCH (a:CodeNode {{qname: {_cypher_string(edge.src)}}}), "
            f"(b:CodeNode {{qname: {_cypher_string(edge.dst)}}}) "
            f"MERGE (a)-[r:{edge.kind.upper()} {{path: {_cypher_string(edge.path)}, "
            f"line: {edge.line}}}]->(b) "
            f"SET r.confidence = {edge.confidence:g}, r.detail = {_cypher_string(edge.detail)};"
        )
    return "\n".join(lines) + "\n"


_SINGLE_FILE: dict[str, tuple[str, Callable[[GraphData], str]]] = {
    "json": ("graph.json", render_json),
    "graphml": ("graph.graphml", render_graphml),
    "dot": ("graph.dot", render_dot),
    "mermaid": ("graph.mmd", render_mermaid),
    "cypher": ("graph.cypher", render_cypher),
}


def render(data: GraphData, fmt: str) -> dict[str, str]:
    """``{file name: content}`` for ``fmt``; CSV yields ``nodes.csv`` and ``edges.csv``."""
    name = normalise_format(fmt)
    if name == "csv":
        return render_csv(data)
    filename, renderer = _SINGLE_FILE[name]
    return {filename: renderer(data)}


__all__ = [
    "FORMATS",
    "ExportError",
    "GraphData",
    "community_membership",
    "degrees",
    "load_graph",
    "normalise_format",
    "render",
    "render_csv",
    "render_cypher",
    "render_dot",
    "render_graphml",
    "render_json",
    "render_mermaid",
]
