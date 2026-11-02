"""CLI commands ``cypher`` and ``schema``: the terminal side of :mod:`codegraph.tools.query`."""

from __future__ import annotations

import argparse
import json
import sys

from codegraph.commands import AddCommand
from codegraph.cypher import CypherError, CypherResult, Limits, render_value, run_cypher
from codegraph.cypher.schema import graph_schema
from codegraph.cypher.vocab import NODE_PROPERTIES
from codegraph.store import GraphStore, default_db_path

MAX_CELL = 60


def register(add: AddCommand) -> None:
    p = add("cypher", cmd_cypher, "run a read-only Cypher query")
    p.add_argument("query", help='e.g. "MATCH (f:Function) RETURN f.qname LIMIT 5"')
    p.add_argument(
        "--max-rows",
        type=int,
        default=Limits().default_rows,
        help="rows returned when the query has no LIMIT (default: %(default)s)",
    )
    add("schema", cmd_schema, "labels, relationship types, patterns and Cypher examples")


def cmd_cypher(args: argparse.Namespace) -> int:
    limits = Limits()
    if not 1 <= args.max_rows <= limits.max_rows:
        print(f"--max-rows must be between 1 and {limits.max_rows}", file=sys.stderr)
        return 2
    try:
        with GraphStore(default_db_path(args.root)) as store:
            result = run_cypher(store, args.query, limits=Limits(default_rows=args.max_rows))
    except CypherError as exc:
        print(f"cypher: {exc}", file=sys.stderr)
        if exc.excerpt:
            print(exc.excerpt, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(format_table(result))
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if result.truncated:
        print(f"(first {len(result.rows)} rows; add LIMIT or --max-rows)", file=sys.stderr)
    return 0 if result.rows else 1


def format_table(result: CypherResult) -> str:
    """Columns padded to their widest cell (cells clipped at ``MAX_CELL`` characters)."""
    cells = [[_clip(render_value(v)) for v in row] for row in result.rows]
    widths = [len(c) for c in result.columns]
    for row in cells:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row, strict=True)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(result.columns, widths, strict=True))]
    lines.append("  ".join("-" * w for w in widths))
    lines.extend("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in cells)
    lines.append(f"({len(cells)} row{'s' if len(cells) != 1 else ''})")
    return "\n".join(line.rstrip() for line in lines)


def cmd_schema(args: argparse.Namespace) -> int:
    with GraphStore(default_db_path(args.root)) as store:
        schema = graph_schema(store)
    if args.json:
        print(json.dumps(schema.as_dict(), indent=2))
        return 0 if schema.node_labels else 1
    if not schema.node_labels:
        print("the graph is empty; run `codegraph index` first", file=sys.stderr)
        return 1
    print("node labels:   " + ", ".join(f"{k}={v}" for k, v in schema.node_labels.items()))
    print("relationships: " + ", ".join(f"{k}={v}" for k, v in schema.relationship_types.items()))
    print("patterns:")
    for pattern in schema.patterns:
        print(f"  {pattern.count:>6}  {pattern.text}")
    print("node properties: " + ", ".join(NODE_PROPERTIES))
    print("relationship properties: " + ", ".join(schema.edge_properties))
    print("examples:")
    for example in schema.examples:
        print(f"  {example}")
    return 0


def _clip(text: str) -> str:
    return text if len(text) <= MAX_CELL else text[: MAX_CELL - 3] + "..."


__all__ = ["cmd_cypher", "cmd_schema", "format_table", "register"]
