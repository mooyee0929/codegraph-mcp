"""CLI: ``codegraph languages`` and ``codegraph resources``.

Same data as the ``list_languages`` / ``list_resources`` MCP tools, printed as tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from codegraph.commands import AddCommand
from codegraph.parsers.inventory import language_inventory, list_resources
from codegraph.store import GraphStore, default_db_path


def cmd_languages(args: argparse.Namespace) -> int:
    with GraphStore(default_db_path(args.root)) as store:
        rows = language_inventory(store, indexed_only=args.indexed)
    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
        return 0 if rows else 1
    if not rows:
        print("no indexed files", file=sys.stderr)
        return 1
    width = max(len(r.language) for r in rows)
    for row in rows:
        matches = ", ".join((*row.suffixes, *row.filenames))
        print(f"{row.language:<{width}}  {row.files:>6}  {row.family:<14} {matches}")
    return 0


def cmd_resources(args: argparse.Namespace) -> int:
    if args.type is not None and not args.type.strip():
        print("--type must not be blank", file=sys.stderr)
        return 2
    with GraphStore(default_db_path(args.root)) as store:
        rows = list_resources(store, args.type)
    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
        return 0 if rows else 1
    if not rows:
        print("no resources found", file=sys.stderr)
        return 1
    for row in rows:
        print(f"{row.type:<24} {row.qname}  ({row.path}:{row.lines[0]})")
        for dependency in row.depends_on:
            marker = "" if dependency.resolved else "  (unresolved)"
            detail = f" [{dependency.detail}]" if dependency.detail else ""
            print(f"    -> {dependency.qname}{detail}{marker}")
    return 0


def register(add: AddCommand) -> None:
    parser = add("languages", cmd_languages, "supported languages and indexed file counts")
    parser.add_argument("--indexed", action="store_true", help="only languages with indexed files")
    parser = add("resources", cmd_resources, "infrastructure resources and their dependencies")
    parser.add_argument(
        "--type", help="resource type prefix, e.g. k8s, k8s/Deployment, compose, terraform"
    )


__all__ = ["register"]
