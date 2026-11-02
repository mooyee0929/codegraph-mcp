"""CLI commands for refactoring: rename (preview), apply, suggest."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from codegraph import refactor
from codegraph.commands import AddCommand
from codegraph.indexer import index_repository
from codegraph.store import GraphStore


def cmd_rename(args: argparse.Namespace) -> int:
    with GraphStore.open(args.root) as store:
        try:
            preview = refactor.preview_rename(args.root, store, args.target, args.new_name)
        except refactor.RefactorError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args.json:
        print(json.dumps(asdict(preview), indent=2))
        return 0
    sys.stdout.write(preview.diff)
    for item in preview.skipped:
        print(f"skipped {item}", file=sys.stderr)
    print(
        f"\n{preview.edit_count} edit(s) in {len(preview.files)} file(s); "
        f"apply within 10 minutes with: codegraph apply {preview.preview_id}"
    )
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    with GraphStore.open(args.root) as store:
        try:
            applied = refactor.apply_refactor(args.root, store, args.preview_id)
        except refactor.RefactorError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        summary = None if args.no_reindex else index_repository(args.root, store).summary()
    if args.json:
        print(json.dumps({**asdict(applied), "index": summary}, indent=2))
        return 0
    print(f"changed {applied.lines_changed} line(s) in {', '.join(applied.files)}")
    if summary:
        print(summary)
    return 0


def cmd_suggest(args: argparse.Namespace) -> int:
    if args.limit < 1 or args.large_lines < 1:
        print("--limit and --large-lines must be positive", file=sys.stderr)
        return 2
    with GraphStore.open(args.root) as store:
        found = refactor.suggest_refactorings(store, limit=args.limit, large_lines=args.large_lines)
    if args.json:
        print(json.dumps([asdict(s) for s in found], indent=2))
        return 0 if found else 1
    for item in found:
        print(f"{item.kind:<7} {item.qname}  ({item.path}:{item.line})")
        print(f"        {item.reason}")
    if not found:
        print("no suggestions", file=sys.stderr)
        return 1
    return 0


def register(add: AddCommand) -> None:
    p = add("rename", cmd_rename, "preview renaming a symbol as a unified diff")
    p.add_argument("target", help="qname, unique bare name, or path:line")
    p.add_argument("new_name")

    p = add("apply", cmd_apply, "apply a refactor preview if no file changed since")
    p.add_argument("preview_id")
    p.add_argument("--no-reindex", action="store_true", help="skip the incremental re-index")

    p = add("suggest", cmd_suggest, "refactoring candidates: dedupe, move, split, remove")
    p.add_argument("--limit", type=int, default=10, help="per kind")
    p.add_argument("--large-lines", type=int, default=80, help="split threshold in lines")


__all__ = ["register"]
