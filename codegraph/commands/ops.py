"""CLI commands for operations: watch, the repo registry, snapshots, install/uninstall."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from codegraph import install, registry, snapshot, watch
from codegraph.commands import AddCommand
from codegraph.store import GraphStore, StoreError


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, default=str))


def _when(timestamp: float | None) -> str:
    if timestamp is None:
        return "never"
    return datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


# -- watch ---------------------------------------------------------------------------


def cmd_watch(args: argparse.Namespace) -> int:
    if args.interval < 0 or args.debounce < 0:
        print("--interval and --debounce must not be negative", file=sys.stderr)
        return 2

    def report(message: str) -> None:
        if args.json:
            _print_json({"event": "index", "message": message})
        else:
            print(message, flush=True)

    report(watch.reindex(args.root).summary())
    watcher = watch.Watcher(
        root=args.root,
        on_batch=watch.make_reindexer(args.root, report),
        interval=args.interval,
        debounce=args.debounce,
    )
    if not args.json:
        print(f"watching {args.root} (Ctrl-C to stop)", file=sys.stderr)
    try:
        watcher.run(should_stop=lambda: False, max_polls=args.max_polls)
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
    return 0


# -- registry ------------------------------------------------------------------------


def cmd_register(args: argparse.Namespace) -> int:
    target = Path(args.path).expanduser() if args.path else args.root
    try:
        entry = registry.register_repo(registry.registry_path(), target, args.name)
    except registry.RegistryError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        _print_json(asdict(entry))
    else:
        print(f"registered {entry.name} -> {entry.root}")
    return 0


def cmd_unregister(args: argparse.Namespace) -> int:
    try:
        entry = registry.unregister_repo(registry.registry_path(), args.repo)
    except registry.RegistryError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if entry is None:
        print(f"no registered repo {args.repo!r}", file=sys.stderr)
        return 1
    if args.json:
        _print_json(asdict(entry))
    else:
        print(f"unregistered {entry.name} ({entry.root})")
    return 0


def cmd_repos(args: argparse.Namespace) -> int:
    try:
        entries = registry.load_registry(registry.registry_path())
    except registry.RegistryError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    statuses = [registry.repo_status(e) for e in entries]
    if args.json:
        _print_json([asdict(s) for s in statuses])
        return 0 if statuses else 1
    for status in statuses:
        if status.indexed:
            detail = (
                f"{status.files} files, {status.nodes} nodes, {status.edges} edges, "
                f"indexed {_when(status.last_indexed)}"
            )
        else:
            detail = status.problem or "not indexed"
        print(f"{status.name:<20} {status.root}\n{'':<20} {detail}")
    if not statuses:
        print(f"no repos registered in {registry.registry_path()}", file=sys.stderr)
        return 1
    return 0


# -- snapshot ------------------------------------------------------------------------


def cmd_snapshot(args: argparse.Namespace) -> int:
    try:
        if args.action == "export":
            with GraphStore.open(args.root) as store:
                exported = snapshot.export_snapshot(args.root, store)
            if args.json:
                _print_json(asdict(exported))
                return 0
            print(
                f"wrote {exported.path} ({exported.compressed_bytes} bytes, "
                f"{exported.nodes} nodes, {exported.edges} edges)"
            )
            if exported.gitattributes_updated:
                print(f"added `{snapshot.GITATTRIBUTES_LINE}` to .gitattributes")
            for hint in exported.hints:
                print(f"hint: {hint}")
            return 0
        imported = snapshot.import_snapshot(args.root)
    except snapshot.SnapshotError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (StoreError, sqlite3.Error) as exc:
        print(f"database error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        _print_json({**asdict(imported), "summary": imported.index.summary()})
        return 0
    print(f"restored {imported.nodes} nodes, {imported.edges} edges from {imported.path}")
    print(imported.index.summary())
    return 0


# -- install / uninstall ---------------------------------------------------------------


def _carry_out(plan: install.Plan, args: argparse.Namespace) -> int:
    if args.json:
        _print_json(
            {
                "dry_run": args.dry_run,
                "steps": [asdict(s) for s in plan.steps],
                "notes": list(plan.notes),
            }
        )
    else:
        prefix = "would " if args.dry_run else ""
        for line in plan.describe():
            print(f"{prefix}{line}")
    if args.dry_run or not plan.steps:
        return 0
    try:
        install.apply_plan(plan)
    except install.InstallError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not args.json:
        print("done")
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    command = tuple(args.command.split()) if args.command else ("codegraph-mcp",)
    try:
        plan = install.plan_install(args.root, hooks=args.hooks, scope=args.scope, command=command)
    except install.InstallError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return _carry_out(plan, args)


def cmd_uninstall(args: argparse.Namespace) -> int:
    try:
        plan = install.plan_uninstall(args.root)
    except install.InstallError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return _carry_out(plan, args)


def register(add: AddCommand) -> None:
    p = add("watch", cmd_watch, "re-index incrementally whenever files change")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between scans")
    p.add_argument("--debounce", type=float, default=0.5, help="quiet seconds before indexing")
    p.add_argument("--max-polls", type=int, default=None, help="stop after N scans")

    p = add("register", cmd_register, "add a repo to the shared registry ($CODEGRAPH_HOME)")
    p.add_argument("path", nargs="?", help="repository root (default: --root)")
    p.add_argument("--name", help="registry name (default: directory name)")

    p = add("unregister", cmd_unregister, "remove a repo from the shared registry")
    p.add_argument("repo", help="registered name or root path")

    add("repos", cmd_repos, "list registered repos with counts and last-indexed time")

    p = add("snapshot", cmd_snapshot, "export or import .codegraph/graph.db.gz")
    p.add_argument("action", choices=("export", "import"))

    p = add("install", cmd_install, "register the MCP server with Claude Code")
    p.add_argument("--hooks", action="store_true", help="also add a re-index PostToolUse hook")
    p.add_argument("--scope", choices=install.SCOPES, default="local", help="claude mcp scope")
    p.add_argument("--command", help="server command line (default: codegraph-mcp)")
    p.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")

    p = add("uninstall", cmd_uninstall, "remove exactly what `codegraph install` added")
    p.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")


__all__ = ["register"]
