"""CLI commands over git history: ``hotspots`` and ``co-changes``."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from codegraph.commands import AddCommand
from codegraph.services import reports
from codegraph.store import GraphStore
from codegraph.tools import dump


def _date(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m-%d")


def cmd_hotspots(args: argparse.Namespace) -> int:
    if args.limit < 1:
        print("--limit must be at least 1", file=sys.stderr)
        return 2
    with GraphStore.open(args.root) as store:
        spots = reports.hotspots(store, args.limit)
        status = store.get_meta("git_history") or "git history pass has not run"
    if args.json:
        print(dump(spots))
        return 0 if spots else 1
    for spot in spots:
        print(
            f"{spot.score:>8.1f}  {spot.path}  commits={spot.commits} lines={spot.lines} "
            f"fan_in={spot.fan_in} authors={spot.authors} last={_date(spot.last_change)}"
        )
    if not spots:
        print(f"no hotspots ({status})", file=sys.stderr)
        return 1
    return 0


def cmd_co_changes(args: argparse.Namespace) -> int:
    with GraphStore.open(args.root) as store:
        partners = reports.co_changes_for(store, args.target, args.limit)
    if partners is None:
        print(f"no file or module matches {args.target!r}", file=sys.stderr)
        return 2
    if args.json:
        print(dump(partners))
        return 0 if partners else 1
    for partner in partners:
        print(
            f"{partner.count:>4}x  ratio={partner.ratio:.2f}  {partner.path}  "
            f"(last {_date(partner.last_change)})"
        )
    if not partners:
        print("no co-change partners", file=sys.stderr)
        return 1
    return 0


def register(add: AddCommand) -> None:
    p = add("hotspots", cmd_hotspots, "files ranked by churn x size x fan-in")
    p.add_argument("--limit", type=int, default=20)

    p = add("co-changes", cmd_co_changes, "files usually committed together with a file")
    p.add_argument("target", help="repo-relative path or module qname")
    p.add_argument("--limit", type=int, default=20)


__all__ = ["register"]
