"""CLI commands for graph analysis; the same data as the analysis MCP tools.

``--json`` prints exactly what the matching tool returns. Exit codes follow
:mod:`codegraph.commands`: 0 found something, 1 found nothing, 2 bad input or git
failure.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

from codegraph.analysis.architecture import architecture, change_impact
from codegraph.analysis.centrality import DEFAULT_MAX_SOURCES, bridge_nodes, hub_nodes
from codegraph.analysis.common import SYMBOL_KINDS
from codegraph.analysis.communities import find_community
from codegraph.analysis.context import DEFAULT_MAX_TOKENS, minimal_context
from codegraph.analysis.flows import affected_flows, changed_qnames, find_flow
from codegraph.analysis.insights import (
    Finding,
    knowledge_gaps,
    large_functions,
    suggested_questions,
    surprising_connections,
)
from codegraph.analysis.navigation import (
    DIRECTIONS,
    MAX_TRACE_DEPTH,
    MODES,
    TRACE_DIRECTIONS,
    Direction,
    Mode,
    TraceDirection,
    TreeNode,
    trace_paths,
    traverse,
)
from codegraph.analysis.present import (
    FLOW_SORT_KEYS,
    communities_payload,
    community_detail_payload,
    flow_detail,
    flow_summary,
    flows_of,
    parse_edge_kinds,
    parse_node_kinds,
    snapshot,
    sort_flows,
)
from codegraph.commands import AddCommand
from codegraph.graph import GraphView
from codegraph.models import EDGE_KINDS, NODE_KINDS
from codegraph.queries import resolve_target
from codegraph.review import GitError
from codegraph.store import GraphStore, default_db_path
from codegraph.tools import dump


def _open(root: Path) -> GraphStore:
    return GraphStore(default_db_path(root))


def _fail(message: str, code: int) -> int:
    print(message, file=sys.stderr)
    return code


# -- communities and flows ---------------------------------------------------


def cmd_communities(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        report = snapshot(store).report
    if args.json:
        print(dump(communities_payload(report, min_size=args.min_size, limit=args.limit)))
        return 0 if report.communities else 1
    shown = [c for c in report.communities if c.size >= args.min_size][: args.limit]
    print(f"{len(report.communities)} communities, modularity {report.modularity}")
    for c in shown:
        print(f"{c.id:>4}  {c.name}  size={c.size} cohesion={c.cohesion} files={c.files}")
        if c.top_members:
            print(f"        {', '.join(c.top_members[:5])}")
    if report.links:
        print("boundaries:")
        for link in report.links[:10]:
            print(f"  {link.src} <-> {link.dst}  edges={link.edges} weight={link.weight}")
    return 0 if shown else 1


def cmd_community(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        snap = snapshot(store)
    found = find_community(snap.report, args.key)
    if found is None:
        return _fail(f"no community matches {args.key!r}", 1)
    payload = community_detail_payload(snap.view, snap.report, found, limit=args.limit)
    if args.json:
        print(dump(payload))
        return 0
    print(f"{found.id}  {found.name}")
    print(f"size={found.size} cohesion={found.cohesion} files={found.files}")
    for qname in snap.report.members.get(found.id, ())[: args.limit]:
        node = snap.view.nodes.get(qname)
        where = f"  ({node.path}:{node.line_start})" if node else ""
        print(f"  {node.kind if node else '?':<8} {qname}{where}")
    for link in snap.report.links_of(found.id):
        other = link.dst if link.src == found.id else link.src
        target = snap.report.by_id(other)
        print(f"  <-> {other} {target.name if target else ''}  edges={link.edges}")
    return 0


def cmd_flows(args: argparse.Namespace) -> int:
    if args.affected:
        return _affected_flows(args)
    with _open(args.root) as store:
        flows = flows_of(store, args.root, snapshot(store))
    ordered = sort_flows(flows, args.sort)[: args.limit]
    if args.json:
        print(dump({"total": len(flows), "flows": [flow_summary(f) for f in ordered]}))
        return 0 if flows else 1
    for f in ordered:
        print(
            f"{f.id:>4}  {f.criticality:.3f}  {f.name}  depth={f.depth} steps={f.node_count} "
            f"files={f.file_count} untested={f.untested_steps}  [{f.reason}]"
        )
    if not flows:
        return _fail("no flows: no entry point reaches anything", 1)
    return 0


def cmd_flow(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        snap = snapshot(store)
        flows = flows_of(store, args.root, snap)
    found = find_flow(flows, args.key)
    if found is None:
        return _fail(f"no flow matches {args.key!r}", 1)
    if args.json:
        print(dump(flow_detail(snap.view, found)))
        return 0
    print(f"flow {found.id}: {found.name}  criticality={found.criticality}  [{found.reason}]")
    for step in found.steps:
        node = snap.view.nodes.get(step.qname)
        where = f"  ({node.path}:{node.line_start})" if node and node.path else ""
        print(f"{'  ' * step.depth}{step.qname}  <{step.via}>{where}")
    if found.truncated:
        print("  ... (truncated)")
    return 0


# -- insights ----------------------------------------------------------------


def cmd_hubs(args: argparse.Namespace) -> int:
    kinds = parse_node_kinds(args.kind) or SYMBOL_KINDS
    with _open(args.root) as store:
        snap = snapshot(store)
    hubs = hub_nodes(snap.view, top=args.top, kinds=kinds, membership=snap.report.membership)
    if args.json:
        print(dump(hubs))
        return 0 if hubs else 1
    for hub in hubs:
        print(f"{hub.degree:>4}  {hub.qname}  in={hub.fan_in} out={hub.fan_out}  ({hub.path})")
    return 0 if hubs else _fail("no connected symbols", 1)


def cmd_bridges(args: argparse.Namespace) -> int:
    if args.max_sources < 1:
        return _fail("--max-sources must be at least 1", 2)
    with _open(args.root) as store:
        snap = snapshot(store)
    bridges = bridge_nodes(
        snap.view, top=args.top, max_sources=args.max_sources, membership=snap.report.membership
    )
    if args.json:
        print(dump(bridges))
        return 0 if bridges else 1
    for bridge in bridges:
        print(
            f"{bridge.betweenness:.4f}  {bridge.qname}  "
            f"communities={bridge.communities_linked}  ({bridge.path})"
        )
    return 0 if bridges else _fail("no node lies between others", 1)


def cmd_gaps(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        snap = snapshot(store)
    gaps = knowledge_gaps(snap.view, snap.report, limit=args.limit)
    if args.json:
        print(dump({"total": gaps.total, **asdict(gaps)}))
        return 0 if gaps.total else 1
    sections: tuple[tuple[str, tuple[Finding, ...]], ...] = (
        ("untested hubs", gaps.untested_hubs),
        ("isolated nodes", gaps.isolated_nodes),
        ("tiny communities", gaps.tiny_communities),
        ("undocumented modules", gaps.undocumented_modules),
        ("unresolved hotspots", gaps.unresolved_hotspots),
    )
    for label, findings in sections:
        print(f"{label} ({len(findings)}):")
        for finding in findings:
            print(f"  {finding.target}  {finding.detail}")
    return 0 if gaps.total else 1


def cmd_surprises(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        snap = snapshot(store)
    surprises = surprising_connections(snap.view, snap.report, top=args.top)
    if args.json:
        print(dump(surprises))
        return 0 if surprises else 1
    for s in surprises:
        print(f"{s.score:.3f}  {s.src} -{s.kind}-> {s.dst}  ({s.path}:{s.line})")
        print(f"        {'; '.join(s.reasons)}")
    return 0 if surprises else _fail("no surprising connections", 1)


def cmd_questions(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        snap = snapshot(store)
        flows = flows_of(store, args.root, snap)
    membership = snap.report.membership
    questions = suggested_questions(
        snap.view,
        snap.report,
        hubs=hub_nodes(snap.view, top=5, membership=membership),
        bridges=bridge_nodes(snap.view, top=3, membership=membership),
        flows=flows,
        gaps=knowledge_gaps(snap.view, snap.report),
        surprises=surprising_connections(snap.view, snap.report, top=3),
        limit=args.limit,
    )
    if args.json:
        print(dump(questions))
        return 0 if questions else 1
    for q in questions:
        print(f"[{q.priority}] {q.question}")
    return 0 if questions else _fail("nothing to ask", 1)


def cmd_large(args: argparse.Namespace) -> int:
    if args.min_lines < 1:
        return _fail("--min-lines must be at least 1", 2)
    kinds = parse_node_kinds(args.kind) or ("function", "method")
    with _open(args.root) as store:
        nodes = store.all_nodes(kinds=kinds)
    found = large_functions(
        nodes,
        min_lines=args.min_lines,
        kinds=kinds,
        path=args.path,
        language=args.language,
        include_tests=args.include_tests,
        limit=args.limit,
    )
    if args.json:
        print(dump(found))
        return 0 if found else 1
    for item in found:
        print(f"{item.lines:>5}  {item.kind:<8} {item.qname}  ({item.path}:{item.line_start})")
    return 0 if found else _fail(f"nothing spans {args.min_lines}+ lines", 1)


# -- navigation --------------------------------------------------------------


def cmd_context(args: argparse.Namespace) -> int:
    if not args.target and not args.task:
        return _fail("give a target, --task, or both", 2)
    with _open(args.root) as store:
        bundle = minimal_context(
            store, args.root, target=args.target, task=args.task, max_tokens=args.max_tokens
        )
    if bundle is None:
        return _fail(f"no node matches {args.target or args.task!r}", 1)
    if args.json:
        print(dump(bundle))
        return 0
    print(bundle.summary)
    symbol = bundle.target
    if symbol is not None:
        print(f"{symbol.kind} {symbol.qname}  ({symbol.location})")
        if symbol.signature:
            print(f"  {symbol.signature}")
        if symbol.doc:
            print(f"  {symbol.doc}")
        for label, items in (
            ("callers", symbol.callers),
            ("callees", symbol.callees),
            ("tests", symbol.tests),
            ("flows", symbol.flows),
        ):
            if items:
                print(f"  {label}: {', '.join(items)}")
        if symbol.community:
            print(f"  community: {symbol.community}")
    if bundle.candidates:
        print(f"other matches: {', '.join(bundle.candidates)}")
    print(f"next: {', '.join(bundle.suggested_tools)}")
    return 0 if symbol is not None else 1


def cmd_traverse(args: argparse.Namespace) -> int:
    if not 1 <= args.depth <= 10:
        return _fail("--depth must be between 1 and 10", 2)
    kinds = parse_edge_kinds(args.edge_kind)
    with _open(args.root) as store:
        node = resolve_target(store, args.start)
        if node is None:
            return _fail(f"no node matches {args.start!r}", 1)
        view = GraphView.load(store)
    result = traverse(
        view,
        node.qname,
        edge_kinds=kinds,
        direction=cast(Direction, args.direction),
        depth=args.depth,
        mode=cast(Mode, args.mode),
        max_nodes=args.max_nodes,
    )
    if args.json:
        print(dump(result))
        return 0
    _print_tree(result.root)
    print(f"{result.visited} nodes{' (truncated)' if result.truncated else ''}", file=sys.stderr)
    return 0 if result.visited > 1 else 1


def _print_tree(node: TreeNode) -> None:
    via = "" if node.via == "root" else f"  <{node.via}>"
    print(f"{'  ' * node.depth}{node.qname}{via}")
    for child in node.children:
        _print_tree(child)


def cmd_trace(args: argparse.Namespace) -> int:
    if not 1 <= args.depth <= MAX_TRACE_DEPTH:
        return _fail(f"--depth must be between 1 and {MAX_TRACE_DEPTH}", 2)
    with _open(args.root) as store:
        node = resolve_target(store, args.target)
        if node is None:
            return _fail(f"no node matches {args.target!r}", 1)
        other = resolve_target(store, args.to) if args.to else None
        if args.to and other is None:
            return _fail(f"no node matches {args.to!r}", 1)
        view = GraphView.load(store)
    result = trace_paths(
        view,
        node.qname,
        direction=cast(TraceDirection, args.direction),
        depth=args.depth,
        max_paths=args.max_paths,
        include_tests=args.include_tests,
        to=other.qname if other is not None else None,
    )
    if args.json:
        print(dump(result))
        return 0
    for label, chains in (("inbound", result.inbound), ("outbound", result.outbound)):
        if chains:
            print(f"{label} ({len(chains)}):")
            for chain in chains:
                print(f"  {' -> '.join(chain)}")
    for reach in result.callers:
        print(f"  caller hop {reach.hop} [{reach.risk}] {reach.qname}")
    if result.to is not None:
        if result.path is None:
            print(f"no path between {result.target} and {result.to}")
        else:
            print(f"path ({result.path.direction}): {' -> '.join(result.path.nodes)}")
    found = result.inbound or result.outbound or result.path is not None
    return 0 if found else 1


# -- architecture and change impact ------------------------------------------


def cmd_architecture(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        arch = architecture(store, args.root, top=args.top)
    if args.json:
        print(dump(asdict(arch)))
        return 0
    print("languages: " + ", ".join(f"{k}={v}" for k, v in arch.languages.items()))
    print("packages:")
    for pkg in arch.packages:
        print(f"  {pkg.name:<30} {pkg.layer:<8} modules={pkg.modules} symbols={pkg.symbols}")
    if arch.entry_points:
        print("entry points:")
        for entry in arch.entry_points:
            print(f"  {entry.qname}  [{entry.reason}]")
    if arch.routes:
        print("routes:")
        for route in arch.routes:
            print(f"  {route.route} -> {', '.join(route.handlers) or '?'}")
    print("hubs:")
    for hub in arch.hubs:
        print(f"  {hub.qname}  in={hub.fan_in} out={hub.fan_out}")
    print(f"communities (modularity {arch.modularity}):")
    for c in arch.communities:
        print(f"  {c.id:>3} {c.name}  size={c.size} cohesion={c.cohesion}")
    if arch.hotspots:
        print("hotspots:")
        for spot in arch.hotspots:
            print(f"  {spot.path}  changes={spot.changes} symbols={spot.symbols}")
    return 0


def cmd_change_impact(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        try:
            impact = change_impact(store, args.root, args.base)
        except GitError as exc:
            return _fail(f"git: {exc}", 2)
    if args.json:
        print(dump(asdict(impact)))
        return 0
    print(impact.summary)
    for item in impact.changed:
        reasons = f"  [{', '.join(item.reasons)}]" if item.reasons else ""
        print(f"  {item.risk:>5.1f}  {item.qname}{reasons}")
    for hit in impact.affected_flows:
        print(f"  flow {hit.flow_id} {hit.name}  criticality={hit.criticality}")
    if impact.test_gaps:
        print(f"  untested: {', '.join(impact.test_gaps)}")
    return 0


def _affected_flows(args: argparse.Namespace) -> int:
    with _open(args.root) as store:
        try:
            changed = changed_qnames(store, args.root, args.base)
        except GitError as exc:
            return _fail(f"git: {exc}", 2)
        flows = flows_of(store, args.root, snapshot(store))
    hits = affected_flows(flows, changed)
    if args.json:
        payload = [{**flow_summary(h.flow), "touched": list(h.touched)} for h in hits]
        print(dump({"changed": changed, "total": len(hits), "flows": payload}))
        return 0 if hits else 1
    for hit in hits:
        print(f"{hit.flow.id:>4}  {hit.flow.name}  touched: {', '.join(hit.touched)}")
    return 0 if hits else _fail("no flow passes through the diff", 1)


def register(add: AddCommand) -> None:
    p = add("communities", cmd_communities, "code communities and their boundaries")
    p.add_argument("--min-size", type=int, default=2)
    p.add_argument("--limit", type=int, default=30)

    p = add("community", cmd_community, "one community's members and couplings")
    p.add_argument("key", help="community id, name, name fragment or member qname")
    p.add_argument("--limit", type=int, default=50)

    p = add("flows", cmd_flows, "execution flows from entry points")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--sort", choices=FLOW_SORT_KEYS, default="criticality")
    p.add_argument(
        "--affected",
        action="store_true",
        help="only flows touched by the diff (see --base)",
    )
    p.add_argument("--base", help="git revision to diff HEAD against (with --affected)")

    p = add("flow", cmd_flow, "one flow's steps")
    p.add_argument("key", help="flow id, entry-point qname or name")

    p = add("hubs", cmd_hubs, "most connected symbols")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--kind", action="append", choices=NODE_KINDS, help="repeatable")

    p = add("bridges", cmd_bridges, "chokepoints by betweenness centrality")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--max-sources", type=int, default=DEFAULT_MAX_SOURCES)

    p = add("gaps", cmd_gaps, "knowledge gaps: untested hubs, isolated code, ...")
    p.add_argument("--limit", type=int, default=20)

    p = add("surprises", cmd_surprises, "unexpected cross-community or cross-package edges")
    p.add_argument("--top", type=int, default=15)

    p = add("questions", cmd_questions, "review questions generated from the graph")
    p.add_argument("--limit", type=int, default=15)

    p = add("large", cmd_large, "functions spanning many lines")
    p.add_argument("--min-lines", type=int, default=50)
    p.add_argument("--kind", action="append", choices=NODE_KINDS, help="repeatable")
    p.add_argument("--path", help="path prefix or glob")
    p.add_argument("--language")
    p.add_argument("--include-tests", action="store_true")
    p.add_argument("--limit", type=int, default=30)

    p = add("context", cmd_context, "compact, token-bounded context for a symbol or task")
    p.add_argument("target", nargs="?", help="qname, unique bare name, or path:line")
    p.add_argument("--task", help="free-text description of the task")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)

    p = add("traverse", cmd_traverse, "BFS/DFS tree from a symbol")
    p.add_argument("start")
    p.add_argument("--edge-kind", action="append", choices=EDGE_KINDS, help="repeatable")
    p.add_argument("--direction", choices=DIRECTIONS, default="out")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--mode", choices=MODES, default="bfs")
    p.add_argument("--max-nodes", type=int, default=200)

    p = add("trace", cmd_trace, "call chains into and out of a symbol")
    p.add_argument("target")
    p.add_argument("--to", help="second symbol: also print the shortest path to it")
    p.add_argument("--direction", choices=TRACE_DIRECTIONS, default="both")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--max-paths", type=int, default=50)
    p.add_argument("--include-tests", action="store_true")

    p = add("architecture", cmd_architecture, "packages, layers, entry points, communities")
    p.add_argument("--top", type=int, default=10)

    p = add("change-impact", cmd_change_impact, "flows, communities and test gaps of the diff")
    p.add_argument("--base", help="git revision to diff HEAD against (default: working tree)")
