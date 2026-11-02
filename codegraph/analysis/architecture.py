"""Architecture at a glance, and what a diff does to it.

:func:`architecture` gathers what a newcomer asks first: which languages, which
packages and in what layer, where execution starts, which routes exist, which nodes
everything leans on, how the code clusters and where the clusters touch, and (when the
history pass has run) which files churn most.

Packages are directories. Each gets a layer from its cross-package call traffic,
following codebase-memory-mcp's heuristic: ``entry`` (has entry points and only calls
out), ``api`` (serves routes), ``core`` (mostly called), ``leaf`` (only called),
``internal`` otherwise.

Churn comes from the ``git_file_churn`` table that the history pass writes; when that
pass was skipped or the repository has no history, hotspots are empty.

:func:`change_impact` extends :func:`codegraph.review.detect_changes` with the flows
and communities a diff touches, its test gaps, and one overall risk level.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from codegraph.analysis.centrality import Hub, hub_nodes
from codegraph.analysis.common import CALLABLE_KINDS, is_test_node, table_exists
from codegraph.analysis.communities import Community, CommunityLink, community_report
from codegraph.analysis.entrypoints import EntryPoint, cli_scripts, detect_entry_points
from codegraph.analysis.flows import affected_flows, changed_qnames, flow_report
from codegraph.graph import GraphView
from codegraph.linker import is_test_path
from codegraph.review import detect_changes
from codegraph.store import GraphStore

HISTORY_TABLE = "git_file_churn"
ROOT_PACKAGE = "(root)"


@dataclass(frozen=True, slots=True)
class PackageInfo:
    name: str
    modules: int
    symbols: int
    fan_in: int
    fan_out: int
    layer: str
    reason: str


@dataclass(frozen=True, slots=True)
class PackageBoundary:
    src: str
    dst: str
    calls: int


@dataclass(frozen=True, slots=True)
class RouteInfo:
    route: str
    handlers: tuple[str, ...]
    callers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Hotspot:
    path: str
    changes: int
    symbols: int
    table: str


@dataclass(frozen=True, slots=True)
class Architecture:
    languages: dict[str, int]
    node_kinds: dict[str, int]
    edge_kinds: dict[str, int]
    packages: tuple[PackageInfo, ...]
    package_boundaries: tuple[PackageBoundary, ...]
    entry_points: tuple[EntryPoint, ...]
    routes: tuple[RouteInfo, ...]
    hubs: tuple[Hub, ...]
    communities: tuple[Community, ...]
    community_boundaries: tuple[CommunityLink, ...]
    modularity: float
    hotspots: tuple[Hotspot, ...]


@dataclass(frozen=True, slots=True)
class ChangedItem:
    qname: str
    kind: str
    path: str
    risk: float
    reasons: tuple[str, ...]
    tests: int


@dataclass(frozen=True, slots=True)
class FlowHit:
    flow_id: int
    name: str
    entry_point: str
    criticality: float
    touched: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CommunityHit:
    id: int
    name: str
    changed: int


@dataclass(frozen=True, slots=True)
class ChangeImpact:
    base: str | None
    changed: tuple[ChangedItem, ...]
    affected_flows: tuple[FlowHit, ...]
    communities: tuple[CommunityHit, ...]
    test_gaps: tuple[str, ...]
    risk_score: float
    risk_level: str
    """``none``, ``low``, ``medium`` or ``high``."""
    summary: str


def package_of(path: str) -> str:
    parent = PurePosixPath(path).parent.as_posix()
    return ROOT_PACKAGE if parent in {"", "."} else parent.replace("/", ".")


def classify_layer(
    fan_in: int, fan_out: int, *, has_routes: bool, has_entry_points: bool
) -> tuple[str, str]:
    if has_entry_points and fan_out > 0 and fan_in == 0:
        return "entry", "has entry points, only outbound calls"
    if has_routes:
        return "api", "serves HTTP routes"
    if fan_in > fan_out and fan_in > 2:
        return "core", f"high fan-in ({fan_in} in, {fan_out} out)"
    if fan_out == 0 and fan_in > 0:
        return "leaf", "only inbound calls"
    if fan_in == 0 and fan_out > 0:
        return "entry", "only outbound calls"
    return "internal", f"fan-in={fan_in}, fan-out={fan_out}"


def architecture(store: GraphStore, root: Path | None, *, top: int = 10) -> Architecture:
    view = GraphView.load(store)
    grouped = store.counts_by_kind()
    report = community_report(store, view)
    entries = detect_entry_points(view, cli_scripts(root))
    routes = _routes(view)
    packages, boundaries = _packages(view, entries, routes)
    return Architecture(
        languages=grouped["languages"],
        node_kinds=grouped["nodes"],
        edge_kinds=grouped["edges"],
        packages=tuple(packages),
        package_boundaries=tuple(boundaries[:top]),
        entry_points=tuple(e for e in entries if e.kind != "root")[:top] or tuple(entries[:top]),
        routes=tuple(routes),
        hubs=tuple(hub_nodes(view, top=top, membership=report.membership)),
        communities=tuple(c for c in report.communities if c.size > 1)[:top],
        community_boundaries=report.links[:top],
        modularity=report.modularity,
        hotspots=tuple(history_hotspots(store, view, top=top)),
    )


def _routes(view: GraphView) -> list[RouteInfo]:
    found: list[RouteInfo] = []
    for node in sorted(view.of_kind("route"), key=lambda n: n.qname):
        found.append(
            RouteInfo(
                route=node.qname.removeprefix("route:"),
                handlers=tuple(view.successors(node.qname, ("handles",))),
                callers=tuple(view.predecessors(node.qname, ("http_calls",))),
            )
        )
    return found


def _packages(
    view: GraphView, entries: list[EntryPoint], routes: list[RouteInfo]
) -> tuple[list[PackageInfo], list[PackageBoundary]]:
    modules: Counter[str] = Counter()
    symbols: Counter[str] = Counter()
    for node in view.nodes.values():
        if not node.path or is_test_path(node.path):
            continue
        if node.kind == "module":
            modules[package_of(node.path)] += 1
        elif node.kind in {"class", "function", "method"}:
            symbols[package_of(node.path)] += 1
    fan_in: Counter[str] = Counter()
    fan_out: Counter[str] = Counter()
    crossings: Counter[tuple[str, str]] = Counter()
    for edge in view.edges(("calls", "http_calls", "handles")):
        src, dst = view.nodes[edge.src], view.nodes[edge.dst]
        if not src.path or not dst.path or is_test_node(src) or is_test_node(dst):
            continue
        a, b = package_of(src.path), package_of(dst.path)
        if a != b:
            fan_out[a] += 1
            fan_in[b] += 1
            crossings[(a, b)] += 1
    with_entries = {
        package_of(view.nodes[e.qname].path) for e in entries if e.kind not in {"root", "module"}
    }
    with_routes = {
        package_of(view.nodes[h].path) for r in routes for h in r.handlers if view.nodes[h].path
    }
    infos: list[PackageInfo] = []
    for name in sorted(set(modules) | set(symbols)):
        layer, reason = classify_layer(
            fan_in[name],
            fan_out[name],
            has_routes=name in with_routes,
            has_entry_points=name in with_entries,
        )
        infos.append(
            PackageInfo(
                name=name,
                modules=modules[name],
                symbols=symbols[name],
                fan_in=fan_in[name],
                fan_out=fan_out[name],
                layer=layer,
                reason=reason,
            )
        )
    boundaries = [
        PackageBoundary(src=a, dst=b, calls=n)
        for (a, b), n in sorted(crossings.items(), key=lambda item: (-item[1], item[0]))
    ]
    return infos, boundaries


def history_hotspots(store: GraphStore, view: GraphView, *, top: int = 10) -> list[Hotspot]:
    """Files with the most commits according to the history pass; empty without it."""
    if not table_exists(store, HISTORY_TABLE):
        return []
    rows = store.connection.execute(
        f"SELECT path, commits FROM {HISTORY_TABLE} ORDER BY commits DESC, path LIMIT ?",
        (top,),
    ).fetchall()
    per_file: Counter[str] = Counter(
        n.path for n in view.nodes.values() if n.kind in {"class", "function", "method"}
    )
    return [
        Hotspot(
            path=str(r["path"]),
            changes=int(r["commits"]),
            symbols=per_file[str(r["path"])],
            table=HISTORY_TABLE,
        )
        for r in rows
    ]


def change_impact(store: GraphStore, root: Path, base: str | None = None) -> ChangeImpact:
    """Risk of the diff (working tree vs HEAD, or ``base...HEAD``). Raises GitError."""
    changed = detect_changes(store, root, base)
    view = GraphView.load(store)
    report = community_report(store, view)
    flows = flow_report(store, root, view, report.membership)
    hits = affected_flows(flows, changed_qnames(store, root, base))

    per_community: defaultdict[int, int] = defaultdict(int)
    for item in changed:
        cid = report.membership.get(item.node.qname)
        if cid is not None:
            per_community[cid] += 1
    communities = [
        CommunityHit(id=cid, name=c.name, changed=count)
        for cid, count in sorted(per_community.items(), key=lambda kv: (-kv[1], kv[0]))
        if (c := report.by_id(cid)) is not None
    ]
    gaps = [
        c.node.qname
        for c in changed
        if c.node.kind in CALLABLE_KINDS and not c.covering_tests and not is_test_node(c.node)
    ]
    score = risk_score(
        max_symbol_risk=max((c.risk for c in changed), default=0.0),
        test_gaps=len(gaps),
        max_flow_criticality=max((h.flow.criticality for h in hits), default=0.0),
        communities=len(communities),
    )
    level = risk_level(score, has_changes=bool(changed))
    summary = (
        f"{len(changed)} changed symbol(s), {len(hits)} affected flow(s), "
        f"{len(communities)} communit{'y' if len(communities) == 1 else 'ies'}, "
        f"{len(gaps)} untested; risk {level}"
    )
    return ChangeImpact(
        base=base,
        changed=tuple(
            ChangedItem(
                qname=c.node.qname,
                kind=c.node.kind,
                path=c.node.path,
                risk=c.risk,
                reasons=c.reasons,
                tests=len(c.covering_tests),
            )
            for c in changed
        ),
        affected_flows=tuple(
            FlowHit(
                flow_id=h.flow.id,
                name=h.flow.name,
                entry_point=h.flow.entry_point,
                criticality=h.flow.criticality,
                touched=h.touched,
            )
            for h in hits
        ),
        communities=tuple(communities),
        test_gaps=tuple(gaps),
        risk_score=score,
        risk_level=level,
        summary=summary,
    )


def risk_score(
    *, max_symbol_risk: float, test_gaps: int, max_flow_criticality: float, communities: int
) -> float:
    """0..1 blend: the riskiest symbol (``review._risk`` tops out near 8), untested
    changes, the most critical affected flow and how many communities the diff spans."""
    score = (
        0.40 * min(max_symbol_risk / 6.0, 1.0)
        + 0.20 * min(test_gaps / 5.0, 1.0)
        + 0.25 * min(max(max_flow_criticality, 0.0), 1.0)
        + 0.15 * min(max(communities - 1, 0) / 3.0, 1.0)
    )
    return round(score, 4)


def risk_level(score: float, *, has_changes: bool) -> str:
    if not has_changes:
        return "none"
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "medium"
    return "low"


__all__ = [
    "Architecture",
    "ChangeImpact",
    "ChangedItem",
    "CommunityHit",
    "FlowHit",
    "Hotspot",
    "PackageBoundary",
    "PackageInfo",
    "RouteInfo",
    "architecture",
    "change_impact",
    "classify_layer",
    "HISTORY_TABLE",
    "history_hotspots",
    "package_of",
    "risk_level",
    "risk_score",
]
