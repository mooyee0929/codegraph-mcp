"""Read-side questions over what the service passes wrote.

Shared by the MCP tools (:mod:`codegraph.tools.services`, :mod:`codegraph.tools.history`)
and the CLI commands, so both answer with the same dataclasses. Every function takes
an open store, creates the extension tables if a pass never ran, and returns plain
values: an empty list or None means "nothing found", never an error.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from codegraph.linker import is_test_path
from codegraph.models import EdgeKind, Node
from codegraph.queries import resolve_target
from codegraph.services.paths import (
    ANY,
    compile_path,
    match_route,
    normalize_method,
    normalize_path,
    split_route_qname,
)
from codegraph.services.storage import ensure_tables
from codegraph.store import GraphStore

# -- routes ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouteSummary:
    qname: str
    method: str
    path: str
    frameworks: tuple[str, ...]
    file: str
    line: int
    handlers: tuple[str, ...]
    callers: int
    observed: int


@dataclass(frozen=True, slots=True)
class RouteSite:
    file: str
    line: int
    framework: str
    handler_ref: str | None


@dataclass(frozen=True, slots=True)
class CallSite:
    src: str
    file: str
    line: int
    confidence: float
    resolved: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ObservedSpan:
    service: str
    method: str
    url: str
    status: int | None
    caller: str | None


@dataclass(frozen=True, slots=True)
class RouteDetail:
    route: RouteSummary
    definitions: tuple[RouteSite, ...]
    handlers: tuple[Node, ...]
    callers: tuple[CallSite, ...]
    tests: tuple[str, ...]
    observed: tuple[ObservedSpan, ...]


def _route_frameworks(store: GraphStore) -> dict[str, list[tuple[str, int, str, str | None]]]:
    ensure_tables(store)
    found: defaultdict[str, list[tuple[str, int, str, str | None]]] = defaultdict(list)
    for row in store.connection.execute(
        "SELECT route, file, line, framework, handler_ref FROM passes_routes ORDER BY file, line"
    ):
        found[row["route"]].append(
            (row["file"], int(row["line"]), row["framework"], row["handler_ref"])
        )
    return found


def _observed_counts(store: GraphStore) -> Counter[str]:
    rows = store.connection.execute(
        "SELECT route, COUNT(*) AS n FROM trace_spans WHERE matched = 1 GROUP BY route"
    ).fetchall()
    return Counter({str(r["route"]): int(r["n"]) for r in rows})


def _summary(
    node: Node, store: GraphStore, sites: list[tuple[str, int, str, str | None]], observed: int
) -> RouteSummary:
    method, _, path = node.name.partition(" ")
    return RouteSummary(
        qname=node.qname,
        method=method,
        path=path,
        frameworks=tuple(sorted({s[2] for s in sites})),
        file=node.path,
        line=node.line_start,
        handlers=tuple(sorted({e.dst for e in store.edges_from(node.qname, "handles")})),
        callers=len({e.src for e in store.edges_to(node.qname, "http_calls")}),
        observed=observed,
    )


def list_routes(
    store: GraphStore,
    method: str | None = None,
    path: str | None = None,
    framework: str | None = None,
    limit: int = 200,
) -> list[RouteSummary]:
    sites = _route_frameworks(store)
    observed = _observed_counts(store)
    wanted_method = method.strip().upper() if method else None
    results: list[RouteSummary] = []
    for node in sorted(store.all_nodes(kinds=("route",)), key=lambda n: n.qname):
        route_method, _, route_path = node.name.partition(" ")
        if wanted_method and route_method != wanted_method:
            continue
        if path and path.lower() not in route_path.lower():
            continue
        node_sites = sites.get(node.qname, [])
        if framework and framework.lower() not in {s[2].lower() for s in node_sites}:
            continue
        results.append(_summary(node, store, node_sites, observed[node.qname]))
        if len(results) >= limit:
            break
    return results


def find_route(store: GraphStore, target: str) -> Node | None:
    """A route qname, ``METHOD /path`` (pattern or concrete), or a bare path."""
    text = target.strip()
    if not text:
        return None
    direct = store.get_node(text) or store.get_node(f"route:{text}")
    if direct is not None and direct.kind == "route":
        return direct
    method, _, rest = text.partition(" ")
    if not rest:
        method, rest = ANY, text
    if method.upper() in {"GRPC", "GRAPHQL"}:
        return store.get_node(f"route:{method.upper()} {rest.strip()}")
    routes = store.all_nodes(kinds=("route",))
    patterns = [compile_path(*parsed) for n in routes if (parsed := split_route_qname(n.qname))]
    match = match_route(normalize_method(method), normalize_path(rest), patterns)
    return store.get_node(match.qname) if match is not None else None


def route_detail(store: GraphStore, target: str) -> RouteDetail | None:
    node = find_route(store, target)
    if node is None:
        return None
    sites = _route_frameworks(store).get(node.qname, [])
    observed_rows = store.connection.execute(
        "SELECT service, method, url, status, caller FROM trace_spans WHERE route = ? "
        "ORDER BY id DESC LIMIT 20",
        (node.qname,),
    ).fetchall()
    handlers = [h for e in store.edges_from(node.qname, "handles") if (h := store.get_node(e.dst))]
    callers = [
        CallSite(e.src, e.path, e.line, e.confidence, e.resolved, e.detail)
        for e in sorted(store.edges_to(node.qname, "http_calls"), key=lambda e: (e.path, e.line))
    ]
    tests = {e.src for h in handlers for e in store.edges_to(h.qname, "tests")}
    tests |= {c.src for c in callers if is_test_path(c.file)}
    return RouteDetail(
        route=_summary(node, store, sites, len(observed_rows)),
        definitions=tuple(RouteSite(f, line, fw, ref) for f, line, fw, ref in sites),
        handlers=tuple(handlers),
        callers=tuple(callers),
        tests=tuple(sorted(tests)),
        observed=tuple(
            ObservedSpan(r["service"], r["method"], r["url"], r["status"], r["caller"])
            for r in observed_rows
        ),
    )


# -- channels -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChannelSummary:
    qname: str
    name: str
    brokers: tuple[str, ...]
    emitters: int
    listeners: int
    file: str
    line: int


@dataclass(frozen=True, slots=True)
class ChannelEnd:
    node: str
    file: str
    line: int
    broker: str
    confidence: float


@dataclass(frozen=True, slots=True)
class ChannelDetail:
    channel: ChannelSummary
    emitters: tuple[ChannelEnd, ...]
    listeners: tuple[ChannelEnd, ...]


def _channel_summary(store: GraphStore, node: Node) -> ChannelSummary:
    return ChannelSummary(
        qname=node.qname,
        name=node.name,
        brokers=tuple(b for b in node.signature.split(", ") if b),
        emitters=len({e.src for e in store.edges_to(node.qname, "emits")}),
        listeners=len({e.src for e in store.edges_to(node.qname, "listens_on")}),
        file=node.path,
        line=node.line_start,
    )


def list_channels(
    store: GraphStore, name: str | None = None, broker: str | None = None, limit: int = 200
) -> list[ChannelSummary]:
    results: list[ChannelSummary] = []
    for node in sorted(store.all_nodes(kinds=("channel",)), key=lambda n: n.qname):
        if name and name.lower() not in node.name.lower():
            continue
        if broker and broker.lower() not in node.signature.lower():
            continue
        results.append(_channel_summary(store, node))
        if len(results) >= limit:
            break
    return results


def channel_detail(store: GraphStore, target: str) -> ChannelDetail | None:
    text = target.strip()
    node = store.get_node(text) or store.get_node(f"channel:{text}")
    if node is None or node.kind != "channel":
        return None

    def ends(kind: EdgeKind) -> tuple[ChannelEnd, ...]:
        edges = sorted(store.edges_to(node.qname, kind), key=lambda e: (e.path, e.line))
        return tuple(ChannelEnd(e.src, e.path, e.line, e.detail, e.confidence) for e in edges)

    return ChannelDetail(_channel_summary(store, node), ends("emits"), ends("listens_on"))


# -- clones ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClonePairView:
    a: str
    b: str
    score: float
    a_location: str
    b_location: str


def clone_pairs(
    store: GraphStore, target: str | None = None, min_score: float = 0.8, limit: int = 50
) -> list[ClonePairView] | None:
    """All pairs, or the pairs touching ``target``; None when ``target`` is unknown."""
    if target:
        node = resolve_target(store, target)
        if node is None:
            return None
        edges = [
            *store.edges_from(node.qname, "similar_to"),
            *store.edges_to(node.qname, "similar_to"),
        ]
    else:
        edges = store.all_edges(kinds=("similar_to",))
    views: list[ClonePairView] = []
    for edge in sorted(edges, key=lambda e: (-e.confidence, e.src, e.dst)):
        if edge.confidence < min_score:
            continue
        views.append(
            ClonePairView(
                edge.src,
                edge.dst,
                edge.confidence,
                _location(store, edge.src),
                _location(store, edge.dst),
            )
        )
        if len(views) >= limit:
            break
    return views


def _location(store: GraphStore, qname: str) -> str:
    node = store.get_node(qname)
    return f"{node.path}:{node.line_start}-{node.line_end}" if node is not None else ""


# -- git history ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Hotspot:
    path: str
    module: str
    commits: int
    authors: int
    lines: int
    fan_in: int
    score: float
    last_change: int


@dataclass(frozen=True, slots=True)
class CoChangePartner:
    path: str
    module: str | None
    count: int
    ratio: float
    last_change: int


@dataclass(frozen=True, slots=True)
class FileChurnView:
    commits: int
    authors: int
    first_change: int
    last_change: int
    added: int
    removed: int


@dataclass(frozen=True, slots=True)
class CommitView:
    sha: str
    timestamp: int
    author: str
    subject: str
    added: int
    removed: int


@dataclass(frozen=True, slots=True)
class FileHistory:
    path: str
    module: str | None
    churn: FileChurnView | None
    commits: tuple[CommitView, ...]
    co_changes: tuple[CoChangePartner, ...]


def _module_fan_in(store: GraphStore) -> Counter[str]:
    """Distinct other files whose calls or imports land in each file."""
    path_of = {n.qname: n.path for n in store.all_nodes()}
    sources: defaultdict[str, set[str]] = defaultdict(set)
    for edge in store.all_edges(kinds=("calls", "imports")):
        dst_path = path_of.get(edge.dst)
        src_path = path_of.get(edge.src, edge.path)
        if dst_path is not None and src_path != dst_path:
            sources[dst_path].add(src_path)
    return Counter({path: len(found) for path, found in sources.items()})


def hotspots(store: GraphStore, limit: int = 20) -> list[Hotspot]:
    """Files ranked by ``commits * log2(1 + lines) * log2(2 + fan_in)``."""
    ensure_tables(store)
    modules = {n.path: n for n in store.all_nodes(kinds=("module",))}
    fan_in = _module_fan_in(store)
    spots: list[Hotspot] = []
    for row in store.connection.execute("SELECT * FROM git_file_churn"):
        module = modules.get(row["path"])
        if module is None:
            continue
        commits = int(row["commits"])
        lines = module.span
        incoming = fan_in[module.path]
        score = commits * math.log2(1 + lines) * math.log2(2 + incoming)
        spots.append(
            Hotspot(
                path=module.path,
                module=module.qname,
                commits=commits,
                authors=int(row["authors"]),
                lines=lines,
                fan_in=incoming,
                score=round(score, 2),
                last_change=int(row["last_ts"]),
            )
        )
    spots.sort(key=lambda h: (-h.score, h.path))
    return spots[:limit]


def _path_for(store: GraphStore, target: str) -> tuple[str, str | None] | None:
    text = target.strip()
    node = store.get_node(text)
    if node is not None:
        module = next((n for n in store.nodes_in_file(node.path) if n.kind == "module"), None)
        return node.path, module.qname if module else None
    modules = [n for n in store.nodes_in_file(text) if n.kind == "module"]
    if modules:
        return text, modules[0].qname
    known = store.connection.execute(
        "SELECT 1 FROM git_file_churn WHERE path = ? "
        "UNION SELECT 1 FROM git_commit_files WHERE path = ?",
        (text, text),
    ).fetchone()
    return (text, None) if known else None


def co_changes_for(store: GraphStore, target: str, limit: int = 20) -> list[CoChangePartner] | None:
    """Files that change with ``target`` (a path or a module qname); None if unknown."""
    ensure_tables(store)
    located = _path_for(store, target)
    if located is None:
        return None
    path = located[0]
    modules = {n.path: n.qname for n in store.all_nodes(kinds=("module",))}
    rows = store.connection.execute(
        "SELECT a, b, count, ratio, last_ts FROM git_co_changes WHERE a = ? OR b = ? "
        "ORDER BY count DESC, ratio DESC LIMIT ?",
        (path, path, limit),
    ).fetchall()
    partners: list[CoChangePartner] = []
    for row in rows:
        other = row["b"] if row["a"] == path else row["a"]
        partners.append(
            CoChangePartner(
                other,
                modules.get(other),
                int(row["count"]),
                float(row["ratio"]),
                int(row["last_ts"]),
            )
        )
    return partners


def file_history(store: GraphStore, target: str, limit: int = 20) -> FileHistory | None:
    ensure_tables(store)
    located = _path_for(store, target)
    if located is None:
        return None
    path, module = located
    churn_row = store.connection.execute(
        "SELECT * FROM git_file_churn WHERE path = ?", (path,)
    ).fetchone()
    churn_view = (
        FileChurnView(
            commits=int(churn_row["commits"]),
            authors=int(churn_row["authors"]),
            first_change=int(churn_row["first_ts"]),
            last_change=int(churn_row["last_ts"]),
            added=int(churn_row["added"]),
            removed=int(churn_row["removed"]),
        )
        if churn_row is not None
        else None
    )
    commits = store.connection.execute(
        "SELECT c.sha, c.ts, c.author, c.subject, f.added, f.removed FROM git_commit_files f "
        "JOIN git_commits c ON c.sha = f.sha WHERE f.path = ? ORDER BY c.ts DESC LIMIT ?",
        (path, limit),
    ).fetchall()
    return FileHistory(
        path=path,
        module=module,
        churn=churn_view,
        commits=tuple(
            CommitView(
                r["sha"],
                int(r["ts"]),
                r["author"],
                r["subject"],
                int(r["added"]),
                int(r["removed"]),
            )
            for r in commits
        ),
        co_changes=tuple(co_changes_for(store, path, limit=10) or ()),
    )


__all__ = [
    "ChannelDetail",
    "ChannelSummary",
    "ClonePairView",
    "CoChangePartner",
    "FileHistory",
    "Hotspot",
    "RouteDetail",
    "RouteSummary",
    "channel_detail",
    "clone_pairs",
    "co_changes_for",
    "file_history",
    "find_route",
    "hotspots",
    "list_channels",
    "list_routes",
    "route_detail",
]
