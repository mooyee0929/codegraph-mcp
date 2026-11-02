"""Execution flows: what runs, in what order, when an entry point fires.

A flow is a breadth-first walk of ``calls`` edges from one entry point
(:mod:`codegraph.analysis.entrypoints`), bounded in depth and size. The walk also
crosses service boundaries when other passes left the edges for it: ``http_calls`` into
a route then ``handles`` into its handler, and ``emits`` into a channel then on to every
function that ``listens_on`` it. Calling a class steps into its constructor.

Criticality (0..1) ranks flows by how much can go wrong along them::

    0.15 depth            (8+ levels deep)
    0.15 fan-out          (20+ steps reached)
    0.20 files spanned    (5+ files)
    0.15 communities      (4+ communities)
    0.20 untested steps   (share of production callables with no test edge)
    0.15 external calls   (5+ calls leaving the indexed code)

The flows pass stores the result; :func:`changed_qnames` and :func:`affected_flows`
answer "which flows does this diff touch" from it.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from codegraph.analysis.common import (
    CALLABLE_KINDS,
    ensure_tables,
    is_test_node,
    table_has_rows,
)
from codegraph.analysis.communities import community_report
from codegraph.analysis.entrypoints import (
    EntryPoint,
    cli_scripts,
    detect_entry_points,
    is_decorator_edge,
)
from codegraph.graph import GraphView
from codegraph.review import changed_lines
from codegraph.store import GraphStore

MAX_DEPTH = 8
MAX_STEPS = 200

DDL = """
CREATE TABLE IF NOT EXISTS analysis_flows (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    entry_point     TEXT NOT NULL,
    entry_kind      TEXT NOT NULL,
    reason          TEXT NOT NULL,
    depth           INTEGER NOT NULL,
    node_count      INTEGER NOT NULL,
    file_count      INTEGER NOT NULL,
    community_count INTEGER NOT NULL,
    untested_steps  INTEGER NOT NULL,
    external_calls  INTEGER NOT NULL,
    criticality     REAL NOT NULL,
    truncated       INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS analysis_flow_steps (
    flow_id  INTEGER NOT NULL,
    position INTEGER NOT NULL,
    qname    TEXT NOT NULL,
    depth    INTEGER NOT NULL,
    parent   TEXT,
    via      TEXT NOT NULL,
    PRIMARY KEY (flow_id, position)
);
CREATE INDEX IF NOT EXISTS idx_analysis_flow_steps_qname ON analysis_flow_steps (qname)
"""


@dataclass(frozen=True, slots=True)
class FlowStep:
    qname: str
    depth: int
    parent: str | None
    via: str
    """Edge kind that reached this step: ``entry``, ``calls``, ``handles``, ..."""


@dataclass(frozen=True, slots=True)
class Flow:
    id: int
    name: str
    entry_point: str
    entry_kind: str
    reason: str
    depth: int
    node_count: int
    file_count: int
    community_count: int
    untested_steps: int
    external_calls: int
    criticality: float
    truncated: bool
    steps: tuple[FlowStep, ...]


@dataclass(frozen=True, slots=True)
class AffectedFlow:
    flow: Flow
    touched: tuple[str, ...]
    """Changed qnames that lie on the flow, in flow order."""


def trace_flow(
    view: GraphView,
    entry: EntryPoint,
    *,
    membership: Mapping[str, int] | None = None,
    max_depth: int = MAX_DEPTH,
    max_steps: int = MAX_STEPS,
) -> Flow | None:
    """Walk one flow from ``entry``; None when it reaches nothing beyond itself."""
    steps = [FlowStep(entry.qname, 0, None, "entry")]
    seen = {entry.qname}
    queue: deque[tuple[str, int]] = deque([(entry.qname, 0)])
    truncated = False
    while queue and not truncated:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for target, via in _next_hops(view, current):
            if target in seen:
                continue
            if len(steps) >= max_steps:
                truncated = True
                break
            seen.add(target)
            steps.append(FlowStep(target, depth + 1, current, via))
            queue.append((target, depth + 1))
    if len(steps) < 2:
        return None
    return _describe(view, entry, tuple(steps), truncated, membership or {})


def _next_hops(view: GraphView, qname: str) -> Iterator[tuple[str, str]]:
    node = view.nodes[qname]
    if node.kind == "class":
        for ctor in ("__init__", "constructor"):
            if f"{qname}.{ctor}" in view.nodes:
                yield f"{qname}.{ctor}", "constructs"
    if node.kind == "route":
        for handler in view.successors(qname, ("handles",)):
            yield handler, "handles"
        return
    if node.kind == "channel":
        for listener in view.predecessors(qname, ("listens_on",)):
            yield listener, "listens_on"
        return
    for edge in view.edges_out(qname, ("calls", "http_calls", "emits")):
        target = view.nodes.get(edge.dst)
        if target is None or target.kind == "module" or is_test_node(target):
            continue
        if is_decorator_edge(node, edge):
            continue
        yield edge.dst, edge.kind


def _describe(
    view: GraphView,
    entry: EntryPoint,
    steps: tuple[FlowStep, ...],
    truncated: bool,
    membership: Mapping[str, int],
) -> Flow:
    nodes = [view.nodes[s.qname] for s in steps]
    files = {n.path for n in nodes if n.path}
    communities = {membership[s.qname] for s in steps if s.qname in membership}
    callables = [n for n in nodes if n.kind in CALLABLE_KINDS]
    untested = sum(1 for n in callables if not view.edges_in(n.qname, ("tests",)))
    external = {
        edge.dst
        for n in nodes
        for edge in view.edges_out(n.qname, ("calls",))
        if edge.dst not in view.nodes and not is_decorator_edge(n, edge)
    }
    depth = max(s.depth for s in steps)
    score = criticality(
        depth=depth,
        node_count=len(steps),
        file_count=len(files),
        community_count=len(communities),
        untested_share=untested / len(callables) if callables else 0.0,
        external_calls=len(external),
    )
    entry_node = view.nodes[entry.qname]
    name = entry_node.name
    if entry.kind == "route":
        name = f"{entry.reason.removeprefix('serves ')} -> {entry_node.name}"
    return Flow(
        id=0,
        name=name,
        entry_point=entry.qname,
        entry_kind=entry.kind,
        reason=entry.reason,
        depth=depth,
        node_count=len(steps),
        file_count=len(files),
        community_count=len(communities),
        untested_steps=untested,
        external_calls=len(external),
        criticality=score,
        truncated=truncated,
        steps=steps,
    )


def criticality(
    *,
    depth: int,
    node_count: int,
    file_count: int,
    community_count: int,
    untested_share: float,
    external_calls: int,
) -> float:
    """Weighted 0..1 risk of a flow; see the module docstring for the weights."""
    score = (
        0.15 * min(depth / 8.0, 1.0)
        + 0.15 * min(max(node_count - 1, 0) / 20.0, 1.0)
        + 0.20 * min(max(file_count - 1, 0) / 4.0, 1.0)
        + 0.15 * min(max(community_count - 1, 0) / 3.0, 1.0)
        + 0.20 * min(max(untested_share, 0.0), 1.0)
        + 0.15 * min(external_calls / 5.0, 1.0)
    )
    return round(score, 4)


def detect_flows(
    view: GraphView,
    *,
    root: Path | None = None,
    membership: Mapping[str, int] | None = None,
    max_depth: int = MAX_DEPTH,
    max_steps: int = MAX_STEPS,
) -> tuple[Flow, ...]:
    """Every non-trivial flow, most critical first, with ids ``1..n`` in that order."""
    traced: list[Flow] = []
    for entry in detect_entry_points(view, cli_scripts(root)):
        flow = trace_flow(
            view, entry, membership=membership, max_depth=max_depth, max_steps=max_steps
        )
        if flow is not None:
            traced.append(flow)
    traced.sort(key=lambda f: (-f.criticality, f.entry_point))
    return tuple(_with_id(flow, i) for i, flow in enumerate(traced, start=1))


def _with_id(flow: Flow, flow_id: int) -> Flow:
    return Flow(
        id=flow_id,
        name=flow.name,
        entry_point=flow.entry_point,
        entry_kind=flow.entry_kind,
        reason=flow.reason,
        depth=flow.depth,
        node_count=flow.node_count,
        file_count=flow.file_count,
        community_count=flow.community_count,
        untested_steps=flow.untested_steps,
        external_calls=flow.external_calls,
        criticality=flow.criticality,
        truncated=flow.truncated,
        steps=flow.steps,
    )


# -- persistence -------------------------------------------------------------


def save_flows(store: GraphStore, flows: tuple[Flow, ...]) -> int:
    """Replace the stored flows; returns rows written."""
    ensure_tables(store, DDL)
    conn = store.connection
    conn.execute("DELETE FROM analysis_flow_steps")
    conn.execute("DELETE FROM analysis_flows")
    conn.executemany(
        "INSERT INTO analysis_flows (id, name, entry_point, entry_kind, reason, depth, "
        "node_count, file_count, community_count, untested_steps, external_calls, "
        "criticality, truncated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                f.id,
                f.name,
                f.entry_point,
                f.entry_kind,
                f.reason,
                f.depth,
                f.node_count,
                f.file_count,
                f.community_count,
                f.untested_steps,
                f.external_calls,
                f.criticality,
                int(f.truncated),
            )
            for f in flows
        ],
    )
    step_rows = [
        (f.id, position, s.qname, s.depth, s.parent, s.via)
        for f in flows
        for position, s in enumerate(f.steps)
    ]
    conn.executemany(
        "INSERT INTO analysis_flow_steps (flow_id, position, qname, depth, parent, via) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        step_rows,
    )
    return len(flows) + len(step_rows)


def load_flows(store: GraphStore) -> tuple[Flow, ...] | None:
    """Stored flows with their steps, most critical first; None when none are stored."""
    if not table_has_rows(store, "analysis_flows"):
        return None
    conn = store.connection
    steps: dict[int, list[FlowStep]] = {}
    for row in conn.execute(
        "SELECT flow_id, qname, depth, parent, via FROM analysis_flow_steps "
        "ORDER BY flow_id, position"
    ):
        steps.setdefault(int(row["flow_id"]), []).append(
            FlowStep(
                qname=str(row["qname"]),
                depth=int(row["depth"]),
                parent=str(row["parent"]) if row["parent"] is not None else None,
                via=str(row["via"]),
            )
        )
    return tuple(
        Flow(
            id=int(row["id"]),
            name=str(row["name"]),
            entry_point=str(row["entry_point"]),
            entry_kind=str(row["entry_kind"]),
            reason=str(row["reason"]),
            depth=int(row["depth"]),
            node_count=int(row["node_count"]),
            file_count=int(row["file_count"]),
            community_count=int(row["community_count"]),
            untested_steps=int(row["untested_steps"]),
            external_calls=int(row["external_calls"]),
            criticality=float(row["criticality"]),
            truncated=bool(row["truncated"]),
            steps=tuple(steps.get(int(row["id"]), [])),
        )
        for row in conn.execute("SELECT * FROM analysis_flows ORDER BY criticality DESC, id")
    )


def flow_report(
    store: GraphStore,
    root: Path | None,
    view: GraphView | None = None,
    membership: Mapping[str, int] | None = None,
) -> tuple[Flow, ...]:
    """Stored flows when the pass has run, otherwise computed now (not saved)."""
    stored = load_flows(store)
    if stored is not None:
        return stored
    graph = view if view is not None else GraphView.load(store)
    if membership is None:
        membership = community_report(store, graph).membership
    return detect_flows(graph, root=root, membership=membership)


def find_flow(flows: tuple[Flow, ...], key: str) -> Flow | None:
    """By id, entry-point qname, exact name, or entry-point bare name when unique."""
    text = key.strip()
    if text.isdigit():
        return next((f for f in flows if f.id == int(text)), None)
    for flow in flows:
        if text in (flow.entry_point, flow.name):
            return flow
    matches = [f for f in flows if f.entry_point.rsplit(".", 1)[-1] == text]
    return matches[0] if len(matches) == 1 else None


def changed_qnames(store: GraphStore, root: Path, base: str | None = None) -> list[str]:
    """Every non-module node whose span holds a changed line, outermost first.

    Unlike :func:`codegraph.review.detect_changes` this keeps enclosing definitions,
    so an edit inside a closure still marks the function that owns it.
    """
    found: list[str] = []
    for path, lines in sorted(changed_lines(root, base).items()):
        nodes = [n for n in store.nodes_in_file(path) if n.kind != "module"]
        hit = {n.qname for n in nodes for line in lines if n.line_start <= line <= n.line_end}
        ordered = sorted(nodes, key=lambda n: (n.line_start, -n.span))
        found.extend(n.qname for n in ordered if n.qname in hit)
    return found


def affected_flows(flows: tuple[Flow, ...], qnames: list[str]) -> list[AffectedFlow]:
    """Flows with at least one step in ``qnames``, most critical first."""
    wanted = set(qnames)
    hits: list[AffectedFlow] = []
    for flow in flows:
        touched = tuple(s.qname for s in flow.steps if s.qname in wanted)
        if touched:
            hits.append(AffectedFlow(flow=flow, touched=touched))
    return sorted(hits, key=lambda a: (-a.flow.criticality, a.flow.id))


__all__ = [
    "DDL",
    "MAX_DEPTH",
    "MAX_STEPS",
    "AffectedFlow",
    "Flow",
    "FlowStep",
    "affected_flows",
    "changed_qnames",
    "criticality",
    "detect_flows",
    "find_flow",
    "flow_report",
    "load_flows",
    "save_flows",
    "trace_flow",
]
