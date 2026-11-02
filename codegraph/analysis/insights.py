"""Insights a reviewer or newcomer would want pointed out without asking.

- **Knowledge gaps**: busy code without tests, code nothing touches, communities too
  small to stand alone, modules nobody documented, files full of calls the linker could
  not resolve (dynamic dispatch, missing imports).
- **Surprising connections**: an edge between two communities that otherwise barely
  talk, or between packages far apart in the tree. One stray import between layers is
  more interesting than the hundredth call inside one.
- **Suggested questions** turn the above (plus bridges and critical flows) into review
  prompts with a priority.
- **Large functions**: definitions spanning many lines, filterable by kind, path and
  language.
"""

from __future__ import annotations

import builtins
import fnmatch
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from codegraph.analysis.centrality import Bridge, Hub, hub_nodes
from codegraph.analysis.common import (
    DEPENDENCY_KINDS,
    SYMBOL_KINDS,
    is_test_node,
    module_of_path,
)
from codegraph.analysis.communities import CommunityReport
from codegraph.analysis.entrypoints import (
    decorators_of,
    framework_decorator,
    has_implicit_decorator,
    is_dunder,
    may_override,
    unresolved_call_names,
)
from codegraph.analysis.flows import Flow
from codegraph.graph import GraphView
from codegraph.linker import COMMON_METHOD_NAMES
from codegraph.models import EdgeKind, Node, NodeKind

SURPRISE_EDGE_KINDS: tuple[EdgeKind, ...] = (
    "calls",
    "imports",
    "inherits",
    "http_calls",
    "handles",
    "emits",
    "listens_on",
)
UNTESTED_HUB_MIN_DEGREE = 4
LOW_COHESION = 0.3
_BUILTIN_NAMES = frozenset(dir(builtins))


@dataclass(frozen=True, slots=True)
class Finding:
    target: str
    path: str
    detail: str
    score: float


@dataclass(frozen=True, slots=True)
class KnowledgeGaps:
    untested_hubs: tuple[Finding, ...]
    isolated_nodes: tuple[Finding, ...]
    tiny_communities: tuple[Finding, ...]
    undocumented_modules: tuple[Finding, ...]
    unresolved_hotspots: tuple[Finding, ...]

    @property
    def total(self) -> int:
        return (
            len(self.untested_hubs)
            + len(self.isolated_nodes)
            + len(self.tiny_communities)
            + len(self.undocumented_modules)
            + len(self.unresolved_hotspots)
        )


@dataclass(frozen=True, slots=True)
class Surprise:
    src: str
    dst: str
    kind: str
    path: str
    line: int
    score: float
    reasons: tuple[str, ...]
    src_community: int | None
    dst_community: int | None


@dataclass(frozen=True, slots=True)
class Question:
    category: str
    priority: str
    """``high``, ``medium`` or ``low``."""
    question: str
    target: str


@dataclass(frozen=True, slots=True)
class LargeSymbol:
    qname: str
    kind: str
    path: str
    language: str
    line_start: int
    line_end: int
    lines: int


# -- knowledge gaps ----------------------------------------------------------


def knowledge_gaps(
    view: GraphView,
    report: CommunityReport,
    *,
    limit: int = 20,
    hub_min_degree: int = UNTESTED_HUB_MIN_DEGREE,
) -> KnowledgeGaps:
    production = [n for n in view.nodes.values() if n.kind in SYMBOL_KINDS and not is_test_node(n)]
    hubs = hub_nodes(view, top=len(view.nodes), membership=report.membership)
    untested = [
        Finding(
            target=h.qname,
            path=h.path,
            detail=f"{h.degree} connections ({h.fan_in} in, {h.fan_out} out), no tests",
            score=float(h.degree),
        )
        for h in hubs
        if h.kind in {"function", "method"}
        and h.degree >= hub_min_degree
        and not view.edges_in(h.qname, ("tests",))
    ]
    loose_names = unresolved_call_names(view)
    isolated = [
        Finding(
            target=n.qname,
            path=n.path,
            detail="no calls, imports or inheritance in either direction",
            score=float(n.span),
        )
        for n in sorted(production, key=lambda n: (n.path, n.line_start))
        if not view.neighbors(n.qname, DEPENDENCY_KINDS)
        and not _has_live_child(view, n)
        and not _reached_implicitly(view, n, loose_names)
    ]
    tiny = [
        Finding(
            target=f"community:{c.id}",
            path="",
            detail=f"{c.name}: {c.size} member(s), cohesion {c.cohesion}",
            score=float(c.size),
        )
        for c in report.communities
        if c.size < 3 and c.internal_weight + c.external_weight > 0 and c.top_members
    ]
    return KnowledgeGaps(
        untested_hubs=tuple(untested[:limit]),
        isolated_nodes=tuple(sorted(isolated, key=lambda f: -f.score)[:limit]),
        tiny_communities=tuple(tiny[:limit]),
        undocumented_modules=tuple(_undocumented_modules(view)[:limit]),
        unresolved_hotspots=tuple(unresolved_hotspots(view)[:limit]),
    )


def _reached_implicitly(view: GraphView, node: Node, loose_names: frozenset[str]) -> bool:
    """Dunders, overrides, decorated hooks and names an unresolved call may target."""
    if is_dunder(node.name) or node.name in loose_names:
        return True
    if node.kind == "method" and may_override(view, node):
        return True
    decorators = decorators_of(view, node)
    return has_implicit_decorator(decorators) or framework_decorator(decorators) is not None


def _has_live_child(view: GraphView, node: Node) -> bool:
    children = view.successors(node.qname, ("defines",))
    return any(view.neighbors(child, DEPENDENCY_KINDS) for child in children)


def _undocumented_modules(view: GraphView) -> list[Finding]:
    found: list[Finding] = []
    for module in view.of_kind("module"):
        if is_test_node(module) or module.docstring:
            continue
        symbols = [
            view.nodes[q]
            for q in view.successors(module.qname, ("defines",))
            if view.nodes[q].kind in SYMBOL_KINDS and not view.nodes[q].name.startswith("_")
        ]
        if len(symbols) < 2 or any(s.docstring for s in symbols):
            continue
        found.append(
            Finding(
                target=module.qname,
                path=module.path,
                detail=f"{len(symbols)} public definitions, none documented",
                score=float(len(symbols)),
            )
        )
    return sorted(found, key=lambda f: (-f.score, f.target))


def unresolved_hotspots(view: GraphView) -> list[Finding]:
    """Files ranked by calls and base classes the linker could not resolve.

    Builtins (``len``, ``print``) and methods every container has (``*.append``) are
    left out: they are unresolved by design, not a sign of dynamic dispatch.
    """
    per_path: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for edges in view.out_edges.values():
        for edge in edges:
            if edge.kind not in {"calls", "inherits"} or edge.resolved:
                continue
            is_method = edge.dst.startswith("*.")
            name = edge.dst.removeprefix("*.")
            if (is_method and name in COMMON_METHOD_NAMES) or (
                not is_method and name in _BUILTIN_NAMES
            ):
                continue
            per_path[edge.path][name] += 1
    found = [
        Finding(
            target=path,
            path=path,
            detail=f"{sum(names.values())} unresolved: "
            + ", ".join(name for name, _ in names.most_common(3)),
            score=float(sum(names.values())),
        )
        for path, names in per_path.items()
    ]
    return sorted(found, key=lambda f: (-f.score, f.target))


# -- surprising connections --------------------------------------------------


def surprising_connections(
    view: GraphView, report: CommunityReport, *, top: int = 15, min_score: float = 0.3
) -> list[Surprise]:
    """Cross-community and cross-package edges, most unexpected first.

    Score = 0.5 x how weak the coupling between the two communities is (this edge's
    share of their boundary) + 0.3 x package distance + 0.2 if the languages differ.
    """
    membership = report.membership
    external = {c.id: c.external_weight for c in report.communities}
    link_weight = {(link.src, link.dst): link.weight for link in report.links}
    link_edges = {(link.src, link.dst): link.edges for link in report.links}

    best: dict[tuple[str, str], Surprise] = {}
    for edge in view.edges(SURPRISE_EDGE_KINDS):
        if edge.src == edge.dst:
            continue
        src, dst = view.nodes[edge.src], view.nodes[edge.dst]
        if is_test_node(src) or is_test_node(dst):
            continue
        score = 0.0
        reasons: list[str] = []
        ca, cb = membership.get(edge.src), membership.get(edge.dst)
        if ca is not None and cb is not None and ca != cb:
            key = (ca, cb) if ca < cb else (cb, ca)
            boundary = min(external.get(ca, 0.0), external.get(cb, 0.0))
            share = link_weight.get(key, 0.0) / boundary if boundary else 1.0
            score += 0.5 * (1.0 - min(share, 1.0))
            reasons.append(
                f"links communities {ca} and {cb} ({link_edges.get(key, 0)} edge(s) between them)"
            )
        distance = package_distance(module_of_path(src.path), module_of_path(dst.path))
        if distance >= 0.5:
            score += 0.3 * distance
            reasons.append(f"crosses packages ({distance:.2f} apart)")
        if src.language != dst.language and src.path and dst.path:
            score += 0.2
            reasons.append(f"{src.language} -> {dst.language}")
        if score < min_score or not reasons:
            continue
        surprise = Surprise(
            src=edge.src,
            dst=edge.dst,
            kind=edge.kind,
            path=edge.path,
            line=edge.line,
            score=round(score, 4),
            reasons=tuple(reasons),
            src_community=ca,
            dst_community=cb,
        )
        previous = best.get((edge.src, edge.dst))
        if previous is None or surprise.score > previous.score:
            best[(edge.src, edge.dst)] = surprise
    ranked = sorted(best.values(), key=lambda s: (-s.score, s.src, s.dst))
    return ranked[: max(top, 0)]


def package_distance(a: str, b: str) -> float:
    """0 for the same module, 1 for modules with no common leading segment."""
    if not a or not b:
        return 0.0
    left, right = a.split("."), b.split(".")
    shared = 0
    for x, y in zip(left, right, strict=False):
        if x != y:
            break
        shared += 1
    return (len(left) + len(right) - 2 * shared) / (len(left) + len(right))


# -- suggested questions -----------------------------------------------------


def suggested_questions(
    view: GraphView,
    report: CommunityReport,
    *,
    hubs: Sequence[Hub],
    bridges: Sequence[Bridge],
    flows: Sequence[Flow],
    gaps: KnowledgeGaps,
    surprises: Sequence[Surprise],
    limit: int = 15,
) -> list[Question]:
    """Review prompts ranked high -> low, at most one per target."""
    questions: list[Question] = []
    names = {c.id: c.name for c in report.communities}

    for bridge in bridges[:3]:
        questions.append(
            Question(
                "bridge",
                "high",
                f"{bridge.qname} sits on the paths between {bridge.communities_linked} "
                "communities. What breaks if it changes, and is it tested end to end?",
                bridge.qname,
            )
        )
    tested = {e.dst for e in view.edges(("tests",))}
    for hub in hubs[:5]:
        if hub.qname not in tested and hub.kind in {"function", "method"}:
            questions.append(
                Question(
                    "untested_hub",
                    "high",
                    f"{hub.qname} has {hub.degree} connections but no direct tests. "
                    "Which behaviour would a test pin down first?",
                    hub.qname,
                )
            )
    for flow in flows[:3]:
        if flow.untested_steps:
            questions.append(
                Question(
                    "critical_flow",
                    "high" if flow.criticality >= 0.5 else "medium",
                    f"Flow {flow.name!r} crosses {flow.file_count} file(s) with "
                    f"{flow.untested_steps} untested step(s). Which test covers it end to end?",
                    flow.entry_point,
                )
            )

    def label(community: int | None, fallback: str) -> str:
        return names.get(community, fallback) if community is not None else fallback

    for surprise in surprises[:3]:
        a = label(surprise.src_community, "its community")
        b = label(surprise.dst_community, "another community")
        questions.append(
            Question(
                "surprising_connection",
                "medium",
                f"{surprise.src} -> {surprise.dst} ({surprise.kind}) ties {a} to {b}. "
                "Is this coupling intentional?",
                surprise.src,
            )
        )
    for community in report.communities:
        if community.size >= 5 and community.cohesion < LOW_COHESION:
            questions.append(
                Question(
                    "low_cohesion",
                    "medium",
                    f"Community {community.name!r} ({community.size} members) has cohesion "
                    f"{community.cohesion}. Should it be split along its boundaries?",
                    f"community:{community.id}",
                )
            )
    for finding in gaps.unresolved_hotspots[:2]:
        questions.append(
            Question(
                "unresolved_calls",
                "medium",
                f"{finding.path} has {finding.detail}. Dynamic dispatch or missing imports?",
                finding.target,
            )
        )
    for finding in gaps.tiny_communities[:2]:
        questions.append(
            Question(
                "tiny_community",
                "low",
                f"{finding.detail}. Should it merge into a neighbour?",
                finding.target,
            )
        )
    for finding in gaps.isolated_nodes[:2]:
        questions.append(
            Question(
                "isolated",
                "low",
                f"Nothing references {finding.target}. Dead code, or reached dynamically?",
                finding.target,
            )
        )

    order = {"high": 0, "medium": 1, "low": 2}
    seen: set[str] = set()
    unique: list[Question] = []
    for question in sorted(questions, key=lambda q: order[q.priority]):
        if question.target not in seen:
            seen.add(question.target)
            unique.append(question)
    return unique[: max(limit, 0)]


# -- large functions ---------------------------------------------------------


def large_functions(
    nodes: Iterable[Node],
    *,
    min_lines: int = 50,
    kinds: Iterable[NodeKind] = ("function", "method"),
    path: str | None = None,
    language: str | None = None,
    include_tests: bool = False,
    limit: int = 30,
) -> list[LargeSymbol]:
    """Definitions spanning at least ``min_lines`` lines, longest first.

    ``path`` is a glob (``src/*/api.py``) when it contains ``*?[``, otherwise a path
    prefix (``src/billing``).
    """
    wanted = set(kinds)
    found: list[LargeSymbol] = []
    for node in nodes:
        if node.kind not in wanted or node.span < min_lines:
            continue
        if language is not None and node.language != language:
            continue
        if path is not None and not _path_matches(node.path, path):
            continue
        if not include_tests and is_test_node(node):
            continue
        found.append(
            LargeSymbol(
                qname=node.qname,
                kind=node.kind,
                path=node.path,
                language=node.language,
                line_start=node.line_start,
                line_end=node.line_end,
                lines=node.span,
            )
        )
    found.sort(key=lambda s: (-s.lines, s.path, s.line_start))
    return found[: max(limit, 0)]


def _path_matches(candidate: str, pattern: str) -> bool:
    if any(ch in pattern for ch in "*?["):
        return fnmatch.fnmatch(candidate, pattern)
    return candidate.startswith(pattern.rstrip("/"))


__all__ = [
    "Finding",
    "KnowledgeGaps",
    "LargeSymbol",
    "Question",
    "Surprise",
    "knowledge_gaps",
    "large_functions",
    "package_distance",
    "suggested_questions",
    "surprising_connections",
    "unresolved_hotspots",
]
