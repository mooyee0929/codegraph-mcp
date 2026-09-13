"""Graph questions: who calls what, what breaks if I touch this, what is the shape of it.

Every function here takes an open :class:`GraphStore` and returns plain dataclasses or
lists of qnames — nothing that requires a live connection to render.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass

from codegraph.models import Edge, EdgeKind, ImpactReport, Node
from codegraph.store import GraphStore

Pattern = str
"""One of: callers_of, callees_of, imports_of, imported_by, tests_for, subclasses_of,
defines, defined_in."""

PATTERNS: tuple[Pattern, ...] = (
    "callers_of",
    "callees_of",
    "imports_of",
    "imported_by",
    "tests_for",
    "subclasses_of",
    "defines",
    "defined_in",
)

_PATTERN_TO_QUERY: dict[Pattern, tuple[EdgeKind, str]] = {
    "callers_of": ("calls", "to"),
    "callees_of": ("calls", "from"),
    "imports_of": ("imports", "from"),
    "imported_by": ("imports", "to"),
    "tests_for": ("tests", "to"),
    "subclasses_of": ("inherits", "to"),
    "defines": ("defines", "from"),
    "defined_in": ("defines", "to"),
}


def resolve_target(store: GraphStore, target: str) -> Node | None:
    """Accept a full qname, a bare name (if unique), or a ``path:line`` locator."""
    node = store.get_node(target)
    if node is not None:
        return node
    if ":" in target:
        path, _, line_text = target.rpartition(":")
        if line_text.isdigit():
            return node_at(store, path, int(line_text))
    named = store.nodes_named(target.rsplit(".", 1)[-1])
    if len(named) == 1:
        return named[0]
    if named:
        # Prefer the one whose qname ends with the whole target text.
        suffix_matches = [n for n in named if n.qname.endswith(target)]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
    return None


def node_at(store: GraphStore, path: str, line: int) -> Node | None:
    """Innermost node whose span contains ``line`` in ``path``."""
    best: Node | None = None
    for node in store.nodes_in_file(path):
        if node.line_start <= line <= node.line_end and (best is None or node.span < best.span):
            best = node
    return best


def answer_side(pattern: Pattern) -> str:
    """Which end of the returned edges holds the answer: ``"src"`` or ``"dst"``."""
    if pattern not in _PATTERN_TO_QUERY:
        raise ValueError(f"unknown pattern {pattern!r}; expected one of {', '.join(PATTERNS)}")
    return "src" if _PATTERN_TO_QUERY[pattern][1] == "to" else "dst"


def query(store: GraphStore, pattern: Pattern, target: str) -> list[Edge]:
    if pattern not in _PATTERN_TO_QUERY:
        raise ValueError(f"unknown pattern {pattern!r}; expected one of {', '.join(PATTERNS)}")
    node = resolve_target(store, target)
    qname = node.qname if node else target
    kind, direction = _PATTERN_TO_QUERY[pattern]
    return store.edges_to(qname, kind) if direction == "to" else store.edges_from(qname, kind)


def impact_radius(store: GraphStore, target: str, depth: int = 3) -> ImpactReport | None:
    """Everything that transitively depends on ``target``, up to ``depth`` hops."""
    node = resolve_target(store, target)
    if node is None:
        return None

    direct = sorted({e.src for e in store.edges_to(node.qname, "calls")})
    seen: set[str] = {node.qname}
    frontier: deque[tuple[str, int]] = deque((q, 1) for q in direct)
    transitive: list[str] = []
    while frontier:
        current, level = frontier.popleft()
        if current in seen:
            continue
        seen.add(current)
        transitive.append(current)
        if level >= depth:
            continue
        for edge in store.edges_to(current, "calls"):
            if edge.src not in seen:
                frontier.append((edge.src, level + 1))

    owning_module = node.qname if node.kind == "module" else _module_of(store, node)
    importing = (
        sorted({e.src for e in store.edges_to(owning_module, "imports")}) if owning_module else []
    )
    tests = sorted({e.src for e in store.edges_to(node.qname, "tests")})
    for caller in transitive:
        tests.extend(e.src for e in store.edges_to(caller, "tests"))

    return ImpactReport(
        target=node.qname,
        direct_callers=tuple(direct),
        transitive_callers=tuple(transitive),
        importing_modules=tuple(importing),
        covering_tests=tuple(sorted(set(tests))),
        depth=depth,
    )


def _module_of(store: GraphStore, node: Node) -> str | None:
    for candidate in store.nodes_in_file(node.path):
        if candidate.kind == "module":
            return candidate.qname
    return None


@dataclass(frozen=True, slots=True)
class HubNode:
    qname: str
    kind: str
    fan_in: int
    fan_out: int


@dataclass(frozen=True, slots=True)
class ArchitectureOverview:
    counts: dict[str, int]
    languages: dict[str, int]
    top_level_packages: dict[str, int]
    hubs: tuple[HubNode, ...]
    entry_points: tuple[str, ...]
    orphans: tuple[str, ...]
    external_imports: tuple[tuple[str, int], ...]


def architecture_overview(store: GraphStore, top: int = 10) -> ArchitectureOverview:
    nodes = store.all_nodes()
    languages: Counter[str] = Counter(n.language for n in nodes if n.kind == "module")
    packages: Counter[str] = Counter(n.qname.split(".")[0] for n in nodes if n.kind == "module")

    fan_in: Counter[str] = Counter()
    fan_out: Counter[str] = Counter()
    known = {n.qname: n for n in nodes}
    external: Counter[str] = Counter()
    for node in nodes:
        for edge in store.edges_from(node.qname):
            if edge.kind == "calls":
                fan_out[edge.src] += 1
                if edge.dst in known:
                    fan_in[edge.dst] += 1
            elif edge.kind == "imports" and edge.dst.split(".")[0] not in packages:
                external[edge.dst.split(".")[0]] += 1

    callables = [n for n in nodes if n.kind in {"function", "method"}]
    scored = sorted(callables, key=lambda n: fan_in[n.qname] + fan_out[n.qname], reverse=True)
    hubs = tuple(
        HubNode(qname=n.qname, kind=n.kind, fan_in=fan_in[n.qname], fan_out=fan_out[n.qname])
        for n in scored[:top]
        if fan_in[n.qname] + fan_out[n.qname] > 0
    )
    entry_points = tuple(
        sorted(n.qname for n in callables if fan_in[n.qname] == 0 and fan_out[n.qname] > 0)[:top]
    )
    orphans = tuple(
        sorted(
            n.qname
            for n in callables
            if fan_in[n.qname] == 0 and fan_out[n.qname] == 0 and not n.name.startswith("_")
        )[:top]
    )
    return ArchitectureOverview(
        counts=store.counts(),
        languages=dict(languages),
        top_level_packages=dict(packages.most_common()),
        hubs=hubs,
        entry_points=entry_points,
        orphans=orphans,
        external_imports=tuple(external.most_common(top)),
    )


def dead_code(store: GraphStore) -> list[Node]:
    """Callables nothing calls, tests, or subclasses. Public entry points show up too —
    treat this as a candidate list, not a verdict."""
    incoming: defaultdict[str, int] = defaultdict(int)
    for node in store.all_nodes():
        for edge in store.edges_from(node.qname):
            if edge.kind in {"calls", "tests", "inherits"}:
                incoming[edge.dst] += 1
    return sorted(
        (
            n
            for n in store.all_nodes(kinds=("function", "method", "class"))
            if incoming[n.qname] == 0
            and not n.name.startswith("_")
            and not n.name.startswith("test")
        ),
        key=lambda n: (n.path, n.line_start),
    )
