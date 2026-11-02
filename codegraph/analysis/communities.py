"""Communities: groups of code that mostly talk to each other.

The directed, multi-kind code graph is folded into one undirected weighted graph and
handed to :func:`codegraph.analysis.louvain.louvain`. Weights say how strongly an edge
ties two things together: a call more than an import, an import more than living in
the same file. Derived edges join in when another pass produced them: ``handles`` and
``http_calls`` tie a client to the server code behind a route, ``co_changes`` ties
modules that change together, and emitters are linked straight to the listeners of
each channel they share.

Each community is reported with a readable name (shared qname prefix plus its most
frequent identifier words), its cohesion (internal weight / all weight touching it) and
its best-connected members. Edges that cross communities are the architecture's
boundaries; :class:`CommunityLink` counts them per pair.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

from codegraph.analysis.common import (
    common_dotted_prefix,
    ensure_tables,
    is_test_node,
    module_of_path,
    split_identifier,
    table_has_rows,
)
from codegraph.analysis.louvain import louvain, modularity
from codegraph.graph import GraphView
from codegraph.models import EdgeKind, Node
from codegraph.store import GraphStore

EDGE_WEIGHTS: dict[EdgeKind, float] = {
    "calls": 1.0,
    "handles": 1.0,
    "http_calls": 1.0,
    "inherits": 0.8,
    "emits": 0.8,
    "listens_on": 0.8,
    "depends_on": 0.6,
    "imports": 0.5,
    "co_changes": 0.5,
    "defines": 0.3,
}
CHANNEL_LINK_WEIGHT = 0.5
CHANNEL_PAIR_CAP = 50
"""Emitters and listeners considered per channel, so one busy bus cannot dominate."""

_STOPWORDS = frozenset(
    {
        "get",
        "set",
        "self",
        "init",
        "new",
        "create",
        "make",
        "build",
        "from",
        "to",
        "for",
        "with",
        "the",
        "and",
        "test",
        "tests",
        "main",
        "run",
        "do",
        "is",
        "has",
        "on",
        "of",
        "in",
        "by",
        "it",
        "py",
    }
)

MODULARITY_META_KEY = "analysis_modularity"

DDL = """
CREATE TABLE IF NOT EXISTS analysis_communities (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    size            INTEGER NOT NULL,
    cohesion        REAL NOT NULL,
    internal_weight REAL NOT NULL,
    external_weight REAL NOT NULL,
    files           INTEGER NOT NULL,
    prefix          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analysis_community_members (
    qname        TEXT PRIMARY KEY,
    community_id INTEGER NOT NULL,
    rank         INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analysis_community_members
    ON analysis_community_members (community_id, rank);
CREATE TABLE IF NOT EXISTS analysis_community_links (
    src    INTEGER NOT NULL,
    dst    INTEGER NOT NULL,
    edges  INTEGER NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY (src, dst)
)
"""


@dataclass(frozen=True, slots=True)
class Community:
    id: int
    name: str
    size: int
    cohesion: float
    internal_weight: float
    external_weight: float
    files: int
    prefix: str
    top_members: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CommunityLink:
    """Coupling between two communities; ``src < dst``."""

    src: int
    dst: int
    edges: int
    weight: float


@dataclass(frozen=True, slots=True)
class CommunityReport:
    communities: tuple[Community, ...]
    members: dict[int, tuple[str, ...]]
    """Community id -> every member qname, best connected first."""
    membership: dict[str, int]
    links: tuple[CommunityLink, ...]
    """Sorted by edge count, heaviest boundary first."""
    modularity: float
    source: Literal["stored", "computed"]

    def by_id(self, community_id: int) -> Community | None:
        for community in self.communities:
            if community.id == community_id:
                return community
        return None

    def links_of(self, community_id: int) -> list[CommunityLink]:
        return [link for link in self.links if community_id in (link.src, link.dst)]


@dataclass(slots=True)
class _WeightedGraph:
    members: list[str]
    weights: dict[tuple[int, int], float]
    edge_counts: dict[tuple[int, int], int]


def build_weighted_graph(view: GraphView) -> tuple[list[str], dict[tuple[int, int], float]]:
    """The undirected graph Louvain sees: sorted member qnames and pair weights."""
    graph = _weighted_graph(view)
    return graph.members, graph.weights


def _weighted_graph(view: GraphView) -> _WeightedGraph:
    members = sorted(view.nodes)
    index = {qname: i for i, qname in enumerate(members)}
    weights: defaultdict[tuple[int, int], float] = defaultdict(float)
    counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for edge in view.edges(EDGE_WEIGHTS):
        a, b = index[edge.src], index[edge.dst]
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        weights[key] += EDGE_WEIGHTS[edge.kind] * max(0.0, min(edge.confidence, 1.0))
        counts[key] += 1
    for channel in sorted(q for q, n in view.nodes.items() if n.kind == "channel"):
        emitters = view.predecessors(channel, ("emits",))[:CHANNEL_PAIR_CAP]
        listeners = view.predecessors(channel, ("listens_on",))[:CHANNEL_PAIR_CAP]
        for emitter in emitters:
            for listener in listeners:
                a, b = index[emitter], index[listener]
                if a != b:
                    weights[(a, b) if a < b else (b, a)] += CHANNEL_LINK_WEIGHT
    return _WeightedGraph(members=members, weights=dict(weights), edge_counts=dict(counts))


def detect_communities(
    view: GraphView, *, resolution: float = 1.0, top_members: int = 8
) -> CommunityReport:
    """Run Louvain over ``view`` and describe every community it finds."""
    graph = _weighted_graph(view)
    labels = louvain(len(graph.members), graph.weights, resolution=resolution)
    score = modularity(len(graph.members), graph.weights, labels, resolution=resolution)

    grouped: defaultdict[int, list[int]] = defaultdict(list)
    for node, label in enumerate(labels):
        grouped[label].append(node)
    ordered = sorted(grouped.values(), key=lambda nodes: (-len(nodes), graph.members[nodes[0]]))
    community_of = {node: cid for cid, nodes in enumerate(ordered, start=1) for node in nodes}

    internal: defaultdict[int, float] = defaultdict(float)
    external: defaultdict[int, float] = defaultdict(float)
    node_weight: defaultdict[int, float] = defaultdict(float)
    link_weight: defaultdict[tuple[int, int], float] = defaultdict(float)
    link_edges: defaultdict[tuple[int, int], int] = defaultdict(int)
    for (a, b), weight in graph.weights.items():
        ca, cb = community_of[a], community_of[b]
        if ca == cb:
            internal[ca] += weight
            node_weight[a] += weight
            node_weight[b] += weight
            continue
        external[ca] += weight
        external[cb] += weight
        key = (ca, cb) if ca < cb else (cb, ca)
        link_weight[key] += weight
        link_edges[key] += graph.edge_counts.get((a, b), 0)

    communities: list[Community] = []
    members: dict[int, tuple[str, ...]] = {}
    for cid, nodes in enumerate(ordered, start=1):
        ranked = sorted(nodes, key=lambda n: (-node_weight[n], graph.members[n]))
        qnames = tuple(graph.members[n] for n in ranked)
        members[cid] = qnames
        member_nodes = [view.nodes[q] for q in qnames]
        prefix, name = community_name(member_nodes)
        touching = internal[cid] + external[cid]
        communities.append(
            Community(
                id=cid,
                name=name,
                size=len(qnames),
                cohesion=round(internal[cid] / touching, 4) if touching else 0.0,
                internal_weight=round(internal[cid], 4),
                external_weight=round(external[cid], 4),
                files=len({n.path for n in member_nodes if n.path}),
                prefix=prefix,
                top_members=tuple(q for q in qnames if view.nodes[q].kind not in {"module"})[
                    :top_members
                ],
            )
        )

    links = tuple(
        sorted(
            (
                CommunityLink(src=a, dst=b, edges=link_edges[(a, b)], weight=round(w, 4))
                for (a, b), w in link_weight.items()
            ),
            key=lambda link: (-link.edges, -link.weight, link.src, link.dst),
        )
    )
    return CommunityReport(
        communities=_dedupe_names(communities),
        members=members,
        membership={q: cid for cid, qnames in members.items() for q in qnames},
        links=links,
        modularity=round(score, 4),
        source="computed",
    )


def community_name(members: list[Node]) -> tuple[str, str]:
    """``(prefix, name)`` such as ``("pkg.billing", "pkg.billing (invoice, total)")``.

    The prefix is the deepest module path that at least half the members live under,
    so a cluster mostly inside ``pkg.billing`` is not just called ``pkg``. Production
    symbols are preferred over tests and modules as the source of words, so a cluster of
    tests around ``Invoice`` is still named after ``Invoice``.
    """
    symbols = [m for m in members if m.kind != "module"]
    naming = [m for m in symbols if not is_test_node(m)] or symbols or members
    modules = [module_of_path(m.path) for m in naming if m.path]
    prefix = dominant_prefix(modules) or _fallback_label(naming)
    prefix_words = {w for segment in prefix.split(".") for w in split_identifier(segment)}

    words: Counter[str] = Counter()
    for member in naming:
        weight = 2 if member.kind == "class" else 1
        for word in split_identifier(member.name):
            if len(word) > 1 and word not in _STOPWORDS and word not in prefix_words:
                words[word] += weight
    top_words = [w for w, _ in sorted(words.items(), key=lambda item: (-item[1], item[0]))[:2]]
    name = f"{prefix} ({', '.join(top_words)})" if top_words else prefix
    return prefix, name


def dominant_prefix(modules: list[str]) -> str:
    """Deepest dotted prefix shared by at least half of ``modules``; ``""`` if none."""
    if not modules:
        return ""
    shared = common_dotted_prefix(modules)
    counts: Counter[str] = Counter()
    for module in modules:
        parts = module.split(".")
        for depth in range(1, len(parts) + 1):
            counts[".".join(parts[:depth])] += 1
    majority = [p for p, n in counts.items() if n * 2 >= len(modules)]
    if not majority:
        return shared
    return min(majority, key=lambda p: (-p.count("."), -counts[p], p))


def _fallback_label(members: list[Node]) -> str:
    """Fallback label for members without a path, such as route or channel nodes."""
    if not members:
        return "cluster"
    kinds = Counter(m.kind for m in members)
    kind, _ = min(kinds.items(), key=lambda item: (-item[1], item[0]))
    return f"{kind}s"


def _dedupe_names(communities: list[Community]) -> tuple[Community, ...]:
    seen: Counter[str] = Counter()
    unique: list[Community] = []
    for community in communities:
        seen[community.name] += 1
        if seen[community.name] == 1:
            unique.append(community)
            continue
        unique.append(
            Community(
                id=community.id,
                name=f"{community.name} #{seen[community.name]}",
                size=community.size,
                cohesion=community.cohesion,
                internal_weight=community.internal_weight,
                external_weight=community.external_weight,
                files=community.files,
                prefix=community.prefix,
                top_members=community.top_members,
            )
        )
    return tuple(unique)


# -- persistence -------------------------------------------------------------


def save_communities(store: GraphStore, report: CommunityReport) -> int:
    """Replace the stored communities with ``report``; returns rows written."""
    ensure_tables(store, DDL)
    conn = store.connection
    conn.execute("DELETE FROM analysis_community_links")
    conn.execute("DELETE FROM analysis_community_members")
    conn.execute("DELETE FROM analysis_communities")
    conn.executemany(
        "INSERT INTO analysis_communities (id, name, size, cohesion, internal_weight, "
        "external_weight, files, prefix) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                c.id,
                c.name,
                c.size,
                c.cohesion,
                c.internal_weight,
                c.external_weight,
                c.files,
                c.prefix,
            )
            for c in report.communities
        ],
    )
    member_rows = [
        (qname, cid, rank)
        for cid, qnames in report.members.items()
        for rank, qname in enumerate(qnames)
    ]
    conn.executemany(
        "INSERT INTO analysis_community_members (qname, community_id, rank) VALUES (?, ?, ?)",
        member_rows,
    )
    conn.executemany(
        "INSERT INTO analysis_community_links (src, dst, edges, weight) VALUES (?, ?, ?, ?)",
        [(link.src, link.dst, link.edges, link.weight) for link in report.links],
    )
    store.set_meta(MODULARITY_META_KEY, str(report.modularity))
    return len(report.communities) + len(member_rows) + len(report.links)


def load_communities(store: GraphStore, *, top_members: int = 8) -> CommunityReport | None:
    """The communities the last index persisted, or None when there are none."""
    if not table_has_rows(store, "analysis_communities"):
        return None
    conn = store.connection
    members: defaultdict[int, list[str]] = defaultdict(list)
    for row in conn.execute(
        "SELECT qname, community_id FROM analysis_community_members ORDER BY community_id, rank"
    ):
        members[int(row["community_id"])].append(str(row["qname"]))
    kinds = {
        str(row["qname"]): str(row["kind"]) for row in conn.execute("SELECT qname, kind FROM nodes")
    }
    communities = tuple(
        Community(
            id=int(row["id"]),
            name=str(row["name"]),
            size=int(row["size"]),
            cohesion=float(row["cohesion"]),
            internal_weight=float(row["internal_weight"]),
            external_weight=float(row["external_weight"]),
            files=int(row["files"]),
            prefix=str(row["prefix"]),
            top_members=tuple(
                q for q in members[int(row["id"])] if kinds.get(q, "module") != "module"
            )[:top_members],
        )
        for row in conn.execute("SELECT * FROM analysis_communities ORDER BY id")
    )
    links = tuple(
        CommunityLink(
            src=int(row["src"]),
            dst=int(row["dst"]),
            edges=int(row["edges"]),
            weight=float(row["weight"]),
        )
        for row in conn.execute(
            "SELECT * FROM analysis_community_links ORDER BY edges DESC, weight DESC, src, dst"
        )
    )
    stored = store.get_meta(MODULARITY_META_KEY)
    frozen_members = {cid: tuple(qnames) for cid, qnames in members.items()}
    return CommunityReport(
        communities=communities,
        members=frozen_members,
        membership={q: cid for cid, qnames in frozen_members.items() for q in qnames},
        links=links,
        modularity=float(stored) if stored else 0.0,
        source="stored",
    )


def community_report(store: GraphStore, view: GraphView | None = None) -> CommunityReport:
    """Stored communities when the pass has run, otherwise computed now (not saved)."""
    stored = load_communities(store)
    if stored is not None:
        return stored
    return detect_communities(view if view is not None else GraphView.load(store))


def find_community(report: CommunityReport, key: str) -> Community | None:
    """Look a community up by id, exact name, member qname, or unique name fragment."""
    text = key.strip().removeprefix("community:")
    if text.isdigit():
        return report.by_id(int(text))
    for community in report.communities:
        if community.name == text:
            return community
    if text in report.membership:
        return report.by_id(report.membership[text])
    fragment = text.lower()
    matches = [c for c in report.communities if fragment and fragment in c.name.lower()]
    return matches[0] if len(matches) == 1 else None


__all__ = [
    "CHANNEL_LINK_WEIGHT",
    "DDL",
    "EDGE_WEIGHTS",
    "Community",
    "CommunityLink",
    "CommunityReport",
    "build_weighted_graph",
    "community_name",
    "community_report",
    "dominant_prefix",
    "detect_communities",
    "find_community",
    "load_communities",
    "save_communities",
]
