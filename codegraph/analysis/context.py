"""A compact starting bundle for an agent: one symbol, its neighbourhood, nothing else.

Given a target (qname, bare name or ``path:line``) or a free-text task, return the
node, its signature and a clipped docstring, direct callers and callees, covering tests,
its community and the flows it belongs to, plus which tools to reach for next. The
bundle is serialised and trimmed until it fits a token budget, so it is safe to call
first in any task.

Communities and flows come from the stored analysis tables when the passes have run;
otherwise they are computed on the spot, which only costs time on large repositories.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from codegraph.analysis.common import estimate_tokens, table_has_rows
from codegraph.analysis.communities import community_report
from codegraph.analysis.flows import flow_report
from codegraph.models import Node
from codegraph.queries import resolve_target
from codegraph.store import GraphStore

DEFAULT_MAX_TOKENS = 800
_NEIGHBOURS = 8
_DOC_CHARS = 240

_TASK_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "the",
        "to",
        "of",
        "in",
        "on",
        "for",
        "fix",
        "bug",
        "add",
        "make",
        "how",
        "does",
        "what",
        "why",
        "where",
        "is",
        "it",
        "this",
        "that",
        "with",
        "review",
        "change",
        "update",
        "work",
        "works",
    }
)
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

_SUGGESTIONS: tuple[tuple[frozenset[str], tuple[str, ...]], ...] = (
    (
        frozenset({"review", "pr", "diff", "merge", "change", "changes"}),
        ("get_change_impact", "get_affected_flows", "get_review_context"),
    ),
    (
        frozenset({"debug", "bug", "error", "fix", "crash", "fails", "broken"}),
        ("trace_path", "get_flow", "get_impact_radius"),
    ),
    (
        frozenset({"refactor", "rename", "dead", "clean", "split", "move"}),
        ("find_dead_code", "find_large_functions", "get_impact_radius"),
    ),
    (
        frozenset({"onboard", "understand", "explore", "architecture", "overview", "learn"}),
        ("get_architecture", "list_communities", "list_flows"),
    ),
)
_DEFAULT_SUGGESTIONS = ("trace_path", "get_change_impact", "get_architecture")


@dataclass(frozen=True, slots=True)
class SymbolContext:
    qname: str
    kind: str
    location: str
    signature: str
    doc: str | None
    callers: tuple[str, ...]
    callees: tuple[str, ...]
    tests: tuple[str, ...]
    community: str | None
    flows: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MinimalContext:
    summary: str
    target: SymbolContext | None
    candidates: tuple[str, ...]
    """Other plausible matches for a free-text task."""
    suggested_tools: tuple[str, ...]
    truncated: bool
    tokens: int


def task_keywords(task: str) -> list[str]:
    words = [w for w in _WORD.findall(task) if w.lower() not in _TASK_STOPWORDS]
    return list(dict.fromkeys(words))


def suggest_tools(task: str | None) -> tuple[str, ...]:
    words = {w.lower() for w in _WORD.findall(task or "")} | set((task or "").lower().split())
    for keys, tools in _SUGGESTIONS:
        if words & keys:
            return tools
    return _DEFAULT_SUGGESTIONS


def minimal_context(
    store: GraphStore,
    root: Path | None,
    *,
    target: str | None = None,
    task: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> MinimalContext | None:
    """The bundle for ``target`` or the best match for ``task``; None if neither matches."""
    node = resolve_target(store, target) if target else None
    candidates: list[str] = []
    if node is None and task:
        hits = [
            n
            for n in store.search_nodes(" ".join(task_keywords(task)) or task, limit=6)
            if n.kind != "module"
        ]
        if hits:
            node = hits[0]
            candidates = [n.qname for n in hits[1:]]
    if node is None and target:
        return None
    if node is None and not task:
        return None

    counts = store.counts()
    summary = f"{counts['files']} files, {counts['nodes']} nodes, {counts['edges']} edges"
    symbol = _symbol_context(store, root, node) if node is not None else None
    bundle = MinimalContext(
        summary=summary,
        target=symbol,
        candidates=tuple(candidates),
        suggested_tools=suggest_tools(task),
        truncated=False,
        tokens=0,
    )
    return fit_budget(bundle, max_tokens)


def _symbol_context(store: GraphStore, root: Path | None, node: Node) -> SymbolContext:
    callers = sorted({e.src for e in store.edges_to(node.qname, "calls")})
    callees = sorted({e.dst for e in store.edges_from(node.qname, "calls") if e.resolved})
    tests = sorted({e.src for e in store.edges_to(node.qname, "tests")})
    doc = node.docstring
    if doc and len(doc) > _DOC_CHARS:
        doc = doc[: _DOC_CHARS - 1] + "…"
    return SymbolContext(
        qname=node.qname,
        kind=node.kind,
        location=f"{node.path}:{node.line_start}-{node.line_end}",
        signature=node.signature,
        doc=doc,
        callers=tuple(callers[:_NEIGHBOURS]),
        callees=tuple(callees[:_NEIGHBOURS]),
        tests=tuple(tests[:_NEIGHBOURS]),
        community=_community_of(store, node.qname),
        flows=tuple(_flows_of(store, root, node.qname)),
    )


def _community_of(store: GraphStore, qname: str) -> str | None:
    if table_has_rows(store, "analysis_communities"):
        row = store.connection.execute(
            "SELECT c.name FROM analysis_community_members m "
            "JOIN analysis_communities c ON c.id = m.community_id WHERE m.qname = ?",
            (qname,),
        ).fetchone()
        return str(row["name"]) if row else None
    report = community_report(store)
    community = report.by_id(report.membership.get(qname, -1))
    return community.name if community else None


def _flows_of(store: GraphStore, root: Path | None, qname: str) -> list[str]:
    if table_has_rows(store, "analysis_flows"):
        rows = store.connection.execute(
            "SELECT DISTINCT f.id, f.name, f.criticality FROM analysis_flow_steps s "
            "JOIN analysis_flows f ON f.id = s.flow_id WHERE s.qname = ? "
            "ORDER BY f.criticality DESC, f.id LIMIT ?",
            (qname, _NEIGHBOURS),
        ).fetchall()
        return [f"#{int(r['id'])} {r['name']}" for r in rows]
    flows = flow_report(store, root)
    return [f"#{f.id} {f.name}" for f in flows if any(s.qname == qname for s in f.steps)][
        :_NEIGHBOURS
    ]


def fit_budget(bundle: MinimalContext, max_tokens: int) -> MinimalContext:
    """Shrink lists, then the docstring, until the JSON form fits ``max_tokens``."""
    current = bundle
    truncated = False
    for _ in range(12):
        tokens = estimate_tokens(json.dumps(asdict(current)))
        if tokens <= max_tokens:
            return replace(current, truncated=truncated, tokens=tokens)
        shrunk = _shrink(current)
        if shrunk == current:
            break
        current, truncated = shrunk, True
    return replace(current, truncated=True, tokens=estimate_tokens(json.dumps(asdict(current))))


def _shrink(bundle: MinimalContext) -> MinimalContext:
    def half(items: tuple[str, ...]) -> tuple[str, ...]:
        return items[: len(items) // 2]

    candidates = half(bundle.candidates)
    symbol = bundle.target
    if symbol is None:
        return replace(bundle, candidates=candidates)
    if any((symbol.callers, symbol.callees, symbol.tests, symbol.flows, bundle.candidates)):
        symbol = replace(
            symbol,
            callers=half(symbol.callers),
            callees=half(symbol.callees),
            tests=half(symbol.tests),
            flows=half(symbol.flows),
        )
    elif symbol.doc:
        symbol = replace(symbol, doc=None)
    elif len(symbol.signature) > 80:
        symbol = replace(symbol, signature=symbol.signature[:79] + "…")
    return replace(bundle, target=symbol, candidates=candidates)


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "MinimalContext",
    "SymbolContext",
    "fit_budget",
    "minimal_context",
    "suggest_tools",
    "task_keywords",
]
