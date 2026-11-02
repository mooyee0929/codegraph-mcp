"""Analysis algorithms on small in-memory graphs (no repository, no database)."""

from __future__ import annotations

from pathlib import Path

import pytest

from codegraph.analysis.centrality import betweenness, bridge_nodes, hub_nodes
from codegraph.analysis.common import common_dotted_prefix, split_identifier, top_package
from codegraph.analysis.communities import (
    build_weighted_graph,
    community_name,
    detect_communities,
    dominant_prefix,
    find_community,
)
from codegraph.analysis.context import MinimalContext, SymbolContext, fit_budget, task_keywords
from codegraph.analysis.deadcode import find_dead_code
from codegraph.analysis.entrypoints import (
    cli_scripts,
    decorators_of,
    detect_entry_points,
    framework_decorator,
    has_implicit_decorator,
    resolve_scripts,
)
from codegraph.analysis.flows import affected_flows, criticality, detect_flows, find_flow
from codegraph.analysis.insights import (
    knowledge_gaps,
    large_functions,
    package_distance,
    surprising_connections,
)
from codegraph.analysis.louvain import louvain, modularity
from codegraph.analysis.navigation import shortest_path, trace_paths, traverse
from codegraph.graph import GraphView
from codegraph.models import Edge, EdgeKind, Node, NodeKind


def _node(
    qname: str,
    kind: NodeKind = "function",
    *,
    path: str | None = None,
    line: int = 1,
    end: int | None = None,
    signature: str = "",
    docstring: str | None = None,
    language: str = "python",
) -> Node:
    module = qname.rsplit(".", 1)[0] if kind != "module" else qname
    return Node(
        qname=qname,
        kind=kind,
        path=path if path is not None else module.replace(".", "/") + ".py",
        line_start=line,
        line_end=end if end is not None else line + 1,
        language=language,
        name=qname.rsplit(".", 1)[-1].removeprefix("route:").removeprefix("channel:"),
        signature=signature,
        docstring=docstring,
    )


def _edge(src: str, dst: str, kind: EdgeKind = "calls", *, line: int = 5) -> Edge:
    return Edge(src=src, dst=dst, kind=kind, path="x.py", line=line)


def _view(nodes: list[Node], edges: list[Edge]) -> GraphView:
    return GraphView.from_parts({n.qname: n for n in nodes}, edges)


def _two_cliques() -> GraphView:
    left = [f"a.m.f{i}" for i in range(4)]
    right = [f"b.n.g{i}" for i in range(4)]
    nodes = [_node(q) for q in left + right]
    edges = [_edge(x, y) for group in (left, right) for x in group for y in group if x < y]
    edges.append(_edge("a.m.f0", "b.n.g0"))
    return _view(nodes, edges)


# -- louvain -----------------------------------------------------------------


def test_louvain_splits_two_cliques_joined_by_one_edge() -> None:
    clique = {(a, b): 1.0 for group in (range(4), range(4, 8)) for a in group for b in group}
    clique = {k: v for k, v in clique.items() if k[0] < k[1]}
    clique[(0, 4)] = 1.0
    labels = louvain(8, clique)
    assert labels == [0, 0, 0, 0, 1, 1, 1, 1]
    assert modularity(8, clique, labels) > 0.3
    assert modularity(8, clique, labels) > modularity(8, clique, [0] * 8)


def test_louvain_is_deterministic_and_keeps_isolated_nodes_apart() -> None:
    edges = {(0, 1): 1.0, (1, 2): 1.0, (0, 2): 1.0, (3, 3): 2.0}
    first = louvain(6, edges)
    assert first == louvain(6, dict(reversed(list(edges.items()))))
    assert first[0] == first[1] == first[2]
    assert len({first[3], first[4], first[5], first[0]}) == 4


def test_louvain_edge_cases() -> None:
    assert louvain(0, {}) == []
    assert louvain(3, {}) == [0, 1, 2]
    assert louvain(2, {(0, 1): -1.0}) == [0, 1]
    assert modularity(2, {}, [0, 1]) == 0.0
    with pytest.raises(ValueError):
        louvain(2, {(0, 5): 1.0})
    with pytest.raises(ValueError):
        modularity(2, {(0, 1): 1.0}, [0])


# -- communities -------------------------------------------------------------


def test_detect_communities_names_cohesion_and_boundaries() -> None:
    report = detect_communities(_two_cliques())
    assert [c.size for c in report.communities] == [4, 4]
    first, second = report.communities
    assert first.prefix == "a.m" and second.prefix == "b.n"
    assert first.name.startswith("a.m")
    assert 0.0 < first.cohesion < 1.0
    assert report.links[0].edges == 1
    assert report.membership["a.m.f1"] == first.id
    assert report.source == "computed"
    assert find_community(report, str(second.id)) == second
    assert find_community(report, "b.n.g2") == second
    assert find_community(report, first.name) == first
    assert find_community(report, "zzz") is None
    assert detect_communities(_two_cliques()) == report


def test_derived_edges_join_communities() -> None:
    nodes = [
        _node("client.app.fetch"),
        _node("server.api.handler"),
        _node("route:GET /items", "route", path="server/api.py"),
        _node("producer.emit_it"),
        _node("consumer.on_it"),
        _node("channel:orders", "channel", path=""),
    ]
    edges = [
        _edge("client.app.fetch", "route:GET /items", "http_calls"),
        _edge("route:GET /items", "server.api.handler", "handles"),
        _edge("producer.emit_it", "channel:orders", "emits"),
        _edge("consumer.on_it", "channel:orders", "listens_on"),
    ]
    members, weights = build_weighted_graph(_view(nodes, edges))
    index = {q: i for i, q in enumerate(members)}
    pair = tuple(sorted((index["producer.emit_it"], index["consumer.on_it"])))
    assert weights[(pair[0], pair[1])] > 0
    report = detect_communities(_view(nodes, edges))
    assert report.membership["client.app.fetch"] == report.membership["server.api.handler"]
    assert report.membership["producer.emit_it"] == report.membership["consumer.on_it"]


def test_community_name_prefers_production_words() -> None:
    members = [
        _node("pkg.billing.Invoice", "class", path="pkg/billing.py"),
        _node("pkg.billing.Invoice.total", "method", path="pkg/billing.py"),
        _node("tests.test_billing.test_invoice_total", path="tests/test_billing.py"),
    ]
    prefix, name = community_name(members)
    assert prefix == "pkg.billing"
    assert name == "pkg.billing (invoice, total)"
    assert community_name([_node("route:GET /x", "route", path="")])[1] == "routes"
    assert dominant_prefix(["a.b.c", "a.b.d", "a.x"]) == "a.b"
    assert dominant_prefix(["a.b.c", "a.b.c", "z"]) == "a.b.c"
    assert dominant_prefix(["p", "q", "r"]) == ""
    assert dominant_prefix([]) == ""


# -- centrality --------------------------------------------------------------


def test_betweenness_matches_known_values() -> None:
    path = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
    assert betweenness(path) == {"a": 0.0, "b": 1.0, "c": 0.0}
    star = {"hub": ["l1", "l2", "l3", "l4"], **{f"l{i}": ["hub"] for i in range(1, 5)}}
    assert betweenness(star)["hub"] == 1.0
    assert betweenness({"a": ["b"], "b": ["a"]}) == {"a": 0.0, "b": 0.0}
    sampled = betweenness(star, max_sources=2)
    assert sampled == betweenness(star, max_sources=2)
    assert sampled["hub"] > 0


def test_hubs_and_bridges() -> None:
    view = _two_cliques()
    report = detect_communities(view)
    hubs = hub_nodes(view, top=2, membership=report.membership)
    assert {h.qname for h in hubs} == {"a.m.f0", "b.n.g0"}
    assert hubs[0].degree == 4
    bridges = bridge_nodes(view, top=2, membership=report.membership)
    assert {b.qname for b in bridges} == {"a.m.f0", "b.n.g0"}
    assert bridges[0].communities_linked == 2
    assert hub_nodes(view, top=0) == []


# -- entry points and dead code ------------------------------------------------


def test_decorators_and_entry_points(tmp_path: Path) -> None:
    nodes = [
        _node("app.views.index", line=10, end=12),
        _node("app.views.helper", line=20, end=21),
        _node("app.cli.cmd", line=30, end=31),
        _node("app.tool.run_tool", line=40, end=41),
        _node("app.tasks.main", line=50, end=51),
        _node("app.tasks.lonely", line=60, end=61),
        _node("tests.test_x.test_it", path="tests/test_x.py"),
        _node("route:GET /", "route", path="app/views.py"),
        _node("web.ts.Ctrl", "class", signature="@Controller('x') class Ctrl", language="ts"),
    ]
    edges = [
        Edge("app.views.index", "*.route", "calls", "app/views.py", 9, resolved=False),
        _edge("app.views.index", "app.views.helper"),
        Edge("app.cli.cmd", "click.command", "calls", "app/cli.py", 29),
        _edge("app.cli.cmd", "app.views.helper"),
        _edge("app.tool.run_tool", "app.views.helper"),
        _edge("app.tasks.main", "app.views.helper"),
        _edge("tests.test_x.test_it", "app.tasks.lonely"),
        _edge("app.tasks.lonely", "app.views.helper"),
        _edge("route:GET /", "app.tool.run_tool", "handles"),
    ]
    view = _view(nodes, edges)
    assert decorators_of(view, view.nodes["app.views.index"]) == ("*.route",)
    assert decorators_of(view, view.nodes["web.ts.Ctrl"]) == ("Controller",)
    assert framework_decorator(["functools.wraps", "click.command"]) == "click.command"
    assert framework_decorator(["functools.lru_cache"]) is None
    assert has_implicit_decorator(["property"]) and not has_implicit_decorator(["cache"])

    kinds = {e.qname: e.kind for e in detect_entry_points(view)}
    assert kinds == {
        "app.tool.run_tool": "route",
        "app.tasks.main": "main",
        "app.views.index": "decorator",
        "app.cli.cmd": "decorator",
        "app.tasks.lonely": "root",
    }
    with_tests = detect_entry_points(view, include_tests=True)
    assert "tests.test_x.test_it" in {e.qname for e in with_tests}

    (tmp_path / "pyproject.toml").write_text(
        '[project.scripts]\ntool = "app.cli:cmd"\n'
        '[project.entry-points."x.plugins"]\nplug = "app.tasks:lonely"\n'
        '[tool.poetry.scripts]\nother = "missing.mod:fn"\n'
    )
    scripts = cli_scripts(tmp_path)
    assert scripts == {"app.cli.cmd": "tool", "app.tasks.lonely": "plug", "missing.mod.fn": "other"}
    assert resolve_scripts(view, {"cli.cmd": "tool"}) == {"app.cli.cmd": "tool"}
    by_kind = {e.qname: e.kind for e in detect_entry_points(view, scripts)}
    assert by_kind["app.cli.cmd"] == "cli"
    assert cli_scripts(None) == {}
    (tmp_path / "pyproject.toml").write_text("not = [valid")
    assert cli_scripts(tmp_path) == {}


def test_find_dead_code_heuristics() -> None:
    nodes = [
        _node("m.Base", "class", line=1, end=6, path="m.py"),
        _node("m.Base.hook", "method", line=2, end=3, path="m.py"),
        _node("m.Sub", "class", line=10, end=20, path="m.py"),
        _node("m.Sub.hook", "method", line=11, end=12, path="m.py"),
        _node("m.Sub.__len__", "method", line=13, end=14, path="m.py"),
        _node("m.Sub.size", "method", line=16, end=17, path="m.py"),
        _node("m.Ext", "class", line=30, end=34, path="m.py"),
        _node("m.Ext.visit", "method", line=31, end=32, path="m.py"),
        _node("m.Typed", "class", line=40, end=41, path="m.py"),
        _node("m.use", line=50, end=51, signature="def use(x: Typed) -> None", path="m.py"),
        _node("m._private", line=60, end=61, path="m.py"),
        _node("m.public_unused", line=70, end=71, path="m.py"),
        _node("m.maybe", line=80, end=81, path="m.py"),
    ]
    edges = [
        Edge("m.Sub", "m.Base", "inherits", "m.py", 10),
        Edge("m.Ext", "ast.NodeVisitor", "inherits", "m.py", 30),
        Edge("m.Sub.size", "property", "calls", "m.py", 15, resolved=False),
        Edge("m.use", "*.maybe", "calls", "m.py", 50, resolved=False),
        Edge("m.Base", "m.Base.hook", "defines", "m.py", 2),
        Edge("m.Sub", "m.Sub.hook", "defines", "m.py", 11),
        Edge("m.Sub", "m.Sub.__len__", "defines", "m.py", 13),
        Edge("m.Sub", "m.Sub.size", "defines", "m.py", 16),
        Edge("m.Ext", "m.Ext.visit", "defines", "m.py", 31),
    ]
    dead = {n.qname for n in find_dead_code(_view(nodes, edges))}
    assert dead == {"m.Base.hook", "m.Sub", "m.Ext", "m.use", "m._private", "m.public_unused"}


# -- flows -------------------------------------------------------------------


def test_flows_cross_routes_and_channels() -> None:
    nodes = [
        _node("ui.main"),
        _node("ui.send"),
        _node("route:POST /orders", "route", path="api/orders.py"),
        _node("api.orders.create", path="api/orders.py"),
        _node("api.orders.Order", "class", path="api/orders.py", line=20, end=30),
        _node("api.orders.Order.__init__", "method", path="api/orders.py", line=21, end=22),
        _node("channel:created", "channel", path=""),
        _node("mail.notify", path="mail.py"),
        _node("tests.test_ui.test_send", path="tests/test_ui.py"),
    ]
    edges = [
        _edge("ui.main", "ui.send"),
        _edge("ui.send", "route:POST /orders", "http_calls"),
        _edge("route:POST /orders", "api.orders.create", "handles"),
        _edge("api.orders.create", "api.orders.Order"),
        _edge("api.orders.create", "channel:created", "emits"),
        _edge("mail.notify", "channel:created", "listens_on"),
        Edge("mail.notify", "smtplib.send", "calls", "mail.py", 3),
        _edge("tests.test_ui.test_send", "ui.send", "tests"),
    ]
    view = _view(nodes, edges)
    flows = detect_flows(view, membership={"ui.main": 1, "mail.notify": 2})
    main = find_flow(flows, "ui.main")
    assert main is not None
    reached = [s.qname for s in main.steps]
    assert reached[:3] == ["ui.main", "ui.send", "route:POST /orders"]
    assert {"api.orders.create", "api.orders.Order.__init__", "mail.notify"} <= set(reached)
    vias = {s.qname: s.via for s in main.steps}
    assert vias["api.orders.create"] == "handles"
    assert vias["api.orders.Order.__init__"] == "constructs"
    assert vias["mail.notify"] == "listens_on"
    assert main.external_calls == 1
    assert main.community_count == 2
    assert main.untested_steps == 4
    assert [f.id for f in flows] == list(range(1, len(flows) + 1))
    assert find_flow(flows, str(main.id)) == main
    assert find_flow(flows, "nope") is None

    hits = affected_flows(flows, ["mail.notify"])
    assert [h.flow.entry_point for h in hits][0] == "ui.main"
    assert hits[0].touched == ("mail.notify",)


def test_flow_bounds_and_criticality() -> None:
    chain = [_node(f"c.f{i}") for i in range(12)]
    edges = [_edge(f"c.f{i}", f"c.f{i + 1}") for i in range(11)]
    view = _view(chain, edges)
    deep = detect_flows(view, max_depth=3)
    assert deep[0].depth == 3 and deep[0].node_count == 4
    small = detect_flows(view, max_steps=5)
    assert small[0].truncated and small[0].node_count == 5
    low = dict(depth=0, node_count=1, file_count=1, community_count=1, external_calls=0)
    assert criticality(untested_share=0.0, **low) == 0.0
    high = dict(depth=99, node_count=99, file_count=99, community_count=99, external_calls=99)
    assert criticality(untested_share=1.0, **high) == 1.0


# -- insights ----------------------------------------------------------------


def test_knowledge_gaps_and_surprises() -> None:
    nodes = [
        _node("core.util.helper"),
        *[_node(f"core.util.user{i}") for i in range(4)],
        _node("core.util.alone"),
        _node("core.util", "module", path="core/util.py"),
        _node("far.away.deep.thing", path="far/away/deep.py"),
        _node("far.away.deep.other", path="far/away/deep.py"),
    ]
    edges = [
        *[_edge(f"core.util.user{i}", "core.util.helper") for i in range(4)],
        _edge("core.util.user0", "far.away.deep.thing"),
        _edge("far.away.deep.thing", "far.away.deep.other"),
        Edge("core.util", "core.util.helper", "defines", "core/util.py", 1),
        Edge("core.util", "core.util.alone", "defines", "core/util.py", 1),
        Edge("core.util.user1", "*.dynamic", "calls", "core/util.py", 9, resolved=False),
        Edge("core.util.user1", "len", "calls", "core/util.py", 9, resolved=False),
    ]
    view = _view(nodes, edges)
    report = detect_communities(view)
    gaps = knowledge_gaps(view, report)
    assert [f.target for f in gaps.untested_hubs] == ["core.util.helper"]
    assert [f.target for f in gaps.isolated_nodes] == ["core.util.alone"]
    assert [f.target for f in gaps.undocumented_modules] == ["core.util"]
    assert gaps.unresolved_hotspots[0].detail == "1 unresolved: dynamic"
    assert gaps.total >= 4

    surprises = surprising_connections(view, report, min_score=0.0)
    assert surprises[0].src == "core.util.user0" and surprises[0].dst == "far.away.deep.thing"
    assert any("crosses packages" in r for r in surprises[0].reasons)
    assert package_distance("a.b", "a.b") == 0.0
    assert package_distance("a.b", "c.d") == 1.0
    assert package_distance("", "c") == 0.0


def test_large_functions_filters() -> None:
    nodes = [
        _node("a.big", end=80),
        _node("a.Klass", "class", end=200),
        _node("b.huge", path="src/b.py", end=120, language="go"),
        _node("tests.t.test_long", path="tests/t.py", end=90),
        _node("a.small", end=3),
    ]
    assert [s.qname for s in large_functions(nodes)] == ["b.huge", "a.big"]
    assert [s.qname for s in large_functions(nodes, kinds=("class",))] == ["a.Klass"]
    assert [s.qname for s in large_functions(nodes, language="go")] == ["b.huge"]
    assert [s.qname for s in large_functions(nodes, path="src")] == ["b.huge"]
    assert [s.qname for s in large_functions(nodes, path="*/b.py")] == ["b.huge"]
    assert "tests.t.test_long" in {s.qname for s in large_functions(nodes, include_tests=True)}
    assert large_functions(nodes, min_lines=500) == []
    assert large_functions(nodes)[0].lines == 120


# -- navigation --------------------------------------------------------------


def _diamond() -> GraphView:
    names = ["d.top", "d.left", "d.right", "d.bottom", "d.leaf"]
    edges = [
        _edge("d.top", "d.left"),
        _edge("d.top", "d.right"),
        _edge("d.left", "d.bottom"),
        _edge("d.right", "d.bottom"),
        _edge("d.bottom", "d.leaf"),
        _edge("d.leaf", "d.top", "imports"),
    ]
    return _view([_node(n) for n in names], edges)


def test_traverse_bfs_dfs_and_limits() -> None:
    view = _diamond()
    bfs = traverse(view, "d.top", edge_kinds=["calls"], depth=5)
    assert bfs.visited == 5
    assert [c.qname for c in bfs.root.children] == ["d.left", "d.right"]
    assert bfs.root.children[0].children[0].qname == "d.bottom"
    dfs = traverse(view, "d.top", edge_kinds=["calls"], depth=5, mode="dfs")
    assert dfs.root.children[0].children[0].children[0].qname == "d.leaf"
    inbound = traverse(view, "d.bottom", edge_kinds=["calls"], direction="in", depth=1)
    assert {c.via for c in inbound.root.children} == {"<-calls"}
    capped = traverse(view, "d.top", depth=5, max_nodes=2)
    assert capped.truncated and capped.visited == 2
    assert "imports" in traverse(view, "d.leaf").edge_kinds
    with pytest.raises(KeyError):
        traverse(view, "missing")
    with pytest.raises(ValueError):
        traverse(view, "d.top", mode="sideways")  # type: ignore[arg-type]


def test_trace_paths_and_shortest_path() -> None:
    view = _diamond()
    trace = trace_paths(view, "d.bottom", direction="inbound", depth=2)
    assert set(trace.inbound) == {("d.top", "d.left", "d.bottom"), ("d.top", "d.right", "d.bottom")}
    assert trace.outbound == ()
    assert [(r.qname, r.risk) for r in trace.callers] == [
        ("d.left", "critical"),
        ("d.right", "critical"),
        ("d.top", "high"),
    ]
    out = trace_paths(view, "d.top", direction="outbound", depth=5, to="d.leaf")
    assert ("d.top", "d.left", "d.bottom", "d.leaf") in out.outbound
    assert out.path is not None and out.path.direction == "forward"
    assert len(out.path.nodes) == 4
    assert trace_paths(view, "d.top", depth=5, max_paths=1).truncated
    back = shortest_path(view, "d.leaf", "d.top")
    assert back is not None and back.direction == "backward"
    lonely = _view([_node("x.a"), _node("x.b")], [])
    assert shortest_path(lonely, "x.a", "x.b") is None
    with pytest.raises(ValueError):
        trace_paths(view, "d.top", depth=6)
    with pytest.raises(KeyError):
        trace_paths(view, "d.top", to="nope")


# -- helpers -----------------------------------------------------------------


def test_identifier_helpers() -> None:
    assert split_identifier("parseHTTPRequest_fast") == ["parse", "http", "request", "fast"]
    assert common_dotted_prefix(["a.b.c", "a.b.d", "a.b"]) == "a.b"
    assert common_dotted_prefix(["a.b", "c"]) == ""
    assert common_dotted_prefix([]) == ""
    assert top_package("pkg/sub/mod.py") == "pkg"
    assert task_keywords("fix the login timeout in the login flow") == ["login", "timeout", "flow"]


def test_fit_budget_trims_lists_then_docs() -> None:
    symbol = SymbolContext(
        qname="m.f",
        kind="function",
        location="m.py:1-2",
        signature="def f()",
        doc="x" * 200,
        callers=tuple(f"m.caller_{i}" for i in range(40)),
        callees=(),
        tests=(),
        community=None,
        flows=(),
    )
    bundle = MinimalContext("s", symbol, ("m.g",), ("trace_path",), False, 0)
    fitted = fit_budget(bundle, 120)
    assert fitted.truncated and fitted.tokens <= 120
    assert fitted.target is not None and len(fitted.target.callers) < 40
    roomy = fit_budget(bundle, 10_000)
    assert not roomy.truncated and roomy.target == symbol
