from __future__ import annotations

import pytest

from codegraph.cypher import (
    CypherError,
    CypherResult,
    Limits,
    Value,
    execute,
    render_value,
    to_json,
)
from codegraph.cypher.executor import sort_key
from codegraph.graph import GraphView
from codegraph.models import Edge, EdgeKind, Node, NodeKind


def _node(
    qname: str, kind: NodeKind, path: str, lines: tuple[int, int], doc: str | None = None
) -> Node:
    return Node(
        qname=qname,
        kind=kind,
        path=path,
        line_start=lines[0],
        line_end=lines[1],
        language="yaml" if path.endswith(".yaml") else "python",
        name=qname.rsplit(".", 1)[-1].split(":", 1)[-1],
        signature=f"def {qname.rsplit('.', 1)[-1]}()" if kind in ("function", "method") else "",
        docstring=doc,
    )


def _edge(
    src: str,
    dst: str,
    kind: EdgeKind,
    line: int = 1,
    *,
    resolved: bool = True,
    confidence: float = 1.0,
    detail: str = "",
) -> Edge:
    return Edge(src, dst, kind, "app.py", line, resolved, confidence, detail)


NODES = [
    _node("app", "module", "app.py", (1, 60), "The app."),
    _node("svc", "module", "svc.py", (1, 40)),
    _node("app.Base", "class", "app.py", (3, 10), "Base class."),
    _node("app.Child", "class", "app.py", (12, 30)),
    _node("app.Base.run", "method", "app.py", (5, 9)),
    _node("app.Child.run", "method", "app.py", (14, 29)),
    _node("app.main", "function", "app.py", (32, 50)),
    _node("svc.fetch", "function", "svc.py", (3, 12)),
    _node("svc.helper", "function", "svc.py", (14, 18)),
    _node("svc.list_users", "function", "svc.py", (20, 30)),
    _node("tests.test_main", "function", "tests/test_app.py", (1, 5)),
    _node("route:GET /users", "route", "svc.py", (20, 20)),
    _node("channel:orders", "channel", "svc.py", (15, 15)),
    _node("resource:Deployment/web", "resource", "deploy/k8s.yaml", (1, 20)),
]

EDGES = [
    _edge("app", "app.Base", "defines"),
    _edge("app", "app.Child", "defines"),
    _edge("app", "app.main", "defines"),
    _edge("app.Base", "app.Base.run", "defines"),
    _edge("app.Child", "app.Child.run", "defines"),
    _edge("svc", "svc.fetch", "defines"),
    _edge("svc", "svc.helper", "defines"),
    _edge("svc", "svc.list_users", "defines"),
    _edge("app.Child", "app.Base", "inherits"),
    _edge("app", "svc", "imports"),
    _edge("app.main", "app.Child.run", "calls", 33),
    _edge("app.main", "svc.fetch", "calls", 34),
    _edge("app.main", "*.unknown", "calls", 35, resolved=False),
    _edge("app.Child.run", "svc.helper", "calls", 15),
    _edge("svc.fetch", "svc.helper", "calls", 5),
    _edge("svc.helper", "svc.fetch", "calls", 16),
    _edge("tests.test_main", "app.main", "calls", 2),
    _edge("tests.test_main", "app.main", "tests", 2),
    _edge("svc.fetch", "route:GET /users", "http_calls", 6, confidence=0.7, detail="GET"),
    _edge("route:GET /users", "svc.list_users", "handles", 20),
    _edge("svc.helper", "channel:orders", "emits", 17),
    _edge("app.main", "channel:orders", "listens_on", 40),
    _edge("resource:Deployment/web", "app", "depends_on", 3),
    _edge("svc.helper", "app.Base.run", "similar_to", 14, confidence=0.9, detail="jaccard=0.90"),
    _edge("app", "svc", "co_changes", 1, confidence=0.5),
]


@pytest.fixture(scope="module")
def view() -> GraphView:
    return GraphView.from_parts({n.qname: n for n in NODES}, EDGES)


def run(view: GraphView, text: str, limits: Limits | None = None) -> CypherResult:
    return execute(view, text, limits=limits)


def column(view: GraphView, text: str) -> list[Value]:
    result = run(view, text)
    assert len(result.columns) == 1
    return [row[0] for row in result.rows]


# -- labels, kinds and relationship types -------------------------------------


def test_every_label_matches_its_kind(view: GraphView) -> None:
    counts = dict(run(view, "MATCH (n) RETURN n.label AS label, count(*) AS n ORDER BY label").rows)
    assert counts == {
        "Channel": 1,
        "Class": 2,
        "Function": 5,
        "Method": 2,
        "Module": 2,
        "Resource": 1,
        "Route": 1,
    }
    assert column(view, "MATCH (c:Callable) RETURN count(c)") == [7]
    assert column(view, "MATCH (r:Route) RETURN r.qname") == ["route:GET /users"]
    assert column(view, "MATCH (r:Resource) RETURN r.language") == ["yaml"]


@pytest.mark.parametrize(
    ("rel_type", "expected"),
    [
        ("HANDLES", [("route:GET /users", "svc.list_users")]),
        ("HTTP_CALLS", [("svc.fetch", "route:GET /users")]),
        ("EMITS", [("svc.helper", "channel:orders")]),
        ("LISTENS_ON", [("app.main", "channel:orders")]),
        ("DEPENDS_ON", [("resource:Deployment/web", "app")]),
        ("SIMILAR_TO", [("svc.helper", "app.Base.run")]),
        ("CO_CHANGES", [("app", "svc")]),
        ("INHERITS", [("app.Child", "app.Base")]),
        ("IMPORTS", [("app", "svc")]),
        ("TESTS", [("tests.test_main", "app.main")]),
        ("SEMANTICALLY_RELATED", []),
    ],
)
def test_every_relationship_type(
    view: GraphView, rel_type: str, expected: list[tuple[str, str]]
) -> None:
    found = run(view, f"MATCH (a)-[:{rel_type}]->(b) RETURN a.qname, b.qname ORDER BY a.qname")
    assert [tuple(r) for r in found.rows] == expected


def test_directions(view: GraphView) -> None:
    out = column(view, "MATCH ({qname: 'svc.helper'})-[:CALLS]->(x) RETURN x.qname")
    into = column(
        view, "MATCH ({qname: 'svc.helper'})<-[:CALLS]-(x) RETURN x.qname ORDER BY x.qname"
    )
    both = column(
        view, "MATCH ({qname: 'svc.helper'})-[:CALLS]-(x) RETURN x.qname ORDER BY x.qname"
    )
    assert out == ["svc.fetch"]
    assert into == ["app.Child.run", "svc.fetch"]
    assert both == ["app.Child.run", "svc.fetch", "svc.fetch"]


def test_alternatives_and_type_function(view: GraphView) -> None:
    found = run(
        view,
        "MATCH ({qname: 'svc.helper'})-[r:EMITS|SIMILAR_TO]->(x) RETURN type(r), x.qname "
        "ORDER BY x.qname",
    )
    assert [tuple(r) for r in found.rows] == [
        ("SIMILAR_TO", "app.Base.run"),
        ("EMITS", "channel:orders"),
    ]


def test_unresolved_edges_never_match(view: GraphView) -> None:
    assert column(
        view, "MATCH ({qname: 'app.main'})-[:CALLS]->(x) RETURN x.qname ORDER BY x.qname"
    ) == [
        "app.Child.run",
        "svc.fetch",
    ]


def test_anchor_on_later_node_gives_same_answer(view: GraphView) -> None:
    forward = column(
        view, "MATCH (c)-[:CALLS]->(f {name: 'helper'}) RETURN c.qname ORDER BY c.qname"
    )
    assert forward == ["app.Child.run", "svc.fetch"]
    chained = run(
        view,
        "MATCH (t)-[:TESTS]->(m)-[:CALLS]->(f {qname: 'svc.fetch'}) RETURN t.qname, m.qname",
    )
    assert [tuple(r) for r in chained.rows] == [("tests.test_main", "app.main")]


# -- inline properties, WHERE ------------------------------------------------


def test_inline_properties_on_nodes_and_edges(view: GraphView) -> None:
    assert column(view, "MATCH (n:Function {path: 'svc.py', line_start: 14}) RETURN n.name") == [
        "helper"
    ]
    assert column(view, "MATCH (n {file_path: 'tests/test_app.py'}) RETURN n.qname") == [
        "tests.test_main"
    ]
    found = column(view, "MATCH (a)-[r:HTTP_CALLS {detail: 'GET'}]->(b) RETURN r.confidence")
    assert found == [0.7]


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("n.name = 'helper'", ["helper"]),
        ("n.name <> 'helper' AND n.path = 'svc.py'", ["fetch", "list_users"]),
        ("n.line_start > 14", ["list_users", "main"]),
        ("n.line_start >= 14 AND n.line_start <= 20", ["helper", "list_users"]),
        ("n.line_start < 3", ["test_main"]),
        ("n.name =~ '.*_.*'", ["list_users", "test_main"]),
        ("n.name =~ 'he'", []),
        ("n.name CONTAINS 'ai'", ["main", "test_main"]),
        ("n.name STARTS WITH 'f'", ["fetch"]),
        ("n.name ENDS WITH 'users'", ["list_users"]),
        ("n.name IN ['main', 'fetch', 'nope']", ["fetch", "main"]),
        ("NOT n.name IN ['main', 'fetch']", ["helper", "list_users", "test_main"]),
        ("n.docstring IS NULL AND n.path = 'app.py'", ["main"]),
        ("(n.name = 'main' OR n.name = 'fetch') AND n.span > 10", ["main"]),
        ("n.name = 'main' XOR n.path = 'app.py'", []),
        ("n.span = 5", ["helper", "test_main"]),
        ("toLower(n.name) = 'main'", ["main"]),
    ],
)
def test_where_operators(view: GraphView, condition: str, expected: list[str]) -> None:
    query = f"MATCH (n:Function) WHERE {condition} RETURN n.name ORDER BY n.name"
    assert column(view, query) == expected


def test_null_comparisons_are_unknown(view: GraphView) -> None:
    assert column(view, "MATCH (n:Function) WHERE n.docstring = 'x' RETURN n") == []
    assert column(view, "MATCH (n:Function) WHERE n.docstring <> 'x' RETURN n") == []
    assert column(view, "MATCH (n:Function) WHERE NOT (n.docstring = 'x') RETURN n") == []
    assert column(view, "MATCH (n:Module) WHERE n.docstring IS NOT NULL RETURN n.qname") == ["app"]
    assert column(
        view, "MATCH (n:Function) WHERE n.docstring = 'x' OR n.name = 'main' RETURN n.name"
    ) == ["main"]
    assert column(view, "RETURN null = null") == [None]
    assert column(view, "RETURN 1 IN [2, null]") == [None]
    assert column(view, "RETURN 2 IN [2, null]") == [True]


def test_type_mismatches_compare_false_or_unknown(view: GraphView) -> None:
    assert column(view, "RETURN 1 = '1'") == [False]
    assert column(view, "RETURN 1 < 'a'") == [None]
    assert column(view, "RETURN true = 1") == [False]
    assert column(view, "RETURN 1 = 1.0") == [True]
    assert column(view, "RETURN [1, 'a'] = [1, 'a']") == [True]


def test_node_equality_and_inequality(view: GraphView) -> None:
    found = run(
        view,
        "MATCH (a:Method), (b:Method) WHERE a <> b AND a.name = b.name RETURN a.qname "
        "ORDER BY a.qname",
    )
    assert [r[0] for r in found.rows] == ["app.Base.run", "app.Child.run"]


def test_computed_properties(view: GraphView) -> None:
    found = run(
        view,
        "MATCH (n {qname: 'svc.helper'}) "
        "RETURN n.fan_in, n.fan_out, n.in_degree, n.out_degree, n.span",
    )
    assert found.rows == ((2, 1, 3, 3, 5),)
    edge = run(
        view,
        "MATCH (a)-[r:SIMILAR_TO]->(b) "
        "RETURN r.src, r.dst, r.kind, r.type, r.line, r.resolved, r.detail",
    )
    assert edge.rows == (
        ("svc.helper", "app.Base.run", "similar_to", "SIMILAR_TO", 14, True, "jaccard=0.90"),
    )


def test_where_requires_a_boolean(view: GraphView) -> None:
    with pytest.raises(CypherError, match="true/false"):
        run(view, "MATCH (n:Function) WHERE n.name RETURN n")


def test_bad_regex_points_at_pattern(view: GraphView) -> None:
    text = "MATCH (n) WHERE n.name =~ '(' RETURN n"
    with pytest.raises(CypherError, match="invalid regular expression") as caught:
        run(view, text)
    assert caught.value.position == text.index("'('")


def test_in_needs_a_list(view: GraphView) -> None:
    with pytest.raises(CypherError, match="IN needs a list"):
        run(view, "MATCH (n:Function) WHERE n.name IN 'main' RETURN n")


# -- patterns: multiple, bound variables, uniqueness -------------------------


def test_comma_patterns_join_on_shared_variables(view: GraphView) -> None:
    found = run(
        view,
        "MATCH (m:Module)-[:DEFINES]->(f:Function), (t)-[:TESTS]->(f) "
        "RETURN m.qname, f.qname, t.qname",
    )
    assert [tuple(r) for r in found.rows] == [("app", "app.main", "tests.test_main")]


def test_disconnected_patterns_form_a_product(view: GraphView) -> None:
    assert column(view, "MATCH (a:Module), (b:Class) RETURN count(*)") == [4]


def test_same_edge_is_not_used_twice_in_a_clause(view: GraphView) -> None:
    found = run(view, "MATCH (a)-[:INHERITS]-(b)-[:INHERITS]-(c) RETURN count(*)")
    assert found.rows == ((0,),)


def test_variable_reused_across_clauses(view: GraphView) -> None:
    found = run(
        view,
        "MATCH (f {qname: 'svc.fetch'}) "
        "MATCH (f)-[:CALLS]->(g) MATCH (g)-[:CALLS]->(f) RETURN g.qname",
    )
    assert [r[0] for r in found.rows] == ["svc.helper"]


def test_relationship_variable_binds_edges(view: GraphView) -> None:
    found = run(view, "MATCH (a {qname: 'app.main'})-[r:LISTENS_ON]->(c) RETURN r")
    [[edge]] = found.rows
    assert isinstance(edge, Edge) and edge.line == 40
    assert to_json(edge) == {
        "type": "LISTENS_ON",
        "src": "app.main",
        "dst": "channel:orders",
        "path": "app.py",
        "line": 40,
        "confidence": 1.0,
        "detail": "",
    }


# -- variable length -----------------------------------------------------------


def test_variable_length_reachability(view: GraphView) -> None:
    query = "MATCH ({qname: 'tests.test_main'})-[:CALLS*1..2]->(x) RETURN x.qname ORDER BY x.qname"
    assert column(view, query) == ["app.Child.run", "app.main", "svc.fetch"]
    deeper = "MATCH ({qname: 'tests.test_main'})-[:CALLS*]->(x) RETURN count(x)"
    assert column(view, deeper) == [4]


def test_variable_length_exact_and_zero_hops(view: GraphView) -> None:
    exact = "MATCH ({qname: 'tests.test_main'})-[:CALLS*3]->(x) RETURN x.qname"
    assert column(view, exact) == ["svc.helper"]
    zero = "MATCH (s {qname: 'svc.fetch'})-[:CALLS*0..1]->(x) RETURN x.qname ORDER BY x.qname"
    assert column(view, zero) == ["svc.fetch", "svc.helper"]


def test_variable_length_finds_cycles(view: GraphView) -> None:
    found = column(view, "MATCH (f:Function)-[:CALLS*1..3]->(f) RETURN f.qname ORDER BY f.qname")
    assert found == ["svc.fetch", "svc.helper"]


def test_variable_length_path_binding_and_size(view: GraphView) -> None:
    found = run(
        view,
        "MATCH ({qname: 'tests.test_main'})-[p:CALLS*1..3]->(x {name: 'helper'}) RETURN size(p), p",
    )
    [[hops, path]] = found.rows
    assert hops == 3
    assert isinstance(path, tuple) and [e.dst for e in path if isinstance(e, Edge)] == [
        "app.main",
        "app.Child.run",
        "svc.helper",
    ]


def test_variable_length_inbound_and_undirected(view: GraphView) -> None:
    callers = "MATCH (x)-[:CALLS*1..5]->({qname: 'app.Child.run'}) RETURN x.qname ORDER BY x.qname"
    assert column(view, callers) == ["app.main", "tests.test_main"]
    around = "MATCH ({qname: 'route:GET /users'})-[*1]-(x) RETURN x.qname ORDER BY x.qname"
    assert column(view, around) == ["svc.fetch", "svc.list_users"]


def test_hop_cap_is_enforced_with_a_warning(view: GraphView) -> None:
    limits = Limits(max_hops=2)
    result = run(
        view, "MATCH ({qname: 'tests.test_main'})-[:CALLS*1..9]->(x) RETURN x.qname", limits
    )
    assert {r[0] for r in result.rows} == {"app.main", "app.Child.run", "svc.fetch"}
    assert any("limited to 2 hops" in w for w in result.warnings)
    unbounded = run(view, "MATCH ({qname: 'tests.test_main'})-[:CALLS*]->(x) RETURN x", limits)
    assert any("unbounded" in w for w in unbounded.warnings)


# -- OPTIONAL MATCH ------------------------------------------------------------


def test_optional_match_fills_nulls(view: GraphView) -> None:
    found = run(
        view,
        "MATCH (f:Function) OPTIONAL MATCH (t)-[:TESTS]->(f) "
        "RETURN f.name, t.qname ORDER BY f.name",
    )
    assert [tuple(r) for r in found.rows] == [
        ("fetch", None),
        ("helper", None),
        ("list_users", None),
        ("main", "tests.test_main"),
        ("test_main", None),
    ]


def test_optional_match_where_applies_inside(view: GraphView) -> None:
    found = run(
        view,
        "MATCH (m:Module) OPTIONAL MATCH (m)-[:DEFINES]->(f:Function) WHERE f.name = 'helper' "
        "RETURN m.qname, f.name ORDER BY m.qname",
    )
    assert [tuple(r) for r in found.rows] == [("app", None), ("svc", "helper")]


def test_later_match_drops_rows_with_null(view: GraphView) -> None:
    found = run(
        view,
        "MATCH (f:Function {name: 'fetch'}) OPTIONAL MATCH (t)-[:TESTS]->(f) "
        "MATCH (t)-[:CALLS]->(x) RETURN x",
    )
    assert found.rows == ()


def test_untested_callables_via_optional_match_and_with(view: GraphView) -> None:
    query = (
        "MATCH (f:Callable) OPTIONAL MATCH (t)-[:TESTS]->(f) WITH f, count(t) AS tests "
        "WHERE tests = 0 AND NOT f.path STARTS WITH 'tests/' RETURN f.qname ORDER BY f.qname"
    )
    assert column(view, query) == [
        "app.Base.run",
        "app.Child.run",
        "svc.fetch",
        "svc.helper",
        "svc.list_users",
    ]


# -- projection, aggregation, ordering, paging -------------------------------


def test_aggregates_group_implicitly(view: GraphView) -> None:
    found = run(
        view,
        "MATCH (n) WHERE n.kind IN ['function', 'method'] "
        "RETURN n.kind AS kind, count(*) AS n, count(n.docstring) AS documented, "
        "min(n.line_start) AS first, max(n.span) AS widest, sum(n.span) AS total, "
        "avg(n.span) AS mean "
        "ORDER BY kind",
    )
    assert found.columns == ("kind", "n", "documented", "first", "widest", "total", "mean")
    assert found.rows == (
        ("function", 5, 0, 1, 19, 50, 10.0),
        ("method", 2, 0, 5, 16, 21, 10.5),
    )


def test_collect_and_count_distinct(view: GraphView) -> None:
    found = run(
        view,
        "MATCH (c)-[:CALLS]->(f) RETURN f.qname, count(DISTINCT c) AS callers, "
        "collect(c.name) AS names "
        "ORDER BY callers DESC, f.qname LIMIT 2",
    )
    assert found.rows == (
        ("svc.fetch", 2, ("main", "helper")),
        ("svc.helper", 2, ("run", "fetch")),
    )
    distinct = column(view, "MATCH (n) RETURN collect(DISTINCT n.path)")
    assert isinstance(distinct[0], tuple) and len(distinct[0]) == 4


def test_aggregate_over_no_rows(view: GraphView) -> None:
    found = run(
        view,
        "MATCH (n:Function {name: 'nope'}) RETURN count(*), collect(n), max(n.span), sum(n.span)",
    )
    assert found.rows == ((0, (), None, 0),)
    grouped = run(view, "MATCH (n:Function {name: 'nope'}) RETURN n.kind, count(*)")
    assert grouped.rows == ()


def test_sum_needs_numbers(view: GraphView) -> None:
    with pytest.raises(CypherError, match="sum\\(\\) needs numbers"):
        run(view, "MATCH (n:Function) RETURN sum(n.name)")


def test_distinct_order_skip_limit(view: GraphView) -> None:
    assert column(view, "MATCH (n) RETURN DISTINCT n.path ORDER BY n.path") == [
        "app.py",
        "deploy/k8s.yaml",
        "svc.py",
        "tests/test_app.py",
    ]
    assert column(
        view, "MATCH (n:Function) RETURN n.name ORDER BY n.span DESC, n.name SKIP 1 LIMIT 2"
    ) == [
        "list_users",
        "fetch",
    ]
    assert column(view, "MATCH (n:Function) RETURN n.name ORDER BY n.name LIMIT 0") == []
    assert column(view, "MATCH (n:Function) RETURN n.name SKIP 10") == []


def test_order_by_alias_and_unprojected_expression(view: GraphView) -> None:
    by_alias = column(view, "MATCH (n:Function) RETURN n.name AS fn ORDER BY fn DESC LIMIT 1")
    assert by_alias == ["test_main"]
    hidden = column(view, "MATCH (n:Function) RETURN n.name ORDER BY n.line_start DESC LIMIT 1")
    assert hidden == ["main"]


def test_nulls_sort_last_ascending_first_descending(view: GraphView) -> None:
    ascending = column(view, "MATCH (n:Module) RETURN n.docstring ORDER BY n.docstring")
    assert ascending == ["The app.", None]
    descending = column(view, "MATCH (n:Module) RETURN n.docstring ORDER BY n.docstring DESC")
    assert descending == [None, "The app."]


def test_return_star_lists_named_variables(view: GraphView) -> None:
    found = run(view, "MATCH (a:Class)-[r:INHERITS]->(b) RETURN *")
    assert found.columns == ("a", "r", "b")
    [[child, edge, base]] = found.rows
    assert isinstance(child, Node) and child.qname == "app.Child"
    assert isinstance(edge, Edge) and isinstance(base, Node)


def test_with_projects_and_filters(view: GraphView) -> None:
    found = run(
        view,
        "MATCH (f:Function)<-[:CALLS]-(c) WITH f, count(c) AS callers WHERE callers >= 2 "
        "RETURN f.qname, callers ORDER BY f.qname",
    )
    assert [tuple(r) for r in found.rows] == [("svc.fetch", 2), ("svc.helper", 2)]
    chained = run(
        view,
        "MATCH (m:Module) WITH m ORDER BY m.qname DESC LIMIT 1 "
        "MATCH (m)-[:DEFINES]->(f) RETURN f.name ORDER BY f.name",
    )
    assert [r[0] for r in chained.rows] == ["fetch", "helper", "list_users"]


def test_scalar_functions(view: GraphView) -> None:
    found = run(
        view,
        "MATCH (n {qname: 'app.main'}) RETURN toUpper(n.name), toString(n.span), labels(n), "
        "coalesce(n.docstring, 'none'), size(n.name), toString(true)",
    )
    assert found.rows == (("MAIN", "19", ("Function",), "none", 4, "true"),)


def test_literal_only_return(view: GraphView) -> None:
    found = run(view, "RETURN 1 AS one, 'two' AS two, [1, 2] AS list")
    assert found.rows == ((1, "two", (1, 2)),)


# -- limits ----------------------------------------------------------------------


def test_default_row_cap_marks_truncation(view: GraphView) -> None:
    limits = Limits(default_rows=3)
    capped = run(view, "MATCH (n) RETURN n.qname", limits)
    assert len(capped.rows) == 3 and capped.truncated
    exact = run(view, "MATCH (n:Module) RETURN n.qname", Limits(default_rows=2))
    assert len(exact.rows) == 2 and not exact.truncated
    explicit = run(view, "MATCH (n) RETURN n.qname LIMIT 3", limits)
    assert len(explicit.rows) == 3 and not explicit.truncated
    ordered = run(view, "MATCH (n) RETURN n.qname ORDER BY n.qname", limits)
    assert ordered.truncated and ordered.rows[0] == ("app",)


def test_limit_above_maximum_is_clamped(view: GraphView) -> None:
    result = run(view, "MATCH (n) RETURN n LIMIT 100", Limits(max_rows=4))
    assert len(result.rows) == 4 and result.truncated
    assert any("lowered to the maximum of 4" in w for w in result.warnings)


def test_work_budget_aborts_broad_queries(view: GraphView) -> None:
    with pytest.raises(CypherError, match="too broad"):
        run(view, "MATCH (a), (b), (c) RETURN count(*)", Limits(max_work=500))


def test_limit_without_order_stops_early(view: GraphView) -> None:
    result = run(view, "MATCH (a), (b), (c) RETURN a.qname LIMIT 2", Limits(max_work=50))
    assert len(result.rows) == 2


# -- rendering -------------------------------------------------------------------


def test_render_and_json_helpers(view: GraphView) -> None:
    node = view.nodes["app.main"]
    assert render_value(node) == "app.main"
    assert render_value(None) == "null"
    assert render_value(False) == "false"
    assert render_value(0.123456) == "0.1235"
    assert render_value((1, "a")) == "[1, a]"
    assert to_json((node,)) == [
        {
            "qname": "app.main",
            "kind": "function",
            "name": "main",
            "path": "app.py",
            "line_start": 32,
            "line_end": 50,
        }
    ]
    result = run(view, "MATCH (n:Route) RETURN n.qname AS q")
    assert result.as_dict() == {
        "columns": ["q"],
        "rows": [["route:GET /users"]],
        "row_count": 1,
        "truncated": False,
        "warnings": [],
    }


def test_sort_key_orders_across_types() -> None:
    values: list[Value] = [None, "b", 2, True, "a", 1.5]
    assert sorted(values, key=sort_key) == [True, 1.5, 2, "a", "b", None]
