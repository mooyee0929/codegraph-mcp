from __future__ import annotations

import re

import pytest

from codegraph.cypher import CypherError, parse, prepare
from codegraph.cypher.lexer import tokenize
from codegraph.cypher.planner import MatchStep
from codegraph.cypher.syntax import (
    Call,
    Compare,
    Const,
    IsNull,
    ListExpr,
    Logical,
    Match,
    Not,
    Prop,
    Return,
    Var,
    With,
)
from codegraph.cypher.vocab import (
    NODE_LABELS,
    REL_TYPES,
    canonical_property,
    label_for,
    resolve_label,
    resolve_rel_type,
)
from codegraph.models import EDGE_KINDS, NODE_KINDS

# -- lexer -------------------------------------------------------------------


def test_tokenize_kinds_and_positions() -> None:
    tokens = tokenize("MATCH (n:Function {name: 'a\\'b'}) RETURN n.x <> 1.5")
    assert [(t.kind, t.text) for t in tokens[:4]] == [
        ("ident", "MATCH"),
        ("symbol", "("),
        ("ident", "n"),
        ("symbol", ":"),
    ]
    string = next(t for t in tokens if t.kind == "string")
    assert string.text == "a'b"
    assert tokens[0].pos == 0 and tokens[0].end == 5
    assert any(t.is_symbol("<>") for t in tokens)
    assert any(t.kind == "number" and t.text == "1.5" for t in tokens)
    assert tokens[-1].kind == "eof"


def test_tokenize_hop_range_is_not_a_float() -> None:
    texts = [t.text for t in tokenize("*1..3")]
    assert texts == ["*", "1", "..", "3", ""]


def test_tokenize_comments_backticks_and_double_quotes() -> None:
    tokens = tokenize('// line\nMATCH /* block */ (`weird name`) RETURN "x"')
    quoted = next(t for t in tokens if t.quoted)
    assert quoted.text == "weird name"
    assert not quoted.is_keyword("WEIRD")
    assert tokens[0].is_keyword("MATCH")
    assert next(t for t in tokens if t.kind == "string").text == "x"


def test_keywords_are_case_insensitive_identifiers() -> None:
    token = tokenize("match")[0]
    assert token.kind == "ident"
    assert token.is_keyword("MATCH")
    assert tokenize("`match`")[0].is_keyword("MATCH") is False


@pytest.mark.parametrize(
    ("text", "position", "fragment"),
    [
        ("MATCH (n) RETURN n # x", 19, "unexpected character"),
        ("RETURN 'open", 7, "unterminated string"),
        ("RETURN `open", 7, "unterminated quoted identifier"),
        ("MATCH /* never closed", 6, "unterminated block comment"),
        ("RETURN 12abc", 7, "malformed number"),
    ],
)
def test_lexer_errors_carry_position(text: str, position: int, fragment: str) -> None:
    with pytest.raises(CypherError, match=re.escape(fragment)) as caught:
        parse(text)
    assert caught.value.position == position
    assert caught.value.line == 1
    assert caught.value.column == position + 1


# -- vocabulary --------------------------------------------------------------


def test_every_kind_has_a_label_and_type() -> None:
    for kind in NODE_KINDS:
        assert resolve_label(label_for(kind)) == frozenset({kind})
    for edge_kind in EDGE_KINDS:
        assert resolve_rel_type(edge_kind.upper()) == edge_kind
    assert NODE_LABELS["Callable"] == frozenset({"function", "method"})
    assert REL_TYPES["HTTP_CALLS"] == "http_calls"
    assert label_for("function") == "Function"
    assert resolve_label("function") == frozenset({"function"})
    assert resolve_label("Nope") is None
    assert canonical_property("file_path") == "path"
    assert canonical_property("qualified_name") == "qname"


# -- parser: patterns --------------------------------------------------------


def _match(text: str, index: int = 0) -> Match:
    clause = parse(text).clauses[index]
    assert isinstance(clause, Match)
    return clause


def test_node_pattern_labels_props_and_callable() -> None:
    match = _match("MATCH (f:Callable {name: 'run', line_start: 3}) RETURN f")
    node = match.patterns[0].nodes[0]
    assert node.var == "f"
    assert node.kinds == frozenset({"function", "method"})
    assert node.props == (("name", Const("run")), ("line_start", Const(3)))


def test_multiple_labels_intersect() -> None:
    node = _match("MATCH (f:Callable:Method) RETURN f").patterns[0].nodes[0]
    assert node.kinds == frozenset({"method"})


def test_anonymous_node_and_unlabelled() -> None:
    node = _match("MATCH () RETURN 1").patterns[0].nodes[0]
    assert node.var is None and node.kinds is None and node.props == ()


@pytest.mark.parametrize(
    ("arrow", "direction"),
    [
        ("-[:CALLS]->", "out"),
        ("<-[:CALLS]-", "in"),
        ("-[:CALLS]-", "both"),
        ("<-[:CALLS]->", "both"),
    ],
)
def test_relationship_directions(arrow: str, direction: str) -> None:
    rel = _match(f"MATCH (a){arrow}(b) RETURN a").patterns[0].rels[0]
    assert rel.direction == direction
    assert rel.kinds == frozenset({"calls"})


def test_bare_arrows_and_alternatives() -> None:
    match = _match("MATCH (a)-->(b)<--(c)--(d), (x)-[r:CALLS|:TESTS|imports]->(y) RETURN a")
    first, second = match.patterns
    assert [r.direction for r in first.rels] == ["out", "in", "both"]
    assert all(r.kinds is None for r in first.rels)
    rel = second.rels[0]
    assert rel.var == "r"
    assert rel.kinds == frozenset({"calls", "tests", "imports"})


@pytest.mark.parametrize(
    ("hops", "expected"),
    [
        ("*", (1, None)),
        ("*2", (2, 2)),
        ("*1..3", (1, 3)),
        ("*..4", (1, 4)),
        ("*2..", (2, None)),
        ("*0..1", (0, 1)),
    ],
)
def test_variable_length_ranges(hops: str, expected: tuple[int, int | None]) -> None:
    rel = _match(f"MATCH (a)-[:CALLS{hops}]->(b) RETURN b").patterns[0].rels[0]
    assert rel.var_length
    assert (rel.min_hops, rel.max_hops) == expected


def test_relationship_properties() -> None:
    rel = _match("MATCH (a)-[r:CALLS {resolved: true}]->(b) RETURN r").patterns[0].rels[0]
    assert rel.props == (("resolved", Const(True)),)


def test_optional_match_and_where_per_clause() -> None:
    query = parse(
        "MATCH (a) WHERE a.name = 'x' OPTIONAL MATCH (a)-->(b) WHERE b.span > 2 RETURN a, b"
    )
    first, second, final = query.clauses
    assert isinstance(first, Match) and not first.optional and first.where is not None
    assert isinstance(second, Match) and second.optional and second.where is not None
    assert isinstance(final, Return)


# -- parser: expressions -----------------------------------------------------


def _where(condition: str) -> object:
    match = _match(f"MATCH (n) WHERE {condition} RETURN n")
    return match.where


def test_comparison_operators() -> None:
    for symbol, op in [
        ("=", "="),
        ("<>", "<>"),
        ("!=", "<>"),
        ("<", "<"),
        ("<=", "<="),
        (">", ">"),
        (">=", ">="),
        ("=~", "=~"),
    ]:
        expr = _where(f"n.span {symbol} 3")
        assert isinstance(expr, Compare) and expr.op == op
        assert expr.left == Prop("n", "span")


def test_string_predicates_in_and_null_checks() -> None:
    contains = _where("n.name CONTAINS 'a'")
    assert isinstance(contains, Compare) and contains.op == "contains"
    starts = _where("n.name STARTS WITH 'a'")
    assert isinstance(starts, Compare) and starts.op == "starts with"
    ends = _where("n.name ends with 'a'")
    assert isinstance(ends, Compare) and ends.op == "ends with"
    member = _where("n.kind IN ['function', 'method']")
    assert isinstance(member, Compare) and member.op == "in"
    assert member.right == ListExpr((Const("function"), Const("method")))
    is_null = _where("n.docstring IS NULL")
    assert isinstance(is_null, IsNull) and not is_null.negated
    not_null = _where("n.docstring IS NOT NULL")
    assert isinstance(not_null, IsNull) and not_null.negated


def test_boolean_precedence_and_parentheses() -> None:
    expr = _where("n.a = 1 OR n.b = 2 AND NOT n.c = 3")
    assert isinstance(expr, Logical) and expr.op == "or"
    assert isinstance(expr.right, Logical) and expr.right.op == "and"
    assert isinstance(expr.right.right, Not)
    grouped = _where("(n.a = 1 OR n.b = 2) AND n.c = 3")
    assert isinstance(grouped, Logical) and grouped.op == "and"
    assert isinstance(grouped.left, Logical) and grouped.left.op == "or"
    xor = _where("n.a = 1 XOR n.b = 2")
    assert isinstance(xor, Logical) and xor.op == "xor"


def test_literals() -> None:
    expr = _where("n.x IN [1, -2, 2.5, 'a', true, FALSE, null]")
    assert isinstance(expr, Compare) and isinstance(expr.right, ListExpr)
    assert [item.value for item in expr.right.items if isinstance(item, Const)] == [
        1,
        -2,
        2.5,
        "a",
        True,
        False,
        None,
    ]


def test_functions_and_aggregates() -> None:
    final = parse(
        "MATCH (n) RETURN count(*), count(DISTINCT n.kind) AS kinds, toLower(n.name), "
        "coalesce(n.docstring, n.signature, '')"
    ).clauses[-1]
    assert isinstance(final, Return)
    items = final.projection.items
    star, kinds, lower, coalesce = (item.expr for item in items)
    assert isinstance(star, Call) and star.star and star.name == "count"
    assert isinstance(kinds, Call) and kinds.distinct and items[1].alias == "kinds"
    assert isinstance(lower, Call) and lower.name == "tolower" and not lower.is_aggregate
    assert isinstance(coalesce, Call) and len(coalesce.args) == 3


# -- parser: projections -----------------------------------------------------


def test_return_columns_order_skip_limit() -> None:
    final = parse(
        "MATCH (n) RETURN DISTINCT n.name,   n.kind AS k ORDER BY k DESC, n.name SKIP 2 LIMIT 5"
    ).clauses[-1]
    assert isinstance(final, Return)
    projection = final.projection
    assert projection.distinct
    assert [item.column for item in projection.items] == ["n.name", "k"]
    assert [(o.expr, o.descending) for o in projection.order] == [
        (Var("k"), True),
        (Prop("n", "name"), False),
    ]
    assert (projection.skip, projection.limit) == (2, 5)


def test_return_star_and_with() -> None:
    query = parse("MATCH (n) WITH n, n.span AS s WHERE s > 1 RETURN *")
    with_clause, final = query.clauses[1], query.clauses[2]
    assert isinstance(with_clause, With) and with_clause.where is not None
    assert isinstance(final, Return) and final.projection.star


def test_keywords_work_as_property_names() -> None:
    final = parse("MATCH (a)-[r]->(b) RETURN r.type, r.kind, a.label, r.line").clauses[-1]
    assert isinstance(final, Return)
    assert [item.text for item in final.projection.items] == [
        "r.type",
        "r.kind",
        "a.label",
        "r.line",
    ]


def test_trailing_semicolon_is_allowed() -> None:
    assert isinstance(parse("MATCH (n) RETURN n;").clauses[-1], Return)


# -- parser errors -----------------------------------------------------------


@pytest.mark.parametrize("word", ["CREATE", "MERGE", "SET", "DELETE", "REMOVE", "DETACH"])
def test_write_clauses_are_rejected_at_their_position(word: str) -> None:
    text = f"MATCH (n) {word} n RETURN n"
    with pytest.raises(CypherError, match="read-only") as caught:
        parse(text)
    assert caught.value.position == text.index(word)


def test_leading_write_clause_and_lower_case() -> None:
    with pytest.raises(CypherError, match="CREATE is a write clause") as caught:
        parse("create (n:Function {name: 'x'}) return n")
    assert caught.value.position == 0


def test_write_after_return_explains_read_only() -> None:
    with pytest.raises(CypherError, match="read-only"):
        parse("MATCH (n) RETURN n DETACH DELETE n")


@pytest.mark.parametrize("word", ["UNWIND", "UNION", "CALL", "FOREACH"])
def test_unsupported_clauses(word: str) -> None:
    with pytest.raises(CypherError, match="not supported"):
        parse(f"MATCH (n) RETURN n {word} x")


@pytest.mark.parametrize(
    ("text", "fragment", "token"),
    [
        ("MATCH (n:Fucntion) RETURN n", "unknown label 'Fucntion'", "Fucntion"),
        ("MATCH (a)-[:CALL]->(b) RETURN a", "unknown relationship type 'CALL'", "CALL"),
        ("MATCH (n) RETURN frobnicate(n)", "unknown function 'frobnicate'", "frobnicate"),
        ("MATCH (n RETURN n", "expected ')'", "RETURN"),
        ("MATCH (n)", "must end with a RETURN", None),
        ("MATCH (n) RETURN n MATCH (m) RETURN m", "RETURN must be the last", "MATCH (m)"),
        ("WHERE n.x = 1 RETURN n", "WHERE must follow", "WHERE"),
        ("MATCH p = (a)-->(b) RETURN p", "named paths", "p ="),
        ("MATCH (a)-[*3..1]->(b) RETURN a", "is empty", "*"),
        ("MATCH (n) RETURN sum(*)", "only count(*)", "*)"),
        ("MATCH (n) RETURN toLower(DISTINCT n.name)", "only allowed inside aggregate", None),
        ("MATCH (n) RETURN size(n.a, n.b)", "exactly 1 argument", "size"),
        ("MATCH (n) RETURN n LIMIT -1", "non-negative integer", "-1"),
        ("MATCH (n {name: n.qname}) RETURN n", "must be literals", "name"),
        ("MATCH (match) RETURN 1", "found keyword MATCH", "match)"),
        ("", "empty query", None),
        ("MATCH (n) RETURN n.name AS", "expected an alias", None),
    ],
)
def test_parse_errors(text: str, fragment: str, token: str | None) -> None:
    with pytest.raises(CypherError, match=re.escape(fragment)) as caught:
        parse(text)
    error = caught.value
    assert error.position is not None
    if token is not None:
        assert error.position == text.index(token)
    assert error.line == 1
    assert "^" in error.excerpt


def test_error_location_on_later_line() -> None:
    text = "MATCH (n)\nWHERE n.name = 'x'\nRETURN n.nope"
    with pytest.raises(CypherError) as caught:
        prepare(text)
    error = caught.value
    assert (error.line, error.column) == (3, 10)
    assert error.excerpt.splitlines() == ["RETURN n.nope", "         ^"]
    assert "line 3, column 10" in str(error)
    payload = error.as_dict()
    assert payload["line"] == 3 and payload["position"] == text.index("nope")


# -- planner -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "fragment", "token"),
    [
        ("MATCH (n) RETURN m", "unknown variable 'm'", "m"),
        ("MATCH (n) RETURN n.nme", "unknown node property 'nme'", "nme"),
        ("MATCH ()-[r]->() RETURN r.name", "unknown relationship property 'name'", "name"),
        ("MATCH (n {nme: 'x'}) RETURN n", "unknown node property 'nme'", "'x'"),
        ("MATCH ()-[r {weight: 1}]->() RETURN r", "unknown relationship property", "1}"),
        ("MATCH ()-[r*1..2]->() RETURN r.kind", "variable-length relationship", "r.kind"),
        ("MATCH (n) WITH n.name AS x RETURN x.name", "is a value", "x.name"),
        ("MATCH (n) WHERE count(n) > 1 RETURN n", "aggregate", "count"),
        ("MATCH (n) RETURN size(collect(n))", "aggregate", "collect"),
        ("MATCH (n)-[n]->(m) RETURN n", "already bound to a node", "-[n]"),
        ("MATCH (a)-[r*]->(b), (b)-[r*]->(c) RETURN a", "cannot be reused", "-[r*]->(c)"),
        ("MATCH (n) RETURN n.name, count(*) ORDER BY n.kind", "ORDER BY must use", "n.kind"),
        ("MATCH (n) RETURN DISTINCT n.name ORDER BY n.kind", "ORDER BY must use", "n.kind"),
        ("MATCH (n) WITH n.name RETURN 1", "WITH needs an alias", "WITH"),
        ("MATCH (n) RETURN n.name, n.name", "appears twice", "RETURN"),
        ("RETURN *", "needs at least one named variable", "RETURN"),
        ("MATCH (n) WITH n AS m RETURN n", "unknown variable 'n'", "n"),
    ],
)
def test_planner_errors(text: str, fragment: str, token: str) -> None:
    with pytest.raises(CypherError, match=re.escape(fragment)) as caught:
        prepare(text)
    assert caught.value.position == text.rindex(token)


def test_planner_accepts_valid_scoping() -> None:
    plan = prepare(
        "MATCH (f:Function) OPTIONAL MATCH (f)<-[:CALLS]-(c) "
        "WITH f, count(c) AS callers WHERE callers = 0 "
        "RETURN f.qname AS name, callers ORDER BY name"
    )
    match_steps = [s for s in plan.steps if isinstance(s, MatchStep)]
    assert [s.new_vars for s in match_steps] == [("f",), ("c",)]
    assert prepare("MATCH (n) RETURN n.name AS x ORDER BY n.span").steps
    assert prepare("MATCH (n) RETURN n.kind, count(*) AS c ORDER BY c DESC, n.kind").steps
    assert prepare("MATCH (n) RETURN n.qualified_name, n.file_path").steps
    assert prepare("RETURN 1 AS one").steps
