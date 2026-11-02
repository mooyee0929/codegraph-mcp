"""Recursive-descent parser for the read-only Cypher subset.

Grammar (keywords case-insensitive)::

    query      := clause+ ';'? EOF            -- must end with RETURN
    clause     := OPTIONAL? MATCH path (',' path)* (WHERE expr)?
                | WITH projection (WHERE expr)?
                | RETURN projection
    path       := node (rel node)*
    node       := '(' var? (':' Label)* props? ')'
    rel        := '<'? '-' ('[' var? (':' TYPE ('|' ':'? TYPE)*)? range? props? ']')? '-' '>'?
    range      := '*' (INT? '..' INT? | INT)?
    props      := '{' key ':' literal (',' key ':' literal)* '}'
    projection := DISTINCT? ('*' | item (',' item)*) (ORDER BY order (',' order)*)?
                  (SKIP INT)? (LIMIT INT)?
    item       := expr (AS name)?
    expr       := xor (OR xor)* ; xor := and (XOR and)* ; and := not (AND not)*
    not        := NOT not | cmp
    cmp        := atom (op atom | IS NOT? NULL | IN atom | CONTAINS atom
                        | STARTS WITH atom | ENDS WITH atom)?
    atom       := literal | '[' list ']' | '(' expr ')' | fn '(' args ')' | var ('.' prop)?

Write and admin clauses are recognised by name so the error says *why* they fail
(codegraph's graph is derived from source; edit the code and re-index instead).
"""

from __future__ import annotations

from typing import NoReturn

from codegraph.cypher.errors import CypherError
from codegraph.cypher.lexer import Token, tokenize
from codegraph.cypher.syntax import (
    AGGREGATES,
    SCALAR_FUNCTIONS,
    Call,
    Clause,
    Compare,
    CompareOp,
    Const,
    Direction,
    Expr,
    IsNull,
    ListExpr,
    Logical,
    Match,
    NodePattern,
    Not,
    OrderItem,
    PathPattern,
    Projection,
    Prop,
    Query,
    RelPattern,
    Return,
    ReturnItem,
    Var,
    With,
)
from codegraph.cypher.vocab import NODE_LABELS, REL_TYPES, resolve_label, resolve_rel_type
from codegraph.models import EdgeKind, NodeKind

WRITE_CLAUSES: frozenset[str] = frozenset({"CREATE", "MERGE", "SET", "DELETE", "REMOVE", "DETACH"})
UNSUPPORTED_CLAUSES: frozenset[str] = frozenset(
    {"UNWIND", "UNION", "CALL", "FOREACH", "LOAD", "USE", "SHOW", "DROP", "MANDATORY", "YIELD"}
)
RESERVED: frozenset[str] = frozenset(
    {
        "MATCH",
        "OPTIONAL",
        "WHERE",
        "RETURN",
        "WITH",
        "ORDER",
        "BY",
        "SKIP",
        "LIMIT",
        "AND",
        "OR",
        "XOR",
        "NOT",
        "AS",
        "DISTINCT",
        "IN",
        "IS",
        "NULL",
        "TRUE",
        "FALSE",
        "CONTAINS",
        "STARTS",
        "ENDS",
        "ASC",
        "DESC",
        "ASCENDING",
        "DESCENDING",
    }
    | WRITE_CLAUSES
    | UNSUPPORTED_CLAUSES
)

_COMPARE_SYMBOLS: dict[str, CompareOp] = {
    "=": "=",
    "<>": "<>",
    "!=": "<>",
    "<": "<",
    "<=": "<=",
    ">": ">",
    ">=": ">=",
    "=~": "=~",
}


def parse(text: str) -> Query:
    """Parse ``text`` into a :class:`Query`; errors carry the offending position."""
    try:
        return _Parser(text, tokenize(text)).query()
    except CypherError as exc:
        raise exc.locate(text) from None


class _Parser:
    def __init__(self, text: str, tokens: list[Token]) -> None:
        self.text = text
        self.tokens = tokens
        self.index = 0
        self.prev_end = 0

    # -- token plumbing ----------------------------------------------------

    def peek(self, ahead: int = 0) -> Token:
        return self.tokens[min(self.index + ahead, len(self.tokens) - 1)]

    def advance(self) -> Token:
        token = self.peek()
        if token.kind != "eof":
            self.index += 1
            self.prev_end = token.end
        return token

    def fail(self, message: str, token: Token | None = None) -> NoReturn:
        raise CypherError(message, (token or self.peek()).pos)

    def accept_symbol(self, *symbols: str) -> Token | None:
        if self.peek().is_symbol(*symbols):
            return self.advance()
        return None

    def expect_symbol(self, symbol: str, context: str) -> Token:
        token = self.peek()
        if not token.is_symbol(symbol):
            self.fail(f"expected '{symbol}' {context}, found {token.describe()}")
        return self.advance()

    def accept_keyword(self, *words: str) -> Token | None:
        if self.peek().is_keyword(*words):
            return self.advance()
        return None

    def expect_keyword(self, word: str, context: str) -> Token:
        token = self.peek()
        if not token.is_keyword(word):
            self.fail(f"expected {word} {context}, found {token.describe()}")
        return self.advance()

    def name(self, what: str) -> Token:
        """An identifier used as a variable or alias: reserved words are rejected."""
        token = self.peek()
        if token.kind != "ident":
            self.fail(f"expected {what}, found {token.describe()}")
        if not token.quoted and token.text.upper() in RESERVED:
            self.fail(f"expected {what}, found keyword {token.text.upper()}")
        return self.advance()

    def word(self, what: str) -> Token:
        """Any identifier, keywords included (labels, types, property keys)."""
        token = self.peek()
        if token.kind != "ident":
            self.fail(f"expected {what}, found {token.describe()}")
        return self.advance()

    def integer(self, what: str) -> int:
        token = self.peek()
        if token.kind != "number" or not token.text.isdigit():
            self.fail(f"expected a non-negative integer for {what}, found {token.describe()}")
        self.advance()
        return int(token.text)

    # -- clauses -----------------------------------------------------------

    def query(self) -> Query:
        clauses: list[Clause] = []
        while True:
            token = self.peek()
            if token.kind == "eof" or token.is_symbol(";"):
                break
            if clauses and isinstance(clauses[-1], Return) and not _clause_keyword(token):
                self.fail(f"RETURN must be the last clause, found {token.describe()}")
            clauses.append(self.clause())
        self.accept_symbol(";")
        if self.peek().kind != "eof":
            self.fail(f"unexpected {self.peek().describe()} after the end of the query")
        if not clauses:
            self.fail("empty query; expected MATCH ... RETURN ...")
        if not isinstance(clauses[-1], Return):
            self.fail("query must end with a RETURN clause")
        return Query(clauses=tuple(clauses), text=self.text)

    def clause(self) -> Clause:
        token = self.peek()
        if token.is_keyword("MATCH"):
            self.advance()
            return self.match(optional=False, pos=token.pos)
        if token.is_keyword("OPTIONAL"):
            self.advance()
            self.expect_keyword("MATCH", "after OPTIONAL")
            return self.match(optional=True, pos=token.pos)
        if token.is_keyword("WITH"):
            self.advance()
            projection = self.projection(token.pos, "WITH")
            where = self.expression() if self.accept_keyword("WHERE") else None
            return With(projection=projection, where=where, pos=token.pos)
        if token.is_keyword("RETURN"):
            self.advance()
            return Return(projection=self.projection(token.pos, "RETURN"), pos=token.pos)
        word = token.text.upper() if token.kind == "ident" and not token.quoted else ""
        if word in WRITE_CLAUSES:
            self.fail(
                f"{word} is a write clause and the graph is read-only: it is derived from "
                "source, so change the code and re-index instead"
            )
        if word in UNSUPPORTED_CLAUSES:
            self.fail(f"{word} is not supported; use MATCH, OPTIONAL MATCH, WITH and RETURN")
        if token.is_keyword("WHERE"):
            self.fail("WHERE must follow MATCH, OPTIONAL MATCH or WITH")
        self.fail(f"expected MATCH, OPTIONAL MATCH, WITH or RETURN, found {token.describe()}")

    def match(self, *, optional: bool, pos: int) -> Match:
        patterns = [self.path()]
        while self.accept_symbol(","):
            patterns.append(self.path())
        where = self.expression() if self.accept_keyword("WHERE") else None
        return Match(patterns=tuple(patterns), optional=optional, where=where, pos=pos)

    # -- patterns ----------------------------------------------------------

    def path(self) -> PathPattern:
        start = self.peek()
        if start.kind == "ident" and self.peek(1).is_symbol("="):
            self.fail("named paths (p = ...) are not supported; bind the relationship instead")
        nodes = [self.node()]
        rels: list[RelPattern] = []
        while self.peek().is_symbol("-", "<"):
            rels.append(self.rel())
            nodes.append(self.node())
        return PathPattern(nodes=tuple(nodes), rels=tuple(rels), pos=start.pos)

    def node(self) -> NodePattern:
        open_paren = self.expect_symbol("(", "to start a node pattern")
        var = self.name("a variable").text if self.peek().kind == "ident" else None
        kinds: frozenset[NodeKind] | None = None
        while self.accept_symbol(":"):
            label = self.word("a label")
            matched = resolve_label(label.text)
            if matched is None:
                known = ", ".join(sorted(NODE_LABELS))
                self.fail(f"unknown label '{label.text}'; known labels: {known}", label)
            kinds = matched if kinds is None else kinds & matched
        props = self.properties() if self.peek().is_symbol("{") else ()
        self.expect_symbol(")", "to close the node pattern")
        return NodePattern(var=var, kinds=kinds, props=props, pos=open_paren.pos)

    def rel(self) -> RelPattern:
        start = self.peek()
        left = self.accept_symbol("<") is not None
        self.expect_symbol("-", "in a relationship pattern")
        var: str | None = None
        kinds: frozenset[EdgeKind] | None = None
        props: tuple[tuple[str, Const], ...] = ()
        var_length = False
        hops: tuple[int, int | None] = (1, 1)
        if self.accept_symbol("["):
            if self.peek().kind == "ident":
                var = self.name("a relationship variable").text
            if self.accept_symbol(":"):
                kinds = frozenset(self.rel_types())
            if self.peek().is_symbol("*"):
                var_length = True
                hops = self.hop_range()
            if self.peek().is_symbol("{"):
                props = self.properties()
            self.expect_symbol("]", "to close the relationship pattern")
        self.expect_symbol("-", "in a relationship pattern")
        right = self.accept_symbol(">") is not None
        direction: Direction = "both" if left == right else ("in" if left else "out")
        min_hops, max_hops = hops
        return RelPattern(
            var=var,
            kinds=kinds,
            direction=direction,
            props=props,
            pos=start.pos,
            var_length=var_length,
            min_hops=min_hops,
            max_hops=max_hops,
        )

    def rel_types(self) -> list[EdgeKind]:
        found = [self.rel_type()]
        while self.accept_symbol("|"):
            self.accept_symbol(":")
            found.append(self.rel_type())
        return found

    def rel_type(self) -> EdgeKind:
        token = self.word("a relationship type")
        kind = resolve_rel_type(token.text)
        if kind is None:
            known = ", ".join(sorted(REL_TYPES))
            self.fail(f"unknown relationship type '{token.text}'; known types: {known}", token)
        return kind

    def hop_range(self) -> tuple[int, int | None]:
        star = self.expect_symbol("*", "for a variable-length relationship")
        low: int | None = None
        high: int | None = None
        if self.peek().kind == "number":
            low = self.integer("the minimum hop count")
            if not self.peek().is_symbol(".."):
                return low, low
        if self.accept_symbol("..") and self.peek().kind == "number":
            high = self.integer("the maximum hop count")
        minimum = 1 if low is None else low
        if high is not None and high < minimum:
            self.fail(f"hop range *{minimum}..{high} is empty", star)
        return minimum, high

    def properties(self) -> tuple[tuple[str, Const], ...]:
        self.expect_symbol("{", "to start a property map")
        pairs: list[tuple[str, Const]] = []
        if not self.peek().is_symbol("}"):
            while True:
                key = self.word("a property name")
                self.expect_symbol(":", f"after property '{key.text}'")
                value = self.atom()
                if not isinstance(value, Const):
                    self.fail("inline property values must be literals", key)
                pairs.append((key.text, value))
                if not self.accept_symbol(","):
                    break
        self.expect_symbol("}", "to close the property map")
        return tuple(pairs)

    # -- projections -------------------------------------------------------

    def projection(self, pos: int, clause: str) -> Projection:
        distinct = self.accept_keyword("DISTINCT") is not None
        items: list[ReturnItem] = []
        star = self.accept_symbol("*") is not None
        if not star:
            items.append(self.return_item())
            while self.accept_symbol(","):
                items.append(self.return_item())
        order: list[OrderItem] = []
        if self.accept_keyword("ORDER"):
            self.expect_keyword("BY", "after ORDER")
            order.append(self.order_item())
            while self.accept_symbol(","):
                order.append(self.order_item())
        skip = self.integer(f"{clause} ... SKIP") if self.accept_keyword("SKIP") else None
        limit = self.integer(f"{clause} ... LIMIT") if self.accept_keyword("LIMIT") else None
        return Projection(
            items=tuple(items),
            distinct=distinct,
            star=star,
            order=tuple(order),
            skip=skip,
            limit=limit,
            pos=pos,
        )

    def return_item(self) -> ReturnItem:
        start = self.peek().pos
        expr = self.expression()
        text = self.text[start : self.prev_end]
        alias = self.name("an alias").text if self.accept_keyword("AS") else None
        return ReturnItem(expr=expr, alias=alias, text=" ".join(text.split()))

    def order_item(self) -> OrderItem:
        expr = self.expression()
        descending = False
        if self.accept_keyword("DESC", "DESCENDING"):
            descending = True
        else:
            self.accept_keyword("ASC", "ASCENDING")
        return OrderItem(expr=expr, descending=descending)

    # -- expressions -------------------------------------------------------

    def expression(self) -> Expr:
        left = self.xor_expr()
        while token := self.accept_keyword("OR"):
            left = Logical("or", left, self.xor_expr(), token.pos)
        return left

    def xor_expr(self) -> Expr:
        left = self.and_expr()
        while token := self.accept_keyword("XOR"):
            left = Logical("xor", left, self.and_expr(), token.pos)
        return left

    def and_expr(self) -> Expr:
        left = self.not_expr()
        while token := self.accept_keyword("AND"):
            left = Logical("and", left, self.not_expr(), token.pos)
        return left

    def not_expr(self) -> Expr:
        if token := self.accept_keyword("NOT"):
            return Not(self.not_expr(), token.pos)
        return self.comparison()

    def comparison(self) -> Expr:
        left = self.atom()
        token = self.peek()
        if token.kind == "symbol" and token.text in _COMPARE_SYMBOLS:
            self.advance()
            return Compare(_COMPARE_SYMBOLS[token.text], left, self.atom(), token.pos)
        if token.is_keyword("IS"):
            self.advance()
            negated = self.accept_keyword("NOT") is not None
            self.expect_keyword("NULL", "after IS" + (" NOT" if negated else ""))
            return IsNull(left, negated, token.pos)
        if token.is_keyword("IN"):
            self.advance()
            return Compare("in", left, self.atom(), token.pos)
        if token.is_keyword("CONTAINS"):
            self.advance()
            return Compare("contains", left, self.atom(), token.pos)
        if token.is_keyword("STARTS", "ENDS"):
            self.advance()
            self.expect_keyword("WITH", f"after {token.text.upper()}")
            op: CompareOp = "starts with" if token.is_keyword("STARTS") else "ends with"
            return Compare(op, left, self.atom(), token.pos)
        return left

    def atom(self) -> Expr:
        token = self.peek()
        if token.kind == "string":
            self.advance()
            return Const(token.text, token.pos)
        if token.kind == "number":
            self.advance()
            return Const(_number(token.text), token.pos)
        if token.is_symbol("-") and self.peek(1).kind == "number":
            self.advance()
            number = self.advance()
            return Const(-_number(number.text), token.pos)
        if token.is_symbol("["):
            return self.list_literal()
        if token.is_symbol("("):
            self.advance()
            inner = self.expression()
            self.expect_symbol(")", "to close the parenthesised expression")
            return inner
        if token.is_keyword("TRUE", "FALSE"):
            self.advance()
            return Const(token.text.upper() == "TRUE", token.pos)
        if token.is_keyword("NULL"):
            self.advance()
            return Const(None, token.pos)
        if token.kind == "ident" and self.peek(1).is_symbol("("):
            return self.call()
        if token.kind == "ident":
            var = self.name("an expression")
            if self.accept_symbol("."):
                prop = self.word("a property name")
                return Prop(var.text, prop.text, var.pos, prop.pos)
            return Var(var.text, var.pos)
        self.fail(f"expected an expression, found {token.describe()}")

    def list_literal(self) -> ListExpr:
        open_bracket = self.expect_symbol("[", "to start a list")
        items: list[Expr] = []
        if not self.peek().is_symbol("]"):
            items.append(self.expression())
            while self.accept_symbol(","):
                items.append(self.expression())
        self.expect_symbol("]", "to close the list")
        return ListExpr(tuple(items), open_bracket.pos)

    def call(self) -> Call:
        token = self.advance()
        name = token.text.lower()
        if name not in AGGREGATES and name not in SCALAR_FUNCTIONS:
            known = ", ".join(sorted(AGGREGATES | SCALAR_FUNCTIONS))
            self.fail(f"unknown function '{token.text}'; supported: {known}", token)
        self.expect_symbol("(", f"after {token.text}")
        if self.peek().is_symbol("*"):
            star = self.advance()
            if name != "count":
                self.fail(f"only count(*) takes '*', not {token.text}(*)", star)
            self.expect_symbol(")", "after count(*")
            return Call(name, (), star=True, pos=token.pos)
        distinct = self.accept_keyword("DISTINCT") is not None
        if distinct and name not in AGGREGATES:
            self.fail(f"DISTINCT is only allowed inside aggregate functions, not {token.text}")
        args: list[Expr] = []
        if not self.peek().is_symbol(")"):
            args.append(self.expression())
            while self.accept_symbol(","):
                args.append(self.expression())
        self.expect_symbol(")", f"to close {token.text}(")
        expected = None if name == "coalesce" else 1
        if expected is not None and len(args) != expected:
            self.fail(f"{token.text}() takes exactly {expected} argument", token)
        if name == "coalesce" and not args:
            self.fail("coalesce() needs at least one argument", token)
        return Call(name, tuple(args), distinct=distinct, pos=token.pos)


def _clause_keyword(token: Token) -> bool:
    """A write or unsupported clause keyword, which :meth:`_Parser.clause` explains."""
    return (
        token.kind == "ident"
        and not token.quoted
        and (token.text.upper() in WRITE_CLAUSES or token.text.upper() in UNSUPPORTED_CLAUSES)
    )


def _number(text: str) -> int | float:
    return float(text) if "." in text else int(text)


__all__ = ["RESERVED", "UNSUPPORTED_CLAUSES", "WRITE_CLAUSES", "parse"]
