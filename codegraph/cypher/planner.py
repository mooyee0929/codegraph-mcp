"""Static checks between parsing and execution.

The planner walks the clauses once, tracking which variables are in scope and whether
each is a node, a single relationship, a variable-length path or a plain value. That
catches the mistakes agents actually make — a misspelled variable, ``r.name`` on a
relationship, ``count()`` inside ``WHERE``, ``ORDER BY`` on something an aggregating
``RETURN`` threw away — before any matching starts, with the position of the culprit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from codegraph.cypher.errors import CypherError
from codegraph.cypher.syntax import (
    Call,
    Expr,
    Match,
    OrderItem,
    Projection,
    Prop,
    Query,
    Return,
    ReturnItem,
    Var,
    With,
    children,
)
from codegraph.cypher.vocab import (
    EDGE_PROPERTIES,
    NODE_PROPERTIES,
    is_edge_property,
    is_node_property,
)

VarType = Literal["node", "rel", "path", "value"]


@dataclass(frozen=True, slots=True)
class MatchStep:
    clause: Match
    new_vars: tuple[str, ...]
    """Variables this clause introduces — set to null when an OPTIONAL MATCH misses."""


@dataclass(frozen=True, slots=True)
class ProjectStep:
    clause: With | Return
    items: tuple[ReturnItem, ...]
    """Items with ``*`` expanded to every variable in scope."""
    aggregating: bool


Step = MatchStep | ProjectStep


@dataclass(frozen=True, slots=True)
class Plan:
    query: Query
    steps: tuple[Step, ...]


def plan(query: Query) -> Plan:
    try:
        return _plan(query)
    except CypherError as exc:
        raise exc.locate(query.text) from None


def _plan(query: Query) -> Plan:
    scope: dict[str, VarType] = {}
    steps: list[Step] = []
    for clause in query.clauses:
        if isinstance(clause, Match):
            steps.append(_plan_match(clause, scope))
        else:
            step, scope = _plan_projection(clause, scope)
            steps.append(step)
    return Plan(query=query, steps=tuple(steps))


def _plan_match(clause: Match, scope: dict[str, VarType]) -> MatchStep:
    new_vars: list[str] = []

    def bind(name: str | None, kind: VarType, pos: int) -> None:
        if name is None:
            return
        existing = scope.get(name)
        if existing is None:
            scope[name] = kind
            new_vars.append(name)
        elif existing != kind:
            raise CypherError(f"variable '{name}' is already bound to a {existing}", pos)
        elif kind == "path":
            raise CypherError(f"variable-length relationship '{name}' cannot be reused", pos)

    for path in clause.patterns:
        for index, node in enumerate(path.nodes):
            if index > 0:
                rel = path.rels[index - 1]
                bind(rel.var, "path" if rel.var_length else "rel", rel.pos)
                for key, value in rel.props:
                    if not is_edge_property(key):
                        raise CypherError(_unknown_property("relationship", key), value.pos)
            bind(node.var, "node", node.pos)
            for key, value in node.props:
                if not is_node_property(key):
                    raise CypherError(_unknown_property("node", key), value.pos)
    if clause.where is not None:
        _check(clause.where, scope, allow_aggregate=False)
    return MatchStep(clause=clause, new_vars=tuple(new_vars))


def _plan_projection(
    clause: With | Return, scope: dict[str, VarType]
) -> tuple[ProjectStep, dict[str, VarType]]:
    projection = clause.projection
    keyword = "WITH" if isinstance(clause, With) else "RETURN"
    items = _expand_star(projection, scope, keyword)
    columns: dict[str, VarType] = {}
    for item in items:
        _check(item.expr, scope, allow_aggregate=True)
        if isinstance(clause, With) and item.alias is None and not isinstance(item.expr, Var):
            raise CypherError(f"WITH needs an alias for '{item.text}' (add AS name)", clause.pos)
        if item.column in columns:
            raise CypherError(
                f"column '{item.column}' appears twice in {keyword}; add AS aliases", clause.pos
            )
        expr = item.expr
        columns[item.column] = scope[expr.name] if isinstance(expr, Var) else "value"
    aggregating = any(_is_aggregate(item.expr) for item in items)
    for order in projection.order:
        _check_order(order, items, scope, columns, grouped=aggregating or projection.distinct)
    step = ProjectStep(clause=clause, items=items, aggregating=aggregating)
    if isinstance(clause, With) and clause.where is not None:
        _check(clause.where, columns, allow_aggregate=False)
    return step, dict(columns)


def _expand_star(
    projection: Projection, scope: dict[str, VarType], keyword: str
) -> tuple[ReturnItem, ...]:
    if not projection.star:
        return projection.items
    if not scope:
        raise CypherError(f"{keyword} * needs at least one named variable", projection.pos)
    return tuple(ReturnItem(expr=Var(name), alias=None, text=name) for name in scope)


def _check_order(
    order: OrderItem,
    items: tuple[ReturnItem, ...],
    scope: dict[str, VarType],
    columns: dict[str, VarType],
    *,
    grouped: bool,
) -> None:
    expr = order.expr
    if isinstance(expr, Var) and expr.name in columns:
        return
    if any(item.expr == expr for item in items):
        return
    if grouped:
        raise CypherError(
            "with aggregation or DISTINCT, ORDER BY must use a returned column or alias",
            expr.pos,
        )
    _check(expr, {**scope, **columns}, allow_aggregate=False)


def _check(expr: Expr, scope: dict[str, VarType], *, allow_aggregate: bool) -> None:
    if isinstance(expr, Var):
        if expr.name not in scope:
            raise CypherError(f"unknown variable '{expr.name}'", expr.pos)
        return
    if isinstance(expr, Prop):
        _check_property(expr, scope)
        return
    if isinstance(expr, Call) and expr.is_aggregate:
        if not allow_aggregate:
            raise CypherError(
                f"{expr.name}() is an aggregate and may only be a whole RETURN or WITH item",
                expr.pos,
            )
        for arg in expr.args:
            _check(arg, scope, allow_aggregate=False)
        return
    for child in children(expr):
        _check(child, scope, allow_aggregate=False)


def _check_property(expr: Prop, scope: dict[str, VarType]) -> None:
    kind = scope.get(expr.var)
    if kind is None:
        raise CypherError(f"unknown variable '{expr.var}'", expr.pos)
    if kind == "node" and not is_node_property(expr.prop):
        raise CypherError(_unknown_property("node", expr.prop), expr.prop_pos)
    if kind == "rel" and not is_edge_property(expr.prop):
        raise CypherError(_unknown_property("relationship", expr.prop), expr.prop_pos)
    if kind == "path":
        raise CypherError(
            f"'{expr.var}' is a variable-length relationship (a list of edges) and has no "
            f"properties; use size({expr.var}) for the hop count",
            expr.pos,
        )
    if kind == "value":
        raise CypherError(f"'{expr.var}' is a value, not a node or relationship", expr.pos)


def _unknown_property(owner: str, name: str) -> str:
    known = NODE_PROPERTIES if owner == "node" else EDGE_PROPERTIES
    return f"unknown {owner} property '{name}'; available: {', '.join(known)}"


def _is_aggregate(expr: Expr) -> bool:
    return isinstance(expr, Call) and expr.is_aggregate


__all__ = ["MatchStep", "Plan", "ProjectStep", "Step", "VarType", "plan"]
