"""Run a planned query against an in-memory :class:`~codegraph.graph.GraphView`.

Execution is a pipeline of generators over *bindings* (variable -> value dicts): each
``MATCH`` extends every incoming binding with the pattern's matches, ``WITH`` projects
and filters, ``RETURN`` projects into rows. Because the stages are lazy, a plain
``MATCH ... RETURN ... LIMIT 5`` stops after five matches; only ``ORDER BY``,
``DISTINCT`` and aggregation need to see every row.

Two deliberate departures from full Cypher keep queries over a code graph cheap and
predictable:

- A variable-length relationship (``-[:CALLS*1..3]->``) yields every reachable node
  once, through its shortest path, instead of every distinct path. ``count()`` over it
  therefore counts nodes, which is what "how many functions reach X" means.
- Work is budgeted. Each candidate node or edge examined costs one unit; a query that
  exceeds :attr:`Limits.max_work` fails with advice instead of running for minutes.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from itertools import islice
from typing import TypeAlias

from codegraph.cypher.errors import CypherError
from codegraph.cypher.planner import MatchStep, Plan, ProjectStep
from codegraph.cypher.syntax import (
    Call,
    Compare,
    Const,
    Direction,
    Expr,
    IsNull,
    ListExpr,
    Logical,
    NodePattern,
    Not,
    OrderItem,
    PathPattern,
    Prop,
    RelPattern,
    ReturnItem,
    Scalar,
    Var,
    With,
)
from codegraph.cypher.vocab import canonical_property, label_for, rel_type_for
from codegraph.graph import GraphView
from codegraph.models import Edge, Node, NodeKind

Value: TypeAlias = Scalar | Node | Edge | tuple["Value", ...]
Binding: TypeAlias = dict[str, Value]
SortKey: TypeAlias = tuple[int, float, str]


@dataclass(frozen=True, slots=True)
class Limits:
    default_rows: int = 200
    """Rows returned when the query has no LIMIT; the result says if more existed."""
    max_rows: int = 5000
    """Hard ceiling on returned rows; a larger LIMIT is clamped with a warning."""
    max_hops: int = 6
    """Ceiling on variable-length relationships; ``*`` and ``*1..50`` both clamp to it."""
    max_work: int = 1_000_000
    """Candidate nodes and edges a query may examine before it is aborted."""


@dataclass(frozen=True, slots=True)
class CypherResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Value, ...], ...]
    truncated: bool = False
    """True when the default row cap cut the result short."""
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "columns": list(self.columns),
            "rows": [[to_json(value) for value in row] for row in self.rows],
            "row_count": len(self.rows),
            "truncated": self.truncated,
            "warnings": list(self.warnings),
        }


def to_json(value: Value) -> object:
    if isinstance(value, Node):
        return {
            "qname": value.qname,
            "kind": value.kind,
            "name": value.name,
            "path": value.path,
            "line_start": value.line_start,
            "line_end": value.line_end,
        }
    if isinstance(value, Edge):
        return {
            "type": rel_type_for(value.kind),
            "src": value.src,
            "dst": value.dst,
            "path": value.path,
            "line": value.line,
            "confidence": value.confidence,
            "detail": value.detail,
        }
    if isinstance(value, tuple):
        return [to_json(item) for item in value]
    return value


def render_value(value: Value) -> str:
    """Compact text for CLI tables: nodes by qname, edges as ``src-[TYPE]->dst``."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Node):
        return value.qname
    if isinstance(value, Edge):
        return f"{value.src}-[{rel_type_for(value.kind)}]->{value.dst}"
    if isinstance(value, tuple):
        return "[" + ", ".join(render_value(item) for item in value) + "]"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def execute_plan(view: GraphView, plan: Plan, limits: Limits) -> CypherResult:
    try:
        return _Executor(view, limits).run(plan)
    except CypherError as exc:
        raise exc.locate(plan.query.text) from None


class _Executor:
    def __init__(self, view: GraphView, limits: Limits) -> None:
        self.view = view
        self.limits = limits
        self.work = 0
        self.warnings: list[str] = []
        self._by_kind: dict[NodeKind, list[Node]] | None = None
        self._by_name: dict[str, list[Node]] | None = None
        self._regex: dict[str, re.Pattern[str]] = {}
        self._degrees: dict[tuple[str, str], int] = {}

    def run(self, plan: Plan) -> CypherResult:
        rows: Iterable[Binding] = [{}]
        *leading, final = plan.steps
        for step in leading:
            if isinstance(step, MatchStep):
                rows = self._match(step, rows)
            else:
                rows = self._with(step, rows)
        if not isinstance(final, ProjectStep) or isinstance(final.clause, With):
            raise CypherError("query must end with a RETURN clause")
        return self._return(final, rows)

    # -- budget and indexes ------------------------------------------------

    def _tick(self) -> None:
        self.work += 1
        if self.work > self.limits.max_work:
            raise CypherError(
                f"query is too broad: it examined more than {self.limits.max_work} candidate "
                "nodes and edges; add labels, inline properties such as {name: 'x'}, a "
                "WHERE filter, or a LIMIT without ORDER BY"
            )

    def _nodes_of_kind(self, kind: NodeKind) -> list[Node]:
        if self._by_kind is None:
            index: dict[NodeKind, list[Node]] = {}
            for node in self.view.nodes.values():
                index.setdefault(node.kind, []).append(node)
            self._by_kind = index
        return self._by_kind.get(kind, [])

    def _nodes_named(self, name: str) -> list[Node]:
        if self._by_name is None:
            index: dict[str, list[Node]] = {}
            for node in self.view.nodes.values():
                index.setdefault(node.name, []).append(node)
            self._by_name = index
        return self._by_name.get(name, [])

    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    # -- MATCH ---------------------------------------------------------------

    def _match(self, step: MatchStep, rows: Iterable[Binding]) -> Iterator[Binding]:
        clause = step.clause
        for row in rows:
            matched = False
            for binding in self._match_patterns(clause.patterns, 0, row, frozenset()):
                if clause.where is not None and self._truth(clause.where, binding) is not True:
                    continue
                matched = True
                yield binding
            if clause.optional and not matched:
                missing: Binding = dict.fromkeys(step.new_vars)
                yield {**row, **missing}

    def _match_patterns(
        self,
        patterns: tuple[PathPattern, ...],
        index: int,
        binding: Binding,
        used: frozenset[Edge],
    ) -> Iterator[Binding]:
        if index == len(patterns):
            yield binding
            return
        for extended, now_used in self._match_path(patterns[index], binding, used):
            yield from self._match_patterns(patterns, index + 1, extended, now_used)

    def _match_path(
        self, path: PathPattern, binding: Binding, used: frozenset[Edge]
    ) -> Iterator[tuple[Binding, frozenset[Edge]]]:
        nodes, rels = path.nodes, path.rels
        if rels and _anchor_cost(nodes[-1], binding) < _anchor_cost(nodes[0], binding):
            nodes = nodes[::-1]
            rels = tuple(_flipped(rel) for rel in reversed(rels))
        for start in self._candidates(nodes[0], binding):
            bound = _bind(binding, nodes[0].var, start)
            if bound is not None:
                yield from self._walk(nodes, rels, 0, start, bound, used)

    def _walk(
        self,
        nodes: tuple[NodePattern, ...],
        rels: tuple[RelPattern, ...],
        index: int,
        current: Node,
        binding: Binding,
        used: frozenset[Edge],
    ) -> Iterator[tuple[Binding, frozenset[Edge]]]:
        if index == len(rels):
            yield binding, used
            return
        rel, target = rels[index], nodes[index + 1]
        if rel.var_length:
            for end, edges in self._reach(current, rel):
                if not self._node_fits(target, end):
                    continue
                extended = _bind(binding, rel.var, edges)
                extended = _bind(extended, target.var, end) if extended is not None else None
                if extended is not None:
                    yield from self._walk(nodes, rels, index + 1, end, extended, used)
            return
        for edge, other in self._step(current, rel):
            if edge in used or not self._node_fits(target, other):
                continue
            extended = _bind(binding, rel.var, edge)
            extended = _bind(extended, target.var, other) if extended is not None else None
            if extended is not None:
                yield from self._walk(nodes, rels, index + 1, other, extended, used | {edge})

    def _candidates(self, pattern: NodePattern, binding: Binding) -> Iterator[Node]:
        if pattern.var is not None and pattern.var in binding:
            value = binding[pattern.var]
            if isinstance(value, Node) and self._node_fits(pattern, value):
                yield value
            return
        props = {canonical_property(key): const.value for key, const in pattern.props}
        pool: Iterable[Node]
        if isinstance(props.get("qname"), str):
            found = self.view.nodes.get(str(props["qname"]))
            pool = [found] if found is not None else []
        elif isinstance(props.get("name"), str):
            pool = self._nodes_named(str(props["name"]))
        elif pattern.kinds is not None:
            pool = [node for kind in sorted(pattern.kinds) for node in self._nodes_of_kind(kind)]
        else:
            pool = self.view.nodes.values()
        for node in pool:
            self._tick()
            if self._node_fits(pattern, node):
                yield node

    def _node_fits(self, pattern: NodePattern, node: Node) -> bool:
        if pattern.kinds is not None and node.kind not in pattern.kinds:
            return False
        return all(
            _equals(self._node_property(node, key), const.value) is True
            for key, const in pattern.props
        )

    def _edge_fits(self, rel: RelPattern, edge: Edge) -> bool:
        return all(
            _equals(_edge_property(edge, key), const.value) is True for key, const in rel.props
        )

    def _step(self, current: Node, rel: RelPattern) -> Iterator[tuple[Edge, Node]]:
        if rel.direction in ("out", "both"):
            for edge in self.view.edges_out(current.qname, rel.kinds):
                other = self.view.nodes.get(edge.dst)
                self._tick()
                if other is not None and self._edge_fits(rel, edge):
                    yield edge, other
        if rel.direction in ("in", "both"):
            for edge in self.view.edges_in(current.qname, rel.kinds):
                if rel.direction == "both" and edge.src == edge.dst:
                    continue
                other = self.view.nodes.get(edge.src)
                self._tick()
                if other is not None and self._edge_fits(rel, edge):
                    yield edge, other

    def _reach(self, start: Node, rel: RelPattern) -> Iterator[tuple[Node, tuple[Edge, ...]]]:
        """Breadth-first: each node once, at its shortest distance within the hop range."""
        cap = self.limits.max_hops
        high = rel.max_hops
        if high is None or high > cap:
            requested = "unbounded" if high is None else str(high)
            self._warn(f"variable-length hop range ({requested}) was limited to {cap} hops")
            high = cap
        low = rel.min_hops
        seen: set[str] = set()
        if low == 0:
            seen.add(start.qname)
            yield start, ()
        parents: dict[str, tuple[str, Edge, int]] = {}
        frontier = [start]
        for depth in range(1, high + 1):
            following: list[Node] = []
            for node in frontier:
                for edge, other in self._step(node, rel):
                    if other.qname in seen:
                        continue
                    seen.add(other.qname)
                    parents[other.qname] = (node.qname, edge, depth)
                    following.append(other)
                    if depth >= low:
                        yield other, _path_to(parents, other.qname)
            if not following:
                return
            frontier = following

    # -- WITH / RETURN -------------------------------------------------------

    def _with(self, step: ProjectStep, rows: Iterable[Binding]) -> Iterator[Binding]:
        clause = step.clause
        if not isinstance(clause, With):
            raise CypherError("RETURN must be the last clause", clause.pos)
        values, _ = self._project(step, rows, limit=clause.projection.limit, probe=False)
        columns = [item.column for item in step.items]
        for row in values:
            binding: Binding = dict(zip(columns, row, strict=True))
            if clause.where is None or self._truth(clause.where, binding) is True:
                yield binding

    def _return(self, step: ProjectStep, rows: Iterable[Binding]) -> CypherResult:
        requested = step.clause.projection.limit
        limit = self.limits.default_rows if requested is None else requested
        if limit > self.limits.max_rows:
            self._warn(f"LIMIT {limit} was lowered to the maximum of {self.limits.max_rows} rows")
            limit = self.limits.max_rows
        probe = requested is None or requested > self.limits.max_rows
        values, more = self._project(step, rows, limit=limit, probe=probe)
        return CypherResult(
            columns=tuple(item.column for item in step.items),
            rows=tuple(values),
            truncated=more,
            warnings=tuple(self.warnings),
        )

    def _project(
        self, step: ProjectStep, rows: Iterable[Binding], *, limit: int | None, probe: bool
    ) -> tuple[list[tuple[Value, ...]], bool]:
        projection = step.clause.projection
        skip = projection.skip or 0
        if limit == 0:
            return [], False
        if not step.aggregating and not projection.distinct and not projection.order:
            return self._stream(step.items, rows, skip=skip, limit=limit, probe=probe)
        table: list[tuple[tuple[Value, ...], Binding | None]]
        if step.aggregating:
            table = self._aggregate(step.items, rows)
        else:
            table = [(self._row(step.items, b), b) for b in rows]
        if projection.distinct:
            table = _distinct(table)
        if projection.order:
            table = self._sorted(table, step.items, projection.order)
        end = None if limit is None else skip + limit
        more = probe and end is not None and len(table) > end
        return [values for values, _ in table[skip:end]], more

    def _stream(
        self,
        items: tuple[ReturnItem, ...],
        rows: Iterable[Binding],
        *,
        skip: int,
        limit: int | None,
        probe: bool,
    ) -> tuple[list[tuple[Value, ...]], bool]:
        remaining = islice(rows, skip, None)
        out: list[tuple[Value, ...]] = []
        for binding in remaining:
            out.append(self._row(items, binding))
            if limit is not None and len(out) >= limit:
                break
        more = (
            probe and limit is not None and len(out) == limit and next(remaining, None) is not None
        )
        return out, more

    def _row(self, items: tuple[ReturnItem, ...], binding: Binding) -> tuple[Value, ...]:
        return tuple(self._eval(item.expr, binding) for item in items)

    def _aggregate(
        self, items: tuple[ReturnItem, ...], rows: Iterable[Binding]
    ) -> list[tuple[tuple[Value, ...], Binding | None]]:
        key_positions = [i for i, item in enumerate(items) if not _is_aggregate(item.expr)]
        calls = [
            item.expr for item in items if isinstance(item.expr, Call) and item.expr.is_aggregate
        ]
        groups: dict[tuple[Value, ...], list[_Aggregate]] = {}
        for binding in rows:
            key = tuple(self._eval(items[i].expr, binding) for i in key_positions)
            accumulators = groups.get(key)
            if accumulators is None:
                accumulators = [_Aggregate(call) for call in calls]
                groups[key] = accumulators
            for accumulator in accumulators:
                accumulator.add(self, binding)
        if not groups and not key_positions:
            groups[()] = [_Aggregate(call) for call in calls]
        table: list[tuple[tuple[Value, ...], Binding | None]] = []
        for key, accumulators in groups.items():
            keys = iter(key)
            results = iter(accumulators)
            row = tuple(
                next(results).result() if _is_aggregate(item.expr) else next(keys) for item in items
            )
            table.append((row, None))
        return table

    def _sorted(
        self,
        table: list[tuple[tuple[Value, ...], Binding | None]],
        items: tuple[ReturnItem, ...],
        order: tuple[OrderItem, ...],
    ) -> list[tuple[tuple[Value, ...], Binding | None]]:
        keyed = [
            (self._order_keys(values, binding, items, order), values, binding)
            for values, binding in table
        ]
        for position in reversed(range(len(order))):
            keyed.sort(key=_key_at(position), reverse=order[position].descending)
        return [(values, binding) for _, values, binding in keyed]

    def _order_keys(
        self,
        values: tuple[Value, ...],
        binding: Binding | None,
        items: tuple[ReturnItem, ...],
        order: tuple[OrderItem, ...],
    ) -> tuple[SortKey, ...]:
        keys: list[SortKey] = []
        columns = {item.column: values[i] for i, item in enumerate(items)}
        for entry in order:
            expr = entry.expr
            if isinstance(expr, Var) and expr.name in columns:
                keys.append(sort_key(columns[expr.name]))
                continue
            matched = next((values[i] for i, item in enumerate(items) if item.expr == expr), None)
            if matched is not None or binding is None:
                keys.append(sort_key(matched))
                continue
            keys.append(sort_key(self._eval(expr, {**binding, **columns})))
        return tuple(keys)

    # -- expressions ---------------------------------------------------------

    def _eval(self, expr: Expr, binding: Binding) -> Value:
        if isinstance(expr, Const):
            return expr.value
        if isinstance(expr, ListExpr):
            return tuple(self._eval(item, binding) for item in expr.items)
        if isinstance(expr, Var):
            return binding.get(expr.name)
        if isinstance(expr, Prop):
            subject = binding.get(expr.var)
            if isinstance(subject, Node):
                return self._node_property(subject, expr.prop)
            if isinstance(subject, Edge):
                return _edge_property(subject, expr.prop)
            return None
        if isinstance(expr, Call):
            if expr.is_aggregate:
                raise CypherError(f"{expr.name}() cannot be used here", expr.pos)
            return self._scalar_call(expr, binding)
        return self._truth(expr, binding)

    def _truth(self, expr: Expr, binding: Binding) -> bool | None:
        if isinstance(expr, Compare):
            return self._compare(expr, binding)
        if isinstance(expr, IsNull):
            return (self._eval(expr.operand, binding) is None) != expr.negated
        if isinstance(expr, Not):
            inner = self._truth(expr.operand, binding)
            return None if inner is None else not inner
        if isinstance(expr, Logical):
            return self._logical(expr, binding)
        value = self._eval(expr, binding)
        if value is None or isinstance(value, bool):
            return value
        raise CypherError("expected a true/false condition here", expr.pos)

    def _logical(self, expr: Logical, binding: Binding) -> bool | None:
        left = self._truth(expr.left, binding)
        if expr.op == "and" and left is False:
            return False
        if expr.op == "or" and left is True:
            return True
        right = self._truth(expr.right, binding)
        if expr.op == "xor":
            return None if left is None or right is None else left != right
        if expr.op == "and":
            if right is False:
                return False
            return None if left is None or right is None else True
        if right is True:
            return True
        return None if left is None or right is None else False

    def _compare(self, expr: Compare, binding: Binding) -> bool | None:
        left = self._eval(expr.left, binding)
        right = self._eval(expr.right, binding)
        op = expr.op
        if op == "=":
            return _equals(left, right)
        if op == "<>":
            same = _equals(left, right)
            return None if same is None else not same
        if op == "in":
            return _contains(right, left, expr.right.pos)
        if left is None or right is None:
            return None
        if op == "=~":
            if not isinstance(left, str) or not isinstance(right, str):
                return None
            return self._pattern(right, expr.right.pos).fullmatch(left) is not None
        if op in ("contains", "starts with", "ends with"):
            if not isinstance(left, str) or not isinstance(right, str):
                return None
            if op == "contains":
                return right in left
            return left.startswith(right) if op == "starts with" else left.endswith(right)
        return _ordered(op, left, right)

    def _pattern(self, source: str, pos: int) -> re.Pattern[str]:
        compiled = self._regex.get(source)
        if compiled is None:
            try:
                compiled = re.compile(source)
            except re.error as exc:
                raise CypherError(f"invalid regular expression {source!r}: {exc}", pos) from exc
            self._regex[source] = compiled
        return compiled

    def _scalar_call(self, expr: Call, binding: Binding) -> Value:
        args = [self._eval(arg, binding) for arg in expr.args]
        if expr.name == "coalesce":
            return next((arg for arg in args if arg is not None), None)
        value = args[0]
        if value is None:
            return None
        if expr.name in ("tolower", "toupper"):
            if not isinstance(value, str):
                return None
            return value.lower() if expr.name == "tolower" else value.upper()
        if expr.name == "tostring":
            return render_value(value) if not isinstance(value, str) else value
        if expr.name == "size":
            return len(value) if isinstance(value, str | tuple) else None
        if expr.name == "type":
            return rel_type_for(value.kind) if isinstance(value, Edge) else None
        if expr.name == "labels":
            return (label_for(value.kind),) if isinstance(value, Node) else None
        raise CypherError(f"unknown function '{expr.name}'", expr.pos)

    def _node_property(self, node: Node, name: str) -> Value:
        prop = canonical_property(name)
        simple = _NODE_GETTERS.get(prop)
        if simple is not None:
            return simple(node)
        cached = self._degrees.get((prop, node.qname))
        if cached is not None:
            return cached
        if prop == "in_degree":
            value = self.view.in_degree(node.qname)
        elif prop == "out_degree":
            value = self.view.out_degree(node.qname)
        elif prop == "fan_in":
            value = self.view.in_degree(node.qname, ("calls",))
        elif prop == "fan_out":
            value = self.view.out_degree(node.qname, ("calls",))
        else:
            raise CypherError(f"unknown node property '{name}'")
        self._degrees[(prop, node.qname)] = value
        return value


class _Aggregate:
    """Running state for one aggregate call within one group."""

    def __init__(self, call: Call) -> None:
        self.call = call
        self.rows = 0
        self.values: list[Value] = []
        self.seen: set[Value] = set()

    def add(self, executor: _Executor, binding: Binding) -> None:
        self.rows += 1
        if self.call.star:
            return
        value = executor._eval(self.call.args[0], binding)
        if value is None:
            return
        if self.call.distinct:
            if value in self.seen:
                return
            self.seen.add(value)
        self.values.append(value)

    def result(self) -> Value:
        name = self.call.name
        if name == "count":
            return self.rows if self.call.star else len(self.values)
        if name == "collect":
            return tuple(self.values)
        if name in ("min", "max"):
            if not self.values:
                return None
            pick = min if name == "min" else max
            return pick(self.values, key=sort_key)
        numbers: list[int | float] = []
        for value in self.values:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise CypherError(
                    f"{name}() needs numbers, got {render_value(value)!r}", self.call.pos
                )
            numbers.append(value)
        if name == "sum":
            return sum(numbers)
        return sum(numbers) / len(numbers) if numbers else None


_NODE_GETTERS: dict[str, Callable[[Node], Value]] = {
    "qname": lambda n: n.qname,
    "name": lambda n: n.name,
    "kind": lambda n: n.kind,
    "label": lambda n: label_for(n.kind),
    "path": lambda n: n.path,
    "line_start": lambda n: n.line_start,
    "line_end": lambda n: n.line_end,
    "language": lambda n: n.language,
    "signature": lambda n: n.signature,
    "docstring": lambda n: n.docstring,
    "span": lambda n: n.span,
}


def _edge_property(edge: Edge, name: str) -> Value:
    prop = canonical_property(name)
    if prop == "kind":
        return edge.kind
    if prop == "type":
        return rel_type_for(edge.kind)
    if prop == "src":
        return edge.src
    if prop == "dst":
        return edge.dst
    if prop == "path":
        return edge.path
    if prop == "line":
        return edge.line
    if prop == "resolved":
        return edge.resolved
    if prop == "confidence":
        return edge.confidence
    if prop == "detail":
        return edge.detail
    raise CypherError(f"unknown relationship property '{name}'")


def sort_key(value: Value) -> SortKey:
    """Total order across types; nulls sort last ascending and first descending."""
    if value is None:
        return (9, 0.0, "")
    if isinstance(value, bool):
        return (1, float(value), "")
    if isinstance(value, int | float):
        return (2, float(value), "")
    if isinstance(value, str):
        return (3, 0.0, value)
    if isinstance(value, Node):
        return (4, 0.0, value.qname)
    if isinstance(value, Edge):
        return (5, float(value.line), f"{value.src}\0{value.dst}\0{value.kind}")
    return (6, float(len(value)), render_value(value))


def _equals(left: Value, right: Value) -> bool | None:
    if left is None or right is None:
        return None
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, int | float) and isinstance(right, int | float):
        return left == right
    if isinstance(left, str) and isinstance(right, str):
        return left == right
    if isinstance(left, Node) and isinstance(right, Node):
        return left.qname == right.qname
    if isinstance(left, Edge) and isinstance(right, Edge):
        return left == right
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            return False
        results = [_equals(a, b) for a, b in zip(left, right, strict=True)]
        if False in results:
            return False
        return None if None in results else True
    return False


def _contains(container: Value, item: Value, pos: int) -> bool | None:
    if container is None:
        return None
    if not isinstance(container, tuple):
        raise CypherError("IN needs a list on its right-hand side, e.g. IN ['a', 'b']", pos)
    if item is None:
        return None if container else False
    results = [_equals(item, candidate) for candidate in container]
    if True in results:
        return True
    return None if None in results else False


def _ordered(op: str, left: Value, right: Value) -> bool | None:
    numeric = (
        isinstance(left, int | float)
        and isinstance(right, int | float)
        and not isinstance(left, bool)
        and not isinstance(right, bool)
    )
    if numeric or (isinstance(left, str) and isinstance(right, str)):
        a, b = sort_key(left), sort_key(right)
        if op == "<":
            return a < b
        if op == "<=":
            return a <= b
        if op == ">":
            return a > b
        return a >= b
    return None


def _bind(binding: Binding, var: str | None, value: Value) -> Binding | None:
    if var is None:
        return binding
    if var in binding:
        return binding if _equals(binding[var], value) is True else None
    return {**binding, var: value}


def _distinct(
    table: list[tuple[tuple[Value, ...], Binding | None]],
) -> list[tuple[tuple[Value, ...], Binding | None]]:
    seen: set[tuple[Value, ...]] = set()
    unique: list[tuple[tuple[Value, ...], Binding | None]] = []
    for values, binding in table:
        if values not in seen:
            seen.add(values)
            unique.append((values, binding))
    return unique


def _path_to(parents: dict[str, tuple[str, Edge, int]], end: str) -> tuple[Edge, ...]:
    edges: list[Edge] = []
    current = end
    while True:
        previous, edge, depth = parents[current]
        edges.append(edge)
        if depth == 1:
            break
        current = previous
    return tuple(reversed(edges))


def _anchor_cost(pattern: NodePattern, binding: Binding) -> int:
    """How cheap a node pattern is as the place to start matching (lower is better)."""
    if pattern.var is not None and pattern.var in binding:
        return 0
    keys = {canonical_property(key) for key, _ in pattern.props}
    if "qname" in keys:
        return 1
    if "name" in keys:
        return 2
    return 3 if pattern.kinds is not None else 4


_FLIPPED: dict[Direction, Direction] = {"out": "in", "in": "out", "both": "both"}


def _flipped(rel: RelPattern) -> RelPattern:
    return replace(rel, direction=_FLIPPED[rel.direction])


def _is_aggregate(expr: Expr) -> bool:
    return isinstance(expr, Call) and expr.is_aggregate


_Keyed: TypeAlias = tuple[tuple[SortKey, ...], tuple[Value, ...], Binding | None]


def _key_at(position: int) -> Callable[[_Keyed], SortKey]:
    def key(entry: _Keyed) -> SortKey:
        return entry[0][position]

    return key


__all__ = [
    "Binding",
    "CypherResult",
    "Limits",
    "Value",
    "execute_plan",
    "render_value",
    "sort_key",
    "to_json",
]
