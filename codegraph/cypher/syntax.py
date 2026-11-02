"""The parsed form of a query.

Expression nodes compare structurally with their source position excluded
(``field(compare=False)``), so the planner can tell that ``ORDER BY n.name`` names the
same expression as ``RETURN n.name`` even though the two sit at different offsets.

Labels and relationship types are already resolved to node and edge kinds here; the
parser rejects unknown ones at their position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from codegraph.models import EdgeKind, NodeKind

Scalar: TypeAlias = str | int | float | bool | None

Direction = Literal["out", "in", "both"]

CompareOp = Literal[
    "=", "<>", "<", "<=", ">", ">=", "=~", "in", "contains", "starts with", "ends with"
]

AGGREGATES: frozenset[str] = frozenset({"count", "collect", "min", "max", "avg", "sum"})
SCALAR_FUNCTIONS: frozenset[str] = frozenset(
    {"tolower", "toupper", "tostring", "size", "type", "labels", "coalesce"}
)


@dataclass(frozen=True, slots=True)
class Const:
    value: Scalar
    pos: int = field(default=0, compare=False)


@dataclass(frozen=True, slots=True)
class ListExpr:
    items: tuple[Expr, ...]
    pos: int = field(default=0, compare=False)


@dataclass(frozen=True, slots=True)
class Var:
    name: str
    pos: int = field(default=0, compare=False)


@dataclass(frozen=True, slots=True)
class Prop:
    var: str
    prop: str
    pos: int = field(default=0, compare=False)
    prop_pos: int = field(default=0, compare=False)


@dataclass(frozen=True, slots=True)
class Call:
    name: str
    """Lower-cased function name."""
    args: tuple[Expr, ...]
    distinct: bool = False
    star: bool = False
    pos: int = field(default=0, compare=False)

    @property
    def is_aggregate(self) -> bool:
        return self.name in AGGREGATES


@dataclass(frozen=True, slots=True)
class Compare:
    op: CompareOp
    left: Expr
    right: Expr
    pos: int = field(default=0, compare=False)


@dataclass(frozen=True, slots=True)
class IsNull:
    operand: Expr
    negated: bool
    pos: int = field(default=0, compare=False)


@dataclass(frozen=True, slots=True)
class Logical:
    op: Literal["and", "or", "xor"]
    left: Expr
    right: Expr
    pos: int = field(default=0, compare=False)


@dataclass(frozen=True, slots=True)
class Not:
    operand: Expr
    pos: int = field(default=0, compare=False)


Expr: TypeAlias = Const | ListExpr | Var | Prop | Call | Compare | IsNull | Logical | Not


@dataclass(frozen=True, slots=True)
class NodePattern:
    var: str | None
    kinds: frozenset[NodeKind] | None
    """None matches every kind; a multi-label pattern is the intersection."""
    props: tuple[tuple[str, Const], ...]
    pos: int


@dataclass(frozen=True, slots=True)
class RelPattern:
    var: str | None
    kinds: frozenset[EdgeKind] | None
    direction: Direction
    props: tuple[tuple[str, Const], ...]
    pos: int
    var_length: bool = False
    min_hops: int = 1
    max_hops: int | None = 1
    """None means unbounded (``*`` or ``*2..``); the executor clamps it to its cap."""


@dataclass(frozen=True, slots=True)
class PathPattern:
    nodes: tuple[NodePattern, ...]
    rels: tuple[RelPattern, ...]
    pos: int


@dataclass(frozen=True, slots=True)
class Match:
    patterns: tuple[PathPattern, ...]
    optional: bool
    where: Expr | None
    pos: int


@dataclass(frozen=True, slots=True)
class ReturnItem:
    expr: Expr
    alias: str | None
    text: str
    """The expression as written; the column name when there is no alias."""

    @property
    def column(self) -> str:
        return self.alias if self.alias is not None else self.text


@dataclass(frozen=True, slots=True)
class OrderItem:
    expr: Expr
    descending: bool


@dataclass(frozen=True, slots=True)
class Projection:
    items: tuple[ReturnItem, ...]
    distinct: bool
    star: bool
    order: tuple[OrderItem, ...]
    skip: int | None
    limit: int | None
    pos: int


@dataclass(frozen=True, slots=True)
class With:
    projection: Projection
    where: Expr | None
    pos: int


@dataclass(frozen=True, slots=True)
class Return:
    projection: Projection
    pos: int


Clause: TypeAlias = Match | With | Return


@dataclass(frozen=True, slots=True)
class Query:
    clauses: tuple[Clause, ...]
    text: str


def contains_aggregate(expr: Expr) -> bool:
    if isinstance(expr, Call):
        return expr.is_aggregate or any(contains_aggregate(a) for a in expr.args)
    return any(contains_aggregate(child) for child in children(expr))


def children(expr: Expr) -> tuple[Expr, ...]:
    if isinstance(expr, ListExpr):
        return expr.items
    if isinstance(expr, Call):
        return expr.args
    if isinstance(expr, Compare | Logical):
        return (expr.left, expr.right)
    if isinstance(expr, IsNull | Not):
        return (expr.operand,)
    return ()


__all__ = [
    "AGGREGATES",
    "SCALAR_FUNCTIONS",
    "Call",
    "Clause",
    "Compare",
    "CompareOp",
    "Const",
    "Direction",
    "Expr",
    "IsNull",
    "ListExpr",
    "Logical",
    "Match",
    "NodePattern",
    "Not",
    "OrderItem",
    "PathPattern",
    "Projection",
    "Prop",
    "Query",
    "RelPattern",
    "Return",
    "ReturnItem",
    "Scalar",
    "Var",
    "With",
    "children",
    "contains_aggregate",
]
