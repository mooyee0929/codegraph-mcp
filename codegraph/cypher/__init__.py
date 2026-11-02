"""A read-only Cypher subset over the code graph.

``MATCH (c:Function)-[:CALLS]->(f {name: 'charge'}) RETURN c.qname`` reads better than
a chain of ``callers_of`` calls once a question spans more than one hop or needs
filtering and counting. The pipeline is :mod:`lexer` -> :mod:`parser` (syntax tree,
labels resolved) -> :mod:`planner` (scope and property checks) -> :mod:`executor`
(lazy matching over :class:`~codegraph.graph.GraphView`).

Labels are node kinds in CamelCase (``Function``, ``Route``) plus ``Callable`` for
function-or-method; relationship types are edge kinds in upper case (``CALLS``,
``HTTP_CALLS``). Write clauses fail: the graph is derived from source.
"""

from __future__ import annotations

from codegraph.cypher.errors import CypherError
from codegraph.cypher.executor import (
    CypherResult,
    Limits,
    Value,
    execute_plan,
    render_value,
    to_json,
)
from codegraph.cypher.parser import parse
from codegraph.cypher.planner import Plan, plan
from codegraph.graph import GraphView
from codegraph.store import GraphStore


def prepare(text: str) -> Plan:
    """Parse and check ``text`` without touching a graph."""
    return plan(parse(text))


def execute(view: GraphView, text: str, *, limits: Limits | None = None) -> CypherResult:
    return execute_plan(view, prepare(text), limits or Limits())


def run_cypher(store: GraphStore, text: str, *, limits: Limits | None = None) -> CypherResult:
    """Run ``text`` against the store's graph. Bad queries fail before the graph loads."""
    checked = prepare(text)
    return execute_plan(GraphView.load(store), checked, limits or Limits())


__all__ = [
    "CypherError",
    "CypherResult",
    "Limits",
    "Plan",
    "Value",
    "execute",
    "parse",
    "prepare",
    "render_value",
    "run_cypher",
    "to_json",
]
