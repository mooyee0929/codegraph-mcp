"""Core value types for the code graph.

A graph is a set of :class:`Node` (things you can point at in source) and a set of
:class:`Edge` (relationships between them). Both are immutable; the store owns
persistence and the parsers own extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Language = str
"""Language identifier such as ``python``, ``go`` or ``dockerfile``.

The set is open: :mod:`codegraph.parsers.languages` maps file names to identifiers
and decides which parser handles each one.
"""

NodeKind = Literal["module", "class", "function", "method", "route", "channel", "resource"]
"""What a node is.

- ``module``: one source file.
- ``class``: any named type — class, struct, interface, trait, enum, protocol.
- ``function`` / ``method``: a callable; ``method`` has a class as its parent.
- ``route``: an HTTP (or RPC) endpoint, e.g. ``GET /users/{id}``.
- ``channel``: a named event/message channel that code emits to or listens on.
- ``resource``: an infrastructure object — Dockerfile stage, Kubernetes object,
  compose service, Terraform resource.
"""

NODE_KINDS: tuple[NodeKind, ...] = (
    "module",
    "class",
    "function",
    "method",
    "route",
    "channel",
    "resource",
)

CALLABLE_KINDS: tuple[NodeKind, ...] = ("function", "method")

EdgeKind = Literal[
    "calls",
    "imports",
    "defines",
    "inherits",
    "tests",
    "handles",
    "http_calls",
    "emits",
    "listens_on",
    "similar_to",
    "semantically_related",
    "co_changes",
    "depends_on",
]
"""How two nodes relate.

Structural, produced by parsers and the linker:

- ``calls``: src invokes dst.
- ``imports``: src (a module) pulls dst into scope.
- ``defines``: src lexically contains dst (module -> class -> method).
- ``inherits``: src is a subclass or implementation of dst.
- ``tests``: src is a test that exercises dst (inferred, see :mod:`codegraph.linker`).

Derived, produced by post-index passes (:mod:`codegraph.passes`):

- ``handles``: src (a route) is served by dst (a function).
- ``http_calls``: src (a function) sends a request that dst (a route) serves.
- ``emits`` / ``listens_on``: src (a function) publishes to / subscribes to dst (a channel).
- ``similar_to``: src and dst are near-clones; ``confidence`` is the similarity.
- ``semantically_related``: src and dst share vocabulary but not structure.
- ``co_changes``: src and dst (modules) are often changed in the same commit.
- ``depends_on``: src (a resource or module) needs dst — image, volume, manifest entry.
"""

EDGE_KINDS: tuple[EdgeKind, ...] = (
    "calls",
    "imports",
    "defines",
    "inherits",
    "tests",
    "handles",
    "http_calls",
    "emits",
    "listens_on",
    "similar_to",
    "semantically_related",
    "co_changes",
    "depends_on",
)


@dataclass(frozen=True, slots=True)
class Node:
    """A named, locatable definition in the codebase.

    ``qname`` is the primary key: a dotted path rooted at the repository, such as
    ``pkg.billing.Invoice.total``. It is stable across runs as long as the file path
    and the enclosing definitions do not change.
    """

    qname: str
    kind: NodeKind
    path: str
    line_start: int
    line_end: int
    language: Language
    name: str
    signature: str = ""
    docstring: str | None = None

    @property
    def span(self) -> int:
        return max(1, self.line_end - self.line_start + 1)


@dataclass(frozen=True, slots=True)
class Edge:
    """A directed relationship between two qnames.

    ``dst`` may be unresolved at parse time — a parser that sees ``helper()`` with no
    import to attribute it to emits ``dst="helper"`` and sets ``resolved=False``; a
    method call on an object of unknown type (``inv.total()``) emits ``dst="*.total"``.
    The linker pass upgrades those to full qnames where the target is unambiguous.

    ``confidence`` (0..1) is 1.0 for edges read straight from syntax and lower for
    heuristic ones (HTTP call-site matching, clone similarity). ``detail`` is a short
    free-form note such as ``"GET"`` or ``"jaccard=0.83"``.
    """

    src: str
    dst: str
    kind: EdgeKind
    path: str
    line: int
    resolved: bool = True
    confidence: float = 1.0
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Everything one parser extracted from one file."""

    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]

    @classmethod
    def empty(cls) -> ParseResult:
        return cls(nodes=(), edges=())


@dataclass(frozen=True, slots=True)
class ImpactReport:
    """Blast radius of changing ``target``."""

    target: str
    direct_callers: tuple[str, ...]
    transitive_callers: tuple[str, ...]
    importing_modules: tuple[str, ...]
    covering_tests: tuple[str, ...]
    depth: int

    @property
    def total_affected(self) -> int:
        return len(set(self.transitive_callers) | set(self.importing_modules))


@dataclass(frozen=True, slots=True)
class ChangedSymbol:
    """A node overlapping a changed hunk, with the signals that make it risky."""

    node: Node
    changed_lines: tuple[int, ...]
    fan_in: int
    fan_out: int
    covering_tests: tuple[str, ...]
    risk: float
    reasons: tuple[str, ...]
