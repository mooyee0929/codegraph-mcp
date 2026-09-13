"""Core value types for the code graph.

A graph is a set of :class:`Node` (things you can point at in source) and a set of
:class:`Edge` (relationships between them). Both are immutable; the store owns
persistence and the parsers own extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Language = Literal["python", "javascript", "typescript"]

NodeKind = Literal["module", "class", "function", "method"]
"""What a node is. ``method`` is a function whose parent is a class."""

EdgeKind = Literal["calls", "imports", "defines", "inherits", "tests"]
"""How two nodes relate.

- ``calls``: src invokes dst.
- ``imports``: src (a module) pulls dst into scope.
- ``defines``: src lexically contains dst (module -> class -> method).
- ``inherits``: src is a subclass of dst.
- ``tests``: src is a test that exercises dst (inferred, see :mod:`codegraph.linker`).
"""

LANGUAGE_BY_SUFFIX: dict[str, Language] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}


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
    """

    src: str
    dst: str
    kind: EdgeKind
    path: str
    line: int
    resolved: bool = True


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
