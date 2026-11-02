"""Source of one symbol, located by qname, unique name or ``path:line``.

Reading a symbol through the graph returns exactly its lines (numbered, optionally with
a little context) plus who calls it and which tests cover it, instead of a whole file.
Rendering is :func:`codegraph.review.review_context`, so ``show``, ``review --context``
and this tool print the same thing. An ambiguous name fails with the candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codegraph.models import Node
from codegraph.queries import resolve_target
from codegraph.review import ReviewSnippet, review_context
from codegraph.store import GraphStore

MAX_CANDIDATES = 10


class SymbolNotFound(LookupError):
    def __init__(self, message: str, candidates: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.candidates = candidates


@dataclass(frozen=True, slots=True)
class CodeSnippet:
    node: Node
    snippet: ReviewSnippet

    def as_dict(self) -> dict[str, object]:
        return {
            "qname": self.node.qname,
            "kind": self.node.kind,
            "path": self.node.path,
            "symbol_lines": [self.node.line_start, self.node.line_end],
            "shown_lines": [self.snippet.line_start, self.snippet.line_end],
            "signature": self.node.signature,
            "source": self.snippet.source,
            "callers": list(self.snippet.callers),
            "tests": list(self.snippet.tests),
        }


def candidates_for(store: GraphStore, target: str) -> tuple[str, ...]:
    """Qnames an ambiguous or misspelled target might have meant."""
    named = store.nodes_named(target.rsplit(".", 1)[-1])
    if not named:
        named = store.search_nodes(target, limit=MAX_CANDIDATES)
    return tuple(sorted(n.qname for n in named)[:MAX_CANDIDATES])


def code_snippet(
    store: GraphStore, root: Path, target: str, *, context: int = 0, max_lines: int = 200
) -> CodeSnippet:
    if context < 0 or max_lines < 1:
        raise ValueError("context must be >= 0 and max_lines >= 1")
    node = resolve_target(store, target)
    if node is None:
        found = candidates_for(store, target)
        reason = "is ambiguous" if len(found) > 1 else "matches no symbol"
        raise SymbolNotFound(f"{target!r} {reason}", found)
    snippets = review_context(store, root, [node.qname], context_lines=context, max_lines=max_lines)
    if not snippets:
        raise SymbolNotFound(f"cannot read {node.path} for {node.qname}; re-index if it moved")
    return CodeSnippet(node=node, snippet=snippets[0])


__all__ = ["CodeSnippet", "SymbolNotFound", "candidates_for", "code_snippet"]
