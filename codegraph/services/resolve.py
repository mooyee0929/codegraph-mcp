"""Attach a ``path:line`` or a handler reference found in source text to a graph node.

Three questions come up for every route, call site and channel:

- *Who owns this line?* -- the innermost definition containing it, else the file's
  module (:meth:`NodeIndex.owner`).
- *Which function does this decorator or annotation sit on?* -- the definition that
  starts right below it, or that starts a few lines above and contains it when the
  parser counts decorators as part of the span (:meth:`NodeIndex.decorated`).
- *What does ``views.user_detail`` / ``UserController@show`` refer to?* -- a unique
  definition with that name, preferring the same file and then one whose qname or
  path mentions the owner segment (:meth:`NodeIndex.lookup`).

Files whose language has no parser yet have no nodes; for those :meth:`owner` falls
back to the module qname the file will get once a parser lands, so edges written now
start resolving without a schema change.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath

from codegraph.models import Node, NodeKind
from codegraph.parsers.base import module_qname_for
from codegraph.store import GraphStore

DEFINITION_KINDS = frozenset({"function", "method", "class"})
CALLABLE_KINDS = frozenset({"function", "method"})
_DERIVED_KINDS: tuple[NodeKind, ...] = ("module", "class", "function", "method")
_REF_SPLIT = re.compile(r"::|->|[.#@/\\]")


@dataclass(slots=True)
class NodeIndex:
    by_file: dict[str, list[Node]]
    by_name: dict[str, list[Node]]
    modules: dict[str, Node]

    @classmethod
    def load(cls, store: GraphStore) -> NodeIndex:
        return cls.from_nodes(store.all_nodes(kinds=_DERIVED_KINDS))

    @classmethod
    def from_nodes(cls, nodes: list[Node]) -> NodeIndex:
        by_file: defaultdict[str, list[Node]] = defaultdict(list)
        by_name: defaultdict[str, list[Node]] = defaultdict(list)
        modules: dict[str, Node] = {}
        for node in nodes:
            if node.kind == "module":
                modules[node.path] = node
                continue
            by_file[node.path].append(node)
            if node.kind in DEFINITION_KINDS:
                by_name[node.name].append(node)
        for file_nodes in by_file.values():
            file_nodes.sort(key=lambda n: (n.line_start, -n.line_end))
        return cls(by_file=dict(by_file), by_name=dict(by_name), modules=modules)

    def module_qname(self, path: str) -> str:
        module = self.modules.get(path)
        return module.qname if module is not None else module_qname_for(path)

    def indexed(self, path: str) -> bool:
        """Whether a parser produced nodes for ``path`` (edges from it will resolve)."""
        return path in self.modules

    def owner_node(self, path: str, line: int) -> Node | None:
        """Innermost definition containing ``line``, else the module node, else None."""
        return self.enclosing(path, line) or self.modules.get(path)

    def enclosing(self, path: str, line: int) -> Node | None:
        best: Node | None = None
        for node in self.by_file.get(path, ()):
            if node.line_start > line:
                break
            if node.line_end >= line and (best is None or node.span <= best.span):
                best = node
        return best

    def owner(self, path: str, line: int) -> str:
        """Qname of the innermost definition containing ``line``, else the module."""
        node = self.enclosing(path, line)
        return node.qname if node is not None else self.module_qname(path)

    def decorated(
        self, path: str, line: int, name: str | None = None, window: int = 30
    ) -> Node | None:
        """The definition a decorator/annotation on ``line`` applies to."""
        containing: list[Node] = []
        following: list[Node] = []
        for node in self.by_file.get(path, ()):
            if node.kind not in DEFINITION_KINDS:
                continue
            if node.line_start <= line <= node.line_end and line - node.line_start <= 6:
                containing.append(node)
            elif line <= node.line_start <= line + window:
                following.append(node)
        if name is not None:
            named = [n for n in (*containing, *following) if n.name == name]
            if named:
                return min(named, key=lambda n: (abs(n.line_start - line), n.span))
        inner = min(containing, key=lambda n: (n.span, -n.line_start)) if containing else None
        nearest = min(following, key=lambda n: n.line_start) if following else None
        if inner is not None and nearest is not None and inner.line_end >= nearest.line_end:
            return nearest  # the decorator sits inside a class/function, on a member below it
        return inner or nearest

    def lookup(
        self, ref: str, path: str | None = None, owner_hint: str | None = None
    ) -> Node | None:
        """Resolve a handler reference such as ``views.detail`` or ``UsersController#show``."""
        segments = [s for s in _REF_SPLIT.split(ref.strip()) if s]
        if not segments:
            return None
        name = segments[-1].strip("'\"$&*() ")
        hint = owner_hint or (segments[-2] if len(segments) > 1 else None)
        candidates = self.by_name.get(name, [])
        if not candidates:
            return None
        if path is not None:
            local = [c for c in candidates if c.path == path]
            if len(local) == 1:
                return local[0]
            if local:
                candidates = local
        if hint:
            hinted = [c for c in candidates if _mentions(c, hint)]
            if len(hinted) == 1:
                return hinted[0]
            if hinted:
                candidates = hinted
        return candidates[0] if len(candidates) == 1 else None

    def named_in_file(self, path: str, name: str) -> Node | None:
        matches = [n for n in self.by_file.get(path, ()) if n.name == name]
        return matches[0] if matches else None


def _mentions(node: Node, hint: str) -> bool:
    wanted = hint.lower().strip("'\"$")
    if not wanted:
        return False
    parts = [p.lower() for p in node.qname.split(".")]
    if wanted in parts:
        return True
    stem = PurePosixPath(node.path).stem.lower()
    return stem == wanted or stem.replace("_", "") == wanted.replace("_", "")


__all__ = ["CALLABLE_KINDS", "DEFINITION_KINDS", "NodeIndex"]
