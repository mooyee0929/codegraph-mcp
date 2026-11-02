"""Dead-code candidates with the usual false positives filtered out.

"Nothing calls it" is the start, not the answer. Static call graphs miss everything a
runtime or framework reaches on its own, so a callable is kept alive when any of these
hold:

- something references it: a call, test, subclass, route ``handles`` edge, or a
  ``from x import it`` (Python's way of exporting a symbol to another module);
- an unresolved call or base class carries its bare name (``obj.save()`` the linker
  could not pin to one ``save`` may well be this one);
- it is an entry point (:mod:`codegraph.analysis.entrypoints`) other than a mere root;
- a decorator registers it or the runtime invokes it (``@property``, ``@app.get``);
- it is a dunder or JS ``constructor``;
- it overrides a method of a base class, or its class inherits from something outside
  the repository whose methods we cannot see;
- it is a class named in some signature's type annotations, or a class with a live
  member;
- it is ``export``-ed from a JS/TS module, or is a public method of an exported class;
- its name occurs in the source more often than it is defined: passed as a callback,
  stored in a dispatch table, named in a string registry or a type cast.

Test code is never reported. Private (``_name``) callables *are*: an unreferenced
private helper is the most likely dead code there is.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from codegraph.analysis.common import SYMBOL_KINDS, is_test_node
from codegraph.analysis.entrypoints import (
    class_of,
    cli_scripts,
    decorators_of,
    detect_entry_points,
    framework_decorator,
    has_implicit_decorator,
    is_dunder,
    may_override,
    unresolved_call_names,
)
from codegraph.graph import GraphView
from codegraph.models import EdgeKind, Node
from codegraph.store import GraphStore

REFERENCE_KINDS: tuple[EdgeKind, ...] = (
    "calls",
    "tests",
    "inherits",
    "handles",
    "http_calls",
    "imports",
    "depends_on",
)

_IDENTIFIER = re.compile(r"[A-Za-z_$][\w$]*")
_JS_LANGUAGES = frozenset({"javascript", "typescript"})


def dead_code_candidates(store: GraphStore) -> list[Node]:
    """Entry for :func:`codegraph.queries.dead_code`: load the graph, then filter."""
    root_text = store.get_meta("root")
    return find_dead_code(GraphView.load(store), root=Path(root_text) if root_text else None)


def find_dead_code(view: GraphView, *, root: Path | None = None) -> list[Node]:
    """Unreferenced functions, methods and classes, ordered by file and line.

    ``root`` enables the checks that need source or packaging files (JS ``export``,
    name mentions, ``pyproject.toml`` scripts); without it those checks are skipped.
    """
    referenced = _referenced(view)
    loose_names = unresolved_call_names(view)
    alive_entries = {
        e.qname
        for e in detect_entry_points(view, cli_scripts(root), include_tests=True)
        if e.kind != "root"
    }
    annotated = _annotation_names(view)
    sources = _Sources(view, root)

    candidates: list[Node] = []
    for node in view.of_kind(*SYMBOL_KINDS):
        if is_test_node(node) or node.path.endswith(".d.ts") or node.name.startswith("test"):
            continue
        if node.qname in referenced or node.qname in alive_entries:
            continue
        if node.name in loose_names or is_dunder(node.name) or node.name == "constructor":
            continue
        decorators = decorators_of(view, node)
        if has_implicit_decorator(decorators) or framework_decorator(decorators) is not None:
            continue
        if node.kind == "method" and may_override(view, node):
            continue
        if node.kind == "class" and (
            node.name in annotated or _has_live_member(view, node, referenced, loose_names)
        ):
            continue
        if node.language in _JS_LANGUAGES and sources.is_exported(node):
            continue
        if sources.mentioned_elsewhere(node):
            continue
        candidates.append(node)
    return sorted(candidates, key=lambda n: (n.path, n.line_start, n.qname))


def _referenced(view: GraphView) -> set[str]:
    referenced: set[str] = set()
    for edge in view.edges(REFERENCE_KINDS):
        if edge.src != edge.dst:
            referenced.add(edge.dst)
    return referenced


def _annotation_names(view: GraphView) -> set[str]:
    """Identifiers used in callable signatures: parameter and return annotations."""
    names: set[str] = set()
    for node in view.of_kind("function", "method"):
        _, _, params = node.signature.partition("(")
        names.update(name for name in _IDENTIFIER.findall(params) if name != node.name)
    return names


def _has_live_member(
    view: GraphView, cls: Node, referenced: set[str], loose_names: frozenset[str]
) -> bool:
    for child in view.successors(cls.qname, ("defines",)):
        member = view.nodes[child]
        if child in referenced or member.name in loose_names:
            return True
    return False


class _Sources:
    """Lazy access to the indexed source files: definition lines and name mentions."""

    def __init__(self, view: GraphView, root: Path | None) -> None:
        self._view = view
        self._root = root
        self._lines: dict[str, list[str]] = {}
        self._mentions: Counter[str] | None = None
        self._definitions: Counter[str] = Counter(
            n.name for n in view.nodes.values() if n.kind != "module"
        )

    def _file(self, path: str) -> list[str]:
        if self._root is None:
            return []
        if path not in self._lines:
            try:
                text = (self._root / path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""  # the file vanished since indexing; nothing to inspect
            self._lines[path] = text.splitlines()
        return self._lines[path]

    def mentioned_elsewhere(self, node: Node) -> bool:
        """The name occurs in the sources more often than it is defined.

        Catches what the call graph cannot: a handler passed as a value
        (``add("index", cmd_index)``), dispatch tables, string registries, TypeScript
        types used only in casts.
        """
        if self._root is None:
            return False
        if self._mentions is None:
            self._mentions = Counter()
            paths = sorted({n.path for n in self._view.nodes.values() if n.kind == "module"})
            for path in paths:
                for line in self._file(path):
                    self._mentions.update(_IDENTIFIER.findall(line))
        return self._mentions[node.name] > self._definitions[node.name]

    def is_exported(self, node: Node) -> bool:
        lines = self._file(node.path)
        line = lines[node.line_start - 1] if 0 < node.line_start <= len(lines) else ""
        if line.lstrip().startswith("export "):
            return True
        if node.kind != "method" or node.name.startswith(("_", "#")):
            return False
        owner = class_of(self._view, node)
        return owner is not None and self.is_exported(owner)


__all__ = ["REFERENCE_KINDS", "dead_code_candidates", "find_dead_code"]
