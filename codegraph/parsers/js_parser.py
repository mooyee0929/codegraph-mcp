"""JavaScript/TypeScript parser built on tree-sitter.

Resolution follows the same idea as the Python parser: the import statements at the top
of a file become an alias table, so ``helper()`` in ``web/app.ts`` resolves to
``web.util.helper`` rather than a bare name.

Two approximations are deliberate and documented here so callers can reason about the
edges they get back:

- A default or namespace import (``import def from "./x"``) binds the local name to the
  *module* qname. ``def.run()`` therefore resolves to ``x.run``, which is right for
  namespace imports and a reasonable guess for defaults.
- ``this.foo()`` resolves against the lexically enclosing class. Anything deeper
  (``this.client.send()``) is left unresolved for the linker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

import tree_sitter_javascript
import tree_sitter_typescript
from tree_sitter import Language as TSLanguage
from tree_sitter import Node as TSNode
from tree_sitter import Parser as TSParser

from codegraph.models import Edge, Language, Node, NodeKind, ParseResult
from codegraph.parsers.base import child_qname, clip, module_qname_for

_FUNCTION_VALUE_TYPES = frozenset({"arrow_function", "function_expression", "function"})
_CLASS_TYPES = frozenset({"class_declaration", "abstract_class_declaration", "class"})
_FUNCTION_TYPES = frozenset({"function_declaration", "generator_function_declaration"})
_CONTAINER_TYPES = frozenset(
    {"export_statement", "program", "statement_block", "ambient_declaration"}
)


def _language_for(language: Language, rel_path: str) -> TSLanguage:
    if language == "javascript":
        return TSLanguage(tree_sitter_javascript.language())
    if rel_path.endswith(".tsx"):
        return TSLanguage(tree_sitter_typescript.language_tsx())
    return TSLanguage(tree_sitter_typescript.language_typescript())


@dataclass(slots=True)
class _Scope:
    qname: str
    class_qname: str | None = None


@dataclass(slots=True)
class _Collector:
    rel_path: str
    module_qname: str
    language: Language
    source: bytes
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)
    local_defs: set[str] = field(default_factory=set)

    # -- text helpers -----------------------------------------------------

    def text(self, node: TSNode) -> str:
        return self.source[node.start_byte : node.end_byte].decode("utf-8", "replace")

    def _named(self, node: TSNode) -> str | None:
        name = node.child_by_field_name("name")
        return self.text(name) if name is not None else None

    def _signature(self, node: TSNode) -> str:
        body = node.child_by_field_name("body")
        end = body.start_byte if body is not None else node.end_byte
        return clip(self.source[node.start_byte : end].decode("utf-8", "replace"))

    def _leading_jsdoc(self, node: TSNode) -> str | None:
        cursor = node
        while cursor.parent is not None and cursor.parent.type == "export_statement":
            cursor = cursor.parent
        previous = cursor.prev_sibling
        if previous is not None and previous.type == "comment":
            raw = self.text(previous)
            if raw.startswith("/**"):
                stripped = raw.removeprefix("/**").removesuffix("*/")
                lines = [line.strip().lstrip("*").strip() for line in stripped.splitlines()]
                body = " ".join(line for line in lines if line)
                return clip(body, 600) if body else None
        return None

    # -- emit helpers -----------------------------------------------------

    def _emit(self, qname: str, kind: NodeKind, name: str, node: TSNode, parent: str) -> None:
        self.nodes.append(
            Node(
                qname=qname,
                kind=kind,
                path=self.rel_path,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language=self.language,
                name=name,
                signature=self._signature(node),
                docstring=self._leading_jsdoc(node),
            )
        )
        self.edges.append(
            Edge(
                src=parent,
                dst=qname,
                kind="defines",
                path=self.rel_path,
                line=node.start_point[0] + 1,
            )
        )

    # -- imports ----------------------------------------------------------

    def collect_imports(self, root: TSNode) -> None:
        for node in _descend(root, {"import_statement"}):
            source_node = node.child_by_field_name("source")
            if source_node is None:
                continue
            target = self._module_of(self.text(source_node).strip("\"'`"))
            self.edges.append(
                Edge(
                    src=self.module_qname,
                    dst=target,
                    kind="imports",
                    path=self.rel_path,
                    line=node.start_point[0] + 1,
                )
            )
            clause = next((c for c in node.named_children if c.type == "import_clause"), None)
            for child in _descend(clause or node, {"import_specifier", "namespace_import"}):
                if child.type == "namespace_import":
                    local = child.named_children[-1] if child.named_children else None
                    if local is not None:
                        self.aliases[self.text(local)] = target
                    continue
                name_node = child.child_by_field_name("name")
                alias_node = child.child_by_field_name("alias")
                if name_node is None:
                    continue
                bound = self.text(alias_node) if alias_node is not None else self.text(name_node)
                self.aliases[bound] = child_qname(target, self.text(name_node))
            if clause is not None:
                for child in clause.named_children:
                    if child.type == "identifier":  # default import
                        self.aliases[self.text(child)] = target

    def _module_of(self, specifier: str) -> str:
        """Resolve an import specifier to a module qname."""
        if not specifier.startswith("."):
            return specifier  # bare package name; stays external
        here = PurePosixPath(self.rel_path).parent
        resolved = PurePosixPath(str(here / specifier))
        parts: list[str] = []
        for part in resolved.parts:
            if part == ".":
                continue
            if part == "..":
                if parts:
                    parts.pop()
                continue
            parts.append(part)
        return module_qname_for("/".join(parts) + ".ts")

    # -- traversal --------------------------------------------------------

    def walk(self, node: TSNode, scope: _Scope) -> None:
        for child in node.named_children:
            self._visit(child, scope)

    def _visit(self, node: TSNode, scope: _Scope) -> None:
        kind = node.type
        if kind in _CONTAINER_TYPES:
            self.walk(node, scope)
            return
        if kind in _CLASS_TYPES:
            self._visit_class(node, scope)
            return
        if kind == "interface_declaration":
            self._visit_interface(node, scope)
            return
        if kind in _FUNCTION_TYPES:
            self._visit_function(node, scope)
            return
        if kind in {"lexical_declaration", "variable_declaration"}:
            self._visit_variables(node, scope)
            return
        if kind in {"call_expression", "new_expression"}:
            self._record_call(node, scope)
        self.walk(node, scope)

    def _visit_class(self, node: TSNode, scope: _Scope) -> None:
        name = self._named(node) or "<anonymous>"
        qname = child_qname(scope.qname, name)
        self._emit(qname, "class", name, node, scope.qname)
        for heritage in _descend(node, {"extends_clause", "implements_clause"}):
            for base in heritage.named_children:
                text = self.text(base).split("<")[0].strip()
                if not text:
                    continue
                dst, resolved = self._lookup(text)
                self.edges.append(
                    Edge(
                        src=qname,
                        dst=dst,
                        kind="inherits",
                        path=self.rel_path,
                        line=node.start_point[0] + 1,
                        resolved=resolved,
                    )
                )
        body = node.child_by_field_name("body")
        if body is None:
            return
        inner = _Scope(qname=qname, class_qname=qname)
        for member in body.named_children:
            if member.type in {"method_definition", "method_signature"}:
                self._visit_method(member, inner)
            else:
                self._visit(member, inner)

    def _visit_interface(self, node: TSNode, scope: _Scope) -> None:
        name = self._named(node) or "<anonymous>"
        qname = child_qname(scope.qname, name)
        self._emit(qname, "class", name, node, scope.qname)
        body = node.child_by_field_name("body")
        if body is None:
            return
        inner = _Scope(qname=qname, class_qname=qname)
        for member in body.named_children:
            if member.type in {"method_signature", "method_definition"}:
                self._visit_method(member, inner)

    def _visit_method(self, node: TSNode, scope: _Scope) -> None:
        name = self._named(node) or "<anonymous>"
        qname = child_qname(scope.qname, name)
        self._emit(qname, "method", name, node, scope.qname)
        body = node.child_by_field_name("body")
        if body is not None:
            self.walk(body, _Scope(qname=qname, class_qname=scope.class_qname))

    def _visit_function(self, node: TSNode, scope: _Scope) -> None:
        name = self._named(node) or "<anonymous>"
        qname = child_qname(scope.qname, name)
        self._emit(qname, "function", name, node, scope.qname)
        body = node.child_by_field_name("body")
        if body is not None:
            self.walk(body, _Scope(qname=qname, class_qname=scope.class_qname))

    def _visit_variables(self, node: TSNode, scope: _Scope) -> None:
        for declarator in node.named_children:
            if declarator.type != "variable_declarator":
                self._visit(declarator, scope)
                continue
            name_node = declarator.child_by_field_name("name")
            value = declarator.child_by_field_name("value")
            if name_node is None or value is None:
                continue
            if value.type not in _FUNCTION_VALUE_TYPES or name_node.type != "identifier":
                self._visit(value, scope)
                continue
            name = self.text(name_node)
            qname = child_qname(scope.qname, name)
            self._emit(qname, "function", name, declarator, scope.qname)
            body = value.child_by_field_name("body")
            if body is not None:
                # Arrow bodies may be a bare expression rather than a block.
                self._visit(body, _Scope(qname=qname, class_qname=scope.class_qname))

    # -- calls ------------------------------------------------------------

    def _record_call(self, node: TSNode, scope: _Scope) -> None:
        field_name = "constructor" if node.type == "new_expression" else "function"
        callee = node.child_by_field_name(field_name)
        if callee is None:
            return
        dst, resolved = self._resolve_callee(callee, scope)
        if dst:
            self.edges.append(
                Edge(
                    src=scope.qname,
                    dst=dst,
                    kind="calls",
                    path=self.rel_path,
                    line=node.start_point[0] + 1,
                    resolved=resolved,
                )
            )

    def _resolve_callee(self, callee: TSNode, scope: _Scope) -> tuple[str, bool]:
        if callee.type in {"identifier", "type_identifier"}:
            return self._lookup(self.text(callee))
        if callee.type == "member_expression":
            obj = callee.child_by_field_name("object")
            prop = callee.child_by_field_name("property")
            if prop is None:
                return "", False
            attr = self.text(prop)
            if obj is not None and obj.type == "this":
                owner = scope.class_qname
                return (child_qname(owner, attr), True) if owner else (f"*.{attr}", False)
            if obj is not None and obj.type == "identifier":
                base = self.text(obj)
                if base in self.aliases:
                    return child_qname(self.aliases[base], attr), True
            return f"*.{attr}", False
        return "", False

    def _lookup(self, name: str) -> tuple[str, bool]:
        if name in self.aliases:
            return self.aliases[name], True
        if name in self.local_defs:
            return child_qname(self.module_qname, name), True
        return name, False


def _descend(node: TSNode, wanted: set[str]) -> list[TSNode]:
    """Depth-first collect of named descendants (and the node itself) matching ``wanted``."""
    found: list[TSNode] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in wanted:
            found.append(current)
            continue
        stack.extend(reversed(current.named_children))
    return found


def _top_level_names(root: TSNode, collector: _Collector) -> set[str]:
    names: set[str] = set()
    for child in root.named_children:
        target = child
        if target.type == "export_statement":
            declarations = [c for c in target.named_children if c.type != "string"]
            target = declarations[0] if declarations else target
        if target.type in _CLASS_TYPES | _FUNCTION_TYPES | {"interface_declaration"}:
            name = target.child_by_field_name("name")
            if name is not None:
                names.add(collector.text(name))
        elif target.type in {"lexical_declaration", "variable_declaration"}:
            for declarator in target.named_children:
                name = declarator.child_by_field_name("name")
                if name is not None and name.type == "identifier":
                    names.add(collector.text(name))
    return names


class JavaScriptParser:
    """Parses JS/JSX/TS/TSX sources into graph nodes and edges."""

    def __init__(self, language: Language) -> None:
        if language not in {"javascript", "typescript"}:
            raise ValueError(f"JavaScriptParser cannot handle {language!r}")
        self.language: Language = language

    def parse(self, rel_path: str, source: str, module_qname: str) -> ParseResult:
        raw = source.encode("utf-8")
        parser = TSParser(_language_for(self.language, rel_path))
        tree = parser.parse(raw)
        root = tree.root_node

        module = Node(
            qname=module_qname,
            kind="module",
            path=rel_path,
            line_start=1,
            line_end=max(1, source.count("\n") + 1),
            language=self.language,
            name=module_qname.rsplit(".", 1)[-1],
            signature="",
            docstring=None,
        )
        collector = _Collector(
            rel_path=rel_path,
            module_qname=module_qname,
            language=self.language,
            source=raw,
        )
        collector.collect_imports(root)
        collector.local_defs = _top_level_names(root, collector)
        collector.walk(root, _Scope(qname=module_qname))
        return ParseResult(nodes=(module, *collector.nodes), edges=tuple(collector.edges))
