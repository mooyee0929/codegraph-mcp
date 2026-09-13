"""Python parser built on the standard library ``ast`` module.

No third-party dependency, and the resolution it can do is real rather than textual:
imports give us an alias table, so a call to ``billing.charge()`` becomes an edge to
``pkg.billing.charge`` instead of the bare string ``charge``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from codegraph.models import Edge, Language, Node, NodeKind, ParseResult
from codegraph.parsers.base import child_qname, clip


@dataclass(slots=True)
class _Scope:
    """One lexical level: the module, a class, or a function."""

    qname: str
    is_class: bool = False
    class_qname: str | None = None


@dataclass(slots=True)
class _Collector(ast.NodeVisitor):
    rel_path: str
    module_qname: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)
    local_defs: set[str] = field(default_factory=set)
    scopes: list[_Scope] = field(default_factory=list)

    # -- scope helpers ----------------------------------------------------

    @property
    def current(self) -> _Scope:
        return self.scopes[-1]

    def _enclosing_class(self) -> str | None:
        for scope in reversed(self.scopes):
            if scope.is_class:
                return scope.qname
        return None

    # -- imports ----------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            target = alias.name
            bound = alias.asname or target.split(".")[0]
            self.aliases[bound] = target if alias.asname else target.split(".")[0]
            self.edges.append(
                Edge(
                    src=self.module_qname,
                    dst=target,
                    kind="imports",
                    path=self.rel_path,
                    line=node.lineno,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = self._resolve_relative(node.module, node.level)
        for alias in node.names:
            if alias.name == "*":
                continue
            target = child_qname(base, alias.name)
            self.aliases[alias.asname or alias.name] = target
            self.edges.append(
                Edge(
                    src=self.module_qname,
                    dst=target,
                    kind="imports",
                    path=self.rel_path,
                    line=node.lineno,
                )
            )

    def _resolve_relative(self, module: str | None, level: int) -> str:
        """Turn ``from ..a.b import c`` into an absolute dotted prefix."""
        if level == 0:
            return module or ""
        package = self.module_qname.split(".")
        # A module's own name is not part of its package path.
        package = package[:-1] if package else []
        trimmed = package[: len(package) - (level - 1)] if level > 1 else package
        prefix = ".".join(trimmed)
        return child_qname(prefix, module) if module else prefix

    # -- definitions ------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qname = child_qname(self.current.qname, node.name)
        self.nodes.append(
            Node(
                qname=qname,
                kind="class",
                path=self.rel_path,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                language="python",
                name=node.name,
                signature=clip(f"class {node.name}({_bases_text(node)})"),
                docstring=_docstring(node),
            )
        )
        self.edges.append(
            Edge(
                src=self.current.qname,
                dst=qname,
                kind="defines",
                path=self.rel_path,
                line=node.lineno,
            )
        )
        for base in node.bases:
            dst, resolved = self._resolve_expr(base)
            if dst:
                self.edges.append(
                    Edge(
                        src=qname,
                        dst=dst,
                        kind="inherits",
                        path=self.rel_path,
                        line=node.lineno,
                        resolved=resolved,
                    )
                )
        for decorator in node.decorator_list:
            self._record_call(decorator, src=qname)
        self.scopes.append(_Scope(qname=qname, is_class=True, class_qname=qname))
        for child in node.body:
            self.visit(child)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qname = child_qname(self.current.qname, node.name)
        kind: NodeKind = "method" if self.current.is_class else "function"
        self.nodes.append(
            Node(
                qname=qname,
                kind=kind,
                path=self.rel_path,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                language="python",
                name=node.name,
                signature=clip(_signature(node)),
                docstring=_docstring(node),
            )
        )
        self.edges.append(
            Edge(
                src=self.current.qname,
                dst=qname,
                kind="defines",
                path=self.rel_path,
                line=node.lineno,
            )
        )
        for decorator in node.decorator_list:
            self._record_call(decorator, src=qname)
        enclosing_class = self._enclosing_class()
        self.scopes.append(_Scope(qname=qname, is_class=False, class_qname=enclosing_class))
        for child in node.body:
            self.visit(child)
        self.scopes.pop()

    # -- calls ------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        self._record_call(node, src=self.current.qname)
        self.generic_visit(node)

    def _record_call(self, node: ast.expr, src: str) -> None:
        func = node.func if isinstance(node, ast.Call) else node
        dst, resolved = self._resolve_expr(func)
        if not dst:
            return
        self.edges.append(
            Edge(
                src=src,
                dst=dst,
                kind="calls",
                path=self.rel_path,
                line=getattr(node, "lineno", 0),
                resolved=resolved,
            )
        )

    def _resolve_expr(self, expr: ast.expr) -> tuple[str, bool]:
        """Best-effort qname for a callable expression, plus whether it is trustworthy."""
        if isinstance(expr, ast.Name):
            if expr.id in self.aliases:
                return self.aliases[expr.id], True
            if expr.id in self.local_defs:
                return child_qname(self.module_qname, expr.id), True
            return expr.id, False
        if isinstance(expr, ast.Attribute):
            root = _dotted(expr)
            if root is None:
                return f"*.{expr.attr}", False
            head, _, tail = root.partition(".")
            if head == "self":
                owner = self.current.class_qname or self._enclosing_class()
                # Only ``self.method()`` is trustworthy; ``self.client.send()`` needs
                # type inference we do not have, so leave it for the linker.
                if owner and tail and "." not in tail:
                    return child_qname(owner, tail), True
                return f"*.{expr.attr}", False
            if head in self.aliases:
                return child_qname(self.aliases[head], tail), True
            return f"*.{expr.attr}", False
        if isinstance(expr, ast.Call):
            return self._resolve_expr(expr.func)
        return "", False


def _dotted(expr: ast.expr) -> str | None:
    """Flatten ``a.b.c`` to ``"a.b.c"``; return None for anything dynamic."""
    parts: list[str] = []
    cursor: ast.expr = expr
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if not isinstance(cursor, ast.Name):
        return None
    parts.append(cursor.id)
    return ".".join(reversed(parts))


def _bases_text(node: ast.ClassDef) -> str:
    try:
        return ", ".join(ast.unparse(b) for b in node.bases)
    except Exception:  # pragma: no cover - unparse is total in practice
        return ""


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    try:
        args = ast.unparse(node.args)
        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    except Exception:  # pragma: no cover
        args, returns = "...", ""
    return f"{prefix} {node.name}({args}){returns}"


def _docstring(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        return None
    text = ast.get_docstring(node)
    return clip(text, 600) if text else None


class PythonParser:
    """Parses ``.py``/``.pyi`` sources into graph nodes and edges."""

    language: Language = "python"

    def parse(self, rel_path: str, source: str, module_qname: str) -> ParseResult:
        try:
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError:
            # A file we cannot parse still exists; record the module so paths resolve.
            module = Node(
                qname=module_qname,
                kind="module",
                path=rel_path,
                line_start=1,
                line_end=max(1, source.count("\n") + 1),
                language="python",
                name=module_qname.rsplit(".", 1)[-1],
                signature="",
                docstring=None,
            )
            return ParseResult(nodes=(module,), edges=())

        module = Node(
            qname=module_qname,
            kind="module",
            path=rel_path,
            line_start=1,
            line_end=max(1, source.count("\n") + 1),
            language="python",
            name=module_qname.rsplit(".", 1)[-1],
            signature="",
            docstring=_docstring(tree),
        )
        collector = _Collector(rel_path=rel_path, module_qname=module_qname)
        collector.local_defs = {
            stmt.name
            for stmt in tree.body
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        }
        collector.scopes.append(_Scope(qname=module_qname))
        for stmt in tree.body:
            collector.visit(stmt)
        return ParseResult(
            nodes=(module, *collector.nodes),
            edges=tuple(collector.edges),
        )
