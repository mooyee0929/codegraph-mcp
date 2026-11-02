"""Generic tree-sitter parser driven by a per-language :class:`LanguageSpec`.

Two dozen grammars disagree on almost every node name, but they agree on the questions
a code graph asks: *is this node a type, a callable, an import or a call, and what is
it called?* A spec answers those questions through small hook functions
(:mod:`codegraph.parsers.ts_languages` holds the table); this module owns everything
that is the same across languages:

- traversal and lexical scoping, so ``module -> class -> method`` qnames come out the
  same way the Python and JS parsers build them;
- signatures (source text up to the body) and doc comments (the comment block directly
  above a definition, read from the source lines so it works whatever node the grammar
  attaches the comment to);
- call resolution, done once the whole file is walked so a call can see definitions
  that appear later in the file.

Resolution follows the shared convention. A call resolved through the file's own
definitions or ``self``/``this`` is ``resolved=True``. A call through an import whose
target is only a *guess* at a repo-local qname (``crate::util::pad`` ->
``src.util.pad``) keeps the guess as ``dst`` but stays ``resolved=False`` so the linker
can still upgrade it, and the guess is copied into ``Edge.detail`` because the linker
overwrites ``dst``. A method call on an unknown object becomes ``*.name``; when the
receiver was a static path (``Strings.pad``) that path goes into ``detail`` too, which
is what :mod:`codegraph.parsers.resolve` uses to finish the job across files.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from functools import cache
from typing import Literal

from tree_sitter import Language as TSLanguage
from tree_sitter import Node as TSNode
from tree_sitter import Parser as TSParser
from tree_sitter_language_pack import SupportedLanguage, get_language

from codegraph.models import Edge, Language, Node, NodeKind, ParseResult
from codegraph.parsers.base import child_qname, clip

DefinitionKind = Literal["class", "function", "namespace", "owner"]
"""What a definition hook found.

- ``class`` / ``function``: emit a node; functions directly inside a class become methods.
- ``namespace``: adds a qname segment but no node (Rust inline ``mod``, Julia ``module``).
- ``owner``: a block that adds members to a type declared elsewhere (Rust ``impl``,
  Swift ``extension``); members get the type's qname, no node is emitted for the block.
"""


@dataclass(frozen=True, slots=True)
class Definition:
    kind: DefinitionKind
    name: str
    node: TSNode
    body: tuple[TSNode, ...] = ()
    owner: str | None = None
    """For callables declared outside their type: Go receivers, ``Foo::bar`` in C++."""
    bases: tuple[str, ...] = ()
    self_name: str | None = None
    """Extra receiver name bound to the enclosing type, e.g. Go's ``s`` in ``(s *Server)``."""
    signature: str | None = None
    end_node: TSNode | None = None
    """Last node of the definition when the grammar splits it (Dart signature + body)."""
    consumed: tuple[TSNode, ...] = ()
    """Siblings that belong to this definition and must not be walked again."""
    sequential: bool = False
    """Statement-form class (Perl ``package Foo;``): owns the siblings that follow it."""
    anchor: TSNode | None = None
    """Where the span starts and the doc comment is looked up, when that is not ``node``
    (a Haskell type signature above its equations)."""


@dataclass(frozen=True, slots=True)
class Import:
    """One imported module and the local names it binds.

    ``target`` is the module qname when it can be derived from the path (a guess when
    ``exact`` is False) or the external specifier as written. ``bindings`` map local
    names to the qnames they stand for.
    """

    spec: str
    target: str
    line: int
    bindings: tuple[tuple[str, str], ...] = ()
    exact: bool = True


@dataclass(frozen=True, slots=True)
class Reference:
    """A use of a name: a call, or a base type introduced by a statement (Ruby ``include``).

    ``receiver`` is the static path before the name (``("util", "fmt")`` for
    ``util::fmt::run``); ``dynamic`` means the receiver was an expression we cannot
    name (``a().b()``, ``self.client.send()``).
    """

    name: str
    line: int
    receiver: tuple[str, ...] = ()
    dynamic: bool = False
    kind: Literal["calls", "inherits"] = "calls"
    external: bool = False
    """The receiver is known to be outside the repository (R's ``pkg::fn``)."""


@dataclass(frozen=True, slots=True)
class Alias:
    target: str
    exact: bool


@dataclass(frozen=True, slots=True)
class CommentStyle:
    """How doc comments look in a language: line prefixes and an optional block pair."""

    line_prefixes: tuple[str, ...]
    block: tuple[str, str] | None = None
    skip_prefixes: tuple[str, ...] = ()


C_COMMENTS = CommentStyle(("///", "//!", "//"), ("/*", "*/"), ("@", "#["))
HASH_COMMENTS = CommentStyle(("#",))
DASH_COMMENTS = CommentStyle(("--",), ("{-", "-}"))
PERCENT_COMMENTS = CommentStyle(("%",))


@dataclass(slots=True)
class FileContext:
    """Per-file state handed to every hook."""

    rel_path: str
    module_qname: str
    source: bytes
    spec: LanguageSpec
    lines: list[str]
    root: TSNode | None = None
    memo: dict[str, str] = field(default_factory=dict)

    def text(self, node: TSNode | None) -> str:
        if node is None:
            return ""
        return self.source[node.start_byte : node.end_byte].decode("utf-8", "replace")

    def field_text(self, node: TSNode, *names: str) -> str | None:
        for name in names:
            child = node.child_by_field_name(name)
            if child is not None:
                return self.text(child)
        return None


DefineHook = Callable[[FileContext, TSNode], Definition | None]
ImportHook = Callable[[FileContext, TSNode], list[Import] | None]
ReferenceHook = Callable[[FileContext, TSNode], list[Reference] | None]
PreludeHook = Callable[[FileContext], dict[str, Alias]]
SourceHook = Callable[[str], str]


def no_imports(_ctx: FileContext, _node: TSNode) -> list[Import] | None:
    return None


def no_definitions(_ctx: FileContext, _node: TSNode) -> Definition | None:
    return None


def generic_references(ctx: FileContext, node: TSNode) -> list[Reference] | None:
    """Calls for grammars whose call nodes carry a callee expression (most of them)."""
    if node.type not in ctx.spec.call_types:
        return None
    callee = callee_node(node)
    if callee is None:
        return []
    if callee.id != node.id and callee.type in ctx.spec.call_types:
        return []  # curried/chained: the inner call records itself
    reference = split_callee(ctx, callee, node.start_point[0] + 1)
    return [reference] if reference is not None else []


@dataclass(frozen=True, slots=True)
class LanguageSpec:
    language: Language
    grammar: SupportedLanguage
    define: DefineHook = no_definitions
    references: ReferenceHook = generic_references
    imports: ImportHook = no_imports
    import_types: frozenset[str] = frozenset()
    call_types: frozenset[str] = frozenset()
    comments: CommentStyle = C_COMMENTS
    self_names: frozenset[str] = frozenset({"self", "this"})
    external_roots: frozenset[str] = frozenset()
    """Receiver roots that are always external (``std``), so calls through them stay
    resolved to the written path instead of becoming ``*.name``."""
    prelude: PreludeHook | None = None
    preprocess: SourceHook | None = None
    grammar_for: Callable[[str, str], SupportedLanguage] | None = None
    """Pick a grammar per file from (rel_path, source): ``.h`` may be C, C++ or ObjC."""


# -- generic node helpers ---------------------------------------------------------

IDENTIFIER_TYPES = frozenset(
    {
        "identifier",
        "type_identifier",
        "field_identifier",
        "simple_identifier",
        "property_identifier",
        "constant",
        "name",
        "word",
        "atom",
        "variable",
        "bareword",
        "IDENTIFIER",
        "package_identifier",
        "namespace_identifier",
        "constructor",
    }
)

_CALLEE_FIELDS = ("function", "callee", "constructor", "macro", "expr", "target")
_NAME_FIELDS = (
    "name",
    "field",
    "property",
    "method",
    "fun",
    "attribute",
    "suffix",
    "rhs",
    "right",
)
_RECEIVER_FIELDS = (
    "object",
    "receiver",
    "operand",
    "value",
    "argument",
    "expression",
    "scope",
    "path",
    "table",
    "target",
    "module",
    "invocant",
    "left",
    "lhs",
    "prefix",
)
_WRAPPER_TYPES = frozenset(
    {"expression", "parenthesized_expression", "primary_expression", "ErrorUnionExpr"}
)
_GENERIC_TYPES = frozenset(
    {"generic_type", "generic_name", "template_type", "parameterized_type", "generic_function"}
)
_SEPARATORS = re.compile(r"::|->|\?\.|!!\.|\.|\\|:")
_PATH = re.compile(r"^[$@&*]*[A-Za-z_][\w$]*(?:(?:::|->|\?\.|\.|\\|:)[$@]*[A-Za-z_][\w$]*)*$")
_NAME = re.compile(r"^[A-Za-z_][\w$]*[!?]?$")
_GENERIC_ARGS = re.compile(r"(::)?<[^<>]*>|\[[^\[\]]*\]")


def descendants(node: TSNode, types: frozenset[str]) -> Iterator[TSNode]:
    """Depth-first named descendants of the wanted types; does not descend into matches."""
    stack = list(reversed(node.named_children))
    while stack:
        current = stack.pop()
        if current.type in types:
            yield current
            continue
        stack.extend(reversed(current.named_children))


def first_child(node: TSNode, types: frozenset[str]) -> TSNode | None:
    for child in node.named_children:
        if child.type in types:
            return child
    return None


def name_of(
    ctx: FileContext,
    node: TSNode,
    fields: tuple[str, ...] = ("name",),
    types: frozenset[str] = IDENTIFIER_TYPES,
) -> str | None:
    """A definition's name: the first matching field, else the first identifier child."""
    for name_field in fields:
        child = node.child_by_field_name(name_field)
        if child is not None:
            return leaf_name(ctx, child)
    child = first_child(node, types)
    return leaf_name(ctx, child) if child is not None else None


def leaf_name(ctx: FileContext, node: TSNode) -> str | None:
    """The rightmost identifier inside ``node`` (``a.b.c`` -> ``c``)."""
    text = strip_generics(ctx.text(node)).strip()
    if _NAME.match(text):
        return text
    if _PATH.match(text):
        return _SEPARATORS.split(text)[-1].lstrip("$@&*") or None
    for child in reversed(node.named_children):
        found = leaf_name(ctx, child)
        if found:
            return found
    return None


def strip_generics(text: str) -> str:
    """``Vec<T>`` -> ``Vec``, ``Map[K, V]`` -> ``Map``, ``Vec::<u8>::new`` -> ``Vec::new``."""
    previous = None
    while previous != text:
        previous = text
        text = _GENERIC_ARGS.sub("", text)
    return text


def path_parts(text: str) -> tuple[str, ...] | None:
    """Split a static path (``a::b.c``, ``$this->x``) into names, or None if it is dynamic."""
    cleaned = strip_generics("".join(text.split())).rstrip(":.")
    if not _PATH.match(cleaned):
        return None
    return tuple(part.lstrip("$@&*") for part in _SEPARATORS.split(cleaned) if part)


def type_names(ctx: FileContext, node: TSNode, types: frozenset[str]) -> list[str]:
    """Texts of the type references under ``node``, generics stripped."""
    names: list[str] = []
    for found in descendants(node, types):
        text = strip_generics(ctx.text(found)).strip()
        if text:
            names.append(text)
    return names


def string_value(ctx: FileContext, node: TSNode) -> str:
    text = ctx.text(node).strip()
    for prefix in ("r#", "b", "r", "f", "@"):
        if text.startswith(prefix) and len(text) > len(prefix) and text[len(prefix)] in "\"'":
            text = text[len(prefix) :]
            break
    for quote in ('"""', "'''", '"', "'", "`"):
        if len(text) >= 2 * len(quote) and text.startswith(quote) and text.endswith(quote):
            return text[len(quote) : -len(quote)]
    return text


def callee_node(call: TSNode) -> TSNode | None:
    """The expression a call node invokes."""
    for callee_field in _CALLEE_FIELDS:
        child = call.child_by_field_name(callee_field)
        if child is not None:
            return child
    named = call.child_by_field_name("name") or call.child_by_field_name("method")
    if named is not None:
        if any(call.child_by_field_name(f) is not None for f in _RECEIVER_FIELDS):
            return call
        return named
    typed = call.child_by_field_name("type")
    if typed is not None:
        return typed
    return call.named_children[0] if call.named_children else None


def split_callee(ctx: FileContext, callee: TSNode, line: int) -> Reference | None:
    """Turn a callee expression into a :class:`Reference` (receiver path + name)."""
    while callee.type in _WRAPPER_TYPES | _GENERIC_TYPES and callee.named_children:
        callee = callee.named_children[0]
    if callee.named_child_count == 0 or callee.type in IDENTIFIER_TYPES:
        parts = path_parts(ctx.text(callee))
        if not parts:
            return None
        return _reference(parts[-1], parts[:-1], line)
    name_node = next(
        (c for f in _NAME_FIELDS if (c := callee.child_by_field_name(f)) is not None), None
    )
    receiver = next(
        (c for f in _RECEIVER_FIELDS if (c := callee.child_by_field_name(f)) is not None), None
    )
    if name_node is None:
        kids = callee.named_children
        if len(kids) < 2:
            parts = path_parts(ctx.text(callee))
            return _reference(parts[-1], parts[:-1], line) if parts else None
        receiver, name_node = kids[0], kids[-1]
    if receiver is not None and receiver.id == name_node.id:
        receiver = None
    name = leaf_name(ctx, name_node)
    if not name:
        return None
    if receiver is None:
        return _reference(name, (), line)
    if receiver.type in ctx.spec.call_types:
        return Reference(name=name, line=line, dynamic=True)
    parts = path_parts(ctx.text(receiver))
    if parts is None:
        return Reference(name=name, line=line, dynamic=True)
    return _reference(name, parts, line)


def _reference(name: str, receiver: tuple[str, ...], line: int) -> Reference | None:
    return Reference(name=name, receiver=receiver, line=line) if _NAME.match(name) else None


# -- doc comments ------------------------------------------------------------------

_MARKER = re.compile(r"^(?:/{2,3}!?|/\*+!?|\*+/?|#+'?|-{2,3}\s*\|?|%+|@doc\b|\{-\s*\|?)\s?")
_XML_TAG = re.compile(r"<[^>]+>")


def leading_comment(lines: list[str], start_row: int, style: CommentStyle) -> str | None:
    """The contiguous comment block ending on the line above ``start_row`` (0-based)."""
    collected: list[str] = []
    row = start_row - 1
    while row >= 0:
        text = lines[row].strip()
        if not text:
            break
        if style.block is not None and text.endswith(style.block[1]):
            block: list[str] = []
            while row >= 0:
                block.append(lines[row].strip())
                if style.block[0] in lines[row]:
                    break
                row -= 1
            collected.extend(block)
            row -= 1
            continue
        if text.startswith(style.line_prefixes):
            collected.append(text)
            row -= 1
            continue
        if style.skip_prefixes and text.startswith(style.skip_prefixes) and not collected:
            row -= 1
            continue
        break
    if not collected:
        return None
    cleaned: list[str] = []
    for raw in reversed(collected):
        line = raw
        if style.block is not None:
            line = line.removesuffix(style.block[1])
        line = _MARKER.sub("", line).strip().removesuffix("*/").strip()
        line = _XML_TAG.sub("", line).strip().strip('"').strip()
        if line and not line.startswith("#!"):
            cleaned.append(line)
    body = " ".join(cleaned)
    return clip(body, 600) if body else None


# -- the collector -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Scope:
    qname: str
    parent: str
    class_qname: str | None
    self_names: frozenset[str]


@dataclass(slots=True)
class _Collector:
    ctx: FileContext
    nodes: dict[str, Node] = field(default_factory=dict)
    defines: list[Edge] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    aliases: dict[str, Alias] = field(default_factory=dict)
    members: dict[str, set[str]] = field(default_factory=dict)
    pending: list[tuple[_Scope, Reference]] = field(default_factory=list)
    bases: list[tuple[str, Reference]] = field(default_factory=list)
    consumed: set[int] = field(default_factory=set)

    # -- traversal ---------------------------------------------------------

    def walk(self, node: TSNode, scope: _Scope) -> None:
        current = scope
        children = node.named_children
        for index, child in enumerate(children):
            if child.id in self.consumed:
                continue
            definition = self._definition(child)
            if definition is not None and definition.sequential:
                following = next((c for c in children[index + 1 :] if self._is_sequential(c)), None)
                end_row = following.start_point[0] - 1 if following else node.end_point[0]
                current = self._open_class(definition, scope, end_row)
                continue
            self.visit(child, current, definition)

    def _is_sequential(self, node: TSNode) -> bool:
        definition = self._definition(node)
        return definition is not None and definition.sequential

    def _definition(self, node: TSNode) -> Definition | None:
        return self.ctx.spec.define(self.ctx, node)

    def visit(self, node: TSNode, scope: _Scope, definition: Definition | None = None) -> None:
        spec = self.ctx.spec
        if node.type in spec.import_types:
            found = spec.imports(self.ctx, node)
            if found is not None:
                for item in found:
                    self._import(item)
                return
        if definition is None:
            definition = self._definition(node)
        if definition is not None:
            self._define(definition, scope)
            return
        references = spec.references(self.ctx, node)
        if references:
            self.pending.extend((scope, ref) for ref in references)
        self.walk(node, scope)

    def _walk_bodies(self, definition: Definition, scope: _Scope) -> None:
        for extra in definition.consumed:
            self.consumed.add(extra.id)
        for body in definition.body:
            if body.id == definition.node.id:
                self.walk(body, scope)
            else:
                self.visit(body, scope)

    # -- definitions -------------------------------------------------------

    def _define(self, definition: Definition, scope: _Scope) -> None:
        if definition.kind == "namespace":
            qname = child_qname(scope.qname, definition.name)
            self._member(scope.qname, definition.name)
            inner = _Scope(qname, scope.parent, None, scope.self_names)
            self._walk_bodies(definition, inner)
            return
        if definition.kind == "owner":
            qname = child_qname(scope.qname, definition.name)
            self._record_bases(qname, definition)
            self._walk_bodies(definition, _Scope(qname, qname, qname, scope.self_names))
            return
        if definition.kind == "class":
            end_row = (definition.end_node or definition.node).end_point[0]
            self._open_class(definition, scope, end_row)
            return
        self._define_function(definition, scope)

    def _open_class(self, definition: Definition, scope: _Scope, end_row: int) -> _Scope:
        qname = child_qname(scope.qname, definition.name)
        self._member(scope.qname, definition.name)
        self._emit(qname, "class", definition, scope.parent, end_row)
        self._record_bases(qname, definition)
        inner = _Scope(qname, qname, qname, scope.self_names)
        self._walk_bodies(definition, inner)
        return inner

    def _define_function(self, definition: Definition, scope: _Scope) -> None:
        class_qname: str | None
        if definition.owner:
            owner = child_qname(scope.qname, definition.owner)
            qname = child_qname(owner, definition.name)
            kind: NodeKind = "method"
            parent, class_qname = owner, owner
            self._member(owner, definition.name)
            self._member(scope.qname, definition.owner.split(".")[0])
        else:
            qname = child_qname(scope.qname, definition.name)
            in_class = scope.class_qname is not None and scope.qname == scope.class_qname
            kind = "method" if in_class else "function"
            parent, class_qname = scope.parent, scope.class_qname
            self._member(scope.qname, definition.name)
        end_row = (definition.end_node or definition.node).end_point[0]
        self._emit(qname, kind, definition, parent, end_row)
        self_names = scope.self_names
        if definition.self_name:
            self_names = self_names | {definition.self_name}
        self._walk_bodies(definition, _Scope(qname, qname, class_qname, self_names))

    def _member(self, scope_qname: str, name: str) -> None:
        self.members.setdefault(scope_qname, set()).add(name)

    def _emit(
        self, qname: str, kind: NodeKind, definition: Definition, parent: str, end_row: int
    ) -> None:
        first = definition.anchor or definition.node
        start = first.start_point[0] + 1
        end = max(start, end_row + 1)
        existing = self.nodes.get(qname)
        if existing is not None:  # Haskell/Erlang clauses, ObjC @interface + @implementation
            self.nodes[qname] = replace(
                existing,
                line_start=min(existing.line_start, start),
                line_end=max(existing.line_end, end),
            )
            return
        self.nodes[qname] = Node(
            qname=qname,
            kind=kind,
            path=self.ctx.rel_path,
            line_start=start,
            line_end=end,
            language=self.ctx.spec.language,
            name=definition.name.rsplit(".", 1)[-1],
            signature=self._signature(definition),
            docstring=leading_comment(self.ctx.lines, first.start_point[0], self.ctx.spec.comments),
        )
        self.defines.append(
            Edge(src=parent, dst=qname, kind="defines", path=self.ctx.rel_path, line=start)
        )

    def _signature(self, definition: Definition) -> str:
        if definition.signature is not None:
            return clip(definition.signature)
        node = definition.node
        body = next((b for b in definition.body if b.id != node.id), None)
        source = self.ctx.source
        if body is not None and node.start_byte < body.start_byte <= node.end_byte:
            text = source[node.start_byte : body.start_byte].decode("utf-8", "replace")
        else:
            text = self.ctx.text(node).split("\n", 1)[0]
        style = self.ctx.spec.comments
        markers = style.line_prefixes + ((style.block[0],) if style.block else ())
        lines = [line for line in text.splitlines() if not line.strip().startswith(markers)]
        text = " ".join(lines)
        return clip(text.rstrip().removesuffix("{").removesuffix(" do").rstrip(" ,"))

    def _record_bases(self, qname: str, definition: Definition) -> None:
        line = definition.node.start_point[0] + 1
        for base in definition.bases:
            parts = path_parts(strip_generics(base))
            if parts:
                self.bases.append((qname, Reference(parts[-1], line, parts[:-1], kind="inherits")))

    # -- imports -----------------------------------------------------------

    def _import(self, item: Import) -> None:
        if item.target:
            self.edges.append(
                Edge(
                    src=self.ctx.module_qname,
                    dst=item.target,
                    kind="imports",
                    path=self.ctx.rel_path,
                    line=item.line,
                    detail=item.spec,
                )
            )
        for local, target in item.bindings:
            self.aliases[local] = Alias(target=target, exact=item.exact)

    # -- resolution --------------------------------------------------------

    def finish(self, module: Node) -> ParseResult:
        emitted = set(self.nodes) | {module.qname}
        edges: list[Edge] = []
        for edge in self.defines:
            src = edge.src if edge.src in emitted else self._nearest(edge.src, emitted)
            edges.append(replace(edge, src=src))
        for qname, reference in self.bases:
            scope = _Scope(qname, qname, qname, frozenset())
            self._append_reference(edges, scope, reference, qname, emitted)
        for scope, reference in self.pending:
            if reference.kind == "inherits":
                if scope.class_qname is None:
                    continue
                src = scope.class_qname
            else:
                src = scope.qname if scope.qname in emitted else self._nearest(scope.qname, emitted)
            self._append_reference(edges, scope, reference, src, emitted)
        edges.extend(self.edges)
        return ParseResult(nodes=(module, *self.nodes.values()), edges=tuple(edges))

    def _nearest(self, qname: str, emitted: set[str]) -> str:
        cursor = qname
        while "." in cursor:
            cursor = cursor.rsplit(".", 1)[0]
            if cursor in emitted:
                return cursor
        return self.ctx.module_qname

    def _append_reference(
        self,
        edges: list[Edge],
        scope: _Scope,
        reference: Reference,
        src: str,
        emitted: set[str],
    ) -> None:
        dst, resolved, detail = self._resolve(scope, reference, emitted)
        if not dst or (dst == src and reference.kind == "inherits"):
            return
        edges.append(
            Edge(
                src=src,
                dst=dst,
                kind=reference.kind,
                path=self.ctx.rel_path,
                line=reference.line,
                resolved=resolved,
                detail=detail,
            )
        )

    def _lexical(self, scope_qname: str, name: str) -> str | None:
        cursor = scope_qname
        while True:
            if name in self.members.get(cursor, ()):
                return child_qname(cursor, name)
            if cursor == self.ctx.module_qname or "." not in cursor:
                return None
            cursor = cursor.rsplit(".", 1)[0]

    def _resolve(
        self, scope: _Scope, reference: Reference, emitted: set[str]
    ) -> tuple[str, bool, str]:
        name = reference.name
        if reference.dynamic:
            return f"*.{name}", False, ""
        parts = reference.receiver
        if reference.external:
            return ".".join((*parts, name)), True, ""
        if not parts:
            alias = self.aliases.get(name)
            if alias is not None:
                return alias.target, alias.exact, "" if alias.exact else alias.target
            local = self._lexical(scope.qname, name)
            if local is not None:
                return local, True, ""
            return name, False, ""
        head, rest = parts[0], parts[1:]
        if head in scope.self_names or head in self.ctx.spec.self_names:
            # Only ``self.method()`` is trustworthy; ``self.client.send()`` needs types.
            if scope.class_qname is None or rest:
                return f"*.{name}", False, ""
            return child_qname(scope.class_qname, name), True, ""
        alias = self.aliases.get(head)
        if alias is not None:
            target = ".".join((alias.target, *rest, name))
            return target, alias.exact, "" if alias.exact else target
        local = self._lexical(scope.qname, head)
        if local is not None:
            target = ".".join((local, *rest, name))
            return (target, True, "") if target in emitted else (target, False, target)
        if head in self.ctx.spec.external_roots:
            return ".".join((*parts, name)), True, ""
        return f"*.{name}", False, ".".join((*parts, name))


# -- the parser --------------------------------------------------------------------


@cache
def _language(grammar: SupportedLanguage) -> TSLanguage:
    return get_language(grammar)


class TreeSitterParser:
    """A :class:`codegraph.parsers.base.Parser` for any language with a :class:`LanguageSpec`."""

    def __init__(self, spec: LanguageSpec) -> None:
        self.spec = spec
        self.language: Language = spec.language

    def parse(self, rel_path: str, source: str, module_qname: str) -> ParseResult:
        spec = self.spec
        text = spec.preprocess(source) if spec.preprocess is not None else source
        grammar = spec.grammar_for(rel_path, text) if spec.grammar_for else spec.grammar
        raw = text.encode("utf-8")
        tree = TSParser(_language(grammar)).parse(raw)
        ctx = FileContext(
            rel_path=rel_path,
            module_qname=module_qname,
            source=raw,
            spec=spec,
            lines=text.splitlines(),
            root=tree.root_node,
        )
        module = Node(
            qname=module_qname,
            kind="module",
            path=rel_path,
            line_start=1,
            line_end=max(1, source.count("\n") + 1),
            language=spec.language,
            name=module_qname.rsplit(".", 1)[-1],
            signature="",
            docstring=None,
        )
        collector = _Collector(ctx=ctx)
        if spec.prelude is not None:
            collector.aliases.update(spec.prelude(ctx))
        collector.walk(tree.root_node, _Scope(module_qname, module_qname, None, frozenset()))
        return collector.finish(module)


__all__ = [
    "C_COMMENTS",
    "DASH_COMMENTS",
    "HASH_COMMENTS",
    "IDENTIFIER_TYPES",
    "PERCENT_COMMENTS",
    "Alias",
    "CommentStyle",
    "Definition",
    "FileContext",
    "Import",
    "LanguageSpec",
    "Reference",
    "TreeSitterParser",
    "callee_node",
    "descendants",
    "first_child",
    "generic_references",
    "leaf_name",
    "leading_comment",
    "name_of",
    "path_parts",
    "split_callee",
    "string_value",
    "strip_generics",
    "type_names",
]
