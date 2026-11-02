"""The per-language table for :class:`codegraph.parsers.treesitter.TreeSitterParser`.

Most languages are described by a :class:`Table` (which node types are types, callables
and namespaces, and where their names and bodies live) plus the generic call splitter.
Grammars that do not fit that shape get a small hook of their own: Go methods hang off
a receiver, Rust ``impl`` blocks add members to a type declared elsewhere, Dart splits a
function into a signature and a sibling body, Elixir and Erlang express everything as
calls, Zig spells a call as a chain of suffixes.

Declarations without a body (Java interface methods, C prototypes, Go interface method
sets, Rust trait signatures) are deliberately not emitted: they would give every
interface-dispatched call two candidates and stop the linker from resolving any of them.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from tree_sitter import Node as TSNode
from tree_sitter_language_pack import SupportedLanguage

from codegraph.models import Language
from codegraph.parsers import ts_imports as imp
from codegraph.parsers.treesitter import (
    DASH_COMMENTS,
    HASH_COMMENTS,
    PERCENT_COMMENTS,
    CommentStyle,
    DefineHook,
    Definition,
    FileContext,
    LanguageSpec,
    Reference,
    descendants,
    first_child,
    generic_references,
    leaf_name,
    name_of,
    path_parts,
    split_callee,
    string_value,
    strip_generics,
    type_names,
)

BasesHook = Callable[[FileContext, TSNode], list[str]]

_TYPE_REFS = frozenset(
    {
        "type_identifier",
        "scoped_type_identifier",
        "generic_type",
        "user_type",
        "identifier",
        "qualified_name",
        "generic_name",
        "qualified_identifier",
        "template_type",
        "name",
        "constant",
        "scope_resolution",
        "stable_type_identifier",
        "user_defined_type",
        "type_name",
    }
)


def _line(node: TSNode) -> int:
    return node.start_point[0] + 1


def first_of(*hooks: DefineHook) -> DefineHook:
    """Compose definition hooks: the first one that recognises the node wins."""

    def define(ctx: FileContext, node: TSNode) -> Definition | None:
        for hook in hooks:
            found = hook(ctx, node)
            if found is not None:
                return found
        return None

    return define


def bases_in(containers: frozenset[str], refs: frozenset[str] = _TYPE_REFS) -> BasesHook:
    """Base types listed under the given child nodes (``extends``/``implements`` clauses)."""

    def bases(ctx: FileContext, node: TSNode) -> list[str]:
        found: list[str] = []
        for child in node.named_children:
            if child.type in containers:
                if child.type in refs:
                    found.append(strip_generics(ctx.text(child)))
                else:
                    found.extend(type_names(ctx, child, refs))
        for container in containers:
            by_field = node.child_by_field_name(container)
            if by_field is not None and by_field.type not in containers:
                found.extend(
                    type_names(ctx, by_field, refs) or [strip_generics(ctx.text(by_field))]
                )
        return list(dict.fromkeys(b for b in found if b))

    return bases


@dataclass(frozen=True, slots=True)
class Table:
    """Table-driven definitions: node types plus where names and bodies live."""

    classes: frozenset[str] = frozenset()
    functions: frozenset[str] = frozenset()
    namespaces: frozenset[str] = frozenset()
    name_fields: tuple[str, ...] = ("name",)
    body_fields: tuple[str, ...] = ("body",)
    body_types: frozenset[str] = frozenset()
    bases: BasesHook | None = None
    needs_body: bool = True

    def __call__(self, ctx: FileContext, node: TSNode) -> Definition | None:
        if node.type in self.classes:
            kind: str = "class"
        elif node.type in self.functions:
            kind = "function"
        elif node.type in self.namespaces:
            kind = "namespace"
        else:
            return None
        name = name_of(ctx, node, self.name_fields)
        if not name:
            return None
        body = self.body_of(node)
        if kind == "function" and body is None and self.needs_body:
            return None
        bodies = (body,) if body is not None else ()
        if kind == "class":
            bases = tuple(self.bases(ctx, node)) if self.bases else ()
            return Definition("class", name, node, bodies, bases=bases)
        if kind == "namespace":
            return Definition("namespace", name, node, bodies)
        return Definition("function", name, node, bodies)

    def body_of(self, node: TSNode) -> TSNode | None:
        for body_field in self.body_fields:
            child = node.child_by_field_name(body_field)
            if child is not None:
                return child
        return first_child(node, self.body_types) if self.body_types else None


def references_for(*types: str) -> frozenset[str]:
    return frozenset(types)


# -- Go -----------------------------------------------------------------------------


def go_define(ctx: FileContext, node: TSNode) -> Definition | None:
    kind = node.type
    if kind in {"type_spec", "type_alias"}:
        name = ctx.field_text(node, "name")
        type_node = node.child_by_field_name("type")
        if not name:
            return None
        bases: list[str] = []
        if type_node is not None and type_node.type == "struct_type":
            for field_decl in descendants(type_node, frozenset({"field_declaration"})):
                if field_decl.child_by_field_name("name") is None:
                    embedded = field_decl.child_by_field_name("type")
                    if embedded is not None:
                        bases.append(strip_generics(ctx.text(embedded)).lstrip("*"))
        elif type_node is not None and type_node.type == "interface_type":
            for elem in type_node.named_children:
                if elem.type in {"type_elem", "constraint_elem", "interface_type_name"}:
                    bases.append(strip_generics(ctx.text(elem)))
        return Definition("class", name, node, bases=tuple(bases))
    if kind not in {"function_declaration", "method_declaration"}:
        return None
    name = ctx.field_text(node, "name")
    body = node.child_by_field_name("body")
    if not name or body is None:
        return None
    if kind == "function_declaration":
        return Definition("function", name, node, (body,))
    receiver = node.child_by_field_name("receiver")
    parameter = first_child(receiver, frozenset({"parameter_declaration"})) if receiver else None
    if parameter is None:
        return Definition("function", name, node, (body,))
    owner = strip_generics(ctx.field_text(parameter, "type") or "").lstrip("*").strip()
    self_name = ctx.field_text(parameter, "name")
    return Definition("function", name, node, (body,), owner=owner or None, self_name=self_name)


# -- Rust -----------------------------------------------------------------------------


def _rust_type_name(text: str) -> str:
    cleaned = strip_generics(text).replace("&", "").replace("mut ", "").replace("dyn ", "")
    parts = path_parts(cleaned.strip())
    return parts[-1] if parts else cleaned.strip()


def rust_define(ctx: FileContext, node: TSNode) -> Definition | None:
    kind = node.type
    if kind in {"struct_item", "enum_item", "union_item", "trait_item"}:
        name = ctx.field_text(node, "name")
        if not name:
            return None
        body = node.child_by_field_name("body")
        bounds = node.child_by_field_name("bounds")
        bases = tuple(type_names(ctx, bounds, _TYPE_REFS)) if bounds is not None else ()
        return Definition("class", name, node, (body,) if body else (), bases=bases)
    if kind == "impl_item":
        type_text = ctx.field_text(node, "type")
        body = node.child_by_field_name("body")
        if not type_text or body is None:
            return None
        trait = ctx.field_text(node, "trait")
        bases = (strip_generics(trait),) if trait else ()
        return Definition("owner", _rust_type_name(type_text), node, (body,), bases=bases)
    if kind == "function_item":
        name = ctx.field_text(node, "name")
        body = node.child_by_field_name("body")
        if not name or body is None:
            return None
        return Definition("function", name, node, (body,))
    if kind == "mod_item":
        name = ctx.field_text(node, "name")
        body = node.child_by_field_name("body")
        if name and body is not None:
            return Definition("namespace", name, node, (body,))
    return None


def rust_references(ctx: FileContext, node: TSNode) -> list[Reference] | None:
    found = generic_references(ctx, node)
    if not found:
        return found
    return [
        Reference(r.name, r.line, r.receiver[1:], r.dynamic)
        if r.receiver[:1] == ("self",) and "self::" in ctx.text(node).split("(", 1)[0]
        else r
        for r in found
    ]


# -- JVM ------------------------------------------------------------------------------

_JAVA_TABLE = Table(
    classes=frozenset(
        {
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
            "annotation_type_declaration",
        }
    ),
    functions=frozenset(
        {"method_declaration", "constructor_declaration", "compact_constructor_declaration"}
    ),
    bases=bases_in(frozenset({"superclass", "super_interfaces", "extends_interfaces"})),
)


def kotlin_define(ctx: FileContext, node: TSNode) -> Definition | None:
    if node.type in {"class_declaration", "object_declaration"}:
        name_node = first_child(node, frozenset({"type_identifier", "simple_identifier"}))
        if name_node is None:
            return None
        body = first_child(node, frozenset({"class_body", "enum_class_body"}))
        bases: list[str] = []
        for spec in node.named_children:
            if spec.type == "delegation_specifier":
                user_type = next(descendants(spec, frozenset({"user_type"})), None)
                if user_type is not None:
                    bases.append(strip_generics(ctx.text(user_type)))
        return Definition(
            "class", ctx.text(name_node), node, (body,) if body else (), bases=tuple(bases)
        )
    if node.type == "function_declaration":
        name_node = first_child(node, frozenset({"simple_identifier"}))
        body = first_child(node, frozenset({"function_body"}))
        if name_node is None or body is None:
            return None
        return Definition("function", ctx.text(name_node), node, (body,))
    return None


_SCALA_TABLE = Table(
    classes=frozenset(
        {"class_definition", "object_definition", "trait_definition", "enum_definition"}
    ),
    functions=frozenset({"function_definition"}),
    bases=bases_in(frozenset({"extends_clause"})),
)


_GROOVY_STATEMENT_END = re.compile(r"[\w)\]\"'}]$")
_GROOVY_CONTINUES = ("{", ".", "?.", "?:", "&&", "||", "+", "-", "*", "/", ")", "]", ":")
_GROOVY_NO_SEMI_START = ("@", "//", "/*", "*", "else", "}", "catch", "finally", "case", "default")


def groovy_semicolons(source: str) -> str:
    """Terminate Groovy statements so the Java grammar can parse them.

    Groovy makes semicolons optional; the Java grammar needs them to separate
    statements. Only line ends are touched, so line numbers are preserved.
    """
    lines = source.split("\n")
    out: list[str] = []
    in_block = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if in_block:
            in_block = "*/" not in stripped
            out.append(line)
            continue
        if stripped.startswith("/*") and "*/" not in stripped:
            in_block = True
            out.append(line)
            continue
        following = next((n.strip() for n in lines[index + 1 :] if n.strip()), "")
        needs = (
            bool(stripped)
            and _GROOVY_STATEMENT_END.search(stripped) is not None
            and not stripped.startswith(_GROOVY_NO_SEMI_START)
            and not following.startswith(_GROOVY_CONTINUES)
            and not stripped.endswith("->")
            and '"""' not in stripped
            and "'''" not in stripped
        )
        out.append(f"{line};" if needs else line)
    return "\n".join(out)


# -- C family -----------------------------------------------------------------------

_DECLARATOR_LEAVES = frozenset(
    {
        "identifier",
        "field_identifier",
        "qualified_identifier",
        "destructor_name",
        "operator_name",
        "type_identifier",
    }
)


def c_declarator_name(ctx: FileContext, node: TSNode | None) -> tuple[str, str | None] | None:
    """(name, owner) from a C/C++ declarator chain; owner for ``Foo::bar`` definitions."""
    cursor = node
    while cursor is not None and cursor.type not in _DECLARATOR_LEAVES:
        inner = cursor.child_by_field_name("declarator")
        if inner is None:
            inner = first_child(cursor, _DECLARATOR_LEAVES)
        cursor = inner
    if cursor is None:
        return None
    if cursor.type == "qualified_identifier":
        parts = path_parts(strip_generics(ctx.text(cursor)).replace("~", ""))
        if not parts:
            return None
        owner = parts[-2] if len(parts) >= 2 and parts[-2][:1].isupper() else None
        return parts[-1], owner
    text = ctx.text(cursor).strip()
    return (text, None) if text else None


_C_RECORDS = frozenset({"struct_specifier", "union_specifier", "enum_specifier", "class_specifier"})


def c_define(ctx: FileContext, node: TSNode) -> Definition | None:
    kind = node.type
    if kind == "function_definition":
        body = node.child_by_field_name("body")
        found = c_declarator_name(ctx, node.child_by_field_name("declarator"))
        if body is None or found is None:
            return None
        name, owner = found
        return Definition("function", name, node, (body,), owner=owner)
    if kind in _C_RECORDS:
        record_name = ctx.field_text(node, "name")
        body = node.child_by_field_name("body")
        if not record_name or body is None:
            return None
        clause = first_child(node, frozenset({"base_class_clause"}))
        bases = tuple(type_names(ctx, clause, _TYPE_REFS)) if clause is not None else ()
        return Definition("class", strip_generics(record_name), node, (body,), bases=bases)
    if kind == "type_definition":
        record = node.child_by_field_name("type")
        declarator = node.child_by_field_name("declarator")
        if record is None or record.type not in _C_RECORDS or declarator is None:
            return None
        body = record.child_by_field_name("body")
        if body is None:
            return None
        return Definition("class", ctx.text(declarator).strip("* "), node, (body,))
    return None


_CPP_HINTS = re.compile(r"^\s*(class|namespace|template\s*<|public:|private:)|\w::\w", re.M)
_OBJC_HINTS = re.compile(r"^\s*(@interface|@implementation|@protocol|#import)", re.M)


def header_grammar(rel_path: str, source: str) -> SupportedLanguage:
    """``.h`` is shared by C, C++ and Objective-C; pick the grammar from the content."""
    if not rel_path.endswith(".h"):
        return "c"
    if _OBJC_HINTS.search(source):
        return "objc"
    if _CPP_HINTS.search(source):
        return "cpp"
    return "c"


# -- C#, Ruby, PHP, Swift, Solidity --------------------------------------------------

_CSHARP_TABLE = Table(
    classes=frozenset(
        {
            "class_declaration",
            "interface_declaration",
            "struct_declaration",
            "record_declaration",
            "record_struct_declaration",
            "enum_declaration",
        }
    ),
    functions=frozenset(
        {"method_declaration", "constructor_declaration", "local_function_statement"}
    ),
    body_types=frozenset({"arrow_expression_clause"}),
    bases=bases_in(frozenset({"base_list"})),
)

_RUBY_TABLE = Table(
    classes=frozenset({"class", "module"}),
    functions=frozenset({"method", "singleton_method"}),
    bases=bases_in(frozenset({"superclass"})),
    needs_body=False,
)

_PHP_TABLE = Table(
    classes=frozenset(
        {"class_declaration", "interface_declaration", "trait_declaration", "enum_declaration"}
    ),
    functions=frozenset({"function_definition", "method_declaration"}),
    bases=bases_in(frozenset({"base_clause", "class_interface_clause"})),
)


def ruby_references(ctx: FileContext, node: TSNode) -> list[Reference] | None:
    if node.type != "call":
        return None
    method = ctx.field_text(node, "method")
    if method in {"include", "extend", "prepend"} and node.child_by_field_name("receiver") is None:
        found: list[Reference] = []
        arguments = node.child_by_field_name("arguments") or node
        for constant in descendants(arguments, frozenset({"constant", "scope_resolution"})):
            parts = path_parts(ctx.text(constant))
            if parts:
                found.append(Reference(parts[-1], _line(node), parts[:-1], kind="inherits"))
        return found
    return generic_references(ctx, node)


def swift_define(ctx: FileContext, node: TSNode) -> Definition | None:
    kind = node.type
    if kind == "class_declaration":
        body = node.child_by_field_name("body")
        bodies = (body,) if body is not None else ()
        if any(child.type == "extension" for child in node.children):
            extended = first_child(node, frozenset({"user_type", "type_identifier"}))
            if extended is None:
                return None
            return Definition("owner", strip_generics(ctx.text(extended)), node, bodies)
        name = ctx.field_text(node, "name")
        if not name:
            return None
        bases = tuple(
            strip_generics(ctx.text(c))
            for c in node.named_children
            if c.type == "inheritance_specifier"
        )
        return Definition("class", name, node, bodies, bases=bases)
    if kind == "protocol_declaration":
        name = ctx.field_text(node, "name")
        body = node.child_by_field_name("body")
        return Definition("class", name, node, (body,) if body else ()) if name else None
    if kind in {"function_declaration", "init_declaration", "deinit_declaration"}:
        body = node.child_by_field_name("body")
        if body is None:
            return None
        default = {"init_declaration": "init", "deinit_declaration": "deinit"}.get(kind)
        name = default or ctx.field_text(node, "name")
        return Definition("function", name, node, (body,)) if name else None
    return None


def solidity_define(ctx: FileContext, node: TSNode) -> Definition | None:
    kind = node.type
    if kind in {"constructor_definition", "fallback_receive_definition"}:
        body = node.child_by_field_name("body")
        if body is None:
            return None
        name = "constructor" if kind == "constructor_definition" else ctx.text(node).split("(")[0]
        return Definition("function", name.strip() or "fallback", node, (body,))
    return _SOLIDITY_TABLE(ctx, node)


_SOLIDITY_TABLE = Table(
    classes=frozenset(
        {
            "contract_declaration",
            "interface_declaration",
            "library_declaration",
            "struct_declaration",
            "enum_declaration",
        }
    ),
    functions=frozenset({"function_definition", "modifier_definition"}),
    bases=bases_in(frozenset({"inheritance_specifier"})),
)


# -- Dart -----------------------------------------------------------------------------

_DART_SIGNATURES = frozenset(
    {
        "function_signature",
        "getter_signature",
        "setter_signature",
        "constructor_signature",
        "factory_constructor_signature",
    }
)


def dart_define(ctx: FileContext, node: TSNode) -> Definition | None:
    kind = node.type
    if kind in {"class_definition", "mixin_declaration", "enum_declaration"}:
        name = name_of(ctx, node)
        body = node.child_by_field_name("body") or first_child(
            node, frozenset({"class_body", "enum_body"})
        )
        if not name:
            return None
        bases: list[str] = []
        for clause in ("superclass", "interfaces"):
            child = node.child_by_field_name(clause)
            if child is not None:
                bases.extend(type_names(ctx, child, frozenset({"type_identifier"})))
        return Definition(
            "class", name, node, (body,) if body else (), bases=tuple(dict.fromkeys(bases))
        )
    if kind == "extension_declaration":
        name = name_of(ctx, node)
        body = node.child_by_field_name("body") or first_child(
            node, frozenset({"extension_body", "class_body"})
        )
        return Definition("class", name, node, (body,) if body else ()) if name else None
    if kind not in {"function_signature", "method_signature"}:
        return None
    signature = node if kind == "function_signature" else first_child(node, _DART_SIGNATURES)
    if signature is None:
        return None
    name = name_of(ctx, signature)
    body = node.next_named_sibling
    if not name or body is None or body.type != "function_body":
        return None
    return Definition(
        "function",
        name,
        node,
        (body,),
        signature=ctx.text(node),
        end_node=body,
        consumed=(body,),
    )


_DART_ACCESSORS = frozenset(
    {"unconditional_assignable_selector", "conditional_assignable_selector"}
)


def dart_references(ctx: FileContext, node: TSNode) -> list[Reference] | None:
    if node.type in {"new_expression", "const_object_expression"}:
        type_node = first_child(node, frozenset({"type_identifier"}))
        return [Reference(ctx.text(type_node), _line(node))] if type_node is not None else []
    kids = node.named_children
    if not any(k.type == "selector" for k in kids):
        return None
    found: list[Reference] = []
    for index, kid in enumerate(kids):
        if index == 0 or kid.type != "selector":
            continue
        if first_child(kid, frozenset({"argument_part"})) is None:
            continue
        previous = kids[index - 1]
        if previous.type == "identifier" and index == 1:
            found.append(Reference(ctx.text(previous), _line(kid)))
            continue
        accessor = first_child(previous, _DART_ACCESSORS) if previous.type == "selector" else None
        name = leaf_name(ctx, accessor) if accessor is not None else None
        if not name:
            continue
        before = kids[: index - 1]
        receiver: list[str] = []
        static = True
        for part in before:
            if part.type in {"identifier", "this"}:
                receiver.append(ctx.text(part))
            elif part.type == "selector" and first_child(part, _DART_ACCESSORS) is not None:
                receiver.append(leaf_name(ctx, part) or "")
            else:
                static = False
        if static and receiver and all(receiver):
            found.append(Reference(name, _line(kid), tuple(receiver)))
        else:
            found.append(Reference(name, _line(kid), dynamic=True))
    return found


# -- Lua, Bash ------------------------------------------------------------------------


def lua_define(ctx: FileContext, node: TSNode) -> Definition | None:
    if node.type == "function_declaration":
        name_node = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if name_node is None:
            return None
        bodies = (body,) if body is not None else ()
        if name_node.type in {"dot_index_expression", "method_index_expression"}:
            table = name_node.child_by_field_name("table")
            member = name_node.child_by_field_name("field") or name_node.child_by_field_name(
                "method"
            )
            if table is None or member is None:
                return None
            return Definition("function", ctx.text(member), node, bodies, owner=ctx.text(table))
        return Definition("function", ctx.text(name_node), node, bodies)
    if node.type in {"variable_declaration", "assignment_statement"}:
        assignment = node if node.type == "assignment_statement" else None
        if assignment is None:
            assignment = first_child(node, frozenset({"assignment_statement"}))
        if assignment is None:
            return None
        variables = first_child(assignment, frozenset({"variable_list"}))
        values = first_child(assignment, frozenset({"expression_list"}))
        if variables is None or values is None or len(values.named_children) != 1:
            return None
        value = values.named_children[0]
        if value.type != "function_definition" or len(variables.named_children) != 1:
            return None
        target = variables.named_children[0]
        body = value.child_by_field_name("body")
        bodies = (body,) if body is not None else ()
        if target.type == "dot_index_expression":
            table = target.child_by_field_name("table")
            member = target.child_by_field_name("field")
            if table is not None and member is not None:
                return Definition("function", ctx.text(member), node, bodies, owner=ctx.text(table))
            return None
        return Definition("function", ctx.text(target), node, bodies)
    return None


BASH_SKIP = frozenset(
    [
        "echo",
        "printf",
        "cd",
        "pwd",
        "exit",
        "return",
        "export",
        "local",
        "declare",
        "readonly",
        "set",
        "unset",
        "shift",
        "test",
        "true",
        "false",
        "read",
        "eval",
        "exec",
        "trap",
        "wait",
        "sleep",
        "source",
        ".",
        "cat",
        "grep",
        "egrep",
        "sed",
        "awk",
        "ls",
        "rm",
        "mkdir",
        "rmdir",
        "cp",
        "mv",
        "ln",
        "chmod",
        "chown",
        "touch",
        "find",
        "xargs",
        "sort",
        "uniq",
        "head",
        "tail",
        "cut",
        "tr",
        "wc",
        "tee",
        "date",
        "env",
        "basename",
        "dirname",
        "realpath",
        "mktemp",
        "curl",
        "wget",
        "tar",
        "gzip",
        "unzip",
        "git",
        "docker",
        "kubectl",
        "make",
        "sudo",
        "ssh",
        "scp",
        "rsync",
        "python",
        "python3",
        "pip",
        "node",
        "npm",
        "npx",
        "yarn",
        "go",
        "cargo",
        "java",
        "jq",
        "yq",
        "kill",
        "ps",
        "command",
        "type",
        "which",
        "getopts",
        "let",
        "shopt",
        "pushd",
        "popd",
        "umask",
        "ulimit",
        "time",
        "[",
        "[[",
        ":",
    ]
)


def bash_references(ctx: FileContext, node: TSNode) -> list[Reference] | None:
    if node.type != "command":
        return None
    name = ctx.field_text(node, "name")
    if not name or name in BASH_SKIP or not re.match(r"^[A-Za-z_][\w:-]*$", name):
        return []
    return [Reference(name.replace("-", "_").replace(":", "_"), _line(node))]


_BASH_TABLE = Table(functions=frozenset({"function_definition"}))


# -- Elixir ---------------------------------------------------------------------------

_ELIXIR_MODULES = frozenset({"defmodule", "defprotocol"})
_ELIXIR_DEFS = frozenset(
    {"def", "defp", "defmacro", "defmacrop", "defguard", "defguardp", "defn", "defnp"}
)
_ELIXIR_SKIP = frozenset(
    [
        "if",
        "unless",
        "case",
        "cond",
        "with",
        "for",
        "fn",
        "quote",
        "unquote",
        "receive",
        "try",
        "raise",
        "reraise",
        "throw",
        "alias",
        "import",
        "require",
        "use",
        "defstruct",
        "defexception",
        "defdelegate",
        "defimpl",
        "defoverridable",
        "super",
        "send",
        "spawn",
        "and",
        "or",
        "not",
        "in",
        "when",
    ]
)


def _elixir_head_name(ctx: FileContext, head: TSNode) -> str | None:
    if head.type == "binary_operator":
        left = head.child_by_field_name("left")
        if left is None:
            return None
        head = left
    if head.type == "call":
        return ctx.field_text(head, "target")
    if head.type == "identifier":
        return ctx.text(head)
    return None


def elixir_define(ctx: FileContext, node: TSNode) -> Definition | None:
    if node.type != "call":
        return None
    target = ctx.field_text(node, "target")
    arguments = first_child(node, frozenset({"arguments"}))
    if arguments is None or not arguments.named_children:
        return None
    body = first_child(node, frozenset({"do_block"}))
    if target in _ELIXIR_MODULES:
        name = ctx.text(arguments.named_children[0]).strip()
        if not name:
            return None
        return Definition("class", name, node, (body,) if body else ())
    if target == "defimpl":
        name = ctx.text(arguments.named_children[0]).strip()
        return Definition("namespace", name, node, (body,) if body else ()) if name else None
    if target not in _ELIXIR_DEFS:
        return None
    function = _elixir_head_name(ctx, arguments.named_children[0])
    if not function:
        return None
    if body is None:
        body = first_child(arguments, frozenset({"keywords"}))
    return Definition("function", function, node, (body,) if body else ())


def elixir_references(ctx: FileContext, node: TSNode) -> list[Reference] | None:
    if node.type != "call":
        return None
    parent = node.parent
    if parent is not None and parent.type == "unary_operator":
        return []
    target = node.child_by_field_name("target")
    if target is None or (target.type == "identifier" and ctx.text(target) in _ELIXIR_SKIP):
        return []
    found = split_callee(ctx, target, _line(node))
    return [found] if found is not None else []


# -- Objective-C ----------------------------------------------------------------------


def objc_define(ctx: FileContext, node: TSNode) -> Definition | None:
    kind = node.type
    if kind in {"class_interface", "class_implementation", "protocol_declaration"}:
        name_node = first_child(node, frozenset({"identifier"}))
        if name_node is None:
            return None
        bases: list[str] = []
        superclass = ctx.field_text(node, "superclass")
        if superclass:
            bases.append(superclass)
        for protocols in node.named_children:
            if protocols.type in {"parameterized_arguments", "protocol_reference_list"}:
                bases.extend(type_names(ctx, protocols, frozenset({"type_name", "identifier"})))
        return Definition("class", ctx.text(name_node), node, (node,), bases=tuple(bases))
    if kind in {"category_interface", "category_implementation"}:
        name_node = first_child(node, frozenset({"identifier"}))
        if name_node is None:
            return None
        return Definition("owner", ctx.text(name_node), node, (node,))
    if kind == "method_definition":
        name_node = first_child(node, frozenset({"identifier"}))
        body = first_child(node, frozenset({"compound_statement"}))
        if name_node is None or body is None:
            return None
        return Definition("function", ctx.text(name_node), node, (body,))
    return c_define(ctx, node)


# -- Zig ------------------------------------------------------------------------------


def zig_define(ctx: FileContext, node: TSNode) -> Definition | None:
    if node.type == "Decl":
        proto = first_child(node, frozenset({"FnProto"}))
        body = first_child(node, frozenset({"Block"}))
        if proto is None or body is None:
            return None
        name = ctx.field_text(proto, "function")
        return Definition("function", name, node, (body,)) if name else None
    if node.type == "VarDecl":
        name = ctx.field_text(node, "variable_type_function")
        container = next(descendants(node, frozenset({"ContainerDecl"})), None)
        if not name or container is None:
            return None
        return Definition("class", name, node, (container,))
    return None


def zig_references(ctx: FileContext, node: TSNode) -> list[Reference] | None:
    if node.type != "SuffixExpr":
        return None
    kids = node.named_children
    if not kids or kids[0].type != "IDENTIFIER":
        return []
    path = [ctx.text(kids[0])]
    dynamic = False
    found: list[Reference] = []
    for kid in kids[1:]:
        if kid.type == "FnCallArguments":
            found.append(Reference(path[-1], _line(kid), tuple(path[:-1]), dynamic))
            dynamic = True
            continue
        if kid.type != "FieldOrFnCall":
            break
        call = kid.child_by_field_name("function_call")
        access = kid.child_by_field_name("field_access")
        if call is not None:
            receiver = () if dynamic else tuple(path)
            found.append(Reference(ctx.text(call), _line(kid), receiver, dynamic))
            dynamic = True
            path.append(ctx.text(call))
        elif access is not None:
            path.append(ctx.text(access))
    return found


# -- Julia, R, Perl, Haskell, Erlang --------------------------------------------------


def _julia_callee(ctx: FileContext, node: TSNode) -> tuple[str, str | None] | None:
    call = node if node.type == "call_expression" else None
    if call is None:
        call = next(descendants(node, frozenset({"call_expression"})), None)
    if call is None or not call.named_children:
        return None
    callee = call.named_children[0]
    parts = path_parts(ctx.text(callee))
    if not parts:
        return None
    return parts[-1], (".".join(parts[:-1]) or None)


def julia_define(ctx: FileContext, node: TSNode) -> Definition | None:
    kind = node.type
    if kind in {"struct_definition", "abstract_definition", "primitive_definition"}:
        head = first_child(node, frozenset({"type_head"}))
        if head is None:
            return None
        inner = head.named_children[0] if head.named_children else head
        bases: tuple[str, ...] = ()
        if inner.type == "binary_expression" and len(inner.named_children) >= 3:
            bases = (strip_generics(ctx.text(inner.named_children[-1])),)
            inner = inner.named_children[0]
        name = strip_generics(ctx.text(inner)).strip()
        return Definition("class", name, node, bases=bases) if name else None
    if kind == "module_definition":
        module = ctx.field_text(node, "name")
        body = first_child(node, frozenset({"block"}))
        return Definition("namespace", module, node, (body,) if body else ()) if module else None
    if kind in {"function_definition", "macro_definition"}:
        signature = first_child(node, frozenset({"signature"}))
        found = _julia_callee(ctx, signature) if signature is not None else None
        body = first_child(node, frozenset({"block"}))
        if found is None:
            return None
        return Definition("function", found[0], node, (body,) if body else ())
    if kind == "assignment" and len(node.named_children) >= 2:
        lhs = node.named_children[0]
        if lhs.type != "call_expression":
            return None
        found = _julia_callee(ctx, lhs)
        if found is None:
            return None
        return Definition("function", found[0], node, (node.named_children[-1],))
    return None


def r_define(ctx: FileContext, node: TSNode) -> Definition | None:
    if node.type == "binary_operator":
        lhs = node.child_by_field_name("lhs")
        rhs = node.child_by_field_name("rhs")
        operator = ctx.field_text(node, "operator") or ""
        if lhs is None or rhs is None or rhs.type != "function_definition":
            return None
        if operator and operator not in {"<-", "=", "<<-"}:
            return None
        body = rhs.child_by_field_name("body")
        name = string_value(ctx, lhs)
        return Definition("function", name, node, (body,) if body else ()) if name else None
    if node.type == "call":
        function = ctx.field_text(node, "function")
        if function not in {"setClass", "setRefClass", "R6Class"}:
            return None
        arguments = node.child_by_field_name("arguments")
        first = next(descendants(arguments, frozenset({"string"})), None) if arguments else None
        if first is None:
            return None
        return Definition("class", string_value(ctx, first), node)
    return None


def r_references(ctx: FileContext, node: TSNode) -> list[Reference] | None:
    if node.type != "call":
        return None
    function = node.child_by_field_name("function")
    if function is not None and function.type == "namespace_operator":
        package = ctx.field_text(function, "lhs")
        name = ctx.field_text(function, "rhs")
        if package and name:
            return [Reference(name, _line(node), (package,), external=True)]
    return generic_references(ctx, node)


def perl_define(ctx: FileContext, node: TSNode) -> Definition | None:
    kind = node.type
    if kind in {"package_statement", "class_statement"}:
        name = ctx.field_text(node, "name")
        if not name:
            return None
        block = first_child(node, frozenset({"block"}))
        dotted = name.replace("::", ".")
        if block is not None:
            return Definition("class", dotted, node, (block,))
        return Definition("class", dotted, node, sequential=True)
    if kind in {"subroutine_declaration_statement", "method_declaration_statement"}:
        name = ctx.field_text(node, "name")
        body = node.child_by_field_name("body")
        if not name or body is None:
            return None
        return Definition("function", name, node, (body,))
    return None


def perl_references(ctx: FileContext, node: TSNode) -> list[Reference] | None:
    if node.type == "use_statement" and ctx.field_text(node, "module") in {"parent", "base"}:
        found: list[Reference] = []
        for literal in descendants(node, frozenset({"string_literal", "quoted_word_list"})):
            for word in string_value(ctx, literal).removeprefix("qw").strip("()[]{}/ ").split():
                parts = path_parts(word)
                if parts:
                    found.append(Reference(parts[-1], _line(node), parts[:-1], kind="inherits"))
        return found
    return generic_references(ctx, node)


def haskell_define(ctx: FileContext, node: TSNode) -> Definition | None:
    kind = node.type
    if kind in {"data_type", "newtype", "type_synomym", "type_family"}:
        name = ctx.field_text(node, "name")
        return Definition("class", name, node) if name else None
    if kind == "class":
        name = ctx.field_text(node, "name")
        body = node.child_by_field_name("declarations")
        return Definition("class", name, node, (body,) if body else ()) if name else None
    if kind == "instance":
        names = [ctx.text(c) for c in descendants(node, frozenset({"name"}))]
        body = node.child_by_field_name("declarations")
        if len(names) < 2 or body is None:
            return None
        return Definition("owner", names[1], node, (body,), bases=(names[0],))
    if kind in {"function", "bind"}:
        name = ctx.field_text(node, "name")
        if not name or node.child_by_field_name("match") is None and kind == "bind":
            return None
        previous = node.prev_named_sibling
        typed = previous is not None and previous.type == "signature"
        if typed and previous is not None and ctx.field_text(previous, "name") == name:
            return Definition(
                "function", name, node, (node,), anchor=previous, signature=ctx.text(previous)
            )
        return Definition("function", name, node, (node,))
    return None


def erlang_define(ctx: FileContext, node: TSNode) -> Definition | None:
    if node.type != "fun_decl":
        return None
    clause = first_child(node, frozenset({"function_clause"}))
    name = ctx.field_text(clause, "name") if clause is not None else None
    return Definition("function", name, node, (node,)) if name else None


# -- the table ------------------------------------------------------------------------

C_STYLE = CommentStyle(("///", "//!", "//"), ("/*", "*/"), ("@", "#["))
ELIXIR_COMMENTS = CommentStyle(("#", "@doc"), None, ("@spec", "@impl"))
LUA_COMMENTS = CommentStyle(("--",), ("--[[", "]]"))
JULIA_COMMENTS = CommentStyle(("#", '"""'))
JAVA_EXTERNAL = frozenset(
    {
        "System",
        "Math",
        "String",
        "Integer",
        "Long",
        "Double",
        "Boolean",
        "Objects",
        "Arrays",
        "Collections",
        "List",
        "Map",
        "Set",
        "Optional",
        "Thread",
        "Stream",
        "Collectors",
        "LocalDate",
        "LocalDateTime",
        "Instant",
        "Duration",
        "UUID",
        "Files",
        "Paths",
        "Executors",
        "CompletableFuture",
        "Logger",
        "LoggerFactory",
    }
)


def _types(*names: str) -> frozenset[str]:
    return frozenset(names)


_C_CALLS = _types("call_expression", "new_expression")
_C_INCLUDES = _types("preproc_include")

SPECS: dict[Language, LanguageSpec] = {
    "go": LanguageSpec(
        "go",
        "go",
        go_define,
        imports=imp.go_imports,
        import_types=_types("import_declaration"),
        call_types=_types("call_expression"),
        self_names=frozenset(),
    ),
    "rust": LanguageSpec(
        "rust",
        "rust",
        rust_define,
        references=rust_references,
        imports=imp.rust_imports,
        import_types=_types("use_declaration", "mod_item", "extern_crate_declaration"),
        call_types=_types("call_expression"),
        self_names=frozenset({"self", "Self"}),
        external_roots=frozenset({"std", "core", "alloc"}),
        prelude=imp.rust_prelude,
    ),
    "java": LanguageSpec(
        "java",
        "java",
        _JAVA_TABLE,
        imports=imp.java_imports,
        import_types=_types("import_declaration"),
        call_types=_types("method_invocation", "object_creation_expression"),
        self_names=frozenset({"this"}),
        external_roots=JAVA_EXTERNAL,
    ),
    "groovy": LanguageSpec(
        "groovy",
        "java",
        _JAVA_TABLE,
        imports=imp.java_imports,
        import_types=_types("import_declaration"),
        call_types=_types("method_invocation", "object_creation_expression"),
        self_names=frozenset({"this"}),
        external_roots=JAVA_EXTERNAL,
        preprocess=groovy_semicolons,
    ),
    "kotlin": LanguageSpec(
        "kotlin",
        "kotlin",
        kotlin_define,
        imports=imp.kotlin_imports,
        import_types=_types("import_header"),
        call_types=_types("call_expression"),
        self_names=frozenset({"this"}),
        external_roots=JAVA_EXTERNAL | {"println", "listOf", "mapOf"},
    ),
    "scala": LanguageSpec(
        "scala",
        "scala",
        _SCALA_TABLE,
        imports=imp.scala_imports,
        import_types=_types("import_declaration"),
        call_types=_types("call_expression", "instance_expression"),
        self_names=frozenset({"this"}),
        external_roots=frozenset({"Future", "Option", "Some", "Seq", "Await", "Try", "Console"})
        | JAVA_EXTERNAL,
    ),
    "c": LanguageSpec(
        "c",
        "c",
        objc_define,
        imports=imp.include_imports,
        import_types=_C_INCLUDES,
        call_types=_C_CALLS | {"message_expression"},
        self_names=frozenset({"this", "self"}),
        external_roots=frozenset({"std"}),
        grammar_for=header_grammar,
    ),
    "cpp": LanguageSpec(
        "cpp",
        "cpp",
        c_define,
        imports=imp.include_imports,
        import_types=_C_INCLUDES,
        call_types=_C_CALLS,
        self_names=frozenset({"this"}),
        external_roots=frozenset({"std", "boost"}),
    ),
    "objc": LanguageSpec(
        "objc",
        "objc",
        objc_define,
        imports=imp.include_imports,
        import_types=_C_INCLUDES,
        call_types=_types("message_expression", "call_expression"),
        self_names=frozenset({"self"}),
        external_roots=frozenset({"NSString", "NSArray", "NSDictionary", "NSNotificationCenter"}),
    ),
    "csharp": LanguageSpec(
        "csharp",
        "csharp",
        _CSHARP_TABLE,
        imports=imp.csharp_imports,
        import_types=_types("using_directive"),
        call_types=_types("invocation_expression", "object_creation_expression"),
        self_names=frozenset({"this"}),
        external_roots=frozenset(
            {"Console", "Math", "String", "Task", "Enumerable", "Convert", "Guid", "File"}
        ),
    ),
    "ruby": LanguageSpec(
        "ruby",
        "ruby",
        _RUBY_TABLE,
        references=ruby_references,
        imports=imp.ruby_imports,
        import_types=_types("call"),
        call_types=_types("call"),
        comments=HASH_COMMENTS,
        self_names=frozenset({"self"}),
        external_roots=frozenset({"File", "Dir", "JSON", "Time", "Rails", "YAML", "Logger"}),
    ),
    "php": LanguageSpec(
        "php",
        "php",
        _PHP_TABLE,
        imports=imp.php_imports,
        import_types=_types("namespace_use_declaration"),
        call_types=_types(
            "function_call_expression",
            "member_call_expression",
            "scoped_call_expression",
            "nullsafe_member_call_expression",
            "object_creation_expression",
        ),
        self_names=frozenset({"this", "self", "static"}),
    ),
    "swift": LanguageSpec(
        "swift",
        "swift",
        swift_define,
        imports=imp.swift_imports,
        import_types=_types("import_declaration"),
        call_types=_types("call_expression"),
        self_names=frozenset({"self", "Self"}),
        external_roots=frozenset({"DispatchQueue", "NotificationCenter", "UserDefaults"}),
    ),
    "dart": LanguageSpec(
        "dart",
        "dart",
        dart_define,
        references=dart_references,
        imports=imp.dart_imports,
        import_types=_types("import_or_export"),
        self_names=frozenset({"this"}),
        external_roots=frozenset({"Future", "Stream", "Navigator", "Theme", "DateTime"}),
    ),
    "lua": LanguageSpec(
        "lua",
        "lua",
        lua_define,
        imports=imp.lua_imports,
        import_types=_types("variable_declaration", "assignment_statement", "function_call"),
        call_types=_types("function_call"),
        comments=LUA_COMMENTS,
        self_names=frozenset({"self"}),
        external_roots=frozenset({"string", "table", "math", "os", "io", "coroutine"}),
    ),
    "bash": LanguageSpec(
        "bash",
        "bash",
        _BASH_TABLE,
        references=bash_references,
        imports=imp.bash_imports,
        import_types=_types("command"),
        comments=HASH_COMMENTS,
        self_names=frozenset(),
    ),
    "elixir": LanguageSpec(
        "elixir",
        "elixir",
        elixir_define,
        references=elixir_references,
        imports=imp.elixir_imports,
        import_types=_types("call"),
        call_types=_types("call"),
        comments=ELIXIR_COMMENTS,
        self_names=frozenset({"__MODULE__"}),
        external_roots=frozenset(
            {"Enum", "Map", "String", "List", "Kernel", "IO", "Logger", "GenServer", "Keyword"}
        ),
    ),
    "solidity": LanguageSpec(
        "solidity",
        "solidity",
        solidity_define,
        imports=imp.solidity_imports,
        import_types=_types("import_directive"),
        call_types=_types("call_expression"),
        self_names=frozenset({"this"}),
        external_roots=frozenset({"abi", "msg", "block", "tx"}),
    ),
    "zig": LanguageSpec(
        "zig",
        "zig",
        zig_define,
        references=zig_references,
        imports=imp.zig_imports,
        import_types=_types("VarDecl"),
        self_names=frozenset({"self"}),
    ),
    "julia": LanguageSpec(
        "julia",
        "julia",
        julia_define,
        imports=imp.julia_imports,
        import_types=_types("using_statement", "import_statement", "call_expression"),
        call_types=_types("call_expression", "broadcast_call_expression"),
        comments=JULIA_COMMENTS,
        self_names=frozenset(),
        external_roots=frozenset({"Base", "Core"}),
    ),
    "r": LanguageSpec(
        "r",
        "r",
        r_define,
        references=r_references,
        imports=imp.r_imports,
        import_types=_types("call"),
        call_types=_types("call"),
        comments=HASH_COMMENTS,
        self_names=frozenset(),
    ),
    "perl": LanguageSpec(
        "perl",
        "perl",
        perl_define,
        references=perl_references,
        imports=imp.perl_imports,
        import_types=_types("use_statement", "require_expression"),
        call_types=_types(
            "function_call_expression",
            "method_call_expression",
            "ambiguous_function_call_expression",
        ),
        comments=HASH_COMMENTS,
        self_names=frozenset({"self", "class"}),
    ),
    "haskell": LanguageSpec(
        "haskell",
        "haskell",
        haskell_define,
        imports=imp.haskell_imports,
        import_types=_types("import"),
        call_types=_types("apply"),
        comments=DASH_COMMENTS,
        self_names=frozenset(),
    ),
    "erlang": LanguageSpec(
        "erlang",
        "erlang",
        erlang_define,
        imports=imp.erlang_imports,
        import_types=_types("import_attribute", "pp_include", "pp_include_lib"),
        call_types=_types("call"),
        comments=PERCENT_COMMENTS,
        self_names=frozenset(),
        external_roots=frozenset(
            {"lists", "io", "maps", "gen_server", "ets", "string", "erlang", "proplists"}
        ),
    ),
}


__all__ = [
    "SPECS",
    "Table",
    "bases_in",
    "c_declarator_name",
    "first_of",
    "groovy_semicolons",
    "header_grammar",
]
