"""Import hooks for the tree-sitter languages.

Each hook turns one import-like node into :class:`Import` records. The target is the
importee's module qname when the path alone determines it (relative includes, Rust
``mod`` and ``crate::``, packages laid out under a source root), otherwise the
specifier as written. ``exact=False`` marks a qname that is a guess about the
repository layout; names bound by such an import produce unresolved edges that keep
the guess in ``detail`` for :mod:`codegraph.parsers.resolve`.

"Is this import repo-local?" is answered cheaply: an import that shares its root
namespace with the importing file's own package (``com.acme.*`` inside ``com.acme.app``)
is local, a well-known standard-library root is external, and anything else is
treated as external, which is the safe direction (the edge is still recorded).
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from tree_sitter import Node as TSNode

from codegraph.parsers.paths import package_root, relative_module, under
from codegraph.parsers.treesitter import (
    Alias,
    FileContext,
    Import,
    descendants,
    first_child,
    path_parts,
    string_value,
)

_STRINGS = frozenset(
    {
        "string",
        "string_literal",
        "interpreted_string_literal",
        "raw_string_literal",
        "encapsed_string",
        "quoted_word_list",
        "STRINGLITERALSINGLE",
        "string_lit",
    }
)


def _line(node: TSNode) -> int:
    return node.start_point[0] + 1


def _first_string(ctx: FileContext, node: TSNode) -> str | None:
    found = next(descendants(node, _STRINGS), None)
    return string_value(ctx, found) if found is not None else None


def _dotted(text: str) -> str:
    parts = path_parts(text)
    return ".".join(parts) if parts else text.strip()


# -- Go -----------------------------------------------------------------------------

_GO_VERSION = re.compile(r"^v\d+$")


def go_package_name(path: str) -> str:
    """The identifier an import path binds by default (``gopkg.in/yaml.v3`` -> ``yaml``)."""
    segments = [s for s in path.split("/") if s]
    if not segments:
        return path
    last = segments[-1]
    if _GO_VERSION.match(last) and len(segments) > 1:
        last = segments[-2]
    last = last.split(".")[0].removeprefix("go-").removesuffix("-go")
    return last.replace("-", "_")


def go_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    found: list[Import] = []
    specs = (
        [node]
        if node.type == "import_spec"
        else list(descendants(node, frozenset({"import_spec"})))
    )
    for spec in specs:
        path_node = spec.child_by_field_name("path")
        if path_node is None:
            continue
        path = string_value(ctx, path_node)
        alias = ctx.field_text(spec, "name")
        local = alias or go_package_name(path)
        stdlib = "." not in path.split("/")[0]
        bindings = () if local in {"_", "."} else ((local, path),)
        found.append(Import(path, path, _line(spec), bindings, exact=stdlib))
    return found


# -- Rust ---------------------------------------------------------------------------


def rust_module_path(ctx: FileContext) -> list[str]:
    """Module path of the file itself; ``lib.rs``/``main.rs`` are the crate root."""
    parts = ctx.module_qname.split(".")
    if PurePosixPath(ctx.rel_path).stem in {"lib", "main"}:
        parts = parts[:-1]
    return parts


def rust_crate_root(ctx: FileContext) -> list[str]:
    dirs = list(PurePosixPath(ctx.rel_path).parent.parts)
    if "src" in dirs:
        cut = len(dirs) - 1 - dirs[::-1].index("src")
        return dirs[: cut + 1]
    return dirs


def rust_prelude(ctx: FileContext) -> dict[str, Alias]:
    here = rust_module_path(ctx)
    aliases = {"crate": Alias(".".join(rust_crate_root(ctx)) or ctx.module_qname, exact=False)}
    if len(here) > 1:
        aliases["super"] = Alias(".".join(here[:-1]), exact=False)
    return aliases


def _rust_use_paths(ctx: FileContext, node: TSNode, prefix: list[str]) -> list[list[str]]:
    """Expand a use tree into full paths; an ``as`` alias is appended as ``as:<name>``."""
    kind = node.type
    if kind in {"identifier", "crate", "self", "super", "metavariable"}:
        return [[*prefix, ctx.text(node)]]
    if kind == "scoped_identifier":
        path = node.child_by_field_name("path")
        name = node.child_by_field_name("name")
        head = _rust_use_paths(ctx, path, prefix)[0] if path is not None else prefix
        return [[*head, ctx.text(name)]] if name is not None else [head]
    if kind == "use_as_clause":
        path = node.child_by_field_name("path")
        alias = node.child_by_field_name("alias")
        if path is None:
            return []
        return [[*p, f"as:{ctx.text(alias)}"] for p in _rust_use_paths(ctx, path, prefix)]
    if kind == "use_wildcard":
        inner = node.named_children[0] if node.named_children else None
        head = _rust_use_paths(ctx, inner, prefix)[0] if inner is not None else prefix
        return [[*head, "*"]]
    if kind == "scoped_use_list":
        path = node.child_by_field_name("path")
        listing = node.child_by_field_name("list")
        head = _rust_use_paths(ctx, path, prefix)[0] if path is not None else prefix
        return _rust_use_paths(ctx, listing, head) if listing is not None else []
    if kind == "use_list":
        paths: list[list[str]] = []
        for child in node.named_children:
            paths.extend(_rust_use_paths(ctx, child, prefix))
        return paths
    return []


def _rust_target(ctx: FileContext, segments: list[str]) -> tuple[str, bool]:
    head, rest = segments[0], segments[1:]
    here = rust_module_path(ctx)
    if head == "crate":
        return ".".join([*rust_crate_root(ctx), *rest]), False
    if head == "self":
        return ".".join([*here, *rest]), False
    if head == "super":
        base = here[:-1]
        while rest and rest[0] == "super":
            base, rest = base[:-1], rest[1:]
        return ".".join([*base, *rest]), False
    return ".".join(segments), True


def rust_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    line = _line(node)
    if node.type == "mod_item":
        if node.child_by_field_name("body") is not None:
            return None
        name = ctx.field_text(node, "name")
        if not name:
            return []
        target = ".".join([*rust_module_path(ctx), name])
        return [Import(f"mod {name}", target, line, ((name, target),), exact=True)]
    if node.type == "extern_crate_declaration":
        name = ctx.field_text(node, "name") or ""
        alias = ctx.field_text(node, "alias") or name
        return [Import(f"extern crate {name}", name, line, ((alias, name),), exact=True)]
    argument = node.child_by_field_name("argument")
    if argument is None:
        return []
    found: list[Import] = []
    for path in _rust_use_paths(ctx, argument, []):
        rename = path.pop()[3:] if path and path[-1].startswith("as:") else None
        if path and path[-1] == "self":
            path.pop()
        if not path:
            continue
        spec = "::".join(path)
        if path[-1] == "*":
            target, exact = _rust_target(ctx, path[:-1] or path)
            found.append(Import(spec, target, line, (), exact))
            continue
        target, exact = _rust_target(ctx, path)
        local = rename or path[-1]
        bindings = ((local, target),) if local not in {"crate", "self", "super", "_"} else ()
        found.append(Import(spec, target, line, bindings, exact))
    return found


# -- JVM languages (Java, Kotlin, Scala, Groovy) -------------------------------------

JVM_EXTERNAL_ROOTS = frozenset(
    {"java", "javax", "jakarta", "jdk", "sun", "kotlin", "kotlinx", "scala", "android", "androidx"}
)
_PACKAGE_TYPES = frozenset({"package_declaration", "package_header", "package_clause"})


def jvm_package(ctx: FileContext) -> str:
    cached = ctx.memo.get("package")
    if cached is not None:
        return cached
    package = ""
    if ctx.root is not None:
        declaration = first_child(ctx.root, _PACKAGE_TYPES)
        if declaration is not None:
            inner = declaration.child_by_field_name("name") or (
                declaration.named_children[0] if declaration.named_children else None
            )
            package = _dotted(ctx.text(inner)) if inner is not None else ""
    ctx.memo["package"] = package
    return package


def _shares_root(package: str, dotted: str, depth: int = 2) -> bool:
    own = package.split(".")
    other = dotted.split(".")
    keep = min(depth, len(own))
    return keep > 0 and own[:keep] == other[:keep]


def jvm_reference(ctx: FileContext, dotted: str, *, member: bool) -> tuple[str, str, bool]:
    """(module qname, bound qname, exact) for ``import a.b.Type`` / ``import static a.b.T.m``.

    The type lives in ``a/b/Type.<ext>`` under the source root inferred from the file's
    own package; a static member hangs off the type.
    """
    segments = dotted.split(".")
    package = jvm_package(ctx)
    external = segments[0] in JVM_EXTERNAL_ROOTS or (
        bool(package) and not _shares_root(package, dotted)
    )
    if external:
        return dotted, dotted, True
    type_path = segments[:-1] if member else segments
    root = package_root(ctx.rel_path, package) if package else None
    module = under(root, ".".join(type_path))
    bound = f"{module}.{type_path[-1]}" if type_path else module
    if member:
        bound = f"{bound}.{segments[-1]}"
    return module, bound, False


def java_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    text = ctx.text(node)
    target_node = first_child(node, frozenset({"scoped_identifier", "identifier"}))
    if target_node is None:
        return []
    dotted = _dotted(ctx.text(target_node))
    line = _line(node)
    is_static = " static " in f" {text} "
    if first_child(node, frozenset({"asterisk"})) is not None or text.rstrip("; ").endswith("*"):
        module, _, exact = jvm_reference(ctx, f"{dotted}.x", member=False)
        return [Import(f"{dotted}.*", module.rsplit(".", 1)[0], line, (), exact)]
    module, bound, exact = jvm_reference(ctx, dotted, member=is_static)
    return [Import(dotted, module, line, ((dotted.rsplit(".", 1)[-1], bound),), exact)]


def kotlin_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    identifier = first_child(node, frozenset({"identifier"}))
    if identifier is None:
        return []
    dotted = _dotted(ctx.text(identifier))
    line = _line(node)
    if ctx.text(node).rstrip().endswith("*"):
        module, _, exact = jvm_reference(ctx, f"{dotted}.x", member=False)
        return [Import(f"{dotted}.*", module.rsplit(".", 1)[0], line, (), exact)]
    alias_node = first_child(node, frozenset({"import_alias"}))
    alias = ctx.text(alias_node.named_children[-1]) if alias_node is not None else None
    last = dotted.rsplit(".", 1)[-1]
    member = bool(last) and last[0].islower()
    module, bound, exact = jvm_reference(ctx, dotted, member=member)
    if member and not exact:
        bound = f"{module}.{last}"
    return [Import(dotted, module, line, ((alias or last, bound),), exact)]


def scala_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    base = [ctx.text(c) for c in node.children_by_field_name("path") if c.is_named]
    if not base:
        return []
    line = _line(node)
    selectors = first_child(node, frozenset({"namespace_selectors"}))
    names: list[tuple[str, str]] = []
    if selectors is None:
        if first_child(node, frozenset({"namespace_wildcard"})) is not None:
            prefix = ".".join(base)
            module, _, exact = jvm_reference(ctx, f"{prefix}.x", member=False)
            return [Import(f"{prefix}._", module.rsplit(".", 1)[0], line, (), exact)]
        names = [(base[-1], base[-1])]
        base = base[:-1]
    else:
        for child in selectors.named_children:
            if child.type == "identifier":
                names.append((ctx.text(child), ctx.text(child)))
            elif child.type == "arrow_renamed_identifier":
                original = ctx.field_text(child, "name") or ""
                names.append((original, ctx.field_text(child, "alias") or original))
    found: list[Import] = []
    for original, local in names:
        dotted = ".".join([*base, original])
        module, bound, exact = jvm_reference(ctx, dotted, member=original[:1].islower())
        found.append(Import(dotted, module, line, ((local, bound),), exact))
    return found


# -- C family -----------------------------------------------------------------------


def include_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    path = node.child_by_field_name("path")
    if path is None:
        return []
    text = ctx.text(path).strip()
    line = _line(node)
    if text.startswith("<"):
        spec = text.strip("<>")
        return [Import(spec, spec, line)]
    spec = string_value(ctx, path)
    target = relative_module(ctx.rel_path, spec) or spec
    return [Import(spec, target, line)]


_CSHARP_EXTERNAL = frozenset({"System", "Microsoft", "Newtonsoft", "Xunit", "NUnit", "Moq"})


def csharp_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    alias = ctx.field_text(node, "name")
    target_node = next(
        (
            c
            for c in node.named_children
            if c.type in {"qualified_name", "identifier", "generic_name"}
            and (alias is None or ctx.text(c) != alias)
        ),
        None,
    )
    if target_node is None:
        return []
    dotted = _dotted(ctx.text(target_node))
    exact = dotted.split(".")[0] in _CSHARP_EXTERNAL
    bindings = ((alias, dotted),) if alias else ()
    return [Import(dotted, dotted, _line(node), bindings, exact)]


# -- Ruby, PHP, Swift, Dart ---------------------------------------------------------


def ruby_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    method = ctx.field_text(node, "method")
    if method not in {"require", "require_relative", "load"} or node.child_by_field_name(
        "receiver"
    ):
        return None
    spec = _first_string(ctx, node)
    if spec is None:
        return []
    line = _line(node)
    if method == "require_relative" or spec.startswith("."):
        target = relative_module(ctx.rel_path, spec, ".rb") or spec
        return [Import(spec, target, line)]
    return [Import(spec, spec.replace("/", "."), line)]


def php_namespace(ctx: FileContext) -> str:
    cached = ctx.memo.get("namespace")
    if cached is not None:
        return cached
    namespace = ""
    if ctx.root is not None:
        declaration = first_child(ctx.root, frozenset({"namespace_definition"}))
        if declaration is not None:
            name = ctx.field_text(declaration, "name") or ""
            namespace = _dotted(name.replace("\\", "."))
    ctx.memo["namespace"] = namespace
    return namespace


def php_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    prefix_node = first_child(node, frozenset({"namespace_name"}))
    prefix = _dotted(ctx.text(prefix_node).replace("\\", ".")) if prefix_node else ""
    own_root = php_namespace(ctx).split(".")[0]
    found: list[Import] = []
    for clause in descendants(node, frozenset({"namespace_use_clause"})):
        alias = ctx.field_text(clause, "alias")
        target_node = first_child(clause, frozenset({"qualified_name", "name"}))
        if target_node is None:
            continue
        dotted = _dotted(ctx.text(target_node).replace("\\", ".").lstrip("."))
        if prefix:
            dotted = f"{prefix}.{dotted}"
        last = dotted.rsplit(".", 1)[-1]
        local = bool(own_root) and dotted.split(".")[0] == own_root
        bound = f"{dotted}.{last}" if local else dotted
        found.append(Import(dotted, dotted, _line(node), ((alias or last, bound),), not local))
    return found


def swift_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    identifier = first_child(node, frozenset({"identifier"}))
    if identifier is None:
        return []
    name = _dotted(ctx.text(identifier))
    return [Import(name, name, _line(node))]


def dart_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    uri = _first_string(ctx, node)
    if uri is None:
        return []
    line = _line(node)
    specification = next(descendants(node, frozenset({"import_specification"})), None)
    prefix = None
    if specification is not None:
        identifier = first_child(specification, frozenset({"identifier"}))
        prefix = ctx.text(identifier) if identifier is not None else None
    if uri.startswith("dart:"):
        target, exact = uri, True
    elif uri.startswith("package:"):
        rest = uri.removeprefix("package:").split("/", 1)
        path = f"lib/{rest[1]}" if len(rest) == 2 else rest[0]
        target, exact = relative_module("", path) or uri, False
    else:
        target, exact = relative_module(ctx.rel_path, uri) or uri, True
    bindings = ((prefix, target),) if prefix else ()
    return [Import(uri, target, line, bindings, exact)]


# -- Lua, Bash, Zig, R, Julia --------------------------------------------------------


def _lua_require(ctx: FileContext, node: TSNode) -> str | None:
    if node.type != "function_call" or ctx.field_text(node, "name") != "require":
        return None
    return _first_string(ctx, node)


def lua_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    line = _line(node)
    spec = _lua_require(ctx, node)
    if spec is not None:
        return [Import(spec, spec.replace("/", "."), line, (), exact=False)]
    assignment = node if node.type == "assignment_statement" else None
    if assignment is None:
        assignment = first_child(node, frozenset({"assignment_statement"}))
    if assignment is None:
        return None
    variables = first_child(assignment, frozenset({"variable_list"}))
    values = first_child(assignment, frozenset({"expression_list"}))
    if variables is None or values is None or len(values.named_children) != 1:
        return None
    spec = _lua_require(ctx, values.named_children[0])
    if spec is None:
        return None
    target = spec.replace("/", ".")
    local = ctx.text(variables).strip()
    return [Import(spec, target, line, ((local, target),), exact=False)]


# Expansions that mean "the directory of this script": $(dirname "$0"),
# ${BASH_SOURCE%/*}, $SCRIPT_DIR, ${DIR}. Other variables ($HOME) are not relative.
_SCRIPT_DIR = re.compile(
    r"^(\$\(dirname[^)]*\)|\$\{BASH_SOURCE[^}]*\}|\$\{?\w*(DIR|[Hh]ere)\w*\}?)/"
)


def bash_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    name = ctx.field_text(node, "name")
    if name not in {"source", "."}:
        return None
    argument = node.child_by_field_name("argument")
    if argument is None:
        return []
    spec = _SCRIPT_DIR.sub("", string_value(ctx, argument))
    if "$" in spec or not spec:
        return []
    target = relative_module(ctx.rel_path, spec) or spec
    return [Import(spec, target, _line(node))]


def zig_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    builtin = next(descendants(node, frozenset({"BUILTINIDENTIFIER"})), None)
    if builtin is None or ctx.text(builtin) != "@import":
        return None
    spec = _first_string(ctx, node)
    local = ctx.field_text(node, "variable_type_function")
    if spec is None or local is None:
        return None
    target = relative_module(ctx.rel_path, spec) or spec if spec.endswith(".zig") else spec
    return [Import(spec, target, _line(node), ((local, target),))]


def r_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    function = ctx.field_text(node, "function")
    if function not in {"library", "require", "requireNamespace", "source", "sys.source"}:
        return None
    arguments = node.child_by_field_name("arguments")
    argument = next(descendants(arguments, frozenset({"argument"})), None) if arguments else None
    if argument is None:
        return []
    value = argument.child_by_field_name("value") or argument
    spec = string_value(ctx, value)
    line = _line(node)
    if function in {"source", "sys.source"}:
        return [Import(spec, relative_module(ctx.rel_path, spec) or spec, line)]
    return [Import(spec, spec, line)]


def julia_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    line = _line(node)
    if node.type == "call_expression":
        callee = node.named_children[0] if node.named_children else None
        if callee is None or ctx.text(callee) != "include":
            return None
        spec = _first_string(ctx, node)
        if spec is None:
            return []
        return [Import(spec, relative_module(ctx.rel_path, spec) or spec, line)]
    found: list[Import] = []
    for child in node.named_children:
        if child.type == "selected_import":
            names = [ctx.text(c) for c in child.named_children]
            if not names:
                continue
            module = names[0]
            bindings = tuple((n, f"{module}.{n}") for n in names[1:])
            found.append(Import(module, module, line, bindings))
        else:
            module = _dotted(ctx.text(child)).lstrip(".")
            if module:
                found.append(Import(module, module, line))
    return found


# -- Elixir, Perl, Haskell, Erlang, Solidity -----------------------------------------


def _elixir_root(ctx: FileContext) -> str:
    cached = ctx.memo.get("elixir_root")
    if cached is not None:
        return cached
    root = ""
    if ctx.root is not None:
        for call in descendants(ctx.root, frozenset({"call"})):
            if ctx.field_text(call, "target") == "defmodule":
                alias = next(descendants(call, frozenset({"alias"})), None)
                root = ctx.text(alias).split(".")[0] if alias is not None else ""
                break
    ctx.memo["elixir_root"] = root
    return root


def elixir_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    directive = ctx.field_text(node, "target")
    if directive not in {"alias", "import", "require", "use"}:
        return None
    arguments = first_child(node, frozenset({"arguments"}))
    if arguments is None or not arguments.named_children:
        return []
    module = ctx.text(arguments.named_children[0]).strip()
    if not module or not module[0].isupper():
        return []
    local = module.rsplit(".", 1)[-1]
    for pair in descendants(arguments, frozenset({"pair"})):
        if ctx.text(pair.child_by_field_name("key")).strip().rstrip(":") == "as":
            local = ctx.text(pair.child_by_field_name("value")).strip()
    exact = module.split(".")[0] != _elixir_root(ctx)
    bindings = ((local, module),) if directive == "alias" else ()
    return [Import(module, module, _line(node), bindings, exact)]


PERL_PRAGMAS = frozenset(
    {
        "strict",
        "warnings",
        "utf8",
        "lib",
        "constant",
        "feature",
        "vars",
        "overload",
        "integer",
        "bytes",
        "open",
        "version",
        "Exporter",
        "parent",
        "base",
    }
)


def _perl_package(ctx: FileContext) -> str:
    cached = ctx.memo.get("package")
    if cached is not None:
        return cached
    package = ""
    if ctx.root is not None:
        statement = first_child(ctx.root, frozenset({"package_statement"}))
        if statement is not None:
            package = ctx.field_text(statement, "name") or ""
    ctx.memo["package"] = package
    return package


def perl_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    line = _line(node)
    if node.type == "require_expression":
        bareword = first_child(node, frozenset({"bareword", "package"}))
        if bareword is None:
            return []
        dotted = ctx.text(bareword).replace("::", ".")
        return [Import(ctx.text(bareword), dotted, line)]
    module = ctx.field_text(node, "module")
    if not module or module in PERL_PRAGMAS or module[0].islower():
        return None if module in {"parent", "base"} else []
    dotted = module.replace("::", ".")
    own_root = _perl_package(ctx).split("::")[0]
    exact = not own_root or module.split("::")[0] != own_root
    names: list[str] = []
    for words in descendants(node, frozenset({"quoted_word_list"})):
        names.extend(string_value(ctx, words).removeprefix("qw").strip("()[]{}/ ").split())
    bindings = tuple((n, f"{dotted}.{n}") for n in names if n.isidentifier())
    return [Import(module, dotted, line, bindings, exact)]


def _haskell_header(ctx: FileContext) -> str:
    cached = ctx.memo.get("module")
    if cached is not None:
        return cached
    name = ""
    if ctx.root is not None:
        header = first_child(ctx.root, frozenset({"header"}))
        if header is not None:
            name = ctx.field_text(header, "module") or ""
    ctx.memo["module"] = name
    return name


def haskell_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    module = ctx.field_text(node, "module")
    if not module:
        return []
    own = _haskell_header(ctx)
    local = bool(own) and module.split(".")[0] == own.split(".")[0]
    target = module
    if local:
        package = own.rsplit(".", 1)[0] if "." in own else ""
        target = under(package_root(ctx.rel_path, package), module)
    bindings: list[tuple[str, str]] = []
    alias = ctx.field_text(node, "alias")
    if alias:
        bindings.append((alias, target))
    names = node.child_by_field_name("names")
    if names is not None:
        for item in descendants(names, frozenset({"variable", "name", "constructor"})):
            bindings.append((ctx.text(item), f"{target}.{ctx.text(item)}"))
    return [Import(module, target, _line(node), tuple(bindings), exact=not local)]


def erlang_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    line = _line(node)
    if node.type in {"pp_include", "pp_include_lib"}:
        spec = _first_string(ctx, node)
        if spec is None:
            return []
        if node.type == "pp_include_lib":
            return [Import(spec, spec, line)]
        return [Import(spec, relative_module(ctx.rel_path, spec) or spec, line)]
    module = ctx.field_text(node, "module")
    if not module:
        return []
    bindings = tuple(
        (fun, f"{module}.{fun}")
        for fa in node.children_by_field_name("funs")
        if (fun := ctx.field_text(fa, "fun"))
    )
    return [Import(module, module, line, bindings, exact=False)]


def solidity_imports(ctx: FileContext, node: TSNode) -> list[Import] | None:
    source = node.child_by_field_name("source")
    if source is None:
        return []
    spec = string_value(ctx, source)
    if spec.startswith("."):
        target, exact = relative_module(ctx.rel_path, spec) or spec, True
    else:
        target, exact = spec, True
    names = [ctx.text(c) for c in node.children_by_field_name("import_name")]
    aliases = [ctx.text(c) for c in node.children_by_field_name("alias")]
    bindings = tuple(
        (aliases[i] if i < len(aliases) else name, f"{target}.{name}")
        for i, name in enumerate(names)
    )
    return [Import(spec, target, _line(node), bindings, exact)]


__all__ = [
    "JVM_EXTERNAL_ROOTS",
    "PERL_PRAGMAS",
    "bash_imports",
    "csharp_imports",
    "dart_imports",
    "elixir_imports",
    "erlang_imports",
    "go_imports",
    "go_package_name",
    "haskell_imports",
    "include_imports",
    "java_imports",
    "julia_imports",
    "jvm_package",
    "jvm_reference",
    "kotlin_imports",
    "lua_imports",
    "perl_imports",
    "php_imports",
    "r_imports",
    "ruby_imports",
    "rust_crate_root",
    "rust_imports",
    "rust_module_path",
    "rust_prelude",
    "scala_imports",
    "solidity_imports",
    "swift_imports",
    "zig_imports",
]
