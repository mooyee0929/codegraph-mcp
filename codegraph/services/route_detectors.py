"""Per-framework route detectors, one function per language family.

Each detector reads one :class:`SourceFile` and returns a :class:`FileRoutes`: the
routes it registers (relative to their router), the mounts it performs, and the spans
of its named functions so mounts such as ``r.Mount("/api", apiRouter())`` can find
routes registered inside ``apiRouter``. Composition across files happens in
:func:`codegraph.services.routes.detect_routes`.

Covered: Flask, FastAPI, Starlette, Django ``urls.py``, aiohttp, Sanic (Python);
Express, Koa, Fastify, Hapi, Hono, NestJS (JS/TS); net/http, gin, echo, chi, gorilla,
fiber (Go); Spring, JAX-RS, Micronaut, Ktor (JVM); ASP.NET attributes and minimal APIs
(C#); Rails ``routes.rb`` and Sinatra (Ruby); Laravel and Symfony attributes (PHP);
actix, rocket and axum (Rust).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from pathlib import PurePosixPath

from codegraph.parsers.base import module_qname_for
from codegraph.services.paths import ANY, HTTP_METHODS, join_paths, normalize_method
from codegraph.services.routes import FileRoutes, Mount, RouteHit, Target
from codegraph.services.sources import SourceFile
from codegraph.services.text import (
    Arg,
    keyword_arg,
    matching_close,
    module_constants,
    pick_arg,
    split_args,
    split_bare_args,
    string_literal,
    url_expression,
)

_QUOTED_ANY = re.compile(r"\"((?:\\.|[^\"\\\n])*)\"|'((?:\\.|[^'\\\n])*)'|`([^`]*)`")
_IDENT_REF = re.compile(r"[A-Za-z_$][\w$]*(?:\s*(?:\.|::|->)\s*[A-Za-z_$][\w$]*)*")


def detect_file(source: SourceFile) -> FileRoutes:
    detector = _DETECTORS.get(source.family)
    return detector(source) if detector is not None else FileRoutes()


# -- shared helpers -------------------------------------------------------------


def _calls(
    source: SourceFile, pattern: re.Pattern[str], start: int = 0, end: int | None = None
) -> Iterator[tuple[re.Match[str], list[Arg]]]:
    """Matches of ``pattern`` (which must end at an opening paren) with their args."""
    stop = len(source.text) if end is None else end
    for match in pattern.finditer(source.text, start, stop):
        if source.in_comment(match.start()):
            continue
        args = split_args(source.text, match.end() - 1, source.family)
        if args is not None:
            yield match, args


def _route_path(arg: Arg | None, source: SourceFile, constants: dict[str, str]) -> str | None:
    if arg is None:
        return None
    url = url_expression(arg.value, source.family, constants, client=False)
    return url.text if url is not None else None


def _strings(text: str) -> list[str]:
    return [m.group(1) or m.group(2) or m.group(3) or "" for m in _QUOTED_ANY.finditer(text)]


def _methods_from(text: str | None, default: str) -> list[str]:
    if text is None:
        return [default]
    found = [normalize_method(s) for s in _strings(text)]
    found += [normalize_method(m) for m in re.findall(r"\bMethod\w*\.(\w+)", text)]
    found += [normalize_method(m) for m in re.findall(r":(get|post|put|patch|delete|all)\b", text)]
    unique = sorted({m for m in found if m})
    return unique or [default]


def handler_ref(text: str, family: str) -> str | None:
    """The function a handler argument names, or None for an inline function."""
    value = text.strip().removeprefix("&")
    if not value or value[0] in "{['\"`(|" or "=>" in value:
        return None
    if re.match(r"^(?:async\s+)?(?:function|func|lambda|fn)\b", value):
        return None
    value = re.sub(r"\.\s*(?:as_view|bind)\s*\(.*\)$", "", value, flags=re.S)
    value = re.sub(r"\s*\{\s*\}$", "", value)
    if _IDENT_REF.fullmatch(value):
        return re.sub(r"\s+", "", value)
    call = re.match(r"^[\w$.:]+\s*\(", value)
    if call is not None and value.endswith(")"):
        args = split_args(value, call.end() - 1, family)
        if args:
            return handler_ref(args[-1].text, family)
    return None


def _body_span(text: str, start: int, family: str, limit: int = 4000) -> tuple[int, int] | None:
    """Span of the first ``{...}`` block after ``start`` that is outside parentheses."""
    depth = 0
    index = start
    end = min(len(text), start + limit)
    while index < end:
        char = text[index]
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == ";" and depth <= 0:
            return None
        elif char == "{" and depth <= 0:
            close = matching_close(text, index, family)
            return (index, close) if close is not None else None
        elif char in "'\"`":
            literal_end = _skip_quoted(text, index)
            index = literal_end
            continue
        index += 1
    return None


def _skip_quoted(text: str, index: int) -> int:
    quote = text[index]
    cursor = index + 1
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += 2
            continue
        if text[cursor] == quote or (text[cursor] == "\n" and quote != "`"):
            return cursor + 1
        cursor += 1
    return cursor


def _prefix_at(spans: list[tuple[int, int, str]], offset: int) -> str:
    inside = sorted((s for s in spans if s[0] <= offset <= s[1]), key=lambda s: s[0])
    return join_paths(*(p for _, _, p in inside)) if inside else ""


_FUNCTION_DEFS: dict[str, re.Pattern[str]] = {
    "js": re.compile(
        r"(?:\bfunction\s+(?P<a>[\w$]+)\s*\(|\b(?:const|let|var)\s+(?P<b>[\w$]+)\s*=\s*(?:async\s*)?"
        r"(?:function\b[^(]*)?\([^)]*\)\s*(?::\s*[\w.<>\[\]| ]+)?\s*(?:=>)?\s*)"
    ),
    "go": re.compile(r"^func\s+(?:\([^)]*\)\s*)?(?P<a>\w+)\s*\(", re.M),
    "rust": re.compile(r"\bfn\s+(?P<a>\w+)\s*[<(]"),
}


def _function_spans(source: SourceFile) -> dict[str, tuple[int, int]]:
    pattern = _FUNCTION_DEFS.get(source.family)
    if pattern is None:
        return {}
    spans: dict[str, tuple[int, int]] = {}
    for match in pattern.finditer(source.text):
        name = match.group("a") or (match.groupdict().get("b") or "")
        if not name or name in spans or source.in_comment(match.start()):
            continue
        body = _body_span(source.text, match.end() - 1, source.family, limit=600)
        if body is not None:
            spans[name] = (match.start(), body[1])
    return spans


def _hit(
    source: SourceFile, offset: int, method: str, path: str, framework: str, **extra: str | None
) -> RouteHit:
    return RouteHit(
        method=method,
        path=path,
        framework=framework,
        file=source.path,
        line=source.line(offset),
        offset=offset,
        handler=extra.get("handler"),
        style=extra.get("style") or "call",
        router=extra.get("router"),
        router_prefix=extra.get("router_prefix") or "",
        owner_hint=extra.get("owner_hint"),
    )


# -- Python ---------------------------------------------------------------------

_PY_IMPORT = re.compile(r"^\s*(?:from\s+(\.*[\w.]*)\s+import|import\s+([\w.]+))", re.M)
_PY_FROM = re.compile(
    r"^[ \t]*from[ \t]+(\.*)([\w.]*)[ \t]+import[ \t]+\(?([^)\n]+(?:\n[^)\n]*)*?)\)?$", re.M
)
_PY_IMPORT_AS = re.compile(r"^[ \t]*import[ \t]+([\w.]+)(?:[ \t]+as[ \t]+(\w+))?", re.M)
_PY_FRAMEWORKS = (
    "fastapi",
    "flask",
    "starlette",
    "django",
    "sanic",
    "quart",
    "litestar",
    "aiohttp",
    "bottle",
)
_PY_DECORATOR = re.compile(
    r"^[ \t]*@(?P<recv>[A-Za-z_][\w.]*?)\s*\.\s*"
    r"(?P<verb>route|get|post|put|patch|delete|head|options|api_route)\s*\(",
    re.M,
)
_PY_ROUTER_CTOR = re.compile(
    r"^[ \t]*(?P<var>\w+)\s*(?::\s*[\w.]+)?\s*=\s*(?:[\w.]+\.)?"
    r"(?P<ctor>APIRouter|Blueprint|FastAPI|Flask|Sanic|Quart|RouteTableDef)\s*\(",
    re.M,
)
_PY_DEF = re.compile(r"^[ \t]*(?:async[ \t]+)?(def|class)[ \t]+(\w+)", re.M)
_PY_INCLUDE = re.compile(
    r"(?P<parent>[A-Za-z_][\w.]*)\s*\.\s*(?P<fn>include_router|register_blueprint|mount)\s*\("
)
_PY_ADD_RULE = re.compile(
    r"(?P<recv>[A-Za-z_][\w.]*)\s*\.\s*"
    r"(?P<fn>add_url_rule|add_api_route|add_route|add_get|add_post|add_put|add_patch|add_delete)\s*\("
)
_DJANGO_PATH = re.compile(r"(?<![\w.])(?P<fn>path|re_path|url)\s*\(")
_STARLETTE = re.compile(r"(?<![\w.])(?P<kind>Route|WebSocketRoute|Mount)\s*\(")


def python_framework(text: str) -> str:
    roots = {
        (m.group(1) or m.group(2) or "").lstrip(".").split(".")[0]
        for m in _PY_IMPORT.finditer(text)
    }
    for name in _PY_FRAMEWORKS:
        if name in roots:
            return name
    return "python"


def _python_package(path: str) -> list[str]:
    parts = module_qname_for(path).split(".")
    return parts if PurePosixPath(path).stem == "__init__" else parts[:-1]


def python_aliases(source: SourceFile) -> dict[str, list[Target]]:
    """Local name -> what it might be: a module, or a variable inside a module."""
    aliases: dict[str, list[Target]] = {}
    for match in _PY_FROM.finditer(source.text):
        dots, module, names = match.group(1), match.group(2), match.group(3)
        if dots:
            package = _python_package(source.path)
            keep = len(package) - (len(dots) - 1)
            base = ".".join([*package[: max(0, keep)], *([module] if module else [])])
        else:
            base = module
        for item in names.replace("\n", " ").split(","):
            pieces = item.split("#")[0].split()
            if not pieces or pieces[0] == "*":
                continue
            name = pieces[0]
            alias = pieces[2] if len(pieces) == 3 and pieces[1] == "as" else name
            full = f"{base}.{name}" if base else name
            aliases[alias] = [Target(module=full), Target(module=base, var=name)]
    for match in _PY_IMPORT_AS.finditer(source.text):
        module, alias = match.group(1), match.group(2)
        aliases[alias or module.split(".")[0]] = [
            Target(module=module if alias else module.split(".")[0])
        ]
    return aliases


def _python_targets(ref: str, source: SourceFile, local_vars: set[str]) -> tuple[Target, ...]:
    head, _, rest = ref.partition(".")
    if not rest and head in local_vars:
        return (Target(file=source.path, var=head),)
    aliases = python_aliases(source).get(head, [])
    if not rest:
        return tuple(aliases)
    targets = [Target(module=t.module, var=rest.split(".")[-1]) for t in aliases if t.var is None]
    targets += [Target(module=f"{t.module}.{t.var}", var=rest) for t in aliases if t.var]
    return tuple(targets)


def detect_python(source: SourceFile) -> FileRoutes:
    result = FileRoutes()
    text = source.text
    constants = module_constants(text, "python")
    framework = python_framework(text)
    prefixes: dict[str, str] = {}
    local_vars: set[str] = set()
    for match, args in _calls(source, _PY_ROUTER_CTOR):
        local_vars.add(match.group("var"))
        names = ("prefix",) if match.group("ctor") == "APIRouter" else ("url_prefix",)
        prefix = _route_path(keyword_arg(args, *names), source, constants)
        if prefix:
            prefixes[match.group("var")] = prefix

    for match, args in _calls(source, _PY_DECORATOR):
        recv, verb = match.group("recv"), match.group("verb")
        path = _route_path(pick_arg(args, 0, ("path", "rule")), source, constants)
        if path is None or not (path.startswith("/") or (path == "" and recv in local_vars)):
            continue
        definition = _PY_DEF.search(text, match.end())
        name = definition.group(2) if definition else None
        is_class = bool(definition and definition.group(1) == "class")
        if verb in {"route", "api_route"}:
            default = ANY if is_class or verb == "api_route" or framework == "fastapi" else "GET"
            methods = _methods_from(_arg_text(keyword_arg(args, "methods")), default)
        else:
            methods = [verb.upper()]
        for method in methods:
            result.hits.append(
                _hit(
                    source,
                    match.start(),
                    method,
                    path,
                    framework,
                    handler=name,
                    style="decorator",
                    router=recv.split(".")[-1],
                    router_prefix=prefixes.get(recv.split(".")[-1]),
                )
            )

    for match, args in _calls(source, _PY_ADD_RULE):
        fn = match.group("fn")
        if fn == "add_route":
            methods = _methods_from(_arg_text(pick_arg(args, 0)), ANY)
            path = _route_path(pick_arg(args, 1), source, constants)
            handler = pick_arg(args, 2, ("handler",))
        elif fn.startswith("add_") and fn not in {"add_url_rule", "add_api_route"}:
            methods = [fn.removeprefix("add_").upper()]
            path = _route_path(pick_arg(args, 0), source, constants)
            handler = pick_arg(args, 1, ("handler",))
        else:
            path = _route_path(pick_arg(args, 0, ("rule", "path")), source, constants)
            handler = keyword_arg(args, "view_func", "endpoint") or pick_arg(
                args, 2 if fn == "add_url_rule" else 1
            )
            methods = _methods_from(
                _arg_text(keyword_arg(args, "methods")), "GET" if fn == "add_url_rule" else ANY
            )
        if path is None or not path.startswith("/"):
            continue
        ref = handler_ref(handler.text, "python") if handler else None
        recv = match.group("recv").split(".")[-1]
        for method in methods:
            result.hits.append(
                _hit(
                    source,
                    match.start(),
                    method,
                    path,
                    framework,
                    handler=ref,
                    router=recv,
                    router_prefix=prefixes.get(recv),
                )
            )

    for match, args in _calls(source, _PY_INCLUDE):
        ref_arg = pick_arg(
            args, 1 if match.group("fn") == "mount" else 0, ("router", "blueprint", "app")
        )
        if match.group("fn") == "mount":
            prefix = _route_path(pick_arg(args, 0, ("path",)), source, constants)
        else:
            prefix = _route_path(keyword_arg(args, "prefix", "url_prefix"), source, constants)
        if ref_arg is None:
            continue
        ref = handler_ref(ref_arg.text, "python")
        if ref is None:
            continue
        result.mounts.append(
            Mount(
                file=source.path,
                prefix=prefix or "",
                parent=match.group("parent").split(".")[-1],
                targets=_python_targets(ref, source, local_vars),
                replace_prefix=match.group("fn") == "register_blueprint" and prefix is not None,
                offset=match.start(),
            )
        )

    if PurePosixPath(source.path).name.endswith("urls.py"):
        _detect_django(source, constants, result)
    if re.search(r"\bstarlette\b|\bfastapi\b", text):
        _detect_starlette(source, constants, result)
    return result


def _arg_text(arg: Arg | None) -> str | None:
    return arg.value if arg is not None else None


def _detect_django(source: SourceFile, constants: dict[str, str], result: FileRoutes) -> None:
    for match, args in _calls(source, _DJANGO_PATH):
        route = _route_path(pick_arg(args, 0, ("route",)), source, constants)
        view = pick_arg(args, 1, ("view",))
        if route is None or view is None:
            continue
        if match.group("fn") != "path":
            route = route if route.startswith("^") else f"^{route}"
        include = re.match(r"^include\s*\(", view.value)
        if include is not None:
            inner = split_args(view.value, include.end() - 1, "python") or []
            first = inner[0].value if inner else ""
            literals = _strings(first)
            module = literals[0] if literals else handler_ref(first, "python")
            if module:
                result.mounts.append(
                    Mount(
                        file=source.path,
                        prefix=route,
                        parent=None,
                        targets=(Target(module=module),),
                        offset=match.start(),
                    )
                )
            continue
        result.hits.append(
            _hit(
                source,
                match.start(),
                ANY,
                "/" + route.lstrip("/"),
                "django",
                handler=handler_ref(view.value, "python"),
            )
        )


def _detect_starlette(source: SourceFile, constants: dict[str, str], result: FileRoutes) -> None:
    text = source.text
    spans: list[tuple[int, int, str]] = []
    routes: list[tuple[re.Match[str], list[Arg]]] = []
    for match, args in _calls(source, _STARLETTE):
        if match.group("kind") == "Mount":
            prefix = _route_path(pick_arg(args, 0, ("path",)), source, constants)
            close = matching_close(text, match.end() - 1, "python")
            if prefix is None or close is None:
                continue
            spans.append((match.start(), close, prefix))
            listed = keyword_arg(args, "routes")
            if listed is not None and re.fullmatch(r"\w+", listed.value):
                assigned = re.search(rf"^{re.escape(listed.value)}\s*=\s*\[", text, re.M)
                if assigned is not None:
                    end = matching_close(text, assigned.end() - 1, "python")
                    if end is not None:
                        spans.append((assigned.start(), end, prefix))
        elif match.group("kind") == "Route":
            routes.append((match, args))
    for match, args in routes:
        path = _route_path(pick_arg(args, 0, ("path",)), source, constants)
        endpoint = pick_arg(args, 1, ("endpoint",))
        if path is None or not path.startswith("/"):
            continue
        methods = _methods_from(_arg_text(keyword_arg(args, "methods")), "GET")
        ref = handler_ref(endpoint.text, "python") if endpoint else None
        for method in methods:
            result.hits.append(
                _hit(
                    source,
                    match.start(),
                    method,
                    path,
                    "starlette",
                    handler=ref,
                    router_prefix=_prefix_at(spans, match.start()),
                )
            )


# -- JavaScript / TypeScript ----------------------------------------------------

_JS_FRAMEWORKS = (
    ("nestjs", r"@nestjs/"),
    ("fastify", r"['\"]fastify['\"]"),
    ("koa", r"['\"](?:koa|@koa/router|koa-router)['\"]"),
    ("hono", r"['\"]hono['\"]"),
    ("hapi", r"['\"]@hapi/hapi['\"]"),
    ("express", r"['\"]express['\"]"),
)
_JS_SERVER_CTOR = re.compile(
    r"(?:const|let|var)\s+(?P<var>[\w$]+)\s*(?::\s*[\w.<>]+)?\s*=\s*(?:await\s+)?"
    r"(?P<ctor>express\s*\(|express\s*\.\s*Router\s*\(|Router\s*\(|new\s+(?:Koa)?Router\s*\(|"
    r"new\s+Koa\s*\(|[Ff]astify\s*\(|new\s+Hono\s*(?:<[^>]*>)?\s*\(|new\s+Elysia\s*\(|"
    r"polka\s*\(|restify\s*\.\s*createServer\s*\(|Hapi\s*\.\s*server\s*\()"
)
_JS_SERVER_NAME = re.compile(r"^(?:app|router|server|fastify|routes|\w*Router|\w*App|\w*Server)$")
_JS_VERB_CALL = re.compile(
    r"(?<![\w$.])(?:this\s*\.\s*)?(?P<recv>[A-Za-z_$][\w$]*)\s*\.\s*"
    r"(?P<verb>get|post|put|patch|delete|del|head|options|all)\s*(?:<[^>()]*>)?\s*\("
)
_JS_ROUTE_CALL = re.compile(r"(?<![\w$.])(?P<recv>[A-Za-z_$][\w$]*)\s*\.\s*route\s*\(")
_JS_CHAIN = re.compile(r"\s*\.\s*(?P<verb>get|post|put|patch|delete|all|head|options)\s*\(")
_JS_USE = re.compile(r"(?<![\w$.])(?P<parent>[A-Za-z_$][\w$]*)\s*\.\s*(?P<fn>use|register)\s*\(")
_JS_IMPORT = re.compile(
    r"\bimport\s+(?:(?P<default>[\w$]+)\s*,?\s*)?(?:\{(?P<named>[^}]*)\}|\*\s+as\s+(?P<ns>[\w$]+))?"
    r"\s*from\s*['\"](?P<spec>[^'\"]+)['\"]"
)
_JS_REQUIRE = re.compile(
    r"(?:const|let|var)\s+(?:(?P<var>[\w$]+)|\{(?P<named>[^}]*)\})\s*=\s*require\s*\(\s*"
    r"['\"](?P<spec>[^'\"]+)['\"]\s*\)(?:\s*\.\s*(?P<attr>[\w$]+))?"
)
_NEST_CONTROLLER = re.compile(r"@Controller\s*\(")
_NEST_METHOD = re.compile(r"@(?P<verb>Get|Post|Put|Patch|Delete|All|Options|Head)\s*\(")
_NEST_GLOBAL = re.compile(r"\.\s*setGlobalPrefix\s*\(\s*(['\"`])([^'\"`]*)\1")
_MEMBER_NAME = re.compile(
    r"\s*(?:@[\w.]+\s*(?:\((?:[^()]|\([^()]*\))*\))?\s*)*"
    r"(?:(?:public|private|protected|static|async|readonly|override)\s+)*\*?\s*(?P<name>[\w$]+)\s*[(<]"
)
_JS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


def js_framework(text: str) -> str:
    for name, pattern in _JS_FRAMEWORKS:
        if re.search(pattern, text):
            return name
    return "node"


def resolve_js_specifier(from_path: str, spec: str) -> list[str]:
    """Candidate repo paths for a relative import specifier (without checking existence)."""
    if not spec.startswith("."):
        return []
    parts: list[str] = []
    for part in [*PurePosixPath(from_path).parent.parts, *PurePosixPath(spec).parts]:
        if part == ".":
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    base = "/".join(parts)
    stem, suffix = base, PurePosixPath(base).suffix
    if suffix in _JS_EXTENSIONS:
        stem = base[: -len(suffix)]
        return [base, *(stem + ext for ext in _JS_EXTENSIONS)]
    return [
        *(base + ext for ext in _JS_EXTENSIONS),
        *(f"{base}/index{ext}" for ext in _JS_EXTENSIONS),
    ]


def js_imports(source: SourceFile) -> dict[str, tuple[list[str], str | None]]:
    """Local name -> (candidate files, exported name or None for default/namespace)."""
    imports: dict[str, tuple[list[str], str | None]] = {}
    for match in _JS_IMPORT.finditer(source.text):
        files = resolve_js_specifier(source.path, match.group("spec"))
        if not files:
            continue
        if match.group("default"):
            imports[match.group("default")] = (files, None)
        if match.group("ns"):
            imports[match.group("ns")] = (files, None)
        for item in (match.group("named") or "").split(","):
            pieces = item.split()
            if pieces:
                imports[pieces[-1]] = (files, pieces[0])
    for match in _JS_REQUIRE.finditer(source.text):
        files = resolve_js_specifier(source.path, match.group("spec"))
        if not files:
            continue
        if match.group("var"):
            imports[match.group("var")] = (files, match.group("attr"))
        for item in (match.group("named") or "").split(","):
            pieces = [p.strip() for p in item.split(":")]
            if pieces and pieces[0]:
                imports[pieces[-1]] = (files, pieces[0])
    return imports


def _js_targets(ref_text: str, source: SourceFile, local_vars: set[str]) -> tuple[Target, ...]:
    value = ref_text.strip()
    required = re.match(r"^require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)(?:\s*\.\s*([\w$]+))?$", value)
    if required is not None:
        return tuple(
            Target(file=f, var=required.group(2))
            for f in resolve_js_specifier(source.path, required.group(1))
        )
    call = re.match(r"^([\w$]+)(?:\s*\.\s*([\w$]+))?\s*\(", value)
    imports = js_imports(source)
    if call is not None and call.group(2) in {None, "routes", "router"}:
        name = call.group(1)
        if call.group(2) is None:
            files = imports.get(name, ([source.path], None))[0]
            return tuple(Target(file=f, function=name) for f in files)
        value = name
    ref = handler_ref(value, "js")
    if ref is None:
        return ()
    head, _, attr = ref.partition(".")
    if head in local_vars and not attr:
        return (Target(file=source.path, var=head),)
    if head in imports:
        files, exported = imports[head]
        var = attr or exported
        return tuple(
            Target(file=f, var=var if var not in {None, "default"} else None) for f in files
        )
    return ()


def detect_js(source: SourceFile) -> FileRoutes:
    result = FileRoutes(functions=_function_spans(source))
    text = source.text
    constants = module_constants(text, "js")
    framework = js_framework(text)
    servers: set[str] = set()
    prefixes: dict[str, str] = {}
    for match, args in _calls(source, _JS_SERVER_CTOR):
        servers.add(match.group("var"))
        options = pick_arg(args, 0)
        if options is not None and options.value.startswith("{"):
            prefix = _object_prop(options.value, "prefix", source, constants)
            if prefix:
                prefixes[match.group("var")] = prefix

    def is_server(name: str) -> bool:
        return name in servers or bool(_JS_SERVER_NAME.match(name))

    for match, args in _calls(source, _JS_VERB_CALL):
        recv = match.group("recv")
        if not is_server(recv) or len(args) < 2:
            continue
        path = _route_path(args[0], source, constants)
        if path is None or not (
            path.startswith("/") or path == "*" or (path == "" and recv in servers)
        ):
            continue
        last = args[-1].text
        if last[:1] in "{'\"`" or re.fullmatch(r"[\d.]+", last):
            continue
        result.hits.append(
            _hit(
                source,
                match.start(),
                normalize_method(match.group("verb")),
                path,
                framework,
                handler=handler_ref(last, "js"),
                router=recv,
                router_prefix=prefixes.get(recv),
            )
        )

    for match, args in _calls(source, _JS_ROUTE_CALL):
        recv = match.group("recv")
        if not is_server(recv) or not args:
            continue
        if args[0].value.startswith(("{", "[")):
            _js_route_objects(source, match, args[0], constants, framework, prefixes, result)
            continue
        path = _route_path(args[0], source, constants)
        close = matching_close(text, match.end() - 1, "js")
        if path is None or close is None:
            continue
        cursor = close + 1
        while True:
            chained = _JS_CHAIN.match(text, cursor)
            if chained is None:
                break
            chain_args = split_args(text, chained.end() - 1, "js")
            chain_close = matching_close(text, chained.end() - 1, "js")
            if chain_args is None or chain_close is None:
                break
            ref = handler_ref(chain_args[-1].text, "js") if chain_args else None
            result.hits.append(
                _hit(
                    source,
                    chained.start(),
                    normalize_method(chained.group("verb")),
                    path,
                    framework,
                    handler=ref,
                    router=recv,
                    router_prefix=prefixes.get(recv),
                )
            )
            cursor = chain_close + 1

    for match, args in _calls(source, _JS_USE):
        fn = match.group("fn")
        if not args:
            continue
        if fn == "use":
            literal = _route_path(args[0], source, constants)
            refs = args[1:] if literal is not None else args
        else:
            options = pick_arg(args, 1)
            literal = _object_prop(options.value, "prefix", source, constants) if options else None
            refs = args[:1]
            if literal is None:
                continue
        for ref_arg in refs:
            targets = _js_targets(ref_arg.text, source, servers)
            if targets:
                result.mounts.append(
                    Mount(
                        file=source.path,
                        prefix=literal or "",
                        parent=match.group("parent"),
                        targets=targets,
                        offset=match.start(),
                    )
                )

    if "@Controller" in text:
        _detect_class_prefixed(source, constants, result, _NEST_CONTROLLER, _NEST_METHOD, "nestjs")
    for match in _NEST_GLOBAL.finditer(text):
        result.global_prefixes["nestjs"] = match.group(2)
    return result


def _object_prop(obj: str, name: str, source: SourceFile, constants: dict[str, str]) -> str | None:
    match = re.search(rf"(?<![\w$]){name}\s*:\s*", obj)
    if match is None:
        return None
    rest = obj[match.end() :]
    end = _value_end(rest)
    url = url_expression(rest[:end], source.family, constants, client=False)
    return url.text if url is not None else None


def _value_end(text: str) -> int:
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in "'\"`":
            index = _skip_quoted(text, index)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                return index
            depth -= 1
        elif char == "," and depth == 0:
            return index
        index += 1
    return index


def _js_route_objects(
    source: SourceFile,
    match: re.Match[str],
    arg: Arg,
    constants: dict[str, str],
    framework: str,
    prefixes: dict[str, str],
    result: FileRoutes,
) -> None:
    objects = [arg.value] if arg.value.startswith("{") else []
    if arg.value.startswith("["):
        inner = split_args(arg.value, 0, "js") or []
        objects = [a.value for a in inner if a.value.startswith("{")]
    for obj in objects:
        path = _object_prop(obj, "url", source, constants) or _object_prop(
            obj, "path", source, constants
        )
        if path is None:
            continue
        method_match = re.search(r"(?<![\w$])method\s*:\s*", obj)
        method_text = (
            obj[method_match.end() :][: _value_end(obj[method_match.end() :])]
            if method_match
            else None
        )
        handler_match = re.search(r"(?<![\w$])handler\s*:\s*", obj)
        ref = None
        if handler_match is not None:
            rest = obj[handler_match.end() :]
            ref = handler_ref(rest[: _value_end(rest)], "js")
        for method in _methods_from(method_text, ANY):
            result.hits.append(
                _hit(
                    source,
                    match.start(),
                    method,
                    path,
                    framework,
                    handler=ref,
                    router=match.group("recv"),
                    router_prefix=prefixes.get(match.group("recv")),
                )
            )


def _detect_class_prefixed(
    source: SourceFile,
    constants: dict[str, str],
    result: FileRoutes,
    class_pattern: re.Pattern[str],
    method_pattern: re.Pattern[str],
    framework: str,
) -> None:
    """NestJS / Micronaut: ``@Controller("x")`` on a class, ``@Get("y")`` on methods."""
    text = source.text
    for match, args in _calls(source, class_pattern):
        close = matching_close(text, match.end() - 1, source.family)
        if close is None:
            continue
        first = pick_arg(args, 0, ("value", "path"))
        prefix = ""
        if first is not None:
            prefix = (
                _object_prop(first.value, "path", source, constants)
                if first.value.startswith("{")
                else _route_path(first, source, constants)
            ) or ""
        header = re.compile(r"\bclass\s+\w+").search(text, close)
        if header is None:
            continue
        body = _body_span(text, header.end(), source.family, limit=2000)
        if body is None:
            continue
        for method_match, method_args in _calls(source, method_pattern, body[0], body[1]):
            path = (
                _route_path(pick_arg(method_args, 0, ("value", "path", "uri")), source, constants)
                or ""
            )
            method_close = matching_close(text, method_match.end() - 1, source.family)
            name_match = _MEMBER_NAME.match(text, (method_close or method_match.end()) + 1)
            fun_match = re.compile(r"\s*(?:@\w+(?:\([^)]*\))?\s*)*(?:\w+\s+)*fun\s+(\w+)").match(
                text, (method_close or method_match.end()) + 1
            )
            name = (
                fun_match.group(1)
                if fun_match
                else (name_match.group("name") if name_match else None)
            )
            result.hits.append(
                _hit(
                    source,
                    method_match.start(),
                    normalize_method(method_match.group("verb")),
                    path,
                    framework,
                    handler=name,
                    style="decorator",
                    router_prefix=prefix,
                )
            )


# -- Go -------------------------------------------------------------------------

_GO_FRAMEWORKS = (
    ("gin", "github.com/gin-gonic/gin"),
    ("echo", "github.com/labstack/echo"),
    ("chi", "github.com/go-chi/chi"),
    ("gorilla", "github.com/gorilla/mux"),
    ("fiber", "github.com/gofiber/fiber"),
)
_GO_CALL = re.compile(
    r"(?<![\w.])(?P<recv>[A-Za-z_][\w.]*?)\s*\.\s*(?P<verb>GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|Any|"
    r"Get|Post|Put|Patch|Delete|Head|Options|All|Handle|HandleFunc|Method|MethodFunc|Match)\s*\("
)
_GO_GROUP = re.compile(
    r"(?P<var>\w+)\s*:?=\s*(?P<parent>[\w.]+?)\s*\.\s*(?:Group|PathPrefix)\s*\(\s*"
    r"(?P<lit>\"(?:\\.|[^\"\\])*\"|`[^`]*`)"
)
_GO_ROUTE_BLOCK = re.compile(r"(?P<recv>\w+)\s*\.\s*Route\s*\(\s*(?P<lit>\"[^\"]*\")\s*,\s*func")
_GO_MOUNT = re.compile(r"(?P<parent>\w+)\s*\.\s*Mount\s*\(")
_GO_METHODS = re.compile(r"\s*\.\s*Methods\s*\(")


def detect_go(source: SourceFile) -> FileRoutes:
    result = FileRoutes(functions=_function_spans(source))
    text = source.text
    constants = module_constants(text, "go")
    framework = next((name for name, module in _GO_FRAMEWORKS if module in text), "net/http")
    groups: dict[str, str] = {}
    for match in _GO_GROUP.finditer(text):
        literal = string_literal(match.group("lit"), "go")
        if literal is None:
            continue
        parent = match.group("parent").split(".")[-1]
        groups[match.group("var")] = join_paths(groups.get(parent, ""), literal.content)
    spans: list[tuple[int, int, str]] = []
    for match in _GO_ROUTE_BLOCK.finditer(text):
        close = matching_close(text, text.index("(", match.start()), "go")
        literal = string_literal(match.group("lit"), "go")
        if close is not None and literal is not None:
            spans.append((match.start(), close, literal.content))

    for match, args in _calls(source, _GO_CALL):
        verb, recv = match.group("verb"), match.group("recv").split(".")[-1]
        methods: list[str]
        handler_arg: Arg | None
        if verb in {"Method", "MethodFunc", "Match"}:
            methods = _methods_from(args[0].value if args else None, ANY)
            path = _route_path(pick_arg(args, 1), source, constants)
            handler_arg = pick_arg(args, 2)
        elif verb in {"Handle", "HandleFunc"}:
            first = _route_path(pick_arg(args, 0), source, constants) or ""
            second = pick_arg(args, 1)
            if (
                first.upper() in HTTP_METHODS
                and second is not None
                and second.value.startswith(('"', "`"))
            ):
                methods, path, handler_arg = (
                    [first.upper()],
                    _route_path(second, source, constants),
                    pick_arg(args, 2),
                )
            else:
                pattern = re.match(r"^([A-Z]+)\s+(\S.*)$", first)
                if pattern is not None and pattern.group(1) in HTTP_METHODS:
                    methods, path = [pattern.group(1)], pattern.group(2)
                else:
                    methods, path = [ANY], first
                handler_arg = pick_arg(args, 1)
                close = matching_close(text, match.end() - 1, "go")
                chained = _GO_METHODS.match(text, close + 1) if close is not None else None
                if chained is not None:
                    chain_args = split_args(text, chained.end() - 1, "go") or []
                    methods = _methods_from(", ".join(a.value for a in chain_args), ANY)
        else:
            if len(args) < 2:
                continue
            methods = [normalize_method(verb)]
            path = _route_path(args[0], source, constants)
            handler_arg = args[-1]
        if path is None or not (path.startswith("/") or (path == "" and recv in groups)):
            continue
        ref = handler_ref(handler_arg.text, "go") if handler_arg else None
        prefix = join_paths(groups.get(recv, ""), _prefix_at(spans, match.start()))
        for method in methods:
            result.hits.append(
                _hit(
                    source,
                    match.start(),
                    method,
                    path,
                    framework,
                    handler=ref,
                    router=recv,
                    router_prefix=prefix,
                )
            )

    for match, args in _calls(source, _GO_MOUNT):
        mount_prefix = _route_path(pick_arg(args, 0), source, constants)
        target = pick_arg(args, 1)
        if mount_prefix is None or target is None:
            continue
        call = re.match(r"^(?:\w+\.)?(\w+)\s*\(", target.value)
        targets = (
            (Target(function=call.group(1)),)
            if call
            else (Target(file=source.path, var=target.value),)
        )
        result.mounts.append(
            Mount(
                file=source.path,
                prefix=mount_prefix,
                parent=match.group("parent"),
                targets=targets,
                offset=match.start(),
            )
        )
    return result


# -- JVM: Spring, JAX-RS, Micronaut, Ktor ---------------------------------------

_JVM_ANNOTATION = re.compile(r"@(?P<name>[A-Za-z_][\w.]*)")
_SPRING_VERBS = {
    "GetMapping": "GET",
    "PostMapping": "POST",
    "PutMapping": "PUT",
    "DeleteMapping": "DELETE",
    "PatchMapping": "PATCH",
    "RequestMapping": ANY,
}
_JAXRS_VERBS = frozenset({"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"})
_CLASS_HEADER = re.compile(
    r"^\s*(?:(?:public|private|protected|abstract|final|static|open|data|sealed|internal|inner|"
    r"partial|export|default|annotation)\s+)*(?:class|interface|object|record|enum|@interface)\s+(?P<name>\w+)"
)
METHOD_HEADER = re.compile(
    r"^\s*(?:(?:public|private|protected|static|final|abstract|synchronized|default|open|override|"
    r"suspend|internal|native|async|virtual|sealed|operator|inline|function)\s+)*"
    r"(?:fun\s+(?:<[^>]*>\s*)?(?:[\w.]+\.)?(?P<kname>\w+)|(?:<[^>]*>\s*)?(?:[\w.<>\[\]?,]+\s+)*(?P<jname>\w+))"
    r"\s*\("
)


def annotation_groups(source: SourceFile) -> list[tuple[list[tuple[str, str | None, int]], int]]:
    """Adjacent annotations as ``([(name, args, offset)], end_offset)`` groups."""
    text = source.text
    groups: list[tuple[list[tuple[str, str | None, int]], int]] = []
    current: list[tuple[str, str | None, int]] = []
    current_end = -1
    for match in _JVM_ANNOTATION.finditer(text):
        if match.start() < current_end or source.in_comment(match.start()):
            continue
        before = text[match.start() - 1] if match.start() else " "
        if not (before.isspace() or before in "({,;"):
            continue
        end = match.end()
        args: str | None = None
        after = re.match(r"\s*\(", text[end : end + 200])
        if after is not None:
            close = matching_close(text, end + after.end() - 1, source.family)
            if close is not None:
                args = text[end + after.end() : close]
                end = close + 1
        name = match.group("name").split(".")[-1]
        if current and text[current_end : match.start()].strip() == "":
            current.append((name, args, match.start()))
        else:
            if current:
                groups.append((current, current_end))
            current = [(name, args, match.start())]
        current_end = end
    if current:
        groups.append((current, current_end))
    return groups


def _annotation_paths(args: str | None, source: SourceFile, constants: dict[str, str]) -> list[str]:
    if args is None or not args.strip():
        return [""]
    parsed = split_args(f"({args})", 0, source.family) or []
    value = pick_arg(parsed, 0, ("value", "path"))
    if value is None:
        return [""]
    text = value.value
    if text[:1] + text[-1:] in {"{}", "[]"}:
        items = split_args(text, 0, source.family) or []
        paths = [_route_path(item, source, constants) for item in items]
        return [p for p in paths if p is not None] or [""]
    path = _route_path(value, source, constants)
    return [path] if path is not None else []


def detect_jvm(source: SourceFile) -> FileRoutes:
    result = FileRoutes()
    text = source.text
    constants = module_constants(text, "jvm")
    classes: list[
        tuple[int, int, list[str], str, bool, str]
    ] = []  # start, end, prefixes, fw, feign, name
    methods: list[tuple[list[tuple[str, str | None, int]], int, str]] = []
    for group, end in annotation_groups(source):
        names = {name for name, _, _ in group}
        header = text[end : end + 400]
        class_match = _CLASS_HEADER.match(header)
        if class_match is not None:
            body = _body_span(text, end + class_match.end(), source.family, limit=4000)
            if body is None:
                continue
            prefixes = [""]
            framework = "spring"
            for name, args, _ in group:
                if name == "RequestMapping":
                    prefixes = _annotation_paths(args, source, constants)
                elif name == "Path":
                    prefixes, framework = _annotation_paths(args, source, constants), "jaxrs"
                elif name == "Controller" and args is not None:
                    prefixes, framework = _annotation_paths(args, source, constants), "micronaut"
            classes.append(
                (
                    body[0],
                    body[1],
                    prefixes,
                    framework,
                    "FeignClient" in names,
                    class_match.group("name"),
                )
            )
            continue
        method_match = METHOD_HEADER.match(header)
        if method_match is not None and names & (
            set(_SPRING_VERBS) | _JAXRS_VERBS | _MICRONAUT_VERBS
        ):
            methods.append((group, end, method_match.group("kname") or method_match.group("jname")))

    for group, _end, name in methods:
        offset = group[0][2]
        owner = [c for c in classes if c[0] <= offset <= c[1]]
        scope = max(owner, key=lambda c: c[0]) if owner else None
        if scope is not None and scope[4]:
            continue  # @FeignClient interfaces describe calls, not routes
        prefixes = scope[2] if scope is not None else [""]
        for hit in _jvm_method_routes(
            group, name, source, constants, scope[3] if scope else "spring"
        ):
            for prefix in prefixes:
                result.hits.append(replace_prefix(hit, prefix))
    if source.language == "kotlin" and "routing" in text:
        _detect_ktor(source, constants, result)
    return result


_MICRONAUT_VERBS = frozenset({"Get", "Post", "Put", "Delete", "Patch", "Head", "Options"})


def replace_prefix(hit: RouteHit, prefix: str) -> RouteHit:
    return RouteHit(
        method=hit.method,
        path=hit.path,
        framework=hit.framework,
        file=hit.file,
        line=hit.line,
        offset=hit.offset,
        handler=hit.handler,
        style=hit.style,
        router=hit.router,
        router_prefix=prefix,
        owner_hint=hit.owner_hint,
    )


def _jvm_method_routes(
    group: list[tuple[str, str | None, int]],
    name: str,
    source: SourceFile,
    constants: dict[str, str],
    class_framework: str,
) -> list[RouteHit]:
    hits: list[RouteHit] = []
    jaxrs_verbs = [n for n, args, _ in group if n in _JAXRS_VERBS and args is None]
    jaxrs_path = next((args for n, args, _ in group if n == "Path"), None)
    offset = group[0][2]
    for annotation, args, at in group:
        if annotation in _SPRING_VERBS:
            paths = _annotation_paths(args, source, constants)
            if annotation == "RequestMapping":
                method_text = None
                if args:
                    parsed = split_args(f"({args})", 0, source.family) or []
                    method_arg = keyword_arg(parsed, "method")
                    method_text = method_arg.value if method_arg else None
                verbs = _jvm_request_methods(method_text)
            else:
                verbs = [_SPRING_VERBS[annotation]]
            for path in paths:
                for verb in verbs:
                    hits.append(
                        _hit(source, at, verb, path, "spring", handler=name, style="decorator")
                    )
        elif annotation in _MICRONAUT_VERBS and class_framework == "micronaut":
            for path in _annotation_paths(args, source, constants):
                hits.append(
                    _hit(
                        source,
                        at,
                        annotation.upper(),
                        path,
                        "micronaut",
                        handler=name,
                        style="decorator",
                    )
                )
    if jaxrs_verbs:
        paths = _annotation_paths(jaxrs_path, source, constants) if jaxrs_path is not None else [""]
        for verb in jaxrs_verbs:
            for path in paths:
                hits.append(
                    _hit(source, offset, verb, path, "jaxrs", handler=name, style="decorator")
                )
    return hits


def _jvm_request_methods(text: str | None) -> list[str]:
    if not text:
        return [ANY]
    found = re.findall(r"RequestMethod\s*\.\s*(\w+)", text)
    return sorted({normalize_method(m) for m in found}) or [ANY]


_KTOR_BLOCK = re.compile(r"\broute\s*\(\s*(?P<lit>\"[^\"]*\")\s*\)\s*\{")
_KTOR_VERB = re.compile(
    r"\b(?P<verb>get|post|put|delete|patch|head|options)\s*(?:\(\s*(?P<lit>\"[^\"]*\")\s*\))?\s*\{"
)


def _detect_ktor(source: SourceFile, constants: dict[str, str], result: FileRoutes) -> None:
    text = source.text
    spans: list[tuple[int, int, str]] = []
    for match in _KTOR_BLOCK.finditer(text):
        close = matching_close(text, match.end() - 1, "jvm")
        literal = string_literal(match.group("lit"), "jvm")
        if close is not None and literal is not None:
            spans.append((match.end() - 1, close, literal.content))
    for match in _KTOR_VERB.finditer(text):
        if source.in_comment(match.start()):
            continue
        path = ""
        if match.group("lit"):
            literal = string_literal(match.group("lit"), "jvm")
            path = literal.content if literal else ""
        prefix = _prefix_at(spans, match.start())
        if not path and not prefix:
            continue
        result.hits.append(
            _hit(
                source,
                match.start(),
                match.group("verb").upper(),
                path,
                "ktor",
                router_prefix=prefix,
            )
        )


# -- C# / ASP.NET ---------------------------------------------------------------

_CS_ATTRIBUTE = re.compile(r"\[\s*(?=(?:Http\w+|Route|AcceptVerbs|ApiController|Controller)\b)")
_CS_MINIMAL = re.compile(
    r"(?<![\w.])(?P<recv>[A-Za-z_][\w.]*?)\s*\.\s*Map(?P<verb>Get|Post|Put|Delete|Patch|Methods)\s*\("
)
_CS_GROUP = re.compile(
    r"\bvar\s+(?P<var>\w+)\s*=\s*(?P<parent>[\w.]+?)\s*\.\s*MapGroup\s*\(\s*(?P<lit>\"[^\"]*\")"
)
_CS_HTTP = {
    "HttpGet": "GET",
    "HttpPost": "POST",
    "HttpPut": "PUT",
    "HttpDelete": "DELETE",
    "HttpPatch": "PATCH",
    "HttpHead": "HEAD",
    "HttpOptions": "OPTIONS",
}


def _cs_attribute_groups(source: SourceFile) -> list[tuple[list[tuple[str, str | None]], int, int]]:
    text = source.text
    groups: list[tuple[list[tuple[str, str | None]], int, int]] = []
    current: list[tuple[str, str | None]] = []
    start = end = -1
    for match in _CS_ATTRIBUTE.finditer(text):
        if match.start() < end or source.in_comment(match.start()):
            continue
        close = matching_close(text, match.start(), "csharp")
        if close is None:
            continue
        items = split_args(text, match.start(), "csharp") or []
        attributes: list[tuple[str, str | None]] = []
        for item in items:
            named = re.match(r"^(\w+)\s*(?:\((.*)\))?$", item.value, re.S)
            if named:
                attributes.append((named.group(1), named.group(2)))
        if current and text[end : match.start()].strip() == "":
            current.extend(attributes)
        else:
            if current:
                groups.append((current, start, end))
            current, start = attributes, match.start()
        end = close + 1
        while True:
            following = re.match(r"\s*\[", text[end:])
            if following is None:
                break
            other = matching_close(text, end + following.end() - 1, "csharp")
            if other is None:
                break
            for item in split_args(text, end + following.end() - 1, "csharp") or []:
                named = re.match(r"^(\w+)\s*(?:\((.*)\))?$", item.value, re.S)
                if named:
                    current.append((named.group(1), named.group(2)))
            end = other + 1
    if current:
        groups.append((current, start, end))
    return groups


def detect_csharp(source: SourceFile) -> FileRoutes:
    result = FileRoutes()
    text = source.text
    constants = module_constants(text, "csharp")
    classes: list[tuple[int, int, list[str]]] = []
    methods: list[tuple[list[tuple[str, str | None]], int, str]] = []
    for attributes, start, end in _cs_attribute_groups(source):
        header = text[end : end + 400]
        class_match = _CLASS_HEADER.match(header)
        if class_match is not None:
            body = _body_span(text, end + class_match.end(), "csharp", limit=4000)
            if body is None:
                continue
            controller = re.sub(r"Controller$", "", class_match.group("name")).lower()
            templates = [
                p.replace("[controller]", controller)
                for attr_name, attr_args in attributes
                if attr_name == "Route"
                for p in _annotation_paths(attr_args, source, constants)
            ]
            classes.append((body[0], body[1], templates or [""]))
            continue
        method_match = METHOD_HEADER.match(header)
        if method_match is not None:
            methods.append(
                (attributes, start, method_match.group("jname") or method_match.group("kname"))
            )
    for attributes, start, name in methods:
        owner = [c for c in classes if c[0] <= start <= c[1]]
        prefixes = max(owner, key=lambda c: c[0])[2] if owner else [""]
        route_paths = [
            p
            for n, a in attributes
            if n == "Route"
            for p in _annotation_paths(a, source, constants)
        ]
        verbs: list[tuple[str, list[str]]] = []
        for attr, attr_args in attributes:
            if attr in _CS_HTTP:
                own = _annotation_paths(attr_args, source, constants) if attr_args else []
                verbs.append((_CS_HTTP[attr], own))
            elif attr == "AcceptVerbs":
                verbs.extend((normalize_method(v), []) for v in _strings(attr_args or ""))
        if not verbs and route_paths:
            verbs = [(ANY, [])]
        for verb, own_paths in verbs:
            for template in own_paths or route_paths or [""]:
                template = template.replace("[action]", name.lower())
                absolute = template.startswith(("/", "~/"))
                for prefix in [""] if absolute else prefixes:
                    result.hits.append(
                        _hit(
                            source,
                            start,
                            verb,
                            template.removeprefix("~"),
                            "aspnet",
                            handler=name,
                            style="decorator",
                            router_prefix=prefix,
                        )
                    )
    groups: dict[str, str] = {}
    for match in _CS_GROUP.finditer(text):
        literal = string_literal(match.group("lit"), "csharp")
        if literal is not None:
            groups[match.group("var")] = join_paths(
                groups.get(match.group("parent"), ""), literal.content
            )
    for match, args in _calls(source, _CS_MINIMAL):
        path = _route_path(pick_arg(args, 0), source, constants)
        if path is None:
            continue
        if match.group("verb") == "Methods":
            verbs_list = _methods_from(args[1].value if len(args) > 1 else None, ANY)
            handler_arg = pick_arg(args, 2)
        else:
            verbs_list = [normalize_method(match.group("verb"))]
            handler_arg = pick_arg(args, 1)
        recv = match.group("recv").split(".")[-1]
        ref = handler_ref(handler_arg.text, "csharp") if handler_arg else None
        for verb in verbs_list:
            result.hits.append(
                _hit(
                    source,
                    match.start(),
                    verb,
                    path,
                    "aspnet",
                    handler=ref,
                    router=recv,
                    router_prefix=groups.get(recv),
                )
            )
    return result


# -- Ruby: Rails routes.rb and Sinatra ------------------------------------------

_RAILS_ROUTES = re.compile(r"(?:^|/)config/routes(?:\.rb|/.+\.rb)$")
_RAILS_LINE = re.compile(
    r"^(?P<kw>namespace|scope|resources|resource|member|collection|concern|controller|get|post|put|"
    r"patch|delete|match|root)\b\s*\(?(?P<rest>.*)$"
)
_RAILS_ACTIONS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "plural": (
        ("index", "GET", ""),
        ("create", "POST", ""),
        ("new", "GET", "/new"),
        ("show", "GET", "/{id}"),
        ("edit", "GET", "/{id}/edit"),
        ("update", "PATCH", "/{id}"),
        ("update", "PUT", "/{id}"),
        ("destroy", "DELETE", "/{id}"),
    ),
    "singular": (
        ("show", "GET", ""),
        ("create", "POST", ""),
        ("new", "GET", "/new"),
        ("edit", "GET", "/edit"),
        ("update", "PATCH", ""),
        ("update", "PUT", ""),
        ("destroy", "DELETE", ""),
    ),
}
_SINATRA = re.compile(
    r"^[ \t]*(?P<verb>get|post|put|patch|delete|options)\s*\(?\s*"
    r"(?P<lit>'[^'\n]*'|\"[^\"\n]*\")\s*\)?\s*(?:do\b|\{)",
    re.M,
)


def singular(word: str) -> str:
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("ses"):
        return word[:-2]
    return word[:-1] if word.endswith("s") else word


def _controller_class(controller: str) -> str:
    leaf = controller.split("/")[-1]
    return "".join(part.capitalize() for part in leaf.split("_")) + "Controller"


class _RailsFrame:
    __slots__ = ("collection", "controller", "member", "module", "prefix", "skip")

    def __init__(
        self,
        prefix: str = "",
        module: str = "",
        controller: str | None = None,
        member: str | None = None,
        collection: str | None = None,
        skip: bool = False,
    ) -> None:
        self.prefix = prefix
        self.module = module
        self.controller = controller
        self.member = member
        self.collection = collection
        self.skip = skip

    def child(self, **changes: str | bool | None) -> _RailsFrame:
        frame = _RailsFrame(
            self.prefix, self.module, self.controller, self.member, self.collection, self.skip
        )
        for key, value in changes.items():
            setattr(frame, key, value)
        return frame


def _ruby_symbols(text: str) -> list[str]:
    return re.findall(r"(?<![\w:]):(\w+)", text) + [
        s for s in _strings(text) if re.fullmatch(r"\w+", s)
    ]


def _ruby_option(args: list[Arg], name: str) -> str | None:
    for arg in args:
        match = re.match(rf"^(?:{name}:|:{name}\s*=>)\s*(.+)$", arg.text, re.S)
        if match is not None:
            return match.group(1).strip()
    return None


def detect_ruby(source: SourceFile) -> FileRoutes:
    result = FileRoutes()
    if _RAILS_ROUTES.search(source.path):
        _detect_rails(source, result)
    elif re.search(r"\bsinatra\b|Sinatra::", source.text, re.I):
        for match in _SINATRA.finditer(source.text):
            literal = string_literal(match.group("lit"), "ruby")
            if literal is None or source.in_comment(match.start()):
                continue
            result.hits.append(
                _hit(source, match.start(), match.group("verb").upper(), literal.content, "sinatra")
            )
    return result


def _detect_rails(source: SourceFile, result: FileRoutes) -> None:
    text = source.text
    stack = [_RailsFrame()]
    offset = 0
    for raw in text.splitlines(keepends=True):
        line_offset = offset
        offset += len(raw)
        hash_at = _comment_start(raw, source, line_offset)
        line = (raw[:hash_at] if hash_at is not None else raw).strip()
        if not line:
            continue
        if re.fullmatch(r"end\b[\s)]*", line):
            if len(stack) > 1:
                stack.pop()
            continue
        frame = stack[-1]
        opens = bool(re.search(r"\bdo(?:\s*\|[^|]*\|)?\s*$", line))
        match = _RAILS_LINE.match(line)
        new_frame = frame.child() if opens else None
        if match is not None:
            keyword = match.group("kw")
            rest = re.sub(r"\s*\bdo(?:\s*\|[^|]*\|)?\s*$", "", match.group("rest")).rstrip(") ")
            args = split_bare_args(rest + "\n", 0, "ruby")
            pushed = _rails_statement(keyword, args, frame, source, line_offset, result)
            if opens and pushed is not None:
                new_frame = pushed
        elif opens and re.match(r"^\w[\w.:]*\.draw\b", line):
            new_frame = frame.child()
        if new_frame is not None:
            stack.append(new_frame)


def _comment_start(raw: str, source: SourceFile, line_offset: int) -> int | None:
    for index, char in enumerate(raw):
        if char == "#" and source.in_comment(line_offset + index):
            return index
    return None


def _rails_statement(
    keyword: str,
    args: list[Arg],
    frame: _RailsFrame,
    source: SourceFile,
    offset: int,
    result: FileRoutes,
) -> _RailsFrame | None:
    first = args[0].text if args else ""
    first_value = _ruby_value(first)
    if keyword == "namespace":
        return frame.child(
            prefix=join_paths(frame.prefix, first_value), module=f"{frame.module}{first_value}/"
        )
    if keyword == "scope":
        path_opt = _ruby_option(args, "path")
        module_opt = _ruby_option(args, "module")
        path = (
            _ruby_value(path_opt)
            if path_opt
            else (first_value if not re.match(r"^\w+:", first) else "")
        )
        module = frame.module + (_ruby_value(module_opt) + "/" if module_opt else "")
        return frame.child(
            prefix=join_paths(frame.prefix, path.replace("(:locale)", "")), module=module
        )
    if keyword == "concern":
        return frame.child(skip=True)
    if keyword == "controller":
        return frame.child(controller=first_value)
    if keyword in {"member", "collection"}:
        target = frame.member if keyword == "member" else frame.collection
        return frame.child(prefix=target or frame.prefix)
    if frame.skip:
        return None
    if keyword in {"resources", "resource"}:
        return _rails_resources(keyword, args, frame, source, offset, result)
    if keyword == "root":
        target = _ruby_option(args, "to") or first
        _rails_emit(
            source, offset, "GET", join_paths(frame.prefix), _ruby_value(target), frame, result
        )
        return None
    method_names = [keyword.upper()]
    if keyword == "match":
        method_names = [
            normalize_method(m) for m in _ruby_symbols(_ruby_option(args, "via") or ":all")
        ]
    path_text, target = first, _ruby_option(args, "to")
    rocket = re.match(r"^(.+?)\s*=>\s*(.+)$", first)
    if rocket is not None and not first.startswith(":to"):
        path_text, target = rocket.group(1), rocket.group(2)
    path = _ruby_value(path_text)
    on = _ruby_option(args, "on")
    base = frame.prefix
    if on and "member" in on and frame.member:
        base = frame.member
    elif on and "collection" in on and frame.collection:
        base = frame.collection
    action = _ruby_value(_ruby_option(args, "action") or "") or path.split("/")[-1]
    controller = _ruby_value(_ruby_option(args, "controller") or "") or frame.controller
    endpoint = _ruby_value(target) if target else (f"{controller}#{action}" if controller else None)
    for method in method_names:
        _rails_emit(source, offset, method, join_paths(base, path), endpoint, frame, result)
    return None


def _ruby_value(text: str | None) -> str:
    if not text:
        return ""
    value = text.strip()
    literal = string_literal(value, "ruby")
    if literal is not None:
        return literal.content
    return value.lstrip(":")


def _rails_resources(
    keyword: str,
    args: list[Arg],
    frame: _RailsFrame,
    source: SourceFile,
    offset: int,
    result: FileRoutes,
) -> _RailsFrame | None:
    names = [_ruby_value(a.text) for a in args if re.match(r"^:\w+$|^['\"]\w+['\"]$", a.text)]
    if not names:
        return None
    only = _ruby_option(args, "only")
    except_ = _ruby_option(args, "except")
    kind = "plural" if keyword == "resources" else "singular"
    allowed = {a for a, _, _ in _RAILS_ACTIONS[kind]}
    if only is not None:
        allowed &= set(_ruby_symbols(only))
    if except_ is not None:
        allowed -= set(_ruby_symbols(except_))
    param = _ruby_value(_ruby_option(args, "param") or "") or "id"
    nested: _RailsFrame | None = None
    for name in names:
        segment = _ruby_value(_ruby_option(args, "path") or "") or name
        controller = _ruby_value(_ruby_option(args, "controller") or "") or (
            name if kind == "plural" else name + "s"
        )
        controller = frame.module + controller
        base = join_paths(frame.prefix, segment)
        for action, method, suffix in _RAILS_ACTIONS[kind]:
            if action in allowed:
                path = base + suffix.replace("{id}", "{" + param + "}")
                _rails_emit(source, offset, method, path, f"{controller}#{action}", frame, result)
        member = base + (f"/{{{param}}}" if kind == "plural" else "")
        nested_prefix = base + (f"/{{{singular(name)}_id}}" if kind == "plural" else "")
        nested = frame.child(
            prefix=nested_prefix, controller=controller, member=member, collection=base
        )
    return nested


def _rails_emit(
    source: SourceFile,
    offset: int,
    method: str,
    path: str,
    endpoint: str | None,
    frame: _RailsFrame,
    result: FileRoutes,
) -> None:
    handler, hint = None, None
    if endpoint and "#" in endpoint:
        controller, _, action = endpoint.partition("#")
        if "/" not in controller and frame.module and not endpoint.startswith(frame.module):
            controller = frame.module + controller
        handler, hint = action, _controller_class(controller)
    result.hits.append(
        _hit(source, offset, method, path, "rails", handler=handler, owner_hint=hint)
    )


# -- PHP: Laravel and Symfony ---------------------------------------------------

_CHAIN = r"(?P<chain>(?:\w+\s*\((?:[^()]|\([^()]*\))*\)\s*->\s*)*)"
_LARAVEL_GROUP = re.compile(rf"Route\s*::\s*{_CHAIN}group\s*\(")
_LARAVEL_ROUTE = re.compile(
    rf"Route\s*::\s*{_CHAIN}(?P<verb>get|post|put|patch|delete|options|any|match|resource|apiResource|view|redirect)\s*\("
)
_LARAVEL_RESOURCE = (
    ("index", "GET", ""),
    ("create", "GET", "/create"),
    ("store", "POST", ""),
    ("show", "GET", "/{id}"),
    ("edit", "GET", "/{id}/edit"),
    ("update", "PUT", "/{id}"),
    ("update", "PATCH", "/{id}"),
    ("destroy", "DELETE", "/{id}"),
)
_SYMFONY_ROUTE = re.compile(r"#\[\s*Route\s*\(")


def _chain_value(chain: str, name: str) -> str | None:
    match = re.search(rf"\b{name}\s*\(\s*(['\"])(.*?)\1\s*\)", chain)
    return match.group(2) if match else None


def _php_controller(text: str) -> str | None:
    match = re.search(r"([\w\\]+)::class", text)
    return match.group(1).split("\\")[-1] if match else None


def detect_php(source: SourceFile) -> FileRoutes:
    result = FileRoutes()
    text = source.text
    constants = module_constants(text, "php")
    if "Route::" in text:
        _detect_laravel(source, constants, result)
    if "#[Route" in text:
        _detect_symfony(source, constants, result)
    return result


def _detect_laravel(source: SourceFile, constants: dict[str, str], result: FileRoutes) -> None:
    text = source.text
    groups: list[tuple[int, int, str, str | None]] = []
    for match, args in _calls(source, _LARAVEL_GROUP):
        close = matching_close(text, match.end() - 1, "php")
        if close is None:
            continue
        chain = match.group("chain")
        prefix = _chain_value(chain, "prefix")
        chained_controller = re.search(r"controller\s*\(([^)]*)\)", chain)
        controller = _php_controller(chained_controller.group(1)) if chained_controller else None
        if args and args[0].value.startswith("["):
            array = re.search(r"['\"]prefix['\"]\s*=>\s*(['\"])(.*?)\1", args[0].value)
            prefix = array.group(2) if array else prefix
        groups.append((match.end(), close, prefix or "", controller))
    api_prefix = "api" if source.path.endswith("routes/api.php") else ""
    for match, args in _calls(source, _LARAVEL_ROUTE):
        verb = match.group("verb")
        inside = sorted((g for g in groups if g[0] <= match.start() <= g[1]), key=lambda g: g[0])
        prefix = join_paths(
            api_prefix, *(g[2] for g in inside), _chain_value(match.group("chain"), "prefix") or ""
        )
        group_controller = next((g[3] for g in reversed(inside) if g[3]), None)
        if verb in {"resource", "apiResource"}:
            _laravel_resource(source, match, args, verb, prefix, constants, result)
            continue
        if verb == "match":
            methods = _methods_from(args[0].value if args else None, ANY)
            path_arg, handler_arg = pick_arg(args, 1), pick_arg(args, 2)
        else:
            methods = [
                ANY if verb == "any" else "GET" if verb in {"view", "redirect"} else verb.upper()
            ]
            path_arg, handler_arg = pick_arg(args, 0), pick_arg(args, 1)
        path = _route_path(path_arg, source, constants)
        if path is None:
            continue
        handler, hint = _laravel_handler(
            handler_arg.value if handler_arg and verb not in {"view", "redirect"} else None,
            group_controller,
        )
        for method in methods:
            result.hits.append(
                _hit(
                    source,
                    match.start(),
                    method,
                    path,
                    "laravel",
                    handler=handler,
                    owner_hint=hint,
                    router_prefix=prefix,
                )
            )


def _laravel_handler(
    text: str | None, group_controller: str | None
) -> tuple[str | None, str | None]:
    if text is None:
        return None, None
    if text.startswith("["):
        controller = _php_controller(text)
        actions = [s for s in _strings(text) if re.fullmatch(r"\w+", s)]
        return (actions[0] if actions else None), controller
    literal = string_literal(text, "php")
    if literal is not None:
        controller, _, action = literal.content.rpartition("@")
        if controller:
            return action, controller.split("\\")[-1]
        return literal.content, group_controller
    controller = _php_controller(text)
    if controller is not None:
        return "__invoke", controller
    return None, None


def _laravel_resource(
    source: SourceFile,
    match: re.Match[str],
    args: list[Arg],
    verb: str,
    prefix: str,
    constants: dict[str, str],
    result: FileRoutes,
) -> None:
    name = _route_path(pick_arg(args, 0), source, constants)
    controller_arg = pick_arg(args, 1)
    if not name:
        return
    controller = _php_controller(controller_arg.value) if controller_arg else None
    close = matching_close(source.text, match.end() - 1, "php") or match.end()
    tail = source.text[
        close : source.text.find(";", close) if ";" in source.text[close:] else close
    ]
    allowed = {a for a, _, _ in _LARAVEL_RESOURCE}
    if verb == "apiResource":
        allowed -= {"create", "edit"}
    only = re.search(r"->\s*only\s*\(\s*\[([^\]]*)\]", tail)
    if only is not None:
        allowed &= set(_strings(only.group(1)))
    excluded = re.search(r"->\s*except\s*\(\s*\[([^\]]*)\]", tail)
    if excluded is not None:
        allowed -= set(_strings(excluded.group(1)))
    segments = name.strip("/").split(".")
    base = ""
    for parent in segments[:-1]:
        base = join_paths(base, parent, "{" + singular(parent) + "}")
    leaf = segments[-1]
    base = join_paths(base, leaf)
    for action, method, suffix in _LARAVEL_RESOURCE:
        if action in allowed:
            path = base + suffix.replace("{id}", "{" + singular(leaf) + "}")
            result.hits.append(
                _hit(
                    source,
                    match.start(),
                    method,
                    path,
                    "laravel",
                    handler=action,
                    owner_hint=controller,
                    router_prefix=prefix,
                )
            )


def _detect_symfony(source: SourceFile, constants: dict[str, str], result: FileRoutes) -> None:
    text = source.text
    classes: list[tuple[int, int, str]] = []
    methods: list[tuple[re.Match[str], list[Arg], str]] = []
    for match, args in _calls(source, _SYMFONY_ROUTE):
        close = matching_close(text, match.end() - 1, "php")
        end = text.find("]", close) + 1 if close is not None else -1
        if end <= 0:
            continue
        header = re.match(
            r"\s*(?:#\[[^\]]*\]\s*)*(?:(?:final|abstract|readonly)\s+)*class\s+\w+", text[end:]
        )
        if header is not None:
            body = _body_span(text, end + header.end(), "php", limit=4000)
            if body is not None:
                classes.append(
                    (
                        body[0],
                        body[1],
                        _route_path(pick_arg(args, 0, ("path",)), source, constants) or "",
                    )
                )
            continue
        function = re.match(
            r"\s*(?:#\[[^\]]*\]\s*)*(?:(?:public|private|protected|static)\s+)*function\s+(\w+)",
            text[end:],
        )
        if function is not None:
            methods.append((match, args, function.group(1)))
    for match, args, name in methods:
        path = _route_path(pick_arg(args, 0, ("path",)), source, constants)
        if path is None:
            continue
        owner = [c for c in classes if c[0] <= match.start() <= c[1]]
        prefix = max(owner, key=lambda c: c[0])[2] if owner else ""
        methods_arg = keyword_arg(args, "methods")
        for method in _methods_from(methods_arg.value if methods_arg else None, ANY):
            result.hits.append(
                _hit(
                    source,
                    match.start(),
                    method,
                    path,
                    "symfony",
                    handler=name,
                    style="decorator",
                    router_prefix=prefix,
                )
            )


# -- Rust: actix, rocket, axum --------------------------------------------------

_RUST_ATTRIBUTE = re.compile(
    r"#\[\s*(?P<verb>get|post|put|delete|patch|head|options)\s*\(\s*(?P<lit>\"[^\"]*\")"
)
_RUST_FN = re.compile(r"\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)")
_RUST_ROUTE = re.compile(r"\.\s*route\s*\(")
_RUST_VERB = re.compile(
    r"(?:web\s*::\s*(?P<web>get|post|put|delete|patch|head)\s*\(\s*\)\s*\.\s*to\s*\(\s*(?P<wh>[\w:]+)"
    r"|(?<![\w])(?:(?:axum\s*::\s*)?routing\s*::\s*)?(?P<verb>get|post|put|delete|patch|head|options|any)"
    r"\s*\(\s*(?P<h>[\w:]+))"
)
_RUST_NEST = re.compile(r"\.\s*nest\s*\(")


def detect_rust(source: SourceFile) -> FileRoutes:
    result = FileRoutes(functions=_function_spans(source))
    text = source.text
    constants = module_constants(text, "rust")
    framework = "rocket" if "rocket" in text else "actix" if "actix" in text else "axum"
    for match in _RUST_ATTRIBUTE.finditer(text):
        if source.in_comment(match.start()):
            continue
        literal = string_literal(match.group("lit"), "rust")
        close = text.find("]", match.end())
        function = _RUST_FN.match(text, close + 1) if close != -1 else None
        if literal is None:
            continue
        result.hits.append(
            _hit(
                source,
                match.start(),
                match.group("verb").upper(),
                literal.content,
                framework,
                handler=function.group(1) if function else None,
                style="decorator",
            )
        )
    for match, args in _calls(source, _RUST_ROUTE):
        path = _route_path(pick_arg(args, 0), source, constants)
        target = pick_arg(args, 1)
        if path is None or target is None or not path.startswith("/"):
            continue
        for verb in _RUST_VERB.finditer(target.value):
            method = (verb.group("web") or verb.group("verb") or "").upper()
            handler = verb.group("wh") or verb.group("h")
            label = "actix" if verb.group("web") else "axum"
            result.hits.append(
                _hit(source, match.start(), normalize_method(method), path, label, handler=handler)
            )
    for match, args in _calls(source, _RUST_NEST):
        prefix = _route_path(pick_arg(args, 0), source, constants)
        target = pick_arg(args, 1)
        if prefix is None or target is None:
            continue
        call = re.match(r"^(?:[\w:]+::)?(\w+)\s*\(", target.value)
        if call is not None:
            result.mounts.append(
                Mount(
                    file=source.path,
                    prefix=prefix,
                    parent=None,
                    targets=(Target(function=call.group(1)),),
                    offset=match.start(),
                )
            )
    return result


_DETECTORS: dict[str, Callable[[SourceFile], FileRoutes]] = {
    "python": detect_python,
    "js": detect_js,
    "go": detect_go,
    "jvm": detect_jvm,
    "csharp": detect_csharp,
    "ruby": detect_ruby,
    "php": detect_php,
    "rust": detect_rust,
}


__all__ = ["detect_file", "handler_ref", "js_imports", "python_aliases", "resolve_js_specifier"]
