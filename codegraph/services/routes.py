"""HTTP, gRPC and GraphQL endpoints declared in source, with their mount prefixes.

Detection is split in two. Framework detectors in :mod:`codegraph.services.route_detectors`
read one file each and report :class:`RouteHit` (a route relative to its router) and
:class:`Mount` (``app.use("/api", users)``, ``include_router(..., prefix=)``,
``path("api/", include("blog.urls"))``). :func:`detect_routes` then composes the mount
chains across files so a handler registered as ``/{id}`` on a router mounted at
``/users`` under an app mounted at ``/api`` comes out as ``/api/users/{id}``.

File-system routes (Next.js ``pages/api`` and ``app/**/route.ts``, SvelteKit
``+server.ts``), gRPC services in ``.proto`` files and GraphQL root fields are handled
here because they need no call-site parsing.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from codegraph.parsers.base import module_qname_for
from codegraph.services.paths import (
    ANY,
    join_paths,
    normalize_method,
    normalize_path,
    route_qname,
)
from codegraph.services.sources import SourceFile
from codegraph.services.text import matching_close

Style = str
"""How the handler is attached: ``decorator`` (definition right below ``line``),
``call`` (``handler`` names it, or the registration is inline), ``file`` (the module
exports one function per method), ``grpc`` / ``graphql`` (resolved by naming rules)."""


@dataclass(frozen=True, slots=True)
class RouteHit:
    method: str
    path: str
    framework: str
    file: str
    line: int
    offset: int = 0
    handler: str | None = None
    style: Style = "call"
    router: str | None = None
    router_prefix: str = ""
    owner_hint: str | None = None


@dataclass(frozen=True, slots=True)
class Target:
    """What a mount points at; detectors give every plausible reading."""

    file: str | None = None
    module: str | None = None
    var: str | None = None
    function: str | None = None


@dataclass(frozen=True, slots=True)
class Mount:
    file: str
    prefix: str
    parent: str | None
    targets: tuple[Target, ...]
    replace_prefix: bool = False
    offset: int = 0


@dataclass(slots=True)
class FileRoutes:
    hits: list[RouteHit] = field(default_factory=list)
    mounts: list[Mount] = field(default_factory=list)
    global_prefixes: dict[str, str] = field(default_factory=dict)
    functions: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RouteDef:
    """A finished route: normalised path, prefixes applied, handler still a reference."""

    qname: str
    method: str
    path: str
    framework: str
    file: str
    line: int
    handler: str | None
    style: Style
    owner_hint: str | None


# -- file-system routes -------------------------------------------------------

_NEXT_PAGES_API = re.compile(r"(?:^|/)pages/(api/.+)\.(?:js|jsx|ts|tsx)$")
_NEXT_APP_ROUTE = re.compile(r"(?:^|/)app/(?:(.*)/)?route\.(?:js|ts)$")
_SVELTEKIT_SERVER = re.compile(r"(?:^|/)src/routes/(?:(.*)/)?\+server\.(?:js|ts)$")
_EXPORTED_METHOD = re.compile(
    r"^\s*export\s+(?:async\s+)?(?:function\s+|const\s+|let\s+)(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b",
    re.M,
)
_DEFAULT_EXPORT = re.compile(r"^\s*export\s+default\s+(?:async\s+)?function\s+(\w+)", re.M)


def _file_segments(raw: str) -> list[str] | None:
    segments: list[str] = []
    for segment in raw.split("/"):
        if not segment or segment == "index":
            continue
        if segment.startswith("(") and segment.endswith(")"):
            continue  # route group
        if segment.startswith("@"):
            continue  # parallel route slot
        if segment.startswith("_"):
            return None  # private folder
        segments.append(segment)
    return segments


def file_routes(source: SourceFile) -> list[RouteHit]:
    if source.family != "js":
        return []
    hits: list[RouteHit] = []
    pages = _NEXT_PAGES_API.search(source.path)
    if pages:
        segments = _file_segments(pages.group(1))
        if segments is not None:
            default = _DEFAULT_EXPORT.search(source.text)
            hits.append(
                RouteHit(
                    method=ANY,
                    path=join_paths(*segments),
                    framework="nextjs",
                    file=source.path,
                    line=source.line(default.start()) if default else 1,
                    handler=default.group(1) if default else None,
                    style="file",
                )
            )
        return hits
    for pattern, framework in ((_NEXT_APP_ROUTE, "nextjs"), (_SVELTEKIT_SERVER, "sveltekit")):
        found = pattern.search(source.path)
        if found is None:
            continue
        segments = _file_segments(found.group(1) or "")
        if segments is None:
            return hits
        for match in _EXPORTED_METHOD.finditer(source.text):
            hits.append(
                RouteHit(
                    method=match.group(1),
                    path=join_paths(*segments),
                    framework=framework,
                    file=source.path,
                    line=source.line(match.start()),
                    handler=match.group(1),
                    style="file",
                )
            )
    return hits


# -- gRPC and GraphQL schemas ---------------------------------------------------

_PROTO_SERVICE = re.compile(r"\bservice\s+(\w+)\s*\{")
_PROTO_RPC = re.compile(r"\brpc\s+(\w+)\s*\(")


def grpc_routes(source: SourceFile) -> list[RouteHit]:
    if source.family != "proto":
        return []
    hits: list[RouteHit] = []
    for service in _PROTO_SERVICE.finditer(source.text):
        if source.in_comment(service.start()):
            continue
        close = matching_close(source.text, service.end() - 1, "proto")
        if close is None:
            continue
        for rpc in _PROTO_RPC.finditer(source.text, service.end(), close):
            if source.in_comment(rpc.start()):
                continue
            hits.append(
                RouteHit(
                    method="GRPC",
                    path=f"{service.group(1)}/{rpc.group(1)}",
                    framework="grpc",
                    file=source.path,
                    line=source.line(rpc.start()),
                    offset=rpc.start(),
                    handler=rpc.group(1),
                    style="grpc",
                    owner_hint=service.group(1),
                )
            )
    return hits


_GQL_TEMPLATE = re.compile(r"(?:\bgql|\bgraphql|buildSchema\s*\(|typeDefs\s*[:=])\s*`")
_GQL_SCHEMA = re.compile(r"\bschema\s*\{([^}]*)\}")
_GQL_ROOT_ENTRY = re.compile(r"\b(query|mutation|subscription)\s*:\s*(\w+)")
_GQL_TYPE = re.compile(r"(?:\bextend\s+)?\btype\s+(\w+)\b[^{}]*\{")
_GQL_FIELD = re.compile(
    r"\"\"\"[\s\S]*?\"\"\"|\"(?:\\.|[^\"\\\n])*\"|#[^\n]*|[(){}]|(?<![@\w])([A-Za-z_]\w*)\s*(?=[(:])"
)
_ROOTS = {"query": "Query", "mutation": "Mutation", "subscription": "Subscription"}


def _graphql_regions(source: SourceFile) -> list[tuple[int, int]]:
    if source.family == "graphql":
        return [(0, len(source.text))]
    if source.family != "js":
        return []
    regions: list[tuple[int, int]] = []
    for match in _GQL_TEMPLATE.finditer(source.text):
        start = match.end()
        end = source.text.find("`", start)
        if end != -1 and "type " in source.text[start:end]:
            regions.append((start, end))
    return regions


def graphql_routes(source: SourceFile) -> list[RouteHit]:
    hits: list[RouteHit] = []
    for start, end in _graphql_regions(source):
        text = source.text[start:end]
        roots = {v: v for v in _ROOTS.values()}
        schema = _GQL_SCHEMA.search(text)
        if schema:
            for entry in _GQL_ROOT_ENTRY.finditer(schema.group(1)):
                roots[entry.group(2)] = _ROOTS[entry.group(1)]
        for type_match in _GQL_TYPE.finditer(text):
            root = roots.get(type_match.group(1))
            if root is None:
                continue
            close = matching_close(text, type_match.end() - 1, "graphql")
            if close is None:
                continue
            for name, offset in _graphql_fields(text, type_match.end(), close):
                hits.append(
                    RouteHit(
                        method="GRAPHQL",
                        path=f"{root}.{name}",
                        framework="graphql",
                        file=source.path,
                        line=source.line(start + offset),
                        offset=start + offset,
                        handler=name,
                        style="graphql",
                        owner_hint=type_match.group(1),
                    )
                )
    return hits


def _graphql_fields(text: str, start: int, end: int) -> list[tuple[str, int]]:
    fields: list[tuple[str, int]] = []
    depth = 0
    for match in _GQL_FIELD.finditer(text, start, end):
        token = match.group(0)
        if token in "({":
            depth += 1
        elif token in ")}":
            depth -= 1
        elif match.group(1) and depth == 0:
            fields.append((match.group(1), match.start(1)))
    return fields


# -- composition ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Span:
    file: str
    start: int
    end: int


class _MountGraph:
    """Resolves mount targets to files and walks prefix chains."""

    def __init__(self, sources: Iterable[SourceFile], results: dict[str, FileRoutes]) -> None:
        self.files = {s.path for s in sources}
        self.by_module: defaultdict[str, list[str]] = defaultdict(list)
        for path in self.files:
            self.by_module[module_qname_for(path)].append(path)
        self.functions: defaultdict[str, list[_Span]] = defaultdict(list)
        for path, result in results.items():
            for name, (start, end) in result.functions.items():
                self.functions[name].append(_Span(path, start, end))
        self.mounts = [m for r in results.values() for m in r.mounts]
        self.resolved: list[tuple[Mount, Target]] = []
        for mount in self.mounts:
            target = self._resolve(mount)
            if target is not None:
                self.resolved.append((mount, target))

    def _module_file(self, module: str) -> str | None:
        exact = self.by_module.get(module)
        if exact and len(exact) == 1:
            return exact[0]
        suffix = "." + module
        matches = [p for m, paths in self.by_module.items() if m.endswith(suffix) for p in paths]
        return matches[0] if len(matches) == 1 else None

    def _resolve(self, mount: Mount) -> Target | None:
        for target in mount.targets:
            if target.function is not None:
                spans = self.functions.get(target.function, [])
                if target.file is not None:
                    spans = [s for s in spans if s.file == target.file]
                if len(spans) == 1:
                    return target
                continue
            path = target.file
            if path is None and target.module is not None:
                path = self._module_file(target.module)
            if path is not None and path in self.files:
                return replace(target, file=path, module=None)
        return None

    def _applies(self, target: Target, hit: RouteHit) -> bool:
        if target.function is not None:
            spans = [s for s in self.functions.get(target.function, []) if s.file == hit.file]
            return any(s.start <= hit.offset <= s.end for s in spans)
        if target.file != hit.file:
            return False
        return target.var is None or target.var == hit.router

    def prefixes(self, hit: RouteHit) -> list[tuple[str, bool]]:
        """``(prefix, replaces_router_prefix)`` for every mount chain reaching ``hit``."""
        return self._chains(hit, depth=0, seen=frozenset())

    def _chains(self, hit: RouteHit, depth: int, seen: frozenset[int]) -> list[tuple[str, bool]]:
        mounts = [
            (index, mount)
            for index, (mount, target) in enumerate(self.resolved)
            if index not in seen and self._applies(target, hit)
        ]
        if not mounts or depth > 8:
            return [("", False)]
        chains: list[tuple[str, bool]] = []
        for index, mount in mounts:
            parent = RouteHit(
                method=hit.method,
                path="",
                framework=hit.framework,
                file=mount.file,
                line=0,
                offset=mount.offset,
                router=mount.parent,
            )
            for outer, _ in self._chains(parent, depth + 1, seen | {index}):
                chains.append((join_paths(outer, mount.prefix), mount.replace_prefix))
        return chains


def detect_routes(sources: list[SourceFile]) -> list[RouteDef]:
    """Every route in ``sources`` with mount prefixes applied, deduplicated."""
    from codegraph.services.route_detectors import detect_file

    results: dict[str, FileRoutes] = {}
    for source in sources:
        if source.is_test:
            continue
        result = detect_file(source)
        result.hits.extend(file_routes(source))
        result.hits.extend(grpc_routes(source))
        result.hits.extend(graphql_routes(source))
        if result.hits or result.mounts or result.global_prefixes:
            results[source.path] = result
    graph = _MountGraph(sources, results)
    global_prefixes: dict[str, str] = {}
    for result in results.values():
        global_prefixes.update(result.global_prefixes)

    seen: set[tuple[str, str, int]] = set()
    routes: list[RouteDef] = []
    for result in results.values():
        for hit in result.hits:
            for definition in _finish(hit, graph, global_prefixes):
                key = (definition.qname, definition.file, definition.line)
                if key not in seen:
                    seen.add(key)
                    routes.append(definition)
    return sorted(routes, key=lambda r: (r.qname, r.file, r.line))


def _finish(hit: RouteHit, graph: _MountGraph, global_prefixes: dict[str, str]) -> list[RouteDef]:
    if hit.method in {"GRPC", "GRAPHQL"}:
        return [_definition(hit, hit.method, hit.path)]
    method = normalize_method(hit.method)
    definitions: list[RouteDef] = []
    for prefix, replaces in graph.prefixes(hit):
        router_prefix = "" if replaces else hit.router_prefix
        full = join_paths(global_prefixes.get(hit.framework, ""), prefix, router_prefix, hit.path)
        definitions.append(_definition(hit, method, normalize_path(full)))
    return definitions


def _definition(hit: RouteHit, method: str, path: str) -> RouteDef:
    return RouteDef(
        qname=route_qname(method, path),
        method=method,
        path=path,
        framework=hit.framework,
        file=hit.file,
        line=hit.line,
        handler=hit.handler,
        style=hit.style,
        owner_hint=hit.owner_hint,
    )


__all__ = [
    "FileRoutes",
    "Mount",
    "RouteDef",
    "RouteHit",
    "Target",
    "detect_routes",
    "file_routes",
    "graphql_routes",
    "grpc_routes",
]
