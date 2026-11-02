"""Turn detected routes, client calls and channel sites into graph nodes and edges.

Confidence on the resulting edges says how the far end was found:

- ``1.0``: read straight from syntax (a decorator on the function, a topic literal).
- ``0.9``: a handler named by reference and found by unique name.
- ``0.6-0.8``: an inline handler attributed to its enclosing definition, a gRPC or
  GraphQL resolver found by naming convention, a Temporal worker found by file name.
- client calls: the route-match score (see :mod:`codegraph.services.paths`), scaled
  down when the URL had to be guessed in part.

Edges are only written from and to nodes that exist, so a file whose language has no
parser yet still gets its route or channel node but no dangling ``handles`` edge;
the next index after its parser lands fills them in.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from codegraph.models import Edge, Node
from codegraph.services.channels import ChannelSite
from codegraph.services.clients import ClientCall
from codegraph.services.paths import (
    RPC_METHODS,
    PathPattern,
    compile_path,
    match_route,
    route_qname,
    split_route_qname,
)
from codegraph.services.resolve import CALLABLE_KINDS, NodeIndex
from codegraph.services.routes import RouteDef
from codegraph.services.sources import language_of

_GENERATED = re.compile(
    r"_pb2(?:_grpc)?\.pyi?$|\.pb(?:\.gw)?\.go$|_grpc\.pb\.go$|Grpc\.(?:java|kt)$|_pb\.(?:js|ts|d\.ts)$|"
    r"_grpc_pb\.(?:js|ts|d\.ts)$|\.proto$"
)


@dataclass(frozen=True, slots=True)
class RouteRow:
    route: str
    method: str
    path: str
    framework: str
    file: str
    line: int
    handler: str | None
    handler_ref: str | None


@dataclass(frozen=True, slots=True)
class ChannelRow:
    channel: str
    name: str
    broker: str
    direction: str
    node: str
    file: str
    line: int


def _language(path: str) -> str:
    labels = language_of(path)
    return labels[0] if labels is not None else "unknown"


def handler_candidates(definition: RouteDef, index: NodeIndex) -> list[tuple[Node, float, str]]:
    """``(node, confidence, how)`` for every function serving ``definition``."""
    style, ref, file, line = definition.style, definition.handler, definition.file, definition.line
    if style == "decorator":
        node = index.decorated(file, line, ref)
        if node is not None:
            exact = ref is None or node.name == ref
            return [(node, 1.0 if exact else 0.8, "decorator")]
        found = index.lookup(ref, file) if ref else None
        return [(found, 0.8, "name")] if found is not None else []
    if style == "file":
        node = index.named_in_file(file, ref) if ref else None
        if node is not None:
            return [(node, 1.0, "export")]
        module = index.modules.get(file)
        return [(module, 0.7, "module")] if module is not None else []
    if style == "grpc":
        return [(n, 0.8, "grpc") for n in _grpc_handlers(definition, index)]
    if style == "graphql":
        return [(n, 0.7, "resolver") for n in _graphql_resolvers(definition, index)]
    if ref:
        found = index.lookup(ref, file, definition.owner_hint)
        return [(found, 0.9, "name")] if found is not None else []
    owner = index.owner_node(file, line)
    return [(owner, 0.6, "inline")] if owner is not None else []


def _grpc_handlers(definition: RouteDef, index: NodeIndex) -> list[Node]:
    method = definition.handler or ""
    service = (definition.owner_hint or "").lower()
    lower = method[:1].lower() + method[1:]
    candidates = [
        n
        for name in {method, lower}
        for n in index.by_name.get(name, [])
        if n.kind in CALLABLE_KINDS and not _GENERATED.search(n.path)
    ]
    hinted = [n for n in candidates if service and service in n.qname.lower()]
    if hinted:
        return hinted
    return candidates if len(candidates) == 1 else []


def _graphql_resolvers(definition: RouteDef, index: NodeIndex) -> list[Node]:
    field = definition.handler or ""
    root = definition.path.split(".")[0]
    owner = (definition.owner_hint or root).lower()
    found: list[Node] = []
    for name in (f"resolve_{field}", f"resolve{field[:1].upper()}{field[1:]}"):
        found.extend(n for n in index.by_name.get(name, []) if n.kind in CALLABLE_KINDS)
    for node in index.by_name.get(field, []):
        parent = node.qname.rsplit(".", 2)[-2].lower() if node.qname.count(".") else ""
        if node.kind in CALLABLE_KINDS and (
            parent in {owner, root.lower()} or parent.endswith(("resolver", "resolvers"))
        ):
            found.append(node)
    unique = {n.qname: n for n in found}
    return list(unique.values())


def route_graph(
    definitions: list[RouteDef], index: NodeIndex
) -> tuple[list[Node], list[Edge], list[RouteRow]]:
    grouped: defaultdict[str, list[RouteDef]] = defaultdict(list)
    for definition in definitions:
        grouped[definition.qname].append(definition)
    nodes: list[Node] = []
    edges: dict[tuple[str, str, int], Edge] = {}
    rows: list[RouteRow] = []
    for qname, group in sorted(grouped.items()):
        first = min(group, key=lambda d: (d.file, d.line))
        frameworks = sorted({d.framework for d in group})
        nodes.append(
            Node(
                qname=qname,
                kind="route",
                path=first.file,
                line_start=first.line,
                line_end=first.line,
                language=_language(first.file),
                name=f"{first.method} {first.path}",
                signature=", ".join(frameworks),
                docstring=None,
            )
        )
        for definition in group:
            handlers = handler_candidates(definition, index)
            for node, confidence, how in handlers:
                key = (node.qname, definition.file, definition.line)
                if key in edges:
                    continue
                edges[key] = Edge(
                    src=qname,
                    dst=node.qname,
                    kind="handles",
                    path=definition.file,
                    line=definition.line,
                    confidence=confidence,
                    detail=f"{definition.framework} {how}",
                )
            rows.append(
                RouteRow(
                    route=qname,
                    method=definition.method,
                    path=definition.path,
                    framework=definition.framework,
                    file=definition.file,
                    line=definition.line,
                    handler=handlers[0][0].qname if handlers else None,
                    handler_ref=definition.handler,
                )
            )
    return nodes, list(edges.values()), rows


def route_patterns(route_nodes: list[Node]) -> list[PathPattern]:
    patterns: list[PathPattern] = []
    for node in route_nodes:
        parsed = split_route_qname(node.qname)
        if parsed is not None and parsed[0] not in RPC_METHODS:
            patterns.append(compile_path(*parsed))
    return patterns


def http_call_edges(
    calls: list[ClientCall], index: NodeIndex, patterns: list[PathPattern], known: set[str]
) -> list[Edge]:
    """One ``http_calls`` edge per call site; unmatched ones point at the would-be route."""
    edges: dict[tuple[str, str, int], Edge] = {}
    for call in calls:
        if call.decorated is not None:
            source = index.decorated(call.file, call.line, call.decorated)
        else:
            source = index.owner_node(call.file, call.line)
        if source is None:
            continue
        if call.method in RPC_METHODS:
            dst = route_qname(call.method, call.path)
            resolved = dst in known
            confidence = 0.9 if resolved else 0.5
        else:
            match = match_route(call.method, call.path, patterns)
            factor = 1.0 if call.exact else 0.9
            if match is not None:
                dst, resolved, confidence = match.qname, True, round(match.score * factor, 3)
            else:
                dst, resolved, confidence = route_qname(call.method, call.path), False, 0.5 * factor
        key = (source.qname, call.file, call.line)
        if key in edges:
            continue
        edges[key] = Edge(
            src=source.qname,
            dst=dst,
            kind="http_calls",
            path=call.file,
            line=call.line,
            resolved=resolved,
            confidence=confidence,
            detail=f"{call.library} {call.method} {call.raw}"[:200],
        )
    return list(edges.values())


def channel_graph(
    sites: list[ChannelSite], index: NodeIndex
) -> tuple[list[Node], list[Edge], list[ChannelRow]]:
    grouped: defaultdict[str, list[ChannelSite]] = defaultdict(list)
    for site in sites:
        grouped[site.name].append(site)
    nodes: list[Node] = []
    edges: dict[tuple[str, str, str, int], Edge] = {}
    rows: list[ChannelRow] = []
    for name, group in sorted(grouped.items()):
        qname = f"channel:{name}"
        first = min(group, key=lambda s: (s.file, s.line))
        nodes.append(
            Node(
                qname=qname,
                kind="channel",
                path=first.file,
                line_start=first.line,
                line_end=first.line,
                language=first.language,
                name=name,
                signature=", ".join(sorted({s.broker for s in group})),
                docstring=None,
            )
        )
        for site in group:
            node, confidence = _site_node(site, index)
            owner = node.qname if node is not None else index.owner(site.file, site.line)
            rows.append(
                ChannelRow(qname, name, site.broker, site.direction, owner, site.file, site.line)
            )
            if node is None:
                continue
            key = (node.qname, site.direction, site.file, site.line)
            edges.setdefault(
                key,
                Edge(
                    src=node.qname,
                    dst=qname,
                    kind="emits" if site.direction == "emits" else "listens_on",
                    path=site.file,
                    line=site.line,
                    confidence=confidence,
                    detail=site.broker,
                ),
            )
        if any(s.direction == "emits" for s in group) and not any(
            s.direction == "listens_on" for s in group
        ):
            implicit = _implicit_worker(name, index)
            if implicit is not None:
                edges[(implicit.qname, "listens_on", implicit.path, implicit.line_start)] = Edge(
                    src=implicit.qname,
                    dst=qname,
                    kind="listens_on",
                    path=implicit.path,
                    line=implicit.line_start,
                    confidence=0.7,
                    detail="temporal by-name",
                )
    return nodes, list(edges.values()), rows


def _site_node(site: ChannelSite, index: NodeIndex) -> tuple[Node | None, float]:
    if site.direction == "emits":
        return index.owner_node(site.file, site.line), 1.0
    if site.decorated:
        node = index.decorated(site.file, site.line, site.handler)
        if node is not None:
            return node, 1.0
    if site.handler:
        found = index.lookup(site.handler, site.file)
        if found is not None:
            return found, 0.9
    return index.owner_node(site.file, site.line), 0.8


def _implicit_worker(channel: str, index: NodeIndex) -> Node | None:
    """TypeScript Temporal workers register by file: match ``temporal:<kind>:<fn>`` by name."""
    parts = channel.split(":")
    if len(parts) != 3 or parts[0] != "temporal":
        return None
    marker = "workflow" if parts[1] == "workflow" else "activit"
    candidates = [
        n
        for n in index.by_name.get(parts[2], [])
        if n.kind in CALLABLE_KINDS and marker in n.path.lower()
    ]
    return candidates[0] if len(candidates) == 1 else None


__all__ = [
    "ChannelRow",
    "RouteRow",
    "channel_graph",
    "handler_candidates",
    "http_call_edges",
    "route_graph",
    "route_patterns",
]
