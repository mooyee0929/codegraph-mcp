"""One spelling for every HTTP path, and a scored matcher between them.

Frameworks spell a path parameter half a dozen ways (``:id``, ``{id}``, ``<int:id>``,
``[id]``, ``(?P<id>\\d+)``) and client code adds more (``${user.id}``, ``%s``, an
unknown variable). :func:`normalize_path` folds all of them to ``{name}`` so route
nodes get stable qnames such as ``route:GET /users/{id}``; catch-alls become ``{*}``.

:func:`match_route` then scores a client call against every known route. A static
segment must match exactly; a route parameter accepts anything; a client placeholder
can only stand in for a route parameter at a penalty, because the reader does not know
what value it carries. Leading segments may be skipped on either side (an API gateway
prefix or an axios ``baseURL``) at a further penalty.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

ANY = "ANY"
HTTP_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
RPC_METHODS = frozenset({"GRPC", "GRAPHQL"})

_SCHEME_HOST = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:)?//[^/?#]*")
_REGEX_GROUP = re.compile(r"\(\?P<(\w+)>[^)]*\)")
_UNNAMED_GROUP = re.compile(r"\((?!\?)[^)]*\)")
_TEMPLATE = re.compile(r"\$\{([^}]*)\}|\#\{([^}]*)\}")
_DOLLAR_VAR = re.compile(r"\$([A-Za-z_]\w*)")
_PRINTF = re.compile(r"%[-+ #0]*\d*(?:\.\d+)?[sdvfiqx]")
_IDENT = re.compile(r"[A-Za-z_]\w*")

_SEGMENT_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\[\[?\.\.\.(\w+)\]\]?$"), "{*}"),
    (re.compile(r"^\[(\w+)\]$"), "{\\1}"),
    (re.compile(r"^\{\*{1,2}\w*[^}]*\}$|^\{\w+\.\.\.\}$|^\*{1,2}\w*$"), "{*}"),
    (re.compile(r"^\{(\w+)[^}]*\}$"), "{\\1}"),
    (re.compile(r"^:(\w+)(?:\(.*\))?[?*+]?$"), "{\\1}"),
    (re.compile(r"^<(?:[\w.]+:)?(\w+)>$"), "{\\1}"),
    (re.compile(r"^\{\}$"), "{param}"),
)


def normalize_method(raw: str | None) -> str:
    if not raw:
        return ANY
    method = raw.strip().strip("'\"`:").upper().rsplit(".", 1)[-1].removeprefix("METHOD")
    if method in {"ALL", "ANY", "*", "MATCH"}:
        return ANY
    if method == "DEL":
        return "DELETE"
    return method if method in HTTP_METHODS or method in RPC_METHODS else ANY


def normalize_path(raw: str) -> str:
    """Canonical ``/a/{param}/b`` form of a framework route or a client URL."""
    text = raw.strip()
    text = _SCHEME_HOST.sub("", text, count=1)
    text = text.replace("(.:format)", "")
    is_regex = "^" in text or text.endswith("$") or "(?P<" in text
    text = _REGEX_GROUP.sub(lambda m: "{" + m.group(1) + "}", text)
    if is_regex:
        text = _UNNAMED_GROUP.sub("{param}", text.replace("^", ""))
        text = re.sub(r"\$(?=/|$)", "", text).replace("\\", "").replace("?", "")
    text = _TEMPLATE.sub(lambda m: "{" + _param_name(m.group(1) or m.group(2) or "") + "}", text)
    text = _DOLLAR_VAR.sub(lambda m: "{" + m.group(1) + "}", text)
    text = _PRINTF.sub("{param}", text)
    text = _cut_query(text)
    segments = [_normalize_segment(s) for s in text.split("/") if s and s != "."]
    return "/" + "/".join(segments)


def _cut_query(text: str) -> str:
    depth = 0
    for index, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        elif char in "?#" and depth == 0:
            return text[:index]
    return text


def _param_name(expr: str) -> str:
    names = _IDENT.findall(expr)
    return names[-1] if names else "param"


def _normalize_segment(segment: str) -> str:
    for pattern, replacement in _SEGMENT_RULES:
        if pattern.match(segment):
            return pattern.sub(replacement, segment)
    return segment


def join_paths(*parts: str) -> str:
    """Join prefix and route pieces with exactly one slash between them."""
    pieces = [p.strip("/") for p in parts if p and p.strip("/")]
    return "/" + "/".join(pieces)


def route_qname(method: str, path: str) -> str:
    return f"route:{method} {path}"


def split_route_qname(qname: str) -> tuple[str, str] | None:
    """``route:GET /a`` -> ``("GET", "/a")``; None for anything else."""
    if not qname.startswith("route:"):
        return None
    method, _, path = qname[len("route:") :].partition(" ")
    if not path:
        return None
    return method, path


# -- matching -------------------------------------------------------------

_STATIC, _PARAM, _CATCHALL, _MIXED = "static", "param", "catchall", "mixed"


@dataclass(frozen=True, slots=True)
class Segment:
    kind: str
    text: str
    regex: re.Pattern[str] | None = None


@dataclass(frozen=True, slots=True)
class PathPattern:
    method: str
    path: str
    segments: tuple[Segment, ...]

    @property
    def static_count(self) -> int:
        return sum(1 for s in self.segments if s.kind == _STATIC)


def compile_path(method: str, path: str) -> PathPattern:
    segments: list[Segment] = []
    for raw in path.split("/"):
        if not raw:
            continue
        if raw == "{*}":
            segments.append(Segment(_CATCHALL, raw))
        elif re.fullmatch(r"\{[^{}]+\}", raw):
            segments.append(Segment(_PARAM, raw))
        elif "{" in raw:
            pieces = re.split(r"\{[^{}]*\}", raw)
            regex = re.compile("^" + ".+".join(re.escape(p) for p in pieces) + "$")
            segments.append(Segment(_MIXED, raw, regex))
        else:
            segments.append(Segment(_STATIC, raw))
    return PathPattern(method=method, path=path, segments=tuple(segments))


@dataclass(frozen=True, slots=True)
class RouteMatch:
    qname: str
    score: float


MIN_MATCH_SCORE = 0.5
_SKIP_PENALTY = 0.2


def segments_score(call: tuple[Segment, ...], route: tuple[Segment, ...]) -> float | None:
    """Score how well concrete-ish ``call`` segments fit ``route``; None for no fit."""
    penalty = 0.0
    for index, want in enumerate(route):
        if want.kind == _CATCHALL:
            return max(0.0, 1.0 - penalty - 0.2)
        if index >= len(call):
            return None
        have = call[index]
        if want.kind == _STATIC:
            if have.kind == _STATIC:
                if have.text == want.text:
                    continue
                if have.text.lower() == want.text.lower():
                    penalty += 0.1
                    continue
                return None
            if have.kind in {_PARAM, _MIXED}:
                penalty += 0.3
                continue
            return None
        if want.kind == _PARAM:
            penalty += 0.02 if have.kind == _STATIC else 0.0
            continue
        if want.kind == _MIXED and want.regex is not None:
            if have.kind == _STATIC and want.regex.match(have.text):
                penalty += 0.05
                continue
            if have.kind != _STATIC:
                penalty += 0.2
                continue
            return None
    if len(call) != len(route):
        return None
    return max(0.0, 1.0 - penalty)


def method_penalty(call_method: str, route_method: str) -> float | None:
    if call_method == route_method:
        return 0.0
    if route_method == ANY:
        return 0.05
    if call_method == ANY:
        return 0.1
    return None


def score_route(call: PathPattern, route: PathPattern) -> float | None:
    method_cost = method_penalty(call.method, route.method)
    if method_cost is None:
        return None
    best: float | None = None
    for skip_call in range(0, 3):
        for skip_route in range(0, 3):
            if skip_call and skip_route:
                continue
            call_segments = call.segments[skip_call:]
            route_segments = route.segments[skip_route:]
            if (skip_call or skip_route) and not any(s.kind == _STATIC for s in route_segments):
                continue
            if skip_route and any(s.kind != _STATIC for s in route.segments[:skip_route]):
                continue
            score = segments_score(call_segments, route_segments)
            if score is None:
                continue
            score -= _SKIP_PENALTY * (skip_call + skip_route) + method_cost
            if route.static_count == 0 and route.segments:
                score = min(score, 0.5)
            if best is None or score > best:
                best = score
    return best


def match_route(
    method: str, path: str, routes: Iterable[PathPattern], min_score: float = MIN_MATCH_SCORE
) -> RouteMatch | None:
    """Best route for a call, or None when nothing scores at least ``min_score``.

    Ties go to the route with more static segments, then to the lexically first qname,
    so the answer is deterministic.
    """
    call = compile_path(method, path)
    best: tuple[float, int, str] | None = None
    for route in routes:
        if route.method in RPC_METHODS:
            continue
        score = score_route(call, route)
        if score is None or score < min_score:
            continue
        key = (round(score, 4), route.static_count, route_qname(route.method, route.path))
        if best is None or _better(key, best):
            best = key
    if best is None:
        return None
    return RouteMatch(qname=best[2], score=round(best[0], 3))


def _better(key: tuple[float, int, str], best: tuple[float, int, str]) -> bool:
    """Higher score, then more static segments, then the lexically smaller qname."""
    return (key[0], key[1], _invert(key[2])) > (best[0], best[1], _invert(best[2]))


def _invert(text: str) -> tuple[int, ...]:
    return tuple(-ord(c) for c in text) + (0,)


__all__ = [
    "ANY",
    "HTTP_METHODS",
    "MIN_MATCH_SCORE",
    "PathPattern",
    "RouteMatch",
    "compile_path",
    "join_paths",
    "match_route",
    "normalize_method",
    "normalize_path",
    "route_qname",
    "score_route",
    "split_route_qname",
]
