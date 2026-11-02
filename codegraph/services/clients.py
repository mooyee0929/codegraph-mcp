"""Outgoing HTTP, gRPC and GraphQL calls found in source text.

A rule pairs a call pattern with where its URL and method live: ``requests.get(url)``
names the method, ``http.NewRequest("POST", url, body)`` passes it as an argument, and
``fetch(url, {method: "PUT"})`` hides it in an options object. URLs are recovered with
:func:`codegraph.services.text.url_expression`, so literals, same-file constants,
template strings and ``BASE + "/path"`` concatenations all work.

*Strong* rules match a library by name (``axios.get``, ``HttpClient.GetAsync``) and
accept any recovered URL. *Weak* rules match a method call on a receiver that is only
named like a client (``self.session.get``, ``apiClient.post``); they insist on a URL
that starts with ``/`` or a scheme so ``cache.get("key")`` stays out.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from codegraph.services.paths import ANY, normalize_method, normalize_path
from codegraph.services.route_detectors import METHOD_HEADER, annotation_groups
from codegraph.services.sources import SourceFile
from codegraph.services.text import (
    Arg,
    matching_close,
    module_constants,
    pick_arg,
    split_args,
    split_bare_args,
    url_expression,
)


@dataclass(frozen=True, slots=True)
class ClientCall:
    method: str
    path: str
    raw: str
    library: str
    file: str
    line: int
    exact: bool
    decorated: str | None = None
    """For declarative clients (Feign, Retrofit): the annotated method's name."""


@dataclass(frozen=True, slots=True)
class _Rule:
    pattern: re.Pattern[str]
    library: str
    url_arg: int = 0
    method: str | None = None
    method_arg: int | None = None
    options_arg: int | None = None
    strong: bool = True
    receiver: re.Pattern[str] | None = None


_VERBS = r"get|post|put|patch|delete|head|options"
_CLIENTISH = re.compile(
    r"(?i)(?:^|[.\->$@])(?:\w*client|\w*session|http\w*|api|\w*api|conn\w*|agent|instance|"
    r"axios\w*|request|faraday|guzzle|s|c)$"
)
_TEMPLATEISH = re.compile(r"(?i)rest\w*|\w*template")

_RULES: dict[str, tuple[_Rule, ...]] = {
    "python": (
        _Rule(
            re.compile(rf"(?<![\w.])(?P<lib>requests|httpx)\s*\.\s*(?P<verb>{_VERBS})\s*\("),
            "requests",
        ),
        _Rule(
            re.compile(r"(?<![\w.])(?P<lib>requests|httpx)\s*\.\s*request\s*\("),
            "requests",
            url_arg=1,
            method_arg=0,
        ),
        _Rule(
            re.compile(r"(?<![\w])(?:urllib\s*\.\s*request\s*\.\s*)?urlopen\s*\("),
            "urllib",
            method="GET",
        ),
        _Rule(
            re.compile(rf"(?<![\w.])(?P<recv>[A-Za-z_][\w.]*?)\s*\.\s*(?P<verb>{_VERBS})\s*\("),
            "http-client",
            strong=False,
            receiver=_CLIENTISH,
        ),
    ),
    "js": (
        _Rule(
            re.compile(
                r"(?<![\w$.])(?:(?:window|globalThis|self)\s*\.\s*)?(?P<lib>fetch|\$fetch|ofetch|useFetch|useLazyFetch)\s*\("
            ),
            "fetch",
            method="GET",
            options_arg=1,
        ),
        _Rule(
            re.compile(
                rf"(?<![\w$.])(?P<lib>axios|ky|got|superagent|\$http)\s*\.\s*(?P<verb>{_VERBS})\s*(?:<[^>()]*>)?\s*\("
            ),
            "axios",
        ),
        _Rule(
            re.compile(r"(?<![\w$.])(?P<lib>axios)(?:\s*\.\s*request)?\s*\("),
            "axios",
            method="GET",
            url_arg=-1,
            options_arg=0,
        ),
        _Rule(re.compile(r"(?<![\w$.])\$\s*\.\s*(?P<verb>get|post|getJSON)\s*\("), "jquery"),
        _Rule(
            re.compile(r"(?<![\w$.])\$\s*\.\s*ajax\s*\("),
            "jquery",
            method="GET",
            url_arg=-1,
            options_arg=0,
        ),
        _Rule(
            re.compile(
                r"(?<![\w$.])(?:request|supertest)\s*\(\s*[\w$.]+\s*\)\s*\.\s*(?P<verb>get|post|put|patch|delete)\s*\("
            ),
            "supertest",
        ),
        _Rule(
            re.compile(
                rf"(?<![\w$])(?:this\s*\.\s*)?(?P<recv>[A-Za-z_$][\w$]*(?:\s*\.\s*[A-Za-z_$][\w$]*)*?)\s*\.\s*"
                rf"(?P<verb>{_VERBS})\s*(?:<[^>()]*>)?\s*\("
            ),
            "http-client",
            strong=False,
            receiver=_CLIENTISH,
        ),
    ),
    "go": (
        _Rule(
            re.compile(r"(?<![\w.])http\s*\.\s*(?P<verb>Get|Head|Post|PostForm)\s*\("), "net/http"
        ),
        _Rule(
            re.compile(r"(?<![\w.])http\s*\.\s*NewRequest\s*\("),
            "net/http",
            url_arg=1,
            method_arg=0,
        ),
        _Rule(
            re.compile(r"(?<![\w.])http\s*\.\s*NewRequestWithContext\s*\("),
            "net/http",
            url_arg=2,
            method_arg=1,
        ),
        _Rule(
            re.compile(
                r"\.\s*R\s*\(\s*\)(?:\s*\.\s*\w+\s*\((?:[^()]|\([^()]*\))*\))*\s*\.\s*(?P<verb>Get|Post|Put|Patch|Delete|Head|Options)\s*\("
            ),
            "resty",
        ),
        _Rule(
            re.compile(r"(?<![\w.])(?P<recv>[A-Za-z_][\w.]*?)\s*\.\s*(?P<verb>Get|Post|Head)\s*\("),
            "net/http",
            strong=False,
            receiver=re.compile(r"(?i)client$"),
        ),
    ),
    "jvm": (
        _Rule(
            re.compile(
                r"(?<![\w.])(?P<recv>[A-Za-z_][\w.]*?)\s*\.\s*(?P<verb>getForObject|getForEntity|postForObject|"
                r"postForEntity|postForLocation|patchForObject|put|delete|headForHeaders|optionsForAllow)\s*\("
            ),
            "resttemplate",
            receiver=_TEMPLATEISH,
        ),
        _Rule(
            re.compile(r"(?<![\w.])(?P<recv>[A-Za-z_][\w.]*?)\s*\.\s*exchange\s*\("),
            "resttemplate",
            method_arg=1,
            receiver=_TEMPLATEISH,
        ),
        _Rule(re.compile(rf"\.\s*(?P<verb>{_VERBS})\s*\(\s*\)\s*\.\s*uri\s*\("), "webclient"),
        _Rule(
            re.compile(
                r"\bperform\s*\(\s*(?:MockMvcRequestBuilders\s*\.\s*)?(?P<verb>get|post|put|patch|delete)\s*\("
            ),
            "mockmvc",
        ),
        _Rule(
            re.compile(rf"(?<![\w.])(?P<recv>[A-Za-z_][\w.]*?)\s*\.\s*(?P<verb>{_VERBS})\s*\("),
            "http-client",
            strong=False,
            receiver=re.compile(r"(?i)client$"),
        ),
    ),
    "csharp": (
        _Rule(
            re.compile(
                r"(?<![\w])(?P<recv>[A-Za-z_][\w.]*?)\s*\.\s*(?P<verb>GetAsync|GetStringAsync|GetStreamAsync|"
                r"GetByteArrayAsync|GetFromJsonAsync|PostAsync|PostAsJsonAsync|PutAsync|PutAsJsonAsync|PatchAsync|"
                r"PatchAsJsonAsync|DeleteAsync|DeleteFromJsonAsync)\s*(?:<[^>()]*>)?\s*\("
            ),
            "httpclient",
        ),
        _Rule(
            re.compile(r"\bnew\s+HttpRequestMessage\s*\("), "httpclient", url_arg=1, method_arg=0
        ),
        _Rule(re.compile(r"\bnew\s+RestRequest\s*\("), "restsharp", method="GET", method_arg=1),
    ),
    "ruby": (
        _Rule(
            re.compile(
                r"(?<![\w:])(?P<lib>HTTParty|Faraday|RestClient|Typhoeus|HTTPX|HTTP)\s*\.\s*(?P<verb>get|post|put|patch|delete|head)\b"
            ),
            "ruby-http",
        ),
        _Rule(re.compile(r"\bNet::HTTP\s*\.\s*(?P<verb>get|get_response|post_form)\b"), "net-http"),
        _Rule(
            re.compile(r"\bNet::HTTP::(?P<verb>Get|Post|Put|Patch|Delete)\s*\.\s*new\b"), "net-http"
        ),
        _Rule(
            re.compile(
                r"(?<![\w:])(?P<recv>@?[A-Za-z_][\w.]*?)\s*\.\s*(?P<verb>get|post|put|patch|delete)\b"
            ),
            "ruby-http",
            strong=False,
            receiver=_CLIENTISH,
        ),
    ),
    "php": (
        _Rule(
            re.compile(
                r"\bHttp::(?:\w+\s*\((?:[^()]|\([^()]*\))*\)\s*->\s*)*(?P<verb>get|post|put|patch|delete|head)\s*\("
            ),
            "laravel-http",
        ),
        _Rule(
            re.compile(r"(?P<recv>\$\w+(?:->\w+)*)\s*->\s*request\s*\("),
            "guzzle",
            url_arg=1,
            method_arg=0,
            receiver=_CLIENTISH,
        ),
        _Rule(
            re.compile(
                r"(?P<recv>\$\w+(?:->\w+)*)\s*->\s*(?P<verb>get|post|put|patch|delete|head)\s*\("
            ),
            "guzzle",
            strong=False,
            receiver=_CLIENTISH,
        ),
    ),
    "rust": (
        _Rule(re.compile(r"\breqwest\s*::\s*(?P<verb>get)\s*\("), "reqwest"),
        _Rule(
            re.compile(
                r"(?<![\w.])(?P<recv>[A-Za-z_][\w.]*?)\s*\.\s*(?P<verb>get|post|put|patch|delete|head)\s*\("
            ),
            "reqwest",
            strong=False,
            receiver=re.compile(r"(?i)client$"),
        ),
    ),
}

_STATIC_ASSET = re.compile(
    r"\.(?:png|jpe?g|gif|svg|ico|css|js|mjs|map|woff2?|ttf|eot|webp|mp4|html?)$", re.I
)
_SERVER_RECEIVERS = re.compile(
    r"(?:const|let|var)\s+([\w$]+)\s*=\s*(?:express|[Ff]astify|new\s+(?:Koa|Hono|Router))\b"
)


def _verb_method(verb: str | None) -> str | None:
    if not verb:
        return None
    lowered = verb.lower()
    for method in ("get", "post", "put", "patch", "delete", "head", "options"):
        if lowered.startswith(method):
            return method.upper()
    return None


def _call_args(source: SourceFile, match: re.Match[str]) -> list[Arg] | None:
    text = source.text
    end = match.end()
    if text[end - 1] == "(":
        return split_args(text, end - 1, source.family)
    paren = re.match(r"[ \t]*\(", text[end : end + 20])
    if paren is not None:
        return split_args(text, end + paren.end() - 1, source.family)
    return split_bare_args(text, end, source.family)


def _option(obj: str, names: tuple[str, ...]) -> str | None:
    for name in names:
        match = re.search(rf"(?<![\w$]){name}\s*:\s*", obj)
        if match is None:
            continue
        rest = obj[match.end() :]
        depth = 0
        for index, char in enumerate(rest):
            if char in "([{":
                depth += 1
            elif char in ")]}":
                if depth == 0:
                    return rest[:index].strip()
                depth -= 1
            elif char == "," and depth == 0:
                return rest[:index].strip()
        return rest.strip()
    return None


def detect_client_calls(source: SourceFile) -> list[ClientCall]:
    """Outgoing HTTP calls in one file (gRPC and GraphQL are separate)."""
    rules = _RULES.get(source.family, ())
    if not rules:
        return []
    constants = module_constants(source.text, source.family)
    servers = set(_SERVER_RECEIVERS.findall(source.text)) if source.family == "js" else set()
    calls: dict[int, ClientCall] = {}
    for rule in rules:
        for match in rule.pattern.finditer(source.text):
            if match.start() in calls or source.in_comment(match.start()):
                continue
            call = _apply(rule, match, source, constants, servers)
            if call is not None:
                calls[match.start()] = call
    found = list(calls.values())
    if source.family == "jvm":
        found.extend(_jvm_builders(source, constants))
        found.extend(_declarative_clients(source, constants))
    return sorted(found, key=lambda c: (c.line, c.path))


def _apply(
    rule: _Rule,
    match: re.Match[str],
    source: SourceFile,
    constants: dict[str, str],
    servers: set[str],
) -> ClientCall | None:
    groups = match.groupdict()
    receiver = groups.get("recv")
    if rule.receiver is not None and receiver is not None:
        cleaned = re.sub(r"\s+", "", receiver)
        if not rule.receiver.search(cleaned) or cleaned.split(".")[-1] in servers:
            return None
    args = _call_args(source, match)
    if args is None:
        return None
    if (
        not rule.strong
        and args
        and len(args) > 1
        and re.search(r"=>|^\s*(?:async\s+)?function\b", args[-1].text)
    ):
        return None  # a server route registration, not a client call
    options = None
    if rule.options_arg is not None:
        option_arg = pick_arg(args, rule.options_arg)
        if option_arg is not None and option_arg.value.startswith("{"):
            options = option_arg.value
    url_text: str | None
    if rule.url_arg < 0:
        url_text = _option(options or "", ("url",))
    else:
        url_arg = pick_arg(args, rule.url_arg, ("url",))
        url_text = url_arg.value if url_arg is not None else None
    if url_text is None:
        return None
    url = url_expression(url_text, source.family, constants)
    if url is None or not _acceptable(url.text, rule.strong):
        return None
    method = _verb_method(groups.get("verb"))
    if method is None and rule.method_arg is not None:
        method_arg = pick_arg(args, rule.method_arg, ("method",))
        if method_arg is not None:
            literal = url_expression(method_arg.value, source.family, constants, client=False)
            method = normalize_method(literal.text if literal else method_arg.value)
    if method is None and options is not None:
        option = _option(options, ("method", "type"))
        if option is not None:
            method = normalize_method(option)
    method = method or rule.method or ANY
    return ClientCall(
        method=method,
        path=normalize_path(url.text),
        raw=url.text,
        library=groups.get("lib") or rule.library,
        file=source.path,
        line=source.line(match.start()),
        exact=url.exact,
    )


def _acceptable(url: str, strong: bool) -> bool:
    text = url.strip()
    if (
        not text
        or text.startswith("{param}")
        or " " in text
        or _STATIC_ASSET.search(text.split("?")[0])
    ):
        return False
    if re.match(r"^(?:https?:)?//|^/", text):
        return True
    return strong and bool(re.match(r"^[\w.{}$-]+(?:/|$)", text)) and "." not in text.split("/")[0]


_JVM_BUILDER = re.compile(r"HttpRequest\s*\.\s*newBuilder\s*\(|\bRequest\s*\.\s*Builder\s*\(\s*\)")
_JVM_BUILDER_URL = re.compile(r"\.\s*(?:uri|url)\s*\(")
_JVM_BUILDER_METHOD = re.compile(
    r"\.\s*(GET|POST|PUT|DELETE|get|post|put|delete|patch)\s*\(|\.\s*method\s*\(\s*\"(\w+)\""
)


def _jvm_builders(source: SourceFile, constants: dict[str, str]) -> Iterator[ClientCall]:
    """Java 11 ``HttpRequest.newBuilder()...`` and OkHttp ``Request.Builder()...`` chains."""
    text = source.text
    for match in _JVM_BUILDER.finditer(text):
        if source.in_comment(match.start()):
            continue
        end = text.find(".build()", match.end())
        region_end = end if end != -1 and end - match.end() < 600 else match.end() + 600
        url_text: str | None = None
        builder_open = text.find("(", match.start())
        inline = split_args(text, builder_open, "jvm") if builder_open != -1 else None
        if inline:
            url_text = inline[0].value
        url_match = _JVM_BUILDER_URL.search(text, match.end(), region_end)
        if url_match is not None:
            url_args = split_args(text, url_match.end() - 1, "jvm")
            if url_args:
                url_text = url_args[0].value
        if url_text is None:
            continue
        url = url_expression(url_text, "jvm", constants)
        if url is None or not _acceptable(url.text, strong=True):
            continue
        method_match = _JVM_BUILDER_METHOD.search(text, match.end(), region_end)
        method = "GET"
        if method_match is not None:
            method = normalize_method(method_match.group(1) or method_match.group(2))
        yield ClientCall(
            method=method,
            path=normalize_path(url.text),
            raw=url.text,
            library="java-httpclient" if "HttpRequest" in match.group(0) else "okhttp",
            file=source.path,
            line=source.line(match.start()),
            exact=url.exact,
        )


_RETROFIT_VERBS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"})
_FEIGN_VERBS = {
    "GetMapping": "GET",
    "PostMapping": "POST",
    "PutMapping": "PUT",
    "DeleteMapping": "DELETE",
    "PatchMapping": "PATCH",
    "RequestMapping": ANY,
}


def _declarative_clients(source: SourceFile, constants: dict[str, str]) -> Iterator[ClientCall]:
    """Retrofit ``@GET("users/{id}")`` and Feign mapping annotations on interface methods."""
    text = source.text
    feign = "@FeignClient" in text
    for group, end in annotation_groups(source):
        header = METHOD_HEADER.match(text[end : end + 400])
        if header is None:
            continue
        name = header.group("kname") or header.group("jname")
        for annotation, args, at in group:
            method: str | None = None
            if annotation in _RETROFIT_VERBS and args is not None:
                method, library = annotation, "retrofit"
            elif feign and annotation in _FEIGN_VERBS:
                method, library = _FEIGN_VERBS[annotation], "feign"
            if method is None:
                continue
            parsed = split_args(f"({args or ''})", 0, "jvm") or []
            value = pick_arg(parsed, 0, ("value", "path"))
            url = url_expression(value.value, "jvm", constants) if value is not None else None
            raw = url.text if url is not None else ""
            yield ClientCall(
                method=method,
                path=normalize_path(raw),
                raw=raw,
                library=library,
                file=source.path,
                line=source.line(at),
                exact=url.exact if url is not None else False,
                decorated=name,
            )


# -- gRPC stubs -------------------------------------------------------------------


def _stub_patterns(service: str) -> re.Pattern[str]:
    name = re.escape(service)
    return re.compile(
        rf"\bNew{name}Client\s*\(|\b{name}Stub\s*\(|\b{name}Grpc\s*\.\s*new(?:Blocking|Future)?Stub\s*\("
        rf"|\bnew\s+(?:[\w.]+\.)?{name}Client\s*\(|\b{name}Client\s*\("
    )


def grpc_client_calls(source: SourceFile, services: dict[str, list[str]]) -> list[ClientCall]:
    """Calls through generated stubs: ``NewGreeterClient(conn).SayHello(...)`` and kin."""
    if source.family in {"proto", "graphql"} or re.search(
        r"_pb2(?:_grpc)?\.py$|\.pb\.go$|Grpc\.java$|_pb\.(?:js|ts)$", source.path
    ):
        return []
    calls: list[ClientCall] = []
    for service, methods in services.items():
        if not _stub_patterns(service).search(source.text):
            continue
        for method in methods:
            lower = method[:1].lower() + method[1:]
            pattern = re.compile(rf"\.\s*(?:{re.escape(method)}|{re.escape(lower)})(?:Async)?\s*\(")
            for match in pattern.finditer(source.text):
                if source.in_comment(match.start()):
                    continue
                calls.append(
                    ClientCall(
                        method="GRPC",
                        path=f"{service}/{method}",
                        raw=f"{service}/{method}",
                        library="grpc",
                        file=source.path,
                        line=source.line(match.start()),
                        exact=True,
                    )
                )
    return calls


# -- GraphQL operations -------------------------------------------------------------

_GQL_OPERATION = re.compile(
    r"(?:\bgql|\bgraphql)\s*(?:\(\s*)?`\s*(?P<op>query|mutation|subscription)?\b[^{`]*\{"
)
_GQL_FILE_OPERATION = re.compile(r"^\s*(?P<op>query|mutation|subscription)\b[^{]*\{", re.M)
_GQL_SELECTION = re.compile(r"\.\.\.\s*\w+|[{}()]|(\w+)\s*:\s*(\w+)|(\w+)")
_OPERATION_ROOT = {"query": "Query", "mutation": "Mutation", "subscription": "Subscription"}


def graphql_client_calls(source: SourceFile) -> list[ClientCall]:
    """Root fields selected by GraphQL operations in ``gql`` templates or ``.graphql`` files."""
    if source.family == "graphql":
        pattern = _GQL_FILE_OPERATION
    elif source.family == "js":
        pattern = _GQL_OPERATION
    else:
        return []
    calls: list[ClientCall] = []
    for match in pattern.finditer(source.text):
        if source.in_comment(match.start()):
            continue
        root = _OPERATION_ROOT[match.group("op") or "query"]
        close = matching_close(source.text, match.end() - 1, "graphql")
        if close is None:
            continue
        for field_name in _root_selections(source.text, match.end(), close):
            calls.append(
                ClientCall(
                    method="GRAPHQL",
                    path=f"{root}.{field_name}",
                    raw=f"{root}.{field_name}",
                    library="graphql",
                    file=source.path,
                    line=source.line(match.start()),
                    exact=True,
                )
            )
    return calls


def _root_selections(text: str, start: int, end: int) -> list[str]:
    fields: list[str] = []
    braces = parens = 0
    for token in _GQL_SELECTION.finditer(text, start, end):
        value = token.group(0)
        if value == "{":
            braces += 1
        elif value == "}":
            braces -= 1
        elif value == "(":
            parens += 1
        elif value == ")":
            parens -= 1
        elif braces == 0 and parens == 0 and not value.startswith("..."):
            name = token.group(2) or token.group(3)
            if name and name not in fields and name != "on":
                fields.append(name)
    return fields


__all__ = ["ClientCall", "detect_client_calls", "graphql_client_calls", "grpc_client_calls"]
