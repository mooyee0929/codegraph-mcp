"""Split code identifiers into the words people search with.

``parseHTTPRequest``, ``parse_http_request`` and ``pkg/http/request.py`` should all
answer a query for "parse http request", so every searchable string is split on
non-alphanumerics, then on camelCase and acronym boundaries, then lower-cased. Common
abbreviations also contribute their expansion (``ctx`` -> ``context``) so a query in
plain words still meets terse code — the idea comes from codebase-memory-mcp's
tokenizer, with a shorter table of our own.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from codegraph.models import Node

TOKENIZER_VERSION = "1"
"""Bump when tokenisation changes so persisted indexes are rebuilt, not misread."""

_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_SEPARATORS = re.compile(r"[^A-Za-z0-9]+")

ABBREVIATIONS: dict[str, str] = {
    "args": "arguments",
    "auth": "authentication",
    "buf": "buffer",
    "cfg": "config",
    "cmd": "command",
    "conf": "config",
    "conn": "connection",
    "ctx": "context",
    "db": "database",
    "dir": "directory",
    "doc": "document",
    "env": "environment",
    "err": "error",
    "exc": "exception",
    "fmt": "format",
    "idx": "index",
    "impl": "implementation",
    "init": "initialize",
    "len": "length",
    "msg": "message",
    "num": "number",
    "param": "parameter",
    "params": "parameters",
    "repo": "repository",
    "req": "request",
    "res": "response",
    "resp": "response",
    "src": "source",
    "str": "string",
    "tmp": "temporary",
    "util": "utility",
    "val": "value",
}

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "async",
        "at",
        "await",
        "be",
        "by",
        "class",
        "cls",
        "const",
        "def",
        "export",
        "fn",
        "for",
        "from",
        "func",
        "function",
        "if",
        "in",
        "is",
        "it",
        "let",
        "new",
        "of",
        "on",
        "or",
        "private",
        "pub",
        "public",
        "self",
        "static",
        "that",
        "the",
        "this",
        "to",
        "var",
        "with",
    }
)
"""Filler and keywords that appear in nearly every signature or docstring."""

TYPE_WORDS: frozenset[str] = frozenset(
    {
        "any",
        "bool",
        "boolean",
        "dict",
        "float",
        "int",
        "list",
        "none",
        "null",
        "number",
        "object",
        "optional",
        "return",
        "returns",
        "str",
        "string",
        "tuple",
        "void",
    }
)
"""Type names: useful in a search query, noise when deciding two functions are alike."""


def split_identifier(text: str) -> list[str]:
    """``getHTTPResponse_v2`` -> ``['get', 'http', 'response', 'v', '2']``."""
    words: list[str] = []
    for chunk in _SEPARATORS.split(text):
        words.extend(match.group(0).lower() for match in _WORD.finditer(chunk))
    return words


def terms(
    text: str,
    *,
    expand: bool = True,
    drop: frozenset[str] = STOPWORDS,
) -> list[str]:
    """Search terms for ``text``: split words minus ``drop``, plus abbreviation expansions."""
    found: list[str] = []
    for word in split_identifier(text):
        if len(word) < 2 and not word.isdigit():
            continue
        if word not in drop:
            found.append(word)
        expanded = ABBREVIATIONS.get(word) if expand else None
        if expanded is not None and expanded not in drop:
            found.append(expanded)
    return found


def node_terms(node: Node) -> list[str]:
    """The bag of words the BM25 / TF-IDF search index stores for ``node``.

    The name counts three times and the qualified name twice, a cheap stand-in for
    per-field weights: a hit on the name should beat a hit deep in a docstring.
    """
    bag: list[str] = []
    bag.extend(terms(node.name) * 3)
    bag.extend(terms(node.qname) * 2)
    bag.extend(terms(node.signature))
    bag.extend(terms((node.docstring or "")[:600]))
    bag.extend(terms(node.path))
    bag.append(node.kind)
    return bag


def semantic_terms(node: Node) -> list[str]:
    """Words describing what a callable *does*, for semantically-related edges.

    Paths and qualified names are left out on purpose: they describe where code lives,
    and two functions in one package would look alike for that reason alone.
    """
    drop = STOPWORDS | TYPE_WORDS
    bag: list[str] = []
    bag.extend(terms(node.name, drop=drop) * 2)
    bag.extend(terms(node.signature, drop=drop))
    bag.extend(terms((node.docstring or "")[:600], drop=drop))
    return bag


def unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


__all__ = [
    "ABBREVIATIONS",
    "STOPWORDS",
    "TOKENIZER_VERSION",
    "TYPE_WORDS",
    "node_terms",
    "semantic_terms",
    "split_identifier",
    "terms",
    "unique",
]
