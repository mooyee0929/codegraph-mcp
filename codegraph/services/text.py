"""Lexical helpers for reading source text without a parser.

Route, client-call and channel detection only needs to answer small questions about a
call site: where does this argument list end, what are its top-level arguments, is this
argument a string literal (or a concatenation the reader can still follow), and is this
match inside a comment. Each question is answered with one compiled regular expression
per comment style, so scanning thousands of files stays fast in pure Python.

Every function here is total: malformed or truncated source yields ``None`` or an
empty result, never an exception.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Family = Literal["python", "js", "go", "jvm", "csharp", "ruby", "php", "rust", "proto", "graphql"]
"""Language family: languages that share comment, string and call syntax."""

Style = Literal["c", "hash", "both"]

_FAMILY_STYLE: dict[str, Style] = {
    "python": "hash",
    "ruby": "hash",
    "php": "both",
    "graphql": "hash",
}

_C_COMMENT = r"(?P<comment>//[^\n]*|/\*.*?(?:\*/|\Z))"
_HASH_COMMENT = r"(?P<comment>\#[^\n]*)"
_PHP_COMMENT = r"(?P<comment>//[^\n]*|/\*.*?(?:\*/|\Z)|\#(?!\[)[^\n]*)"
_PY_LONG = r"(?P<long>[rRbBuUfF]{0,2}(?:'''.*?(?:'''|\Z)|\"\"\".*?(?:\"\"\"|\Z)))"
_QUOTED = r"\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'"
_RUST_STR = r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\\n])'"
_BACKTICK = r"`(?:\\.|[^`\\])*`"
_JVM_LONG = r"\"\"\".*?(?:\"\"\"|\Z)"

_LEXEMES: dict[str, re.Pattern[str]] = {
    "python": re.compile(
        rf"{_HASH_COMMENT}|{_PY_LONG}|(?P<str>[rRbBuUfF]{{0,2}}(?:{_QUOTED}))", re.S
    ),
    "ruby": re.compile(rf"{_HASH_COMMENT}|(?P<str>{_QUOTED})", re.S),
    "graphql": re.compile(rf"{_HASH_COMMENT}|(?P<str>\"\"\".*?(?:\"\"\"|\Z)|{_QUOTED})", re.S),
    "php": re.compile(rf"{_PHP_COMMENT}|(?P<str>{_QUOTED})", re.S),
    "rust": re.compile(rf"{_C_COMMENT}|(?P<str>r\#*\"(?:.|\n)*?\"\#*|{_RUST_STR})", re.S),
    "jvm": re.compile(rf"{_C_COMMENT}|(?P<str>{_JVM_LONG}|{_QUOTED})", re.S),
    "csharp": re.compile(rf"{_C_COMMENT}|(?P<str>[@$]{{1,2}}\"(?:\"\"|[^\"])*\"|{_QUOTED})", re.S),
    "c": re.compile(rf"{_C_COMMENT}|(?P<str>{_QUOTED}|{_BACKTICK})", re.S),
}

_OPENERS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = frozenset(_OPENERS.values())


def _lexeme_pattern(family: str) -> re.Pattern[str]:
    if family in _LEXEMES:
        return _LEXEMES[family]
    return _LEXEMES["c"]


def _bracket_pattern(family: str) -> re.Pattern[str]:
    cached = _BRACKETS.get(family)
    if cached is None:
        cached = re.compile(rf"{_lexeme_pattern(family).pattern}|(?P<br>[()\[\]{{}}])", re.S)
        _BRACKETS[family] = cached
    return cached


_BRACKETS: dict[str, re.Pattern[str]] = {}


@dataclass(frozen=True, slots=True)
class Arg:
    """One top-level argument of a call: stripped text and where it starts."""

    text: str
    start: int

    @property
    def keyword(self) -> str | None:
        """``name`` for ``name=value`` / ``name: value`` / ``name => value`` arguments."""
        match = _KEYWORD_ARG.match(self.text)
        if match is None:
            return None
        return match.group(1) or match.group(2) or match.group(3)

    @property
    def value(self) -> str:
        """The argument without its keyword prefix."""
        match = _KEYWORD_ARG.match(self.text)
        return self.text[match.end() :].strip() if match else self.text


_KEYWORD_ARG = re.compile(
    r"^(?:'(\w+)'|\"(\w+)\"|:?(?!lambda\b)([A-Za-z_]\w*))\s*(?:=(?![=>])|:(?![:/])|=>)\s*"
)


def comment_spans(text: str, family: str) -> list[tuple[int, int]]:
    """Spans of comments (and Python triple-quoted strings, usually docstrings)."""
    spans: list[tuple[int, int]] = []
    for match in _lexeme_pattern(family).finditer(text):
        if match.lastgroup in {"comment", "long"}:
            spans.append(match.span())
    return spans


class CommentMask:
    """Answers "is this offset inside a comment?" with one lazy scan per file."""

    __slots__ = ("_ends", "_family", "_starts", "_text")

    def __init__(self, text: str, family: str) -> None:
        self._text = text
        self._family = family
        self._starts: list[int] | None = None
        self._ends: list[int] = []

    def __contains__(self, offset: object) -> bool:
        if not isinstance(offset, int):
            return False
        if self._starts is None:
            spans = comment_spans(self._text, self._family)
            self._starts = [s for s, _ in spans]
            self._ends = [e for _, e in spans]
        index = bisect_right(self._starts, offset) - 1
        return index >= 0 and offset < self._ends[index]


def line_starts(text: str) -> list[int]:
    return [0, *(m.end() for m in re.finditer("\n", text))]


def line_at(starts: list[int], offset: int) -> int:
    """1-based line number of ``offset`` given :func:`line_starts`."""
    return bisect_right(starts, offset)


def matching_close(text: str, open_index: int, family: str, limit: int | None = None) -> int | None:
    """Index of the bracket closing the one at ``open_index``, skipping strings/comments."""
    if open_index >= len(text) or text[open_index] not in _OPENERS:
        return None
    end = len(text) if limit is None else min(len(text), open_index + limit)
    stack: list[str] = []
    for match in _bracket_pattern(family).finditer(text, open_index, end):
        bracket = match.group("br")
        if bracket is None:
            continue
        if bracket in _OPENERS:
            stack.append(_OPENERS[bracket])
            continue
        if bracket in _CLOSERS:
            if stack:
                stack.pop()
            if not stack:
                return match.start()
    return None


def split_args(text: str, open_index: int, family: str, limit: int = 8000) -> list[Arg] | None:
    """Top-level arguments of the bracketed list starting at ``open_index``."""
    close = matching_close(text, open_index, family, limit)
    if close is None:
        return None
    return _split_range(text, open_index + 1, close, family)


def split_bare_args(text: str, start: int, family: str, limit: int = 2000) -> list[Arg]:
    """Arguments of a paren-less call (Ruby ``get "/x", to: "a#b"``) up to end of line.

    The list ends at a newline outside brackets unless the line ends with a comma, or
    at a ``do`` / ``{`` block opener.
    """
    end = min(len(text), start + limit)
    depth = 0
    cursor = start
    stop = end
    for match in _bracket_pattern(family).finditer(text, start, end):
        gap_stop = _bare_stop(text, cursor, match.start(), depth)
        if gap_stop is not None:
            stop = gap_stop
            break
        bracket = match.group("br")
        if bracket == "{" and depth == 0 and _block_brace(text, start, match.start()):
            stop = match.start()
            break
        if bracket in _OPENERS:
            depth += 1
        elif bracket in _CLOSERS:
            depth -= 1
            if depth < 0:
                stop = match.start()
                break
        cursor = match.end()
    else:
        gap_stop = _bare_stop(text, cursor, end, depth)
        if gap_stop is not None:
            stop = gap_stop
    return _split_range(text, start, stop, family)


def _bare_stop(text: str, start: int, end: int, depth: int) -> int | None:
    if depth != 0:
        return None
    for index in range(start, end):
        char = text[index]
        if char == "\n":
            if text[start:index].rstrip().endswith(","):
                continue
            return index
        if char == "d" and text.startswith("do", index) and _word_at(text, index, 2):
            return index
    return None


def _word_at(text: str, index: int, length: int) -> bool:
    before = text[index - 1] if index else " "
    after = text[index + length] if index + length < len(text) else " "
    return not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_")


def _block_brace(text: str, start: int, index: int) -> bool:
    """A ``{`` that opens a block rather than a hash literal argument."""
    return not text[start:index].rstrip().endswith((",", "=>", ":", "(", "["))


def _split_range(text: str, start: int, end: int, family: str) -> list[Arg]:
    args: list[Arg] = []
    depth = 0
    cursor = start
    piece_start = start
    for match in _bracket_pattern(family).finditer(text, start, end):
        for index in range(cursor, match.start()):
            if text[index] == "," and depth == 0:
                _append_arg(args, text, piece_start, index)
                piece_start = index + 1
        bracket = match.group("br")
        if bracket in _OPENERS:
            depth += 1
        elif bracket in _CLOSERS:
            depth -= 1
        cursor = match.end()
    for index in range(cursor, end):
        if text[index] == "," and depth == 0:
            _append_arg(args, text, piece_start, index)
            piece_start = index + 1
    _append_arg(args, text, piece_start, end)
    return args


def _append_arg(args: list[Arg], text: str, start: int, end: int) -> None:
    raw = text[start:end]
    stripped = raw.strip()
    if stripped:
        args.append(Arg(text=stripped, start=start + (len(raw) - len(raw.lstrip()))))


def pick_arg(args: list[Arg], index: int, names: tuple[str, ...] = ()) -> Arg | None:
    """The ``index``-th positional argument, or the first keyword argument in ``names``."""
    for arg in args:
        if arg.keyword is not None and arg.keyword in names:
            return Arg(text=arg.value, start=arg.start)
    positional = [a for a in args if a.keyword is None]
    return positional[index] if index < len(positional) else None


def keyword_arg(args: list[Arg], *names: str) -> Arg | None:
    for arg in args:
        if arg.keyword is not None and arg.keyword in names:
            return Arg(text=arg.value, start=arg.start)
    return None


# -- string literals ------------------------------------------------------

_STRING_START = {
    "python": re.compile(r"[rRbBuUfF]{0,2}(?:'''|\"\"\"|'|\")"),
    "csharp": re.compile(r"[@$]{0,2}\""),
    "rust": re.compile(r"r?\#*\""),
}
_INTERPOLATING = frozenset({"ruby", "php", "jvm"})
_GENERIC_START = re.compile(r"\"\"\"|['\"`]")


@dataclass(frozen=True, slots=True)
class StringLit:
    """A string literal's content and whether it interpolates (f-string, template)."""

    content: str
    interpolated: bool


def string_literal(expr: str, family: str) -> StringLit | None:
    """Parse ``expr`` when it is exactly one string literal; otherwise None."""
    expr = expr.strip()
    starter = _STRING_START.get(family, _GENERIC_START).match(expr)
    if starter is None:
        return None
    opener = starter.group(0)
    prefix = opener.rstrip("'\"`#")
    quote = opener[len(prefix) :]
    if family == "rust":
        hashes = quote.count("#")
        closing = '"' + "#" * hashes
        body_start = len(opener)
    else:
        closing = quote
        body_start = len(opener)
    end = _find_closing(expr, body_start, closing, raw="r" in prefix.lower() or "@" in prefix)
    if end is None or end + len(closing) != len(expr):
        return None
    content = expr[body_start:end]
    interpolated = (
        "f" in prefix.lower()
        or "$" in prefix
        or quote == "`"
        or (family in _INTERPOLATING and quote == '"' and ("#{" in content or "$" in content))
    )
    return StringLit(content=content, interpolated=interpolated)


def _find_closing(expr: str, start: int, closing: str, *, raw: bool) -> int | None:
    index = start
    while index < len(expr):
        if not raw and expr[index] == "\\":
            index += 2
            continue
        if expr.startswith(closing, index):
            if closing == '"' and raw and expr.startswith('""', index) and index + 2 < len(expr):
                index += 2  # C# verbatim "" escape
                continue
            return index
        index += 1
    return None


_FORMAT_WRAPPER = re.compile(
    r"^(?:String\s*\.\s*format|fmt\s*\.\s*Sprintf|sprintf|format|String\.Format)\s*\(", re.S
)
_URL_WRAPPER = re.compile(
    r"^(?:new\s+)?(?:java\s*\.\s*net\s*\.\s*)?(?:URI|URL|Uri)\s*(?:\.\s*(?:create|parse))?\s*\(",
    re.S,
)
_IDENT_EXPR = re.compile(r"^[A-Za-z_$][\w$]*(?:\s*(?:\.|::|->)\s*[A-Za-z_$][\w$]*)*$")
_DOTFORMAT = re.compile(r"\.\s*format\s*\(.*\)$", re.S)


@dataclass(frozen=True, slots=True)
class UrlText:
    """A URL or path recovered from an expression.

    ``text`` keeps unknown pieces as ``{param}``; ``exact`` is False when anything had
    to be guessed (a dropped base URL, an unknown variable in the middle).
    """

    text: str
    exact: bool


def url_expression(
    expr: str,
    family: str,
    constants: Mapping[str, str],
    *,
    client: bool = True,
    depth: int = 0,
) -> UrlText | None:
    """Follow literals, same-file string constants and ``+`` / ``.`` concatenation.

    With ``client`` set, an unknown leading piece followed by ``/`` is taken to be a
    base URL and dropped (``BASE + "/users"`` -> ``/users``); route definitions pass
    ``client=False`` so a leading path parameter survives.
    """
    expr = expr.strip()
    if not expr or depth > 3:
        return None
    wrapped = _unwrap(expr, family)
    if wrapped is not None:
        return url_expression(wrapped, family, constants, client=client, depth=depth + 1)
    pieces = _concat_pieces(expr, family)
    parts: list[tuple[str, bool]] = []  # (text, known)
    saw_literal = False
    for piece in pieces:
        literal = string_literal(piece, family)
        if literal is not None:
            content = literal.content
            if literal.interpolated:
                content = _substitute_placeholders(content, constants)
            parts.append((content, True))
            saw_literal = True
            continue
        dot_format = _DOTFORMAT.search(piece)
        if dot_format is not None:
            inner = string_literal(piece[: dot_format.start()], family)
            if inner is not None:
                parts.append((inner.content, True))
                saw_literal = True
                continue
        constant = _lookup_constant(piece, constants)
        if constant is not None:
            parts.append((constant, True))
            saw_literal = True
            continue
        parts.append(("", False))
    if not saw_literal:
        return None
    exact = True
    out: list[str] = []
    for index, (text, known) in enumerate(parts):
        if known:
            out.append(text)
            continue
        exact = False
        following = parts[index + 1][0] if index + 1 < len(parts) else ""
        if client and index == 0 and following.startswith(("/", "?")):
            continue  # a base URL we cannot see
        out.append("{param}")
    text = "".join(out)
    if client and _LEADING_PLACEHOLDER.match(text):
        text = _LEADING_PLACEHOLDER.sub("", text, count=1)
        exact = False
    return UrlText(text=text, exact=exact)


_LEADING_PLACEHOLDER = re.compile(r"^(?:\$\{[^}]*\}|\{[^}]*\}|\#\{[^}]*\}|\$[A-Za-z_]\w*)(?=/)")
_PLACEHOLDER = re.compile(r"\$\{\s*([\w.$]+)\s*\}|\#\{\s*([\w.]+)\s*\}|\{\s*([\w.]+)\s*\}")


def _substitute_placeholders(content: str, constants: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2) or match.group(3) or ""
        value = _lookup_constant(name, constants)
        return value if value is not None else match.group(0)

    return _PLACEHOLDER.sub(replace, content)


def _lookup_constant(expr: str, constants: Mapping[str, str]) -> str | None:
    expr = expr.strip()
    if not _IDENT_EXPR.match(expr):
        return None
    if expr in constants:
        return constants[expr]
    last = re.split(r"\s*(?:\.|::|->)\s*", expr)[-1].lstrip("$")
    return constants.get(last)


def _unwrap(expr: str, family: str) -> str | None:
    for pattern in (_URL_WRAPPER, _FORMAT_WRAPPER):
        match = pattern.match(expr)
        if match is None:
            continue
        args = split_args(expr, match.end() - 1, family)
        if args and matching_close(expr, match.end() - 1, family) == len(expr) - 1:
            return args[0].value
    return None


def _concat_pieces(expr: str, family: str) -> list[str]:
    """Split on top-level ``+`` (``.`` for PHP), leaving nested calls intact."""
    operator = "." if family == "php" else "+"
    pieces: list[str] = []
    depth = 0
    start = 0
    cursor = 0
    pattern = _bracket_pattern(family)
    for match in pattern.finditer(expr):
        for index in range(cursor, match.start()):
            if expr[index] == operator and depth == 0 and _is_operator(expr, index, family):
                pieces.append(expr[start:index].strip())
                start = index + 1
        bracket = match.group("br")
        if bracket in _OPENERS:
            depth += 1
        elif bracket in _CLOSERS:
            depth -= 1
        cursor = match.end()
    for index in range(cursor, len(expr)):
        if expr[index] == operator and depth == 0 and _is_operator(expr, index, family):
            pieces.append(expr[start:index].strip())
            start = index + 1
    pieces.append(expr[start:].strip())
    return [p for p in pieces if p]


def _is_operator(expr: str, index: int, family: str) -> bool:
    if family != "php":
        return True
    before = expr[index - 1] if index else ""
    after = expr[index + 1] if index + 1 < len(expr) else ""
    return not (before.isdigit() and after.isdigit())


# -- same-file string constants ------------------------------------------

_STRING_VALUE = r"(?P<value>\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|`[^`$]*`)"
_CONSTANT_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "python": (
        re.compile(
            rf"^(?P<name>[A-Za-z_]\w*)\s*(?::\s*[\w\[\]., ]+)?\s*=\s*{_STRING_VALUE}\s*(?:#.*)?$",
            re.M,
        ),
    ),
    "js": (
        re.compile(
            r"^(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?::\s*\w+)?\s*=\s*"
            rf"{_STRING_VALUE}\s*;?\s*(?://.*)?$",
            re.M,
        ),
        re.compile(
            rf"^\s*(?:static\s+)?(?:readonly\s+)?(?P<name>[A-Z][A-Z0-9_]*)\s*=\s*{_STRING_VALUE}",
            re.M,
        ),
    ),
    "go": (
        re.compile(
            r"^\s*(?:const\s+|var\s+)?(?P<name>[A-Za-z_]\w*)\s*(?:string\s*)?=\s*"
            r"(?P<value>\"(?:\\.|[^\"\\\n])*\"|`[^`]*`)",
            re.M,
        ),
    ),
    "jvm": (
        re.compile(
            r"(?:static\s+final|final\s+static|const\s+val|val|final)\s+(?:String\s+)?"
            r"(?P<name>[A-Za-z_]\w*)\s*(?::\s*String)?\s*=\s*(?P<value>\"(?:\\.|[^\"\\\n])*\")"
        ),
    ),
    "csharp": (
        re.compile(
            r"(?:const|static\s+readonly|readonly\s+static)\s+string\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
            r"(?P<value>\"(?:\\.|[^\"\\\n])*\")"
        ),
    ),
    "ruby": (re.compile(rf"^\s*(?P<name>[A-Z][A-Z0-9_]*)\s*=\s*{_STRING_VALUE}", re.M),),
    "php": (
        re.compile(rf"\bconst\s+(?P<name>[A-Za-z_]\w*)\s*=\s*{_STRING_VALUE}"),
        re.compile(rf"\bdefine\s*\(\s*['\"](?P<name>\w+)['\"]\s*,\s*{_STRING_VALUE}"),
    ),
    "rust": (
        re.compile(
            r"\b(?:const|static)\s+(?P<name>[A-Za-z_]\w*)\s*:\s*&(?:'static\s+)?str\s*=\s*"
            r"(?P<value>\"(?:\\.|[^\"\\\n])*\")"
        ),
    ),
}


def module_constants(text: str, family: str) -> dict[str, str]:
    """``NAME -> value`` for string constants declared in this file.

    Python and JavaScript only count module-level (column 0) assignments; the other
    families accept the usual constant declarations wherever they appear, since their
    constants normally live inside a class or a ``const (...)`` block.
    """
    found: dict[str, str] = {}
    for pattern in _CONSTANT_PATTERNS.get(family, ()):
        for match in pattern.finditer(text):
            literal = string_literal(match.group("value"), family)
            if literal is not None and match.group("name") not in found:
                found[match.group("name")] = literal.content
    return found


__all__ = [
    "Arg",
    "CommentMask",
    "Family",
    "StringLit",
    "UrlText",
    "comment_spans",
    "keyword_arg",
    "line_at",
    "line_starts",
    "matching_close",
    "module_constants",
    "pick_arg",
    "split_args",
    "split_bare_args",
    "string_literal",
    "url_expression",
]
