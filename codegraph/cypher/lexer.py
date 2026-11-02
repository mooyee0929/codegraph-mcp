"""Turn query text into tokens.

Keywords are not a separate token kind: every word is an ``ident`` and the parser asks
:meth:`Token.is_keyword` where a keyword may appear. That keeps words such as ``count``,
``type`` or ``end`` usable as property names and aliases, which matters because codegraph
properties (``kind``, ``path``, ``line``) collide with keywords in other dialects.
Backtick-quoted identifiers are never keywords.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from codegraph.cypher.errors import CypherError

TokenKind = Literal["ident", "string", "number", "symbol", "eof"]

_TWO_CHAR_SYMBOLS = frozenset({"..", "<>", "!=", "<=", ">=", "=~"})
_ONE_CHAR_SYMBOLS = frozenset("()[]{},:.|*=<>-+;$/%")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", '"': '"', "`": "`"}


@dataclass(frozen=True, slots=True)
class Token:
    kind: TokenKind
    text: str
    """Identifier name, symbol, number text, or the unescaped string value."""
    pos: int
    end: int
    quoted: bool = False

    def is_keyword(self, *words: str) -> bool:
        return self.kind == "ident" and not self.quoted and self.text.upper() in words

    def is_symbol(self, *symbols: str) -> bool:
        return self.kind == "symbol" and self.text in symbols

    def describe(self) -> str:
        if self.kind == "eof":
            return "end of query"
        if self.kind == "string":
            return f"string {self.text!r}"
        return f"'{self.text}'"


def tokenize(text: str) -> list[Token]:
    """Lex ``text``; the list always ends with an ``eof`` token."""
    tokens: list[Token] = []
    i = 0
    length = len(text)
    while i < length:
        char = text[i]
        if char.isspace():
            i += 1
        elif text.startswith("//", i):
            newline = text.find("\n", i)
            i = length if newline == -1 else newline + 1
        elif text.startswith("/*", i):
            close = text.find("*/", i + 2)
            if close == -1:
                raise CypherError("unterminated block comment", i)
            i = close + 2
        elif char in "'\"":
            token, i = _lex_string(text, i)
            tokens.append(token)
        elif char == "`":
            token, i = _lex_quoted_ident(text, i)
            tokens.append(token)
        elif char.isdigit():
            token, i = _lex_number(text, i)
            tokens.append(token)
        elif char.isalpha() or char == "_":
            start = i
            while i < length and (text[i].isalnum() or text[i] == "_"):
                i += 1
            tokens.append(Token("ident", text[start:i], start, i))
        elif text[i : i + 2] in _TWO_CHAR_SYMBOLS:
            tokens.append(Token("symbol", text[i : i + 2], i, i + 2))
            i += 2
        elif char in _ONE_CHAR_SYMBOLS:
            tokens.append(Token("symbol", char, i, i + 1))
            i += 1
        else:
            raise CypherError(f"unexpected character {char!r}", i)
    tokens.append(Token("eof", "", length, length))
    return tokens


def _lex_string(text: str, start: int) -> tuple[Token, int]:
    quote = text[start]
    chars: list[str] = []
    i = start + 1
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 1 < len(text):
            following = text[i + 1]
            chars.append(_ESCAPES.get(following, "\\" + following))
            i += 2
            continue
        if char == quote:
            return Token("string", "".join(chars), start, i + 1), i + 1
        chars.append(char)
        i += 1
    raise CypherError("unterminated string literal", start)


def _lex_quoted_ident(text: str, start: int) -> tuple[Token, int]:
    chars: list[str] = []
    i = start + 1
    while i < len(text):
        if text[i] == "`":
            if text.startswith("``", i):
                chars.append("`")
                i += 2
                continue
            if not chars:
                raise CypherError("empty quoted identifier", start)
            return Token("ident", "".join(chars), start, i + 1, quoted=True), i + 1
        chars.append(text[i])
        i += 1
    raise CypherError("unterminated quoted identifier", start)


def _lex_number(text: str, start: int) -> tuple[Token, int]:
    i = start
    while i < len(text) and text[i].isdigit():
        i += 1
    if i + 1 < len(text) and text[i] == "." and text[i + 1].isdigit():
        i += 1
        while i < len(text) and text[i].isdigit():
            i += 1
    if i < len(text) and (text[i].isalpha() or text[i] == "_"):
        raise CypherError(f"malformed number {text[start : i + 1]!r}", start)
    return Token("number", text[start:i], start, i), i


__all__ = ["Token", "TokenKind", "tokenize"]
