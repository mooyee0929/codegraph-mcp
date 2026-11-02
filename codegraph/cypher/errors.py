"""The one error type the Cypher engine raises.

Every lexer, parser, planner and executor failure carries the character offset of
the token that caused it. Agents write these queries, so an error that says *where*
the query went wrong ("line 1, column 23: unknown label 'Fucntion'") lets them fix
it in one retry instead of guessing.
"""

from __future__ import annotations


class CypherError(ValueError):
    """A query that cannot be lexed, parsed, planned or executed."""

    def __init__(self, message: str, position: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.position = position
        self.line: int | None = None
        self.column: int | None = None
        self.excerpt = ""

    def locate(self, text: str) -> CypherError:
        """Fill in line, column and a caret excerpt from the query text."""
        if self.position is None:
            return self
        offset = max(0, min(self.position, len(text)))
        before = text[:offset]
        line_start = before.rfind("\n") + 1
        line_end = text.find("\n", offset)
        self.line = before.count("\n") + 1
        self.column = offset - line_start + 1
        source_line = text[line_start : line_end if line_end != -1 else len(text)]
        self.excerpt = f"{source_line}\n{' ' * (self.column - 1)}^"
        return self

    def __str__(self) -> str:
        if self.line is not None:
            return f"{self.message} (line {self.line}, column {self.column})"
        if self.position is not None:
            return f"{self.message} (offset {self.position})"
        return self.message

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"error": str(self)}
        if self.position is not None:
            payload["position"] = self.position
        if self.line is not None:
            payload["line"] = self.line
            payload["column"] = self.column
            payload["excerpt"] = self.excerpt
        return payload


__all__ = ["CypherError"]
