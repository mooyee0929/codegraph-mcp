"""Language parsers. :func:`parser_for` is the only thing the indexer needs."""

from __future__ import annotations

from functools import cache

from codegraph.models import Language
from codegraph.parsers.base import Parser, module_qname_for
from codegraph.parsers.js_parser import JavaScriptParser
from codegraph.parsers.languages import language_of
from codegraph.parsers.python_parser import PythonParser


@cache
def parser_for(language: Language) -> Parser:
    if language == "python":
        return PythonParser()
    if language in {"javascript", "typescript"}:
        return JavaScriptParser(language)
    raise ValueError(f"no parser registered for language {language!r}")


__all__ = ["Parser", "language_of", "module_qname_for", "parser_for"]
