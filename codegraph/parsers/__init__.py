"""Language parsers. :func:`parser_for` is the only thing the indexer needs."""

from __future__ import annotations

from functools import cache

from codegraph.models import LANGUAGE_BY_SUFFIX, Language
from codegraph.parsers.base import Parser, module_qname_for
from codegraph.parsers.js_parser import JavaScriptParser
from codegraph.parsers.python_parser import PythonParser


@cache
def parser_for(language: Language) -> Parser:
    if language == "python":
        return PythonParser()
    return JavaScriptParser(language)


def language_of(rel_path: str) -> Language | None:
    dot = rel_path.rfind(".")
    if dot == -1:
        return None
    return LANGUAGE_BY_SUFFIX.get(rel_path[dot:].lower())


__all__ = ["Parser", "language_of", "module_qname_for", "parser_for"]
