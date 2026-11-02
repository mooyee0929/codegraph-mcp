"""Language parsers. :func:`parser_for` is the only thing the indexer needs.

Python and JavaScript/TypeScript have hand-written parsers; the two dozen other code
languages share :class:`TreeSitterParser` and the spec table in
:mod:`codegraph.parsers.ts_languages`; components, notebooks and infrastructure files
have small parsers of their own that reuse those.
"""

from __future__ import annotations

from functools import cache

from codegraph.models import Language
from codegraph.parsers.base import Parser, module_qname_for
from codegraph.parsers.docker import ComposeParser, DockerfileParser
from codegraph.parsers.js_parser import JavaScriptParser
from codegraph.parsers.kubernetes import KustomizeParser, YamlParser
from codegraph.parsers.languages import language_of, supported_languages
from codegraph.parsers.make import MakeParser
from codegraph.parsers.notebook import NotebookParser
from codegraph.parsers.python_parser import PythonParser
from codegraph.parsers.sfc import ComponentParser
from codegraph.parsers.terraform import TerraformParser
from codegraph.parsers.treesitter import TreeSitterParser
from codegraph.parsers.ts_languages import SPECS


@cache
def parser_for(language: Language) -> Parser:
    if language == "python":
        return PythonParser()
    if language in {"javascript", "typescript"}:
        return JavaScriptParser(language)
    spec = SPECS.get(language)
    if spec is not None:
        return TreeSitterParser(spec)
    if language in {"vue", "svelte"}:
        return ComponentParser(language)
    if language in {"terraform", "hcl"}:
        return TerraformParser(language)
    fixed: dict[Language, Parser] = {
        "notebook": NotebookParser(),
        "dockerfile": DockerfileParser(),
        "compose": ComposeParser(),
        "kustomize": KustomizeParser(),
        "yaml": YamlParser(),
        "make": MakeParser(),
    }
    found = fixed.get(language)
    if found is None:
        raise ValueError(f"no parser registered for language {language!r}")
    return found


__all__ = ["Parser", "language_of", "module_qname_for", "parser_for", "supported_languages"]
