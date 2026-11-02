"""Vue and Svelte single-file components.

Only the ``<script>`` blocks carry code the graph cares about. Each block is cut out,
parsed with the existing JavaScript/TypeScript parser (``lang="ts"`` selects
TypeScript), and every line number is shifted back to the block's position in the
component file. A component with ``<script>`` and ``<script setup>`` contributes both
blocks to the one module. Template expressions are not parsed.
"""

from __future__ import annotations

import re
from dataclasses import replace

from codegraph.models import Edge, Language, Node, ParseResult
from codegraph.parsers.js_parser import JavaScriptParser
from codegraph.parsers.resources import module_node

_SCRIPT = re.compile(r"<script\b([^>]*)>(.*?)</script\s*>", re.DOTALL | re.IGNORECASE)
_LANG = re.compile(r"""\blang\s*=\s*["']?(\w+)""", re.IGNORECASE)


def script_blocks(source: str) -> list[tuple[str, int, str]]:
    """(language, first line of the block body, body) for each ``<script>`` block."""
    blocks: list[tuple[str, int, str]] = []
    for match in _SCRIPT.finditer(source):
        lang = _LANG.search(match.group(1))
        language = "typescript" if lang and lang.group(1).lower() in {"ts", "tsx"} else "javascript"
        start = source.count("\n", 0, match.start(2)) + 1
        blocks.append((language, start, match.group(2)))
    return blocks


def _shift_node(node: Node, offset: int, language: Language) -> Node:
    return replace(
        node,
        line_start=node.line_start + offset,
        line_end=node.line_end + offset,
        language=language,
    )


def _shift_edge(edge: Edge, offset: int) -> Edge:
    return replace(edge, line=edge.line + offset)


class ComponentParser:
    """Parses ``.vue`` / ``.svelte`` files through their script blocks."""

    def __init__(self, language: Language) -> None:
        if language not in {"vue", "svelte"}:
            raise ValueError(f"ComponentParser cannot handle {language!r}")
        self.language: Language = language

    def parse(self, rel_path: str, source: str, module_qname: str) -> ParseResult:
        nodes: dict[str, Node] = {}
        edges: list[Edge] = []
        for script_language, start, body in script_blocks(source):
            parsed = JavaScriptParser(script_language).parse(rel_path, body, module_qname)
            offset = start - 1
            for node in parsed.nodes:
                if node.kind != "module" and node.qname not in nodes:
                    nodes[node.qname] = _shift_node(node, offset, self.language)
            edges.extend(_shift_edge(edge, offset) for edge in parsed.edges)
        module = module_node(rel_path, module_qname, source, self.language)
        return ParseResult(nodes=(module, *nodes.values()), edges=tuple(edges))


__all__ = ["ComponentParser", "script_blocks"]
