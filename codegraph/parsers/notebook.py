"""Jupyter notebooks: the code cells, parsed as one Python module.

Code cells are concatenated in order (a later cell can call a function an earlier cell
defined), IPython magics and shell escapes (``%time``, ``!pip``, ``%%bash`` cells) are
blanked so the result is valid Python, and the combined source goes through
:class:`PythonParser`. If one cell does not parse, cells are parsed one at a time and
the bad ones skipped.

Line numbers are mapped back to the notebook *file*: nbformat writes each source line
as its own JSON string on its own line, so every code line is located in the raw JSON
and nodes point there. That keeps ``path:line`` references and diff-based review
working on ``.ipynb`` files. A minified notebook falls back to the cell's first line.
Non-Python kernels yield just the module node.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, replace

from codegraph.models import Edge, Language, Node, ParseResult
from codegraph.parsers.python_parser import PythonParser
from codegraph.parsers.resources import module_node


@dataclass(frozen=True, slots=True)
class Cell:
    lines: tuple[str, ...]
    file_lines: tuple[int, ...]
    """For each line, its 1-based line number in the notebook file."""


def _is_magic(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith(("%", "!", "?")) or stripped.endswith("?") and " " not in stripped


def _kernel_language(notebook: dict[str, object]) -> str:
    metadata = notebook.get("metadata")
    if not isinstance(metadata, dict):
        return "python"
    for key in ("kernelspec", "language_info"):
        section = metadata.get(key)
        if isinstance(section, dict):
            name = section.get("language") or section.get("name")
            if isinstance(name, str):
                return name.lower()
    return "python"


def _locate(raw_lines: list[str], text: str, cursor: int) -> int | None:
    """0-based index of the raw JSON line holding ``text`` as a string, from ``cursor``."""
    needle = json.dumps(text)[1:-1]
    if not needle.strip():
        return None
    for index in range(cursor, len(raw_lines)):
        if needle in raw_lines[index]:
            return index
    return None


def code_cells(source: str) -> tuple[str, list[Cell]]:
    """(kernel language, code cells with file line numbers); bad JSON yields no cells."""
    try:
        notebook = json.loads(source)
    except json.JSONDecodeError:
        return "python", []
    if not isinstance(notebook, dict):
        return "python", []
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return _kernel_language(notebook), []
    raw_lines = source.splitlines()
    cursor = 0
    found: list[Cell] = []
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        text = cell.get("source", "")
        if isinstance(text, list):
            text = "".join(part for part in text if isinstance(part, str))
        if not isinstance(text, str):
            continue
        lines = text.split("\n")
        if lines and lines[0].lstrip().startswith("%%"):
            lines = [""] * len(lines)
        lines = ["" if _is_magic(line) else line for line in lines]
        file_lines: list[int] = []
        anchor = cursor + 1
        for line in text.split("\n"):
            index = _locate(raw_lines, line, cursor)
            if index is not None:
                cursor = index + 1
                anchor = index + 1
            file_lines.append(anchor)
        found.append(Cell(tuple(lines), tuple(file_lines)))
    return _kernel_language(notebook), found


def _parses(text: str) -> bool:
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


class NotebookParser:
    language: Language = "notebook"

    def parse(self, rel_path: str, source: str, module_qname: str) -> ParseResult:
        module = module_node(rel_path, module_qname, source, self.language)
        kernel, cells = code_cells(source)
        if kernel != "python" or not cells:
            return ParseResult(nodes=(module,), edges=())
        combined = [line for cell in cells for line in cell.lines]
        mapping = [number for cell in cells for number in cell.file_lines]
        if _parses("\n".join(combined)):
            chunks = [(combined, mapping)]
        else:
            chunks = [
                (list(c.lines), list(c.file_lines)) for c in cells if _parses("\n".join(c.lines))
            ]
        nodes: dict[str, Node] = {}
        edges: list[Edge] = []
        for lines, numbers in chunks:
            parsed = PythonParser().parse(rel_path, "\n".join(lines), module_qname)

            def to_file(line: int, numbers: list[int] = numbers) -> int:
                return numbers[min(max(line, 1), len(numbers)) - 1] if numbers else 1

            for node in parsed.nodes:
                if node.kind == "module" or node.qname in nodes:
                    continue
                nodes[node.qname] = replace(
                    node,
                    line_start=to_file(node.line_start),
                    line_end=to_file(node.line_end),
                    language=self.language,
                )
            edges.extend(replace(edge, line=to_file(edge.line)) for edge in parsed.edges)
        return ParseResult(nodes=(module, *nodes.values()), edges=tuple(edges))


__all__ = ["Cell", "NotebookParser", "code_cells"]
