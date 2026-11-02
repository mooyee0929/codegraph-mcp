"""Makefiles: targets as functions, prerequisites and ``$(MAKE) x`` recipes as calls.

That is enough to answer "what runs when I type ``make release``" with the ordinary
``callees_of`` query. Targets are read with a line regex rather than a grammar because
Make's syntax is mostly line-oriented and the interesting part is tiny; variables,
pattern rules and conditionals are skipped. ``## text`` after a target (the
self-documenting Makefile idiom) or a ``#`` block above it becomes the docstring.
"""

from __future__ import annotations

import re

from codegraph.models import Edge, Language, Node, ParseResult
from codegraph.parsers.base import child_qname, clip
from codegraph.parsers.paths import relative_module
from codegraph.parsers.resources import module_node

_RULE = re.compile(r"^([A-Za-z0-9_./%${}()-][A-Za-z0-9_./%${}() -]*?)\s*::?(?![=])\s*(.*)$")
_INCLUDE = re.compile(r"^-?(?:include|sinclude)\s+(.+)$")
_SUBMAKE = re.compile(
    r"\$[({]MAKE[)}]\s+(?:-\S+\s+)*([A-Za-z0-9_.-]+)|^\s*make\s+([A-Za-z0-9_.-]+)"
)
_SPECIAL = re.compile(r"^\.[A-Z_]+$")


def target_segment(target: str) -> str:
    """qname segment for a target: dots and slashes would break dotted qnames."""
    return re.sub(r"[^\w-]", "_", target)


class MakeParser:
    language: Language = "make"

    def parse(self, rel_path: str, source: str, module_qname: str) -> ParseResult:
        lines = source.splitlines()
        rules: list[tuple[str, int, list[str], str | None]] = []
        recipes: dict[int, list[tuple[str, int]]] = {}
        edges: list[Edge] = []
        for number, line in enumerate(lines, start=1):
            if line.startswith("\t"):
                if rules:
                    match = _SUBMAKE.search(line)
                    if match:
                        called = match.group(1) or match.group(2)
                        recipes.setdefault(len(rules) - 1, []).append((called, number))
                continue
            stripped = line.strip()
            include = _INCLUDE.match(stripped)
            if include:
                for spec in include.group(1).split():
                    if "$" not in spec:
                        target = relative_module(rel_path, spec) or spec
                        edges.append(
                            Edge(module_qname, target, "imports", rel_path, number, detail=spec)
                        )
                continue
            match = _RULE.match(stripped)
            if not match or stripped.startswith("#") or "=" in match.group(1):
                continue
            prerequisites, _, comment = match.group(2).partition("#")
            doc = comment.lstrip("#").strip() or _comment_above(lines, number)
            for target in match.group(1).split():
                if not _SPECIAL.match(target) and "%" not in target and "$" not in target:
                    rules.append((target, number, prerequisites.split(), doc))
        defined = {target_segment(name): name for name, _, _, _ in rules}
        nodes: dict[str, Node] = {}
        for index, (target, number, prerequisites, doc) in enumerate(rules):
            qname = child_qname(module_qname, target_segment(target))
            end = rules[index + 1][1] - 1 if index + 1 < len(rules) else len(lines)
            if qname not in nodes:
                nodes[qname] = Node(
                    qname=qname,
                    kind="function",
                    path=rel_path,
                    line_start=number,
                    line_end=max(number, end),
                    language=self.language,
                    name=target,
                    signature=clip(lines[number - 1]),
                    docstring=doc,
                )
                edges.append(Edge(module_qname, qname, "defines", rel_path, number))
            calls = [(p, number) for p in prerequisites] + recipes.get(index, [])
            for called, at in calls:
                segment = target_segment(called)
                if segment in defined:
                    edges.append(
                        Edge(qname, child_qname(module_qname, segment), "calls", rel_path, at)
                    )
        module = module_node(rel_path, module_qname, source, self.language)
        return ParseResult(nodes=(module, *nodes.values()), edges=tuple(edges))


def _comment_above(lines: list[str], number: int) -> str | None:
    collected: list[str] = []
    row = number - 2
    while row >= 0 and lines[row].strip().startswith("#"):
        collected.append(lines[row].strip().lstrip("#").strip())
        row -= 1
    text = " ".join(reversed([c for c in collected if c]))
    return clip(text, 600) if text else None


__all__ = ["MakeParser", "target_segment"]
