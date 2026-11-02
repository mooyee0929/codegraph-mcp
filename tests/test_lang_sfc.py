"""Vue/Svelte components, Jupyter notebooks, and JS imports that name a file suffix."""

from __future__ import annotations

import json

from codegraph.models import ParseResult
from codegraph.parsers import module_qname_for, parser_for
from codegraph.parsers.notebook import code_cells
from codegraph.parsers.sfc import script_blocks


def parse(language: str, path: str, source: str) -> ParseResult:
    return parser_for(language).parse(path, source, module_qname_for(path))


VUE = """<template>
  <button @click="save">{{ label }}</button>
</template>

<script setup lang="ts">
import { format } from "../util/format.ts";
import Child from "./Child.vue";

function save(): void {
  format(1);
}
</script>

<script>
export default { name: "Button" };
function helper() { save(); }
</script>
"""


def test_vue_script_blocks_with_shifted_lines() -> None:
    blocks = script_blocks(VUE)
    assert [(lang, start) for lang, start, _ in blocks] == [("typescript", 5), ("javascript", 14)]
    result = parse("vue", "web/components/Button.vue", VUE)
    nodes = {n.qname: n for n in result.nodes}
    module = nodes["web.components.Button"]
    assert (module.kind, module.language, module.line_end) == ("module", "vue", 18)
    save = nodes["web.components.Button.save"]
    assert (save.kind, save.line_start, save.line_end, save.language) == ("function", 9, 11, "vue")
    assert nodes["web.components.Button.helper"].line_start == 16
    edges = {(e.kind, e.src, e.dst, e.line) for e in result.edges}
    assert ("imports", "web.components.Button", "web.util.format", 6) in edges
    assert ("imports", "web.components.Button", "web.components.Child", 7) in edges
    assert ("calls", "web.components.Button.save", "web.util.format.format", 10) in edges


SVELTE = """<script>
  import { onMount } from "svelte";
  export let name;
  function greet() { onMount(() => {}); }
</script>

<h1>Hello {name}</h1>
"""


def test_svelte_component() -> None:
    result = parse("svelte", "src/App.svelte", SVELTE)
    nodes = {n.qname: n for n in result.nodes}
    assert nodes["src.App.greet"].line_start == 4
    assert ("imports", "svelte") in {(e.kind, e.dst) for e in result.edges}


def test_component_without_script_is_just_a_module() -> None:
    result = parse("vue", "a/Static.vue", "<template><p>hi</p></template>\n")
    assert [n.kind for n in result.nodes] == ["module"]
    assert result.edges == ()


def test_js_import_with_explicit_suffix_resolves_to_module() -> None:
    source = 'import { helper } from "./util.js";\nexport function run() { helper(); }\n'
    result = parse("typescript", "web/app.ts", source)
    edges = {(e.kind, e.dst) for e in result.edges}
    assert ("imports", "web.util") in edges
    assert ("calls", "web.util.helper") in edges


def _notebook(cells: list[dict[str, object]], language: str = "python") -> str:
    document = {
        "cells": cells,
        "metadata": {"kernelspec": {"language": language, "name": language}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(document, indent=1)


def _code(*lines: str) -> dict[str, object]:
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "metadata": {}, "outputs": [], "source": source}


def test_notebook_cells_map_to_file_lines() -> None:
    source = _notebook(
        [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Title"]},
            _code("%matplotlib inline", "import math", "def area(r):", "    return math.pi * r"),
            _code("!pip install x", "def report():", "    return area(2)"),
            _code("%%bash", "echo not python"),
        ]
    )
    raw = source.splitlines()
    result = parse("notebook", "notebooks/analysis.ipynb", source)
    nodes = {n.qname: n for n in result.nodes}
    area = nodes["notebooks.analysis.area"]
    report = nodes["notebooks.analysis.report"]
    assert area.language == "notebook"
    assert '"def area(r):\\n"' in raw[area.line_start - 1]
    assert '"def report():\\n"' in raw[report.line_start - 1]
    edges = {(e.kind, e.src, e.dst) for e in result.edges}
    assert ("calls", "notebooks.analysis.report", "notebooks.analysis.area") in edges
    assert ("imports", "notebooks.analysis", "math") in edges
    call = next(e for e in result.edges if e.dst == "notebooks.analysis.area" and e.kind == "calls")
    assert "area(2)" in raw[call.line - 1]


def test_notebook_with_a_broken_cell_keeps_the_good_ones() -> None:
    source = _notebook([_code("def good():", "    return 1"), _code("def bad(:")])
    result = parse("notebook", "n.ipynb", source)
    assert "n.good" in {n.qname for n in result.nodes}


def test_non_python_and_invalid_notebooks() -> None:
    r_notebook = _notebook([_code("x <- 1")], language="R")
    assert [n.kind for n in parse("notebook", "r.ipynb", r_notebook).nodes] == ["module"]
    assert [n.kind for n in parse("notebook", "bad.ipynb", "{not json").nodes] == ["module"]
    kernel, cells = code_cells(_notebook([_code("a = 1")]))
    assert kernel == "python"
    assert cells[0].lines == ("a = 1",)
