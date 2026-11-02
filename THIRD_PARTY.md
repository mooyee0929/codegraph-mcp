# Third-party notices

codegraph-mcp 0.2 merges the feature sets of two MIT-licensed projects. The code in this
repository was written in Python for this codebase; the designs, algorithms and feature
ideas listed below come from these projects, whose licenses are reproduced here.

| Project | Revision studied | What came from it |
| --- | --- | --- |
| [code-review-graph](https://github.com/tirth8205/code-review-graph) by Tirth Kanani | `6ce25b4` | per-language tree-sitter node tables, Vue / Svelte / notebook handling, execution flows, community detection, hub / bridge / knowledge-gap / surprise analysis, minimal context, refactor preview and apply, wiki, visualization, exports, MCP prompts, notes, multi-repo registry, watch mode, install |
| [codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp) by DeusData | `7d6cdb2` | Cypher-style queries, graph schema, structured and code search, BM25 + semantic scoring, semantically-related edges, HTTP route and cross-service linking, event channels, MinHash clone detection, git-history coupling, infrastructure-as-code indexing, manifest package resolution, ADRs, team-shared graph snapshot, trace ingestion, trace_path, architecture summary |

## code-review-graph

```
MIT License

Copyright (c) 2026 Tirth Kanani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## codebase-memory-mcp

```
MIT License

Copyright (c) 2025 DeusData

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
