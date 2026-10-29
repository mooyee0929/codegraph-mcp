# codegraph-mcp

A code knowledge graph you can query from an agent. Index a repository once, then ask
structural questions — *who calls this, what breaks if I change it, which tests cover
it, what does the architecture look like* — through an [MCP](https://modelcontextprotocol.io)
server or a CLI, instead of grepping and reading whole files.

Supports **Python** (stdlib `ast`) and **JavaScript / TypeScript** (tree-sitter).
Everything lives in one SQLite file under `.codegraph/` in the repo you index; there is
no daemon, no embeddings service, and no network access.

## Why

Agents burn tokens re-discovering the same facts about a codebase: opening files to find
callers, tracing imports by hand, guessing which tests matter. A graph answers those in a
few hundred tokens with exact `path:line` locations, and turns a diff into a ranked list
of "these symbols have many callers and no tests — look here first".

## Install

```bash
uv pip install git+https://github.com/mooyee0929/codegraph-mcp
# or, from a clone
uv pip install -e ".[dev]"
```

Requires Python 3.11+.

## CLI

```bash
codegraph index                       # build or refresh .codegraph/graph.db (incremental)
codegraph index --force               # re-parse unchanged files too
codegraph status                      # row counts and database location
codegraph search "invoice total"      # keyword search over names, signatures, docstrings
codegraph query callers_of Invoice.total
codegraph query tests_for pkg.billing.charge
codegraph impact Invoice.rate --depth 3
codegraph review                      # risk-ranked symbols touched by the working-tree diff
codegraph review --base main --context
codegraph arch                        # languages, hubs, entry points, orphans, external deps
codegraph dead                        # callables nothing references
codegraph show charge                 # print a symbol's source with line numbers
codegraph serve                       # run the MCP server (same as codegraph-mcp)
```

Every subcommand accepts `--json`. To work on another repository, pass `--root <dir>`
(default: cwd) before the subcommand: `codegraph --root ../other-repo index`.

Targets can be a full qname (`pkg.billing.Invoice.total`), a bare name when it is unique
in the repo (`total`), a suffix (`Invoice.total`), or a location (`pkg/billing.py:10`).

Query patterns: `callers_of`, `callees_of`, `imports_of`, `imported_by`, `tests_for`,
`subclasses_of`, `defines`, `defined_in`.

## MCP server

```bash
codegraph-mcp --root /path/to/repo     # stdio transport (default)
codegraph-mcp --root /path/to/repo --transport streamable-http   # or: sse
```

Claude Code:

```bash
claude mcp add codegraph -- codegraph-mcp --root /path/to/repo
```

or in `.mcp.json`:

```json
{
  "mcpServers": {
    "codegraph": { "command": "codegraph-mcp", "args": ["--root", "."] }
  }
}
```

| Tool | What it answers |
| --- | --- |
| `index_repository_tool(force?)` | Build or refresh the graph. Incremental by content hash. |
| `graph_status()` | Row counts and database location. |
| `search_nodes(query_text, limit?)` | Full-text search over qname, name, signature, docstring. |
| `get_node(target)` | One symbol with all incoming and outgoing edges. |
| `query_graph(pattern, target)` | Trace one relationship (see patterns above). |
| `get_impact_radius(target, depth?)` | Direct + transitive callers, importing modules, covering tests. |
| `detect_changes_tool(base?)` | Risk-ranked symbols overlapping the git diff. |
| `get_review_context(qnames, context_lines?)` | Source snippets plus callers and tests for each symbol. |
| `get_architecture_overview(top?)` | Languages, packages, hub functions, entry points, orphans, external imports. |
| `find_dead_code(limit?)` | Callables with no callers, tests or subclasses. |

A typical review turn: `detect_changes_tool` → pick the top few by `risk` →
`get_review_context` on those qnames → `get_impact_radius` on anything with high fan-in.

## How it works

```
source files ──parse──▶ nodes + edges ──store──▶ SQLite (+FTS5)
                 │                                   ▲
   python: ast   │                                   │ link
   js/ts: tree-sitter                                │
                 └──▶ unresolved edges ("helper") ───┘ unique-name resolution,
                                                       test-coverage inference
```

**Nodes** are modules, classes, functions and methods, keyed by a dotted qname derived
from the file path (`pkg/billing.py` → `pkg.billing`, `web/routes/index.ts` → `web.routes`).

**Edges** are `defines`, `imports`, `calls`, `inherits` and `tests`. Parsers resolve what
they can from the file's own import table (`from pkg import models; models.save()` →
`pkg.models.save`, `this.rate()` → the enclosing class). Anything else is stored as a
bare name and upgraded by the linker when exactly one definition with that name exists
in the repository. Ambiguous names stay unresolved rather than guessed, and the CLI marks
them.

**Risk** in `review` is a small additive score: many callers, no covering tests, high
fan-out, large hunks. It is meant to order your attention, not to judge the change.

**Incremental indexing** hashes file contents; unchanged files are skipped, deleted files
are removed, and the link pass re-runs only when something changed.

| Module | Role |
| --- | --- |
| `codegraph/parsers/` | `python_parser.py` (ast) and `js_parser.py` (tree-sitter) emit nodes and edges per file |
| `codegraph/indexer.py` | Walks the repo, hashes files, re-parses only what changed |
| `codegraph/linker.py` | Resolves bare-name edges and infers `tests` edges |
| `codegraph/store.py` | SQLite + FTS5 storage at `.codegraph/graph.db` |
| `codegraph/queries.py` | Relationship patterns, impact radius, architecture overview, dead code |
| `codegraph/review.py` | Git diff → risk-ranked touched symbols plus review context |
| `codegraph/cli.py`, `codegraph/server.py` | `codegraph` CLI and `codegraph-mcp` server |

## Limits

- No type inference. `self.client.send()` and `obj.method()` resolve only when the
  method name is unique across the repo and not a common builtin name (`get`, `run`,
  `send`, ...). Until then they show as `*.send` and are flagged unresolved.
- Dynamic dispatch, decorators that rewrite functions, `getattr`, and re-exports are
  invisible.
- JS default imports bind to the module qname, so `def.run()` becomes `<module>.run`.
- One repository per database; monorepos work but qnames are rooted at the index root.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest -q
uv run ruff check . && uv run mypy codegraph
```

The test-suite indexes a small fixture repository under `tests/fixtures/sample_repo`
and checks parsing, linking, queries, diff analysis, the CLI and the MCP tool surface.

## License

MIT
