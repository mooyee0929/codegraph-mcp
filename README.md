# codegraph-mcp

A code knowledge graph you can query from an agent. Index a repository once, then ask
structural questions — *who calls this, what breaks if I change it, which tests cover
it, which service calls this endpoint, what does the architecture look like* — through
an [MCP](https://modelcontextprotocol.io) server or a CLI, instead of grepping and
reading whole files.

It parses **27 programming languages** plus Vue / Svelte components, Jupyter notebooks,
Dockerfiles, docker-compose, Kubernetes, Kustomize, Terraform and Makefiles. Everything
lives in one SQLite file under `.codegraph/` in the repo you index: no daemon, no
service, no network access, and no embeddings unless you opt in.

Version 0.2 merges the feature sets of
[code-review-graph](https://github.com/tirth8205/code-review-graph) and
[codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp) into this
codebase; see [THIRD_PARTY.md](THIRD_PARTY.md).

## Why

Agents burn tokens re-discovering the same facts about a codebase: opening files to find
callers, tracing imports by hand, guessing which tests matter. A graph answers those in a
few hundred tokens with exact `path:line` locations, turns a diff into a ranked list of
"these symbols have many callers and no tests — look here first", and connects things
grep cannot: an HTTP client call to the route that serves it, a Kafka producer to its
consumers, a Kubernetes Service to the Deployment it selects.

## Install

```bash
uv pip install git+https://github.com/mooyee0929/codegraph-mcp
# optional: vector search with sentence-transformers
uv pip install "codegraph-mcp[embeddings] @ git+https://github.com/mooyee0929/codegraph-mcp"
# or, from a clone
uv pip install -e ".[dev]"
```

Requires Python 3.11+. Grammars come from `tree-sitter-language-pack`; nothing is
downloaded at run time.

## Quick start

```bash
codegraph index                  # build or refresh .codegraph/graph.db (incremental)
codegraph architecture           # languages, layers, entry points, routes, communities
codegraph review --base main     # risk-ranked symbols touched by your branch
codegraph install                # register the MCP server with Claude Code
```

Every command accepts `--root <dir>` (default: cwd) and `--json`. Targets can be a full
qname (`pkg.billing.Invoice.total`), a bare name when it is unique (`total`), a suffix
(`Invoice.total`), or a location (`pkg/billing.py:10`).

## CLI

| Area | Commands |
| --- | --- |
| Index | `index [--force] [--skip-pass NAME]`, `status`, `passes`, `watch`, `languages`, `resources` |
| Look up | `search`, `find`, `grep`, `semantic`, `related`, `show`, `query <pattern> <target>`, `cypher "<query>"`, `schema` |
| Navigate | `impact`, `trace`, `traverse`, `context`, `flows`, `flow` |
| Review | `review [--base REV] [--context]`, `change-impact`, `dead`, `clones`, `large`, `suggest`, `rename`, `apply` |
| Architecture | `arch`, `architecture`, `communities`, `community`, `hubs`, `bridges`, `gaps`, `surprises`, `questions` |
| Services | `routes`, `channels`, `ingest-traces <spans.json>` |
| History | `hotspots`, `co-changes` |
| Docs and output | `wiki`, `visualize`, `ui [--port 9749]`, `export json\|graphml\|dot\|mermaid\|csv\|cypher`, `adr`, `note` |
| Repos and setup | `register`, `unregister`, `repos`, `snapshot export\|import`, `install`, `uninstall`, `serve`, `embed` |

`codegraph <command> --help` documents each one. A few examples:

```bash
codegraph query callers_of Invoice.total
codegraph trace charge --direction inbound --depth 3
codegraph cypher "MATCH (r:Route)-[:HANDLES]->(f) WHERE r.name STARTS WITH 'POST' RETURN r.name, f.qname"
codegraph find --kind method --min-fan-in 5 --untested
codegraph grep "TODO" --file "src/*"
codegraph semantic "retry failed payment"
codegraph routes --method GET
codegraph hotspots
codegraph visualize && open .codegraph/graph.html
```

## MCP server

```bash
codegraph-mcp --root /path/to/repo                 # stdio transport
codegraph-mcp --root . --watch --auto-index        # keep the graph fresh while serving
codegraph-mcp --root . --tools get_node,trace_path,search_graph   # expose only these
```

`codegraph install` registers the server with Claude Code (through `claude mcp add`
when the CLI is present, otherwise by merging `.mcp.json`); `--hooks` also adds a
project-level PostToolUse hook that re-indexes after edits, `--dry-run` shows the plan,
and `codegraph uninstall` removes exactly what was added. Manual setup:

```json
{
  "mcpServers": {
    "codegraph": { "command": "codegraph-mcp", "args": ["--root", ".", "--auto-index"] }
  }
}
```

### Tools

62 tools. With many clients each tool schema costs context, so `--tools` /
`CODEGRAPH_TOOLS` lets you expose a subset.

| Area | Tools |
| --- | --- |
| Index | `index_repository_tool`, `graph_status`, `list_languages`, `list_resources` |
| Look up | `search_nodes`, `search_graph`, `search_code`, `semantic_search_nodes`, `find_related`, `get_node`, `get_code_snippet`, `query_graph`, `run_cypher`, `get_graph_schema`, `embed_graph` |
| Navigate | `get_impact_radius`, `trace_path`, `traverse_graph`, `get_minimal_context`, `list_flows`, `get_flow`, `get_affected_flows` |
| Review | `detect_changes_tool`, `get_review_context`, `get_change_impact`, `find_dead_code`, `find_clones`, `find_large_functions`, `suggest_refactorings`, `refactor_preview`, `apply_refactor` |
| Architecture | `get_architecture_overview`, `get_architecture`, `list_communities`, `get_community`, `get_hub_nodes`, `get_bridge_nodes`, `get_knowledge_gaps`, `get_surprising_connections`, `get_suggested_questions` |
| Services | `list_routes`, `get_route`, `list_channels`, `get_channel`, `ingest_traces` |
| History | `get_hotspots`, `get_co_changes`, `get_file_history` |
| Docs and output | `generate_wiki`, `get_wiki_page`, `generate_visualization`, `export_graph`, `manage_adr`, `save_note`, `search_notes` |
| Repos | `list_repos`, `register_repo`, `cross_repo_search`, `cross_repo_links`, `delete_project`, `export_snapshot`, `import_snapshot` |

### Prompts

`review_changes`, `pre_merge_check`, `debug_issue`, `architecture_map` and
`onboard_developer` walk the agent through the right tools in order. A typical review:
`detect_changes_tool` → `get_change_impact` → `get_review_context` on the riskiest
symbols → `get_impact_radius` on anything with high fan-in.

## How it works

```
source files ──parse──▶ nodes + edges ──store──▶ SQLite (+FTS5)
                 │                                   │
   python: ast   │                          link ◀───┤ unique-name resolution,
   js/ts + 25 more: tree-sitter                      │ test-coverage inference
   infra: yaml / hcl / dockerfile parsers            │
                                            passes ◀─┘ routes, channels, clones,
                                                       history, communities, flows,
                                                       search index, related edges
```

**Nodes** are modules, classes (any named type: struct, interface, trait, enum),
functions, methods, routes (`route:GET /users/{id}`), channels (`channel:orders`) and
infrastructure resources (`resource:k8s/Deployment/web`). Code qnames come from the
file path (`pkg/billing.py` → `pkg.billing`).

**Edges** are `defines`, `imports`, `calls`, `inherits` and `tests` from parsing, plus
derived ones: `handles` (route → handler), `http_calls` (client → route), `emits` /
`listens_on` (function → channel), `similar_to` (near-clones), `semantically_related`,
`co_changes` (files committed together) and `depends_on` (infrastructure). Heuristic
edges carry a `confidence` and a short `detail`.

**Resolution.** Parsers resolve what the file's own imports tell them; go.mod,
Cargo.toml and package.json workspaces map package names to repo paths. Anything else
is stored as a bare name and linked only when exactly one definition has that name.
Ambiguous names stay unresolved rather than guessed, and the CLI marks them.

**Passes** run after linking, in order, whenever the index changed:

| Order | Pass | Writes |
| --- | --- | --- |
| 12 | `resolve_languages` | cross-file imports and calls for the tree-sitter languages, Kubernetes selectors |
| 20 | `routes` | route nodes for Flask, FastAPI, Django, Express, NestJS, Next.js, Go net/http / gin / echo / chi, Spring, JAX-RS, ASP.NET, Rails, Laravel, gRPC protos, GraphQL schemas |
| 22 | `http_calls` | client call sites (fetch, axios, requests, httpx, Go http, RestTemplate, ...) matched to routes |
| 24 | `channels` | Socket.IO / EventEmitter, Kafka, RabbitMQ, Redis, NATS, Celery, signals, Temporal |
| 30 | `clones` | MinHash + LSH near-duplicate functions |
| 35 | `git_history` | per-file churn and co-change edges from a bounded `git log` |
| 40 | `trace_spans` | re-applies ingested runtime spans |
| 60 | `communities` | deterministic Louvain communities with names, cohesion and boundaries |
| 65 | `flows` | execution flows from entry points with a criticality score |
| 90 | `search_index` | BM25 / TF-IDF index over names, signatures, docs and paths |
| 95 | `semantic_edges` | `semantically_related` edges between similar callables |

Skip any with `codegraph index --skip-pass NAME`.

**Risk** in `review` is a small additive score: many callers, no covering tests, high
fan-out, large hunks. `change-impact` adds the flows and communities a diff touches.
Both are meant to order your attention, not to judge the change.

**Incremental indexing** hashes file contents; unchanged files are skipped, deleted
files are removed, and linking and passes re-run only when something changed. The
database is a cache: when its schema version changes it is rebuilt from source.

## Configuration

| Setting | Effect |
| --- | --- |
| `.codegraphignore` | extra ignore patterns, one per line, matched against names and repo-relative paths |
| `CODEGRAPH_TOOLS` | comma-separated tool names the MCP server exposes |
| `CODEGRAPH_WATCH=1`, `CODEGRAPH_AUTO_INDEX=1` | same as `--watch` / `--auto-index` |
| `CODEGRAPH_HOME` | where the multi-repo registry lives (default `~/.codegraph`) |
| `CODEGRAPH_EMBEDDING_MODEL` | sentence-transformers model for `embed` (default `all-MiniLM-L6-v2`) |

**Team snapshot.** `codegraph snapshot export` writes `.codegraph/graph.db.gz` and marks
it `binary merge=ours` in `.gitattributes`. Commit it, and teammates' `--auto-index`
restores it and only re-parses their local changes.

**ADRs and notes.** `codegraph adr` keeps architecture decision records as markdown in
`docs/adr/` (or the directory named in `.adr-dir`); `codegraph note` stores question and
answer pairs in `.codegraph/notes/` so later sessions can search them.

## Limits

- No type inference. `self.client.send()` and `obj.method()` resolve only when the
  method name is unique across the repo and not a common library name (`get`, `run`,
  `send`, `execute`, ...). Until then they show as `*.send` and are flagged unresolved.
- Dynamic dispatch, decorators that rewrite functions, `getattr`, runtime-built routes
  and re-exports are invisible. Route and channel detection is pattern-based.
- JS default imports bind to the module qname, so `def.run()` becomes `<module>.run`.
- Communities come from single-resolution Louvain, so one can be internally
  disconnected; bridge scores above 400 nodes are estimated from a sample.
- The Cypher subset has no write clauses, `UNWIND`, `UNION`, `CASE` or parameters.
- Passes re-run only when indexed files change; after editing only `.proto` files or
  pulling new history, run `codegraph index --force`.
- One repository per database; qnames are rooted at the index root. The registry and
  `cross_repo_*` tools join several.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest -q
uv run ruff check . && uv run ruff format --check . && uv run mypy codegraph
```

Feature code plugs in without touching the core: a module in `codegraph/passes/` that
defines `PASS`, in `codegraph/tools/` that defines `register(server, ctx)`, or in
`codegraph/commands/` that defines `register(add)` is discovered automatically.

## License

MIT. Ideas and designs ported from the two upstream projects are credited in
[THIRD_PARTY.md](THIRD_PARTY.md).
