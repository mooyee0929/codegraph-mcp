"""MCP tools for finding symbols and code.

``search_graph`` filters by structure, ``search_code`` greps indexed files,
``get_code_snippet`` returns one symbol's source, ``semantic_search_nodes`` ranks by
words (and embeddings when computed), ``embed_graph`` computes those embeddings, and
``find_related`` lists symbols with similar vocabulary.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from codegraph.search import embeddings
from codegraph.search.grep import grep_code
from codegraph.search.hybrid import hybrid_search
from codegraph.search.related import find_related as related_to
from codegraph.search.snippet import SymbolNotFound, code_snippet
from codegraph.search.structured import GraphQuery, SortKey, parse_kinds
from codegraph.search.structured import search_graph as structured_search
from codegraph.tools import ToolContext, dump


def register(server: MCPServer[None], ctx: ToolContext) -> None:
    @server.tool()
    def search_graph(
        name_pattern: str | None = None,
        qname_pattern: str | None = None,
        kind: str | None = None,
        file_glob: str | None = None,
        language: str | None = None,
        min_fan_in: int | None = None,
        max_fan_in: int | None = None,
        min_fan_out: int | None = None,
        max_fan_out: int | None = None,
        has_tests: bool | None = None,
        sort: SortKey = "name",
        limit: int = 50,
        offset: int = 0,
        case_sensitive: bool = False,
    ) -> str:
        """Filter symbols by name/qname regex, kind (function, method, class, module,
        route, channel, resource, or Callable; comma-separated), file glob (fnmatch on the
        repo-relative path, e.g. 'src/api/*'), language, fan-in/fan-out bounds (distinct
        callers / callees) and test coverage. Sort by name, fan_in, fan_out or span
        (largest first). Paged: returns total, has_more and one page of results."""
        try:
            query = GraphQuery(
                name_pattern=name_pattern,
                qname_pattern=qname_pattern,
                kinds=parse_kinds(kind),
                file_glob=file_glob,
                language=language,
                min_fan_in=min_fan_in,
                max_fan_in=max_fan_in,
                min_fan_out=min_fan_out,
                max_fan_out=max_fan_out,
                has_tests=has_tests,
                sort=sort,
                limit=limit,
                offset=offset,
                case_sensitive=case_sensitive,
            )
            with ctx.open_store() as store:
                page = structured_search(store, query)
        except ValueError as exc:
            return dump({"error": str(exc)})
        return dump(page.as_dict())

    @server.tool()
    def search_code(
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
        file_glob: str | None = None,
        context: int = 0,
        max_per_file: int = 20,
        max_total: int = 200,
    ) -> str:
        """Grep the indexed source files (literal by default, regex=true for a Python
        regular expression). Every match names its enclosing symbol, so the qname can go
        straight to get_impact_radius or run_cypher. context adds lines before/after
        (max 10); caps limit matches per file and in total, and truncated says if hit."""
        try:
            with ctx.open_store() as store:
                result = grep_code(
                    store,
                    ctx.root,
                    pattern,
                    regex=regex,
                    case_sensitive=case_sensitive,
                    file_glob=file_glob,
                    context=context,
                    max_per_file=max_per_file,
                    max_total=max_total,
                )
        except ValueError as exc:
            return dump({"error": str(exc)})
        return dump(result.as_dict())

    @server.tool()
    def get_code_snippet(target: str, context: int = 0, max_lines: int = 200) -> str:
        """Source of one symbol with line numbers, by qname, unique bare name, suffix
        (Invoice.total) or path:line, plus its direct callers and covering tests.
        An ambiguous name returns the candidate qnames."""
        try:
            with ctx.open_store() as store:
                snippet = code_snippet(
                    store, ctx.root, target, context=context, max_lines=max_lines
                )
        except SymbolNotFound as exc:
            return dump({"error": str(exc), "candidates": list(exc.candidates)})
        except ValueError as exc:
            return dump({"error": str(exc)})
        return dump(snippet.as_dict())

    @server.tool()
    def semantic_search_nodes(
        query: str, limit: int = 20, kind: str | None = None, model: str | None = None
    ) -> str:
        """Rank symbols for a natural-language or identifier query. Fuses BM25 and TF-IDF
        over split identifiers (parseHttpRequest matches 'http request'), SQLite FTS5,
        and stored embeddings when embed_graph has run and sentence-transformers is
        installed. 'backend' names the rankers that answered; each result carries its
        rank in each. kind filters like search_graph."""
        try:
            kinds = parse_kinds(kind)
            with ctx.open_store() as store:
                encoder, note = embeddings.search_encoder(store, model)
                result = hybrid_search(
                    store, query, limit=limit, kinds=kinds or None, encoder=encoder
                )
        except (ValueError, embeddings.EmbeddingsUnavailable) as exc:
            return dump({"error": str(exc)})
        payload = result.as_dict()
        if note:
            payload["notes"] = [*result.notes, note]
        return dump(payload)

    @server.tool()
    def embed_graph(model: str | None = None, batch_size: int = 64) -> str:
        """Compute embeddings for classes, functions, methods, routes and channels so
        semantic_search_nodes can match by meaning. Needs the optional extra
        (uv pip install 'codegraph-mcp[embeddings]'). Only nodes whose text or model
        changed are re-encoded. model defaults to $CODEGRAPH_EMBEDDING_MODEL or
        all-MiniLM-L6-v2."""
        chosen = model or embeddings.default_model()
        try:
            encoder = embeddings.load_encoder(chosen)
            with ctx.open_store() as store:
                stats = embeddings.embed_graph(store, encoder, batch_size=batch_size)
        except (ValueError, embeddings.EmbeddingsUnavailable) as exc:
            return dump({"error": str(exc)})
        return dump(
            {
                "model": stats.model,
                "embedded": stats.embedded,
                "unchanged": stats.unchanged,
                "removed": stats.removed,
                "total": stats.total,
            }
        )

    @server.tool()
    def find_related(target: str, limit: int = 10, min_score: float = 0.0) -> str:
        """Symbols whose names, signatures and docstrings use the same words as target
        (TF-IDF cosine), best first. 'connected' marks pairs already linked by a structural
        edge; unconnected high scorers in other modules are possible duplicates or missing
        links. 'semantic_edge' marks pairs the semantic_edges pass stored (cosine >= 0.8,
        SEMANTICALLY_RELATED in run_cypher)."""
        try:
            with ctx.open_store() as store:
                result = related_to(store, target, limit=limit, min_score=min_score)
        except SymbolNotFound as exc:
            return dump({"error": str(exc), "candidates": list(exc.candidates)})
        except ValueError as exc:
            return dump({"error": str(exc)})
        return dump(result.as_dict())


__all__ = ["register"]
