"""CLI commands ``find``, ``grep``, ``semantic``, ``embed`` and ``related``.

The terminal side of :mod:`codegraph.tools.search`; same operations, one line per hit.
"""

from __future__ import annotations

import argparse
import json
import sys

from codegraph.commands import AddCommand
from codegraph.search import embeddings
from codegraph.search.grep import grep_code
from codegraph.search.hybrid import hybrid_search
from codegraph.search.related import find_related
from codegraph.search.snippet import SymbolNotFound
from codegraph.search.structured import SORT_KEYS, GraphQuery, parse_kinds, search_graph
from codegraph.store import GraphStore, default_db_path


def register(add: AddCommand) -> None:
    p = add("find", cmd_find, "filter symbols by name, kind, file, fan-in/out and tests")
    p.add_argument("--name", help="regex searched in the bare name")
    p.add_argument("--qname", help="regex searched in the qualified name")
    p.add_argument("--kind", help="function, method, class, module, ..., or Callable")
    p.add_argument("--file", help="fnmatch glob on the repo-relative path, e.g. 'pkg/*'")
    p.add_argument("--language")
    p.add_argument("--min-fan-in", type=int)
    p.add_argument("--max-fan-in", type=int)
    p.add_argument("--min-fan-out", type=int)
    p.add_argument("--max-fan-out", type=int)
    tests = p.add_mutually_exclusive_group()
    tests.add_argument("--tested", dest="has_tests", action="store_const", const=True)
    tests.add_argument("--untested", dest="has_tests", action="store_const", const=False)
    p.add_argument("--sort", choices=SORT_KEYS, default="name")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--case-sensitive", action="store_true")

    p = add("grep", cmd_grep, "search indexed files; each hit names its enclosing symbol")
    p.add_argument("pattern")
    p.add_argument("--regex", action="store_true", help="treat the pattern as a regex")
    p.add_argument("-i", "--ignore-case", action="store_true")
    p.add_argument("--file", help="fnmatch glob on the repo-relative path")
    p.add_argument("-C", "--context", type=int, default=0, help="lines around each hit")
    p.add_argument("--max-per-file", type=int, default=20)
    p.add_argument("--max-total", type=int, default=200)

    p = add("semantic", cmd_semantic, "hybrid BM25 / TF-IDF / FTS5 (+ embeddings) search")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--kind")
    p.add_argument("--model", help="embedding model to blend in (default: the stored one)")

    p = add("embed", cmd_embed, "compute embeddings (needs codegraph-mcp[embeddings])")
    p.add_argument("--model", default=None, help=f"default: {embeddings.DEFAULT_MODEL}")
    p.add_argument("--batch-size", type=int, default=64)

    p = add("related", cmd_related, "symbols with similar vocabulary to a target")
    p.add_argument("target", help="qname, unique bare name, or path:line")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--min-score", type=float, default=0.0)


def _open(args: argparse.Namespace) -> GraphStore:
    return GraphStore(default_db_path(args.root))


def cmd_find(args: argparse.Namespace) -> int:
    try:
        query = GraphQuery(
            name_pattern=args.name,
            qname_pattern=args.qname,
            kinds=parse_kinds(args.kind),
            file_glob=args.file,
            language=args.language,
            min_fan_in=args.min_fan_in,
            max_fan_in=args.max_fan_in,
            min_fan_out=args.min_fan_out,
            max_fan_out=args.max_fan_out,
            has_tests=args.has_tests,
            sort=args.sort,
            limit=args.limit,
            offset=args.offset,
            case_sensitive=args.case_sensitive,
        )
        with _open(args) as store:
            page = search_graph(store, query)
    except ValueError as exc:
        print(f"find: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(page.as_dict(), indent=2))
        return 0 if page.hits else 1
    for hit in page.hits:
        node = hit.node
        print(
            f"{node.kind:<8} {node.qname}  ({node.path}:{node.line_start})  "
            f"in={hit.fan_in} out={hit.fan_out} span={node.span} tests={hit.tests}"
        )
    if not page.hits:
        print("no matches", file=sys.stderr)
        return 1
    first = page.offset + 1
    print(f"{first}-{page.offset + len(page.hits)} of {page.total}", file=sys.stderr)
    return 0


def cmd_grep(args: argparse.Namespace) -> int:
    try:
        with _open(args) as store:
            result = grep_code(
                store,
                args.root,
                args.pattern,
                regex=args.regex,
                case_sensitive=not args.ignore_case,
                file_glob=args.file,
                context=args.context,
                max_per_file=args.max_per_file,
                max_total=args.max_total,
            )
    except ValueError as exc:
        print(f"grep: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return 0 if result.matches else 1
    for match in result.matches:
        for offset, text in enumerate(match.before):
            print(f"{match.path}-{match.line - len(match.before) + offset}-  {text}")
        symbol = f"[{match.symbol}]" if match.symbol else "[-]"
        print(f"{match.path}:{match.line}:{match.column}  {symbol}  {match.text}")
        for offset, text in enumerate(match.after, start=1):
            print(f"{match.path}-{match.line + offset}-  {text}")
    for rel in result.unreadable:
        print(f"unreadable: {rel}", file=sys.stderr)
    summary = f"{len(result.matches)} matches in {result.files_matched} files"
    print(summary + (" (truncated)" if result.truncated else ""), file=sys.stderr)
    return 0 if result.matches else 1


def cmd_semantic(args: argparse.Namespace) -> int:
    try:
        kinds = parse_kinds(args.kind)
        with _open(args) as store:
            encoder, note = embeddings.search_encoder(store, args.model)
            result = hybrid_search(
                store, args.query, limit=args.limit, kinds=kinds or None, encoder=encoder
            )
    except (ValueError, embeddings.EmbeddingsUnavailable) as exc:
        print(f"semantic: {exc}", file=sys.stderr)
        return 2
    if args.json:
        payload = result.as_dict()
        if note:
            payload["notes"] = [*result.notes, note]
        print(json.dumps(payload, indent=2))
        return 0 if result.hits else 1
    for message in (*result.notes, note):
        if message:
            print(f"note: {message}", file=sys.stderr)
    print(f"backend: {result.backend}")
    for hit in result.hits:
        node = hit.node
        ranks = " ".join(f"{name}#{rank}" for name, rank in hit.ranks.items())
        print(f"{hit.score:.4f}  {node.kind:<8} {node.qname}  ({node.path}:{node.line_start})")
        print(f"        {ranks}")
    if not result.hits:
        print("no matches", file=sys.stderr)
        return 1
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    if args.batch_size < 1:
        print("--batch-size must be at least 1", file=sys.stderr)
        return 2
    model = args.model or embeddings.default_model()
    try:
        encoder = embeddings.load_encoder(model)
        with _open(args) as store:
            stats = embeddings.embed_graph(store, encoder, batch_size=args.batch_size)
    except embeddings.EmbeddingsUnavailable as exc:
        print(f"embed: {exc}", file=sys.stderr)
        return 2
    payload = {
        "model": stats.model,
        "embedded": stats.embedded,
        "unchanged": stats.unchanged,
        "removed": stats.removed,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"{stats.model}: embedded {stats.embedded}, unchanged {stats.unchanged}, "
            f"removed {stats.removed}"
        )
    return 0


def cmd_related(args: argparse.Namespace) -> int:
    try:
        with _open(args) as store:
            result = find_related(store, args.target, limit=args.limit, min_score=args.min_score)
    except SymbolNotFound as exc:
        print(f"related: {exc}", file=sys.stderr)
        for candidate in exc.candidates:
            print(f"  {candidate}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"related: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return 0 if result.hits else 1
    print(f"target: {result.target.qname}")
    for hit in result.hits:
        flags = [
            flag
            for flag, on in (("connected", hit.connected), ("same module", hit.same_module))
            if on
        ]
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(f"{hit.score:.3f}  {hit.node.qname}  ({hit.node.path}:{hit.node.line_start}){suffix}")
    if not result.hits:
        print(result.note or "no related symbols", file=sys.stderr)
        return 1
    return 0


__all__ = ["cmd_embed", "cmd_find", "cmd_grep", "cmd_related", "cmd_semantic", "register"]
