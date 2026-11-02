"""CLI commands that write documents: adr, note, wiki, visualize, ui, export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from codegraph import adr, exports, notes, visualize, wiki
from codegraph.commands import AddCommand
from codegraph.store import DB_DIRNAME, GraphStore


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, default=str))


def _error(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


# -- adr -------------------------------------------------------------------------------


def cmd_adr(args: argparse.Namespace) -> int:
    try:
        result = adr.manage_adr(
            args.root,
            args.action,
            args.number or 0,
            title=args.title or "",
            status=args.status or "",
            context=args.context or "",
            decision=args.decision or "",
            consequences=args.consequences or "",
            directory=args.dir or "",
        )
    except adr.AdrError as exc:
        return _error(str(exc))
    if args.json:
        _print_json(result)
        return 0
    if args.action == "list":
        records = result.get("adrs")
        listed = records if isinstance(records, list) else []
        for record in listed:
            if isinstance(record, dict):
                print(f"{record['number']:>4}  {record['status']:<11} {record['title']}")
        if not listed:
            print(f"no ADRs in {result.get('directory')}", file=sys.stderr)
            return 1
        return 0
    if args.action == "get":
        print((args.root / str(result["path"])).read_text(encoding="utf-8"), end="")
        return 0
    for label in ("created", "updated", "superseded"):
        record = result.get(label)
        if isinstance(record, dict):
            print(f"{label} {record['path']}  [{record['status']}]")
    return 0


# -- note --------------------------------------------------------------------------------


def cmd_note(args: argparse.Namespace) -> int:
    if args.action == "save":
        if not args.text or not args.answer:
            return _error("note save needs the question and --answer")
        try:
            note = notes.save_note(
                args.root, args.text, args.answer, tuple(args.qname), kind=args.kind
            )
        except notes.NoteError as exc:
            return _error(str(exc))
        rel = note.path.relative_to(args.root).as_posix()
        if args.json:
            _print_json({"path": rel, "question": note.question, "qnames": list(note.qnames)})
        else:
            print(f"saved {rel}")
        return 0
    if args.action == "search":
        if not args.text:
            return _error("note search needs a query")
        found = [h.note for h in notes.search_notes(args.root, args.text, limit=args.limit)]
    else:
        found = notes.list_notes(args.root)[: args.limit]
    if args.json:
        _print_json(
            [
                {
                    "path": n.path.relative_to(args.root).as_posix(),
                    "question": n.question,
                    "answer": n.answer,
                    "qnames": list(n.qnames),
                    "created": n.created,
                }
                for n in found
            ]
        )
        return 0 if found else 1
    for note in found:
        print(f"{note.created or '-':<21} {note.question}")
        print(f"{'':<21} {note.path.relative_to(args.root).as_posix()}")
    if not found:
        print("no notes found", file=sys.stderr)
        return 1
    return 0


# -- wiki ---------------------------------------------------------------------------------


def cmd_wiki(args: argparse.Namespace) -> int:
    if args.page:
        try:
            text = wiki.read_wiki_page(args.root, args.page)
        except wiki.WikiError as exc:
            return _error(str(exc))
        if text is None:
            print(f"no wiki page {args.page!r}; run `codegraph wiki` first", file=sys.stderr)
            return 1
        print(text, end="")
        return 0
    with GraphStore.open(args.root) as store:
        result = wiki.generate_wiki(args.root, store)
    if args.json:
        _print_json(
            {
                "directory": str(result.directory),
                "pages": list(result.pages),
                "removed": list(result.removed),
            }
        )
    else:
        print(f"wrote {len(result.pages)} pages to {result.directory}")
        if result.removed:
            print(f"removed {len(result.removed)} stale pages")
    return 0


# -- visualize / ui ------------------------------------------------------------------------


def cmd_visualize(args: argparse.Namespace) -> int:
    out = Path(args.output).expanduser().resolve() if args.output else None
    try:
        with GraphStore.open(args.root) as store:
            path, data = visualize.write_visualization(
                args.root, store, out=out, max_nodes=args.max_nodes, color_by=args.color_by
            )
    except visualize.VisualizeError as exc:
        return _error(str(exc))
    if args.json:
        _print_json({"path": str(path), "nodes": len(data.nodes), "total": data.total_nodes})
    else:
        shown = f"{len(data.nodes)} of {data.total_nodes}" if data.truncated else len(data.nodes)
        print(f"wrote {path} ({shown} nodes)")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    if not 0 <= args.port <= 65535:
        return _error(f"invalid port {args.port}")
    try:
        visualize.serve_ui(args.root, port=args.port, max_nodes=args.max_nodes)
    except OSError as exc:
        return _error(f"cannot serve on 127.0.0.1:{args.port}: {exc}")
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
    return 0


# -- export ----------------------------------------------------------------------------------


def cmd_export(args: argparse.Namespace) -> int:
    try:
        fmt = exports.normalise_format(args.format)
    except exports.ExportError as exc:
        return _error(str(exc))
    cap = args.max_nodes or (exports.MERMAID_DEFAULT_CAP if fmt == "mermaid" else 0)
    with GraphStore.open(args.root) as store:
        data = exports.load_graph(store, max_nodes=cap or None)
    files = exports.render(data, fmt)
    if args.stdout:
        for content in files.values():
            sys.stdout.write(content)
        return 0
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else args.root / DB_DIRNAME / "exports"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, content in files.items():
        (out_dir / name).write_text(content, encoding="utf-8")
        written.append(str(out_dir / name))
    if args.json:
        _print_json({"format": fmt, "written": written, "nodes": len(data.nodes)})
    else:
        for path in written:
            print(f"wrote {path}")
    return 0


def register(add: AddCommand) -> None:
    p = add("adr", cmd_adr, "architecture decision records under docs/adr")
    p.add_argument("action", choices=adr.ACTIONS)
    p.add_argument("number", nargs="?", type=int, help="ADR number for get/update/supersede")
    p.add_argument("--title")
    p.add_argument("--status", help=f"one of {', '.join(adr.STATUSES)}")
    p.add_argument("--context")
    p.add_argument("--decision")
    p.add_argument("--consequences")
    p.add_argument("--dir", help="ADR directory (default: .adr-dir or docs/adr)")

    p = add("note", cmd_note, "save, search or list remembered answers")
    p.add_argument("action", choices=("save", "search", "list"))
    p.add_argument("text", nargs="?", help="the question (save) or the query (search)")
    p.add_argument("--answer", help="the answer to save")
    p.add_argument("--qname", action="append", default=[], help="related symbol (repeatable)")
    p.add_argument("--kind", default="answer")
    p.add_argument("--limit", type=int, default=10)

    p = add("wiki", cmd_wiki, "generate the markdown wiki, or print one page")
    p.add_argument("--page", help="print this page instead of regenerating")

    p = add("visualize", cmd_visualize, "write a self-contained interactive HTML graph")
    p.add_argument("--max-nodes", type=int, default=visualize.DEFAULT_MAX_NODES)
    p.add_argument("--color-by", choices=(*visualize.COLOR_MODES, "auto"), default="auto")
    p.add_argument("--output", help="file to write (default: .codegraph/graph.html)")

    p = add("ui", cmd_ui, "serve the interactive graph and a JSON API on 127.0.0.1")
    p.add_argument("--port", type=int, default=visualize.DEFAULT_PORT)
    p.add_argument("--max-nodes", type=int, default=visualize.DEFAULT_MAX_NODES)

    p = add("export", cmd_export, "export the graph (json, graphml, dot, mermaid, csv, cypher)")
    p.add_argument("format", help=f"one of {', '.join(exports.FORMATS)} (neo4j = cypher)")
    p.add_argument("--max-nodes", type=int, default=0, help="keep the top-N nodes by degree")
    p.add_argument("--output-dir", help="directory to write into (default: .codegraph/exports)")
    p.add_argument("--stdout", action="store_true", help="print instead of writing files")


__all__ = ["register"]
