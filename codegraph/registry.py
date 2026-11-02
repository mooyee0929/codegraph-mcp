"""An address book of indexed repositories, shared by every codegraph process on a machine.

Each repository keeps its own graph at ``<repo>/.codegraph/graph.db``. The registry at
``$CODEGRAPH_HOME/registry.json`` (default ``~/.codegraph``) only records where those
repositories are, so tools can search several of them at once and match one service's
outbound HTTP calls to the routes another service serves. It never holds graph data:
losing the file loses nothing that ``codegraph register`` cannot rebuild.

Other repositories' databases are read, never migrated: a database written by another
schema version is reported as needing a re-index instead of being silently rebuilt
from under its owner.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from codegraph.models import Node
from codegraph.store import SCHEMA_VERSION, GraphStore, default_db_path

HOME_ENV = "CODEGRAPH_HOME"
REGISTRY_FILENAME = "registry.json"
REGISTRY_FORMAT = 1


class RegistryError(ValueError):
    """Bad input, or a registry file that cannot be read."""


@dataclass(frozen=True, slots=True)
class RepoEntry:
    name: str
    root: str
    added_at: float


@dataclass(frozen=True, slots=True)
class RepoStatus:
    name: str
    root: str
    indexed: bool
    files: int
    nodes: int
    edges: int
    last_indexed: float | None
    problem: str | None = None


@dataclass(frozen=True, slots=True)
class CrossRepoHit:
    repo: str
    node: Node
    score: float


@dataclass(frozen=True, slots=True)
class CrossRepoLink:
    """One repo's unresolved outbound call matched to a route another repo serves."""

    caller_repo: str
    caller: str
    call_route: str
    path: str
    line: int
    target_repo: str
    route: str
    handlers: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class DeleteResult:
    repo: str
    files: tuple[str, ...]
    deleted: bool


# -- registry file ---------------------------------------------------------


def codegraph_home(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    raw = env.get(HOME_ENV, "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".codegraph"


def registry_path(home: Path | None = None) -> Path:
    return (home if home is not None else codegraph_home()) / REGISTRY_FILENAME


def load_registry(path: Path) -> list[RepoEntry]:
    if not path.is_file():
        return []
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"{path} is not valid JSON: {exc}") from exc
    repos = payload.get("repos") if isinstance(payload, dict) else None
    if not isinstance(repos, list):
        raise RegistryError(f"{path} has no 'repos' list")
    entries: list[RepoEntry] = []
    for item in repos:
        if not isinstance(item, dict):
            raise RegistryError(f"{path} has a malformed entry: {item!r}")
        name, root, added = item.get("name"), item.get("root"), item.get("added_at", 0.0)
        if not isinstance(name, str) or not isinstance(root, str):
            raise RegistryError(f"{path} has an entry without name/root: {item!r}")
        if not isinstance(added, int | float):
            raise RegistryError(f"{path} has a non-numeric added_at: {item!r}")
        entries.append(RepoEntry(name=name, root=root, added_at=float(added)))
    return entries


def save_registry(path: Path, entries: list[RepoEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": REGISTRY_FORMAT,
        "repos": [
            {"name": e.name, "root": e.root, "added_at": e.added_at}
            for e in sorted(entries, key=lambda e: e.name)
        ],
    }
    _atomic_write(path, json.dumps(payload, indent=2) + "\n")


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def find_repo(entries: list[RepoEntry], key: str) -> RepoEntry | None:
    """Look a repo up by registered name, or by its root path."""
    for entry in entries:
        if entry.name == key:
            return entry
    resolved = str(Path(key).expanduser().resolve())
    for entry in entries:
        if entry.root == resolved:
            return entry
    return None


def register_repo(
    path: Path,
    root: Path,
    name: str | None = None,
    *,
    clock: Callable[[], float] = time.time,
) -> RepoEntry:
    """Add ``root`` (or rename it, when already registered) and return its entry."""
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise RegistryError(f"not a directory: {root}")
    chosen = (name or resolved.name).strip()
    if not chosen:
        raise RegistryError("repository name must not be empty")
    entries = load_registry(path)
    clash = next((e for e in entries if e.name == chosen and e.root != str(resolved)), None)
    if clash is not None:
        raise RegistryError(f"name {chosen!r} is already used by {clash.root}")
    existing = next((e for e in entries if e.root == str(resolved)), None)
    entry = RepoEntry(
        name=chosen,
        root=str(resolved),
        added_at=existing.added_at if existing else clock(),
    )
    save_registry(path, [e for e in entries if e.root != str(resolved)] + [entry])
    return entry


def unregister_repo(path: Path, key: str) -> RepoEntry | None:
    entries = load_registry(path)
    entry = find_repo(entries, key)
    if entry is None:
        return None
    save_registry(path, [e for e in entries if e is not entry])
    return entry


# -- reading other repositories' databases ---------------------------------


def _schema_problem(db: Path) -> str | None:
    """Why ``db`` cannot be read as-is, or None when it is a current codegraph database."""
    if not db.is_file():
        return "not indexed"
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return f"cannot open: {exc}"
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    except sqlite3.Error as exc:
        return f"not a codegraph database: {exc}"
    finally:
        conn.close()
    if row is None or str(row[0]) != str(SCHEMA_VERSION):
        found = "unknown" if row is None else str(row[0])
        return f"schema v{found}, this codegraph reads v{SCHEMA_VERSION}; re-index it"
    return None


@contextmanager
def open_repo_store(entry: RepoEntry) -> Iterator[GraphStore | None]:
    """The repo's store, or None when it is missing or from another schema version."""
    db = default_db_path(Path(entry.root))
    if _schema_problem(db) is not None:
        yield None
        return
    with GraphStore(db) as store:
        yield store


def repo_status(entry: RepoEntry) -> RepoStatus:
    db = default_db_path(Path(entry.root))
    problem = _schema_problem(db)
    if problem is not None:
        return RepoStatus(entry.name, entry.root, False, 0, 0, 0, None, problem)
    with GraphStore(db) as store:
        counts = store.counts()
        row = store.connection.execute("SELECT MAX(indexed_at) FROM files").fetchone()
    last = float(row[0]) if row is not None and row[0] is not None else None
    return RepoStatus(
        name=entry.name,
        root=entry.root,
        indexed=True,
        files=counts["files"],
        nodes=counts["nodes"],
        edges=counts["edges"],
        last_indexed=last,
    )


def cross_repo_search(entries: list[RepoEntry], query: str, limit: int = 20) -> list[CrossRepoHit]:
    """Full-text search every registered repo; exact name matches rank first.

    FTS ranks are not comparable between databases, so hits are ordered by a simple
    name-match score and then by their rank inside their own repo.
    """
    if not query.strip():
        raise RegistryError("query must not be empty")
    ranked: list[tuple[float, int, str, CrossRepoHit]] = []
    for entry in entries:
        with open_repo_store(entry) as store:
            if store is None:
                continue
            for rank, node in enumerate(store.search_nodes(query, limit=limit)):
                score = _match_score(node, query)
                ranked.append((score, rank, entry.name, CrossRepoHit(entry.name, node, score)))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [item[3] for item in ranked[:limit]]


def _match_score(node: Node, query: str) -> float:
    needle = query.strip().lower()
    name = node.name.lower()
    if name == needle:
        return 3.0
    if name.startswith(needle):
        return 2.0
    if needle in node.qname.lower():
        return 1.5
    return 1.0


# -- cross-repo HTTP links -------------------------------------------------

_PARAM = re.compile(r"^(\{[^}]*\}|:.+|<[^>]+>|\[[^\]]+\]|\*|%[sd]|\$\{[^}]*\})$")
_CATCH_ALL = re.compile(r"^(\*\*|\*.+|\[\.\.\..+\]|\{[^}]*:path\}|\{\*[^}]*\})$")
_ANY_METHODS = frozenset({"ANY", "*", "ALL"})


@dataclass(frozen=True, slots=True)
class RouteKey:
    method: str
    segments: tuple[str, ...]


def parse_route_qname(qname: str) -> RouteKey | None:
    """``route:GET /users/{id}`` -> ``RouteKey("GET", ("users", "{id}"))``.

    Full URLs lose their scheme and host, query strings are dropped, and a qname
    without a method matches any method.
    """
    if not qname.startswith("route:"):
        return None
    body = qname.removeprefix("route:").strip()
    method, _, path = body.partition(" ")
    if not path:
        method, path = "ANY", method
    path = path.strip()
    if "://" in path:
        path = urlsplit(path).path
    path = path.split("?", 1)[0].split("#", 1)[0]
    segments = tuple(s for s in path.split("/") if s)
    return RouteKey(method=method.upper() or "ANY", segments=segments)


def _is_param(segment: str) -> bool:
    return bool(_PARAM.match(segment)) or "${" in segment


def match_routes(call: RouteKey, route: RouteKey) -> float | None:
    """Confidence that ``call`` reaches ``route``; None when they cannot match.

    Literal segments must be equal; a path parameter on either side matches any one
    segment; a trailing catch-all on the route matches the rest. Every parameter used
    and a method wildcard lower the confidence a little.
    """
    if call.method != route.method and not (_ANY_METHODS & {call.method, route.method}):
        return None
    confidence = 1.0 if call.method == route.method else 0.9
    params = 0
    for index, route_segment in enumerate(route.segments):
        if _CATCH_ALL.match(route_segment) and index == len(route.segments) - 1:
            if len(call.segments) <= index:
                return None
            return max(0.5, confidence - 0.1 * (params + 1))
        if index >= len(call.segments):
            return None
        call_segment = call.segments[index]
        if call_segment == route_segment:
            continue
        if _is_param(call_segment) or _is_param(route_segment):
            params += 1
            continue
        return None
    if len(call.segments) != len(route.segments):
        return None
    return max(0.5, confidence - 0.1 * params)


def cross_repo_links(
    entries: list[RepoEntry], source: str | None = None, min_confidence: float = 0.5
) -> list[CrossRepoLink]:
    """Match each repo's unresolved ``http_calls`` edges to other repos' route nodes."""
    calls: dict[str, list[tuple[str, str, str, int, RouteKey]]] = {}
    routes: dict[str, list[tuple[str, RouteKey, tuple[str, ...]]]] = {}
    for entry in entries:
        with open_repo_store(entry) as store:
            if store is None:
                continue
            for edge in store.all_edges(kinds=["http_calls"]):
                key = parse_route_qname(edge.dst)
                if not edge.resolved and key is not None:
                    calls.setdefault(entry.name, []).append(
                        (edge.src, edge.dst, edge.path, edge.line, key)
                    )
            for node in store.all_nodes(kinds=["route"]):
                key = parse_route_qname(node.qname)
                if key is None:
                    continue
                handlers = tuple(sorted({e.dst for e in store.edges_from(node.qname, "handles")}))
                routes.setdefault(entry.name, []).append((node.qname, key, handlers))

    links: list[CrossRepoLink] = []
    for caller_repo, repo_calls in sorted(calls.items()):
        if source is not None and caller_repo != source:
            continue
        for caller, call_route, path, line, call_key in repo_calls:
            for target_repo, repo_routes in sorted(routes.items()):
                if target_repo == caller_repo:
                    continue
                for route_qname, route_key, handlers in repo_routes:
                    confidence = match_routes(call_key, route_key)
                    if confidence is None or confidence < min_confidence:
                        continue
                    links.append(
                        CrossRepoLink(
                            caller_repo=caller_repo,
                            caller=caller,
                            call_route=call_route,
                            path=path,
                            line=line,
                            target_repo=target_repo,
                            route=route_qname,
                            handlers=handlers,
                            confidence=round(confidence, 2),
                        )
                    )
    links.sort(key=lambda link: (-link.confidence, link.caller_repo, link.path, link.line))
    return links


# -- deleting a repo's database ---------------------------------------------


def delete_project_db(entry: RepoEntry, *, confirm: bool) -> DeleteResult:
    """Remove the repo's graph database files; without ``confirm`` only list them.

    Only ``graph.db`` and its WAL/SHM side files go: notes, the wiki and a committed
    snapshot under ``.codegraph/`` are the user's and stay.
    """
    db = default_db_path(Path(entry.root))
    files = tuple(
        str(candidate)
        for candidate in (Path(f"{db}{suffix}") for suffix in ("", "-wal", "-shm"))
        if candidate.exists()
    )
    if confirm:
        for name in files:
            Path(name).unlink(missing_ok=True)
    return DeleteResult(repo=entry.name, files=files, deleted=confirm and bool(files))


__all__ = [
    "CrossRepoHit",
    "CrossRepoLink",
    "DeleteResult",
    "RegistryError",
    "RepoEntry",
    "RepoStatus",
    "RouteKey",
    "codegraph_home",
    "cross_repo_links",
    "cross_repo_search",
    "delete_project_db",
    "find_repo",
    "load_registry",
    "match_routes",
    "open_repo_store",
    "parse_route_qname",
    "register_repo",
    "registry_path",
    "repo_status",
    "save_registry",
    "unregister_repo",
]
