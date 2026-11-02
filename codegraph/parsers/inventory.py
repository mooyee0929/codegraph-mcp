"""What the index holds per language, and the infrastructure resources it found.

Shared by the ``list_languages`` / ``list_resources`` MCP tools and the ``languages`` /
``resources`` CLI commands, so both report the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass

from codegraph.parsers.languages import supported_languages
from codegraph.parsers.resources import RESOURCE_PREFIX
from codegraph.store import GraphStore


@dataclass(frozen=True, slots=True)
class LanguageRow:
    language: str
    family: str
    suffixes: tuple[str, ...]
    filenames: tuple[str, ...]
    files: int


@dataclass(frozen=True, slots=True)
class Dependency:
    qname: str
    line: int
    resolved: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ResourceRow:
    qname: str
    type: str
    name: str
    path: str
    lines: tuple[int, int]
    signature: str
    depends_on: tuple[Dependency, ...]
    required_by: tuple[str, ...]


def language_inventory(store: GraphStore, *, indexed_only: bool = False) -> list[LanguageRow]:
    """Every supported language with how many indexed files it has."""
    counts = store.counts_by_kind()["languages"]
    rows = [
        LanguageRow(
            language=info.language,
            family=info.family,
            suffixes=info.suffixes,
            filenames=info.filenames,
            files=counts.get(info.language, 0),
        )
        for info in supported_languages()
    ]
    if indexed_only:
        rows = [row for row in rows if row.files]
    return sorted(rows, key=lambda r: (-r.files, r.language))


def resource_type(qname: str) -> str:
    """``resource:k8s/Deployment/web`` -> ``k8s/Deployment``;
    ``resource:terraform/aws_s3_bucket.logs`` -> ``terraform/aws_s3_bucket``."""
    body = qname.removeprefix(RESOURCE_PREFIX)
    family, _, rest = body.partition("/")
    if family == "terraform":
        head = rest.split(".", 1)[0]
        return f"terraform/{head}" if head else family
    if family == "kustomize" or "/" not in rest:
        return family
    return f"{family}/{rest.split('/', 1)[0]}"


def list_resources(store: GraphStore, type_filter: str | None = None) -> list[ResourceRow]:
    """Resource nodes (optionally only types starting with ``type_filter``) and their edges."""
    wanted = (type_filter or "").strip().lower()
    rows: list[ResourceRow] = []
    for node in store.all_nodes(kinds=("resource",)):
        kind = resource_type(node.qname)
        if wanted and not kind.lower().startswith(wanted):
            continue
        outgoing = sorted(
            store.edges_from(node.qname, kind="depends_on"), key=lambda e: (e.line, e.dst)
        )
        incoming = sorted({e.src for e in store.edges_to(node.qname, kind="depends_on")})
        rows.append(
            ResourceRow(
                qname=node.qname,
                type=kind,
                name=node.name,
                path=node.path,
                lines=(node.line_start, node.line_end),
                signature=node.signature,
                depends_on=tuple(Dependency(e.dst, e.line, e.resolved, e.detail) for e in outgoing),
                required_by=tuple(incoming),
            )
        )
    return sorted(rows, key=lambda r: (r.type, r.name, r.qname))


__all__ = [
    "Dependency",
    "LanguageRow",
    "ResourceRow",
    "language_inventory",
    "list_resources",
    "resource_type",
]
