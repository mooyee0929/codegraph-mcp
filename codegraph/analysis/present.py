"""Payloads shared by the analysis MCP tools and the CLI's ``--json`` output.

Both front ends load the same snapshot (graph view, communities, flows) and must
describe a community or a flow the same way, so the shaping lives here once. Results
that are already plain dataclasses go straight through :func:`codegraph.tools.dump`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from codegraph.analysis.communities import Community, CommunityLink, CommunityReport
from codegraph.analysis.communities import community_report as _community_report
from codegraph.analysis.flows import Flow, flow_report
from codegraph.graph import GraphView
from codegraph.models import EDGE_KINDS, NODE_KINDS, EdgeKind, NodeKind
from codegraph.store import GraphStore

FLOW_SORT_KEYS: tuple[str, ...] = ("criticality", "depth", "node_count", "file_count", "name")


@dataclass(frozen=True, slots=True)
class Snapshot:
    view: GraphView
    report: CommunityReport


def snapshot(store: GraphStore) -> Snapshot:
    view = GraphView.load(store)
    return Snapshot(view=view, report=_community_report(store, view))


def flows_of(store: GraphStore, root: Path | None, snap: Snapshot) -> tuple[Flow, ...]:
    return flow_report(store, root, snap.view, snap.report.membership)


def parse_node_kinds(raw: Iterable[str] | None) -> tuple[NodeKind, ...] | None:
    """Validate node kind names; raises ValueError naming the bad ones."""
    if raw is None:
        return None
    wanted = list(raw)
    unknown = sorted(set(wanted) - set(NODE_KINDS))
    if unknown:
        raise ValueError(f"unknown node kind(s): {', '.join(unknown)}")
    return tuple(k for k in NODE_KINDS if k in wanted)


def parse_edge_kinds(raw: Iterable[str] | None) -> tuple[EdgeKind, ...] | None:
    """Validate edge kind names; raises ValueError naming the bad ones."""
    if raw is None:
        return None
    wanted = list(raw)
    unknown = sorted(set(wanted) - set(EDGE_KINDS))
    if unknown:
        raise ValueError(f"unknown edge kind(s): {', '.join(unknown)}")
    return tuple(k for k in EDGE_KINDS if k in wanted)


def community_payload(community: Community) -> dict[str, object]:
    return asdict(community)


def link_payload(report: CommunityReport, link: CommunityLink) -> dict[str, object]:
    src, dst = report.by_id(link.src), report.by_id(link.dst)
    return {
        "src": link.src,
        "src_name": src.name if src else "",
        "dst": link.dst,
        "dst_name": dst.name if dst else "",
        "edges": link.edges,
        "weight": link.weight,
    }


def communities_payload(
    report: CommunityReport, *, min_size: int = 2, limit: int = 30, boundaries: int = 15
) -> dict[str, object]:
    shown = [c for c in report.communities if c.size >= min_size][: max(limit, 0)]
    return {
        "source": report.source,
        "modularity": report.modularity,
        "total": len(report.communities),
        "communities": [community_payload(c) for c in shown],
        "boundaries": [link_payload(report, link) for link in report.links[:boundaries]],
    }


def community_detail_payload(
    view: GraphView, report: CommunityReport, community: Community, *, limit: int = 100
) -> dict[str, object]:
    members = report.members.get(community.id, ())
    listed: list[dict[str, object]] = []
    for qname in members[: max(limit, 0)]:
        node = view.nodes.get(qname)
        listed.append(
            {
                "qname": qname,
                "kind": node.kind if node else "",
                "path": node.path if node else "",
                "line": node.line_start if node else 0,
            }
        )
    links: list[dict[str, object]] = []
    for link in report.links_of(community.id):
        other_id = link.dst if link.src == community.id else link.src
        other = report.by_id(other_id)
        links.append(
            {
                "community": other_id,
                "name": other.name if other else "",
                "edges": link.edges,
                "weight": link.weight,
            }
        )
    return {
        **community_payload(community),
        "members": listed,
        "members_total": len(members),
        "links": links,
    }


def flow_summary(flow: Flow) -> dict[str, object]:
    summary = asdict(flow)
    summary.pop("steps")
    return summary


def flow_detail(view: GraphView, flow: Flow) -> dict[str, object]:
    steps: list[dict[str, object]] = []
    for step in flow.steps:
        node = view.nodes.get(step.qname)
        steps.append(
            {
                "qname": step.qname,
                "depth": step.depth,
                "parent": step.parent,
                "via": step.via,
                "kind": node.kind if node else "",
                "path": node.path if node else "",
                "line": node.line_start if node else 0,
            }
        )
    return {**flow_summary(flow), "steps": steps}


def sort_flows(flows: Iterable[Flow], key: str) -> list[Flow]:
    """Order flows by one of :data:`FLOW_SORT_KEYS` (descending, except ``name``)."""
    if key not in FLOW_SORT_KEYS:
        raise ValueError(f"sort_by must be one of {', '.join(FLOW_SORT_KEYS)}")
    items = list(flows)
    if key == "name":
        return sorted(items, key=lambda f: (f.name, f.id))
    if key == "depth":
        return sorted(items, key=lambda f: (-f.depth, f.id))
    if key == "node_count":
        return sorted(items, key=lambda f: (-f.node_count, f.id))
    if key == "file_count":
        return sorted(items, key=lambda f: (-f.file_count, f.id))
    return sorted(items, key=lambda f: (-f.criticality, f.id))


__all__ = [
    "FLOW_SORT_KEYS",
    "Snapshot",
    "communities_payload",
    "community_detail_payload",
    "community_payload",
    "flow_detail",
    "flow_summary",
    "flows_of",
    "link_payload",
    "parse_edge_kinds",
    "parse_node_kinds",
    "snapshot",
    "sort_flows",
]
