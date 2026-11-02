"""Post-index pass: trace execution flows and persist them to ``analysis_flow*``.

Ordered after the communities pass so each flow can count the communities it spans.
When that pass is skipped, the last stored communities are used, or a throwaway
in-memory run when none were ever stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codegraph.analysis.communities import community_report
from codegraph.analysis.flows import detect_flows, save_flows
from codegraph.graph import GraphView
from codegraph.store import GraphStore


@dataclass(frozen=True, slots=True)
class FlowsPass:
    name: str = "flows"
    order: int = 65

    def run(self, root: Path, store: GraphStore) -> int:
        view = GraphView.load(store)
        membership = community_report(store, view).membership
        return save_flows(store, detect_flows(view, root=root, membership=membership))


PASS = FlowsPass()
