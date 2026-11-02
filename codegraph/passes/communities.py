"""Post-index pass: detect communities and persist them to ``analysis_community*``.

Runs in the analysis band (after every pass that adds nodes or edges) so routes,
channels and ``co_changes`` edges already exist when the clustering looks for them.
The flows pass, which runs next, reads the membership this pass wrote.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codegraph.analysis.communities import detect_communities, save_communities
from codegraph.graph import GraphView
from codegraph.store import GraphStore


@dataclass(frozen=True, slots=True)
class CommunitiesPass:
    name: str = "communities"
    order: int = 60

    def run(self, root: Path, store: GraphStore) -> int:
        return save_communities(store, detect_communities(GraphView.load(store)))


PASS = CommunitiesPass()
