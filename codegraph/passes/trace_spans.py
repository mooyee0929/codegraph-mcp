"""Post-index pass: re-apply ingested runtime spans to the rebuilt ``http_calls`` edges.

The ``http_calls`` pass drops and recomputes every edge on each run, which would wipe
what ``ingest_traces`` learned. Spans live in ``trace_spans`` instead, and this pass
(ordered after ``http_calls``) replays them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codegraph.services.traces import apply_spans
from codegraph.store import GraphStore


@dataclass(frozen=True, slots=True)
class TraceSpansPass:
    name: str = "trace_spans"
    order: int = 40

    def run(self, root: Path, store: GraphStore) -> int:
        stats = apply_spans(store)
        return stats.validated + stats.added


PASS = TraceSpansPass()
