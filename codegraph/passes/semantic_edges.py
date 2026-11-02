"""Link callables that share vocabulary but no structure (``semantically_related``).

Order 95: after the graph-mutating passes (so ``similar_to``, ``handles`` and the rest
already count as structure) and after the search index. Each edge joins two functions
or methods in different modules whose TF-IDF cosine is at least ``threshold``, with no
edge of any other kind between them; each symbol keeps only its ``top_k`` best. The
edge runs from the lexically smaller qname, ``confidence`` is the cosine and
``detail`` reads ``tfidf=0.87``. Query it undirected:
``MATCH (a)-[:SEMANTICALLY_RELATED]-(b)``.

The idea is codebase-memory-mcp's SEMANTICALLY_RELATED pass, reduced to its TF-IDF signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codegraph.graph import GraphView
from codegraph.models import Edge
from codegraph.passes import GraphPass
from codegraph.search.related import DEFAULT_THRESHOLD, DEFAULT_TOP_K, related_pairs
from codegraph.store import GraphStore


@dataclass(slots=True)
class SemanticEdgesPass:
    name: str = "semantic_edges"
    order: int = 95
    threshold: float = DEFAULT_THRESHOLD
    top_k: int = DEFAULT_TOP_K

    def run(self, root: Path, store: GraphStore) -> int:
        store.delete_edges("semantically_related")
        view = GraphView.load(store)
        edges: list[Edge] = []
        for pair in related_pairs(view, threshold=self.threshold, top_k=self.top_k):
            source = view.nodes[pair.a]
            edges.append(
                Edge(
                    src=pair.a,
                    dst=pair.b,
                    kind="semantically_related",
                    path=source.path,
                    line=source.line_start,
                    confidence=round(pair.score, 4),
                    detail=f"tfidf={pair.score:.2f}",
                )
            )
        return store.add_edges(edges)


PASS: GraphPass = SemanticEdgesPass()
