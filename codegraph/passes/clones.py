"""Post-index pass: ``similar_to`` edges between near-clone functions.

Function bodies are re-read from disk using each node's line span; test files are
skipped because copy-paste between tests is usually deliberate. Each pair gets one
edge with ``src`` the lexically smaller qname, ``confidence`` the exact Jaccard score
and ``detail`` such as ``jaccard=0.83``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from codegraph.models import Edge, Node
from codegraph.services.clones import DEFAULT_THRESHOLD, ShingleHasher, find_clones, fingerprint
from codegraph.services.sources import is_test_file, language_of
from codegraph.store import GraphStore


@dataclass(frozen=True, slots=True)
class ClonesPass:
    name: str = "clones"
    order: int = 30
    threshold: float = DEFAULT_THRESHOLD

    def run(self, root: Path, store: GraphStore) -> int:
        store.delete_edges("similar_to")
        by_file: defaultdict[str, list[Node]] = defaultdict(list)
        for node in store.all_nodes(kinds=("function", "method")):
            if not is_test_file(node.path):
                by_file[node.path].append(node)
        hasher = ShingleHasher()
        prints = []
        located: dict[str, Node] = {}
        for path, nodes in sorted(by_file.items()):
            labels = language_of(path)
            try:
                lines = (root / path).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            family = labels[1] if labels is not None else "c"
            for node in nodes:
                body = "\n".join(lines[node.line_start - 1 : node.line_end])
                item = fingerprint(
                    node.qname, path, node.line_start, node.line_end, body, family, hasher
                )
                if item is not None:
                    prints.append(item)
                    located[node.qname] = node
        edges = [
            Edge(
                src=pair.a,
                dst=pair.b,
                kind="similar_to",
                path=located[pair.a].path,
                line=located[pair.a].line_start,
                confidence=pair.score,
                detail=f"jaccard={pair.score:.2f}",
            )
            for pair in find_clones(prints, threshold=self.threshold)
        ]
        return store.add_edges(edges)


PASS = ClonesPass()
