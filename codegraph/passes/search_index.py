"""Persist the BM25 / TF-IDF term index once the node set is final.

Order 90, the search band: every pass that adds nodes (routes, channels, resources)
has run, so they are searchable too. The whole index is rebuilt each time — it is a
pure function of the node table and costs a fraction of parsing — which keeps the pass
idempotent without tracking which nodes changed.
"""

from __future__ import annotations

from pathlib import Path

from codegraph.passes import GraphPass
from codegraph.search.index import Corpus, ensure_search_schema, write_index
from codegraph.store import GraphStore


class SearchIndexPass:
    name: str = "search_index"
    order: int = 90

    def run(self, root: Path, store: GraphStore) -> int:
        ensure_search_schema(store)
        return write_index(store, Corpus.from_nodes(store.all_nodes()))


PASS: GraphPass = SearchIndexPass()
