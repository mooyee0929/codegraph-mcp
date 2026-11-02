from __future__ import annotations

from pathlib import Path

import pytest

from codegraph.indexer import index_repository
from codegraph.models import Edge
from codegraph.refactor import (
    RefactorError,
    apply_refactor,
    load_preview,
    preview_rename,
    suggest_refactorings,
    validate_new_name,
)
from codegraph.store import GraphStore


def test_preview_covers_definition_and_references(repo: Path, store: GraphStore) -> None:
    preview = preview_rename(repo, store, "pkg.billing.charge", "bill", clock=lambda: 1000.0)
    assert preview.old_name == "charge" and preview.target == "pkg.billing.charge"
    assert {f.path for f in preview.files} == {"pkg/billing.py", "tests/test_billing.py"}
    assert "-def charge(inv: Invoice) -> str:" in preview.diff
    assert "+def bill(inv: Invoice) -> str:" in preview.diff
    assert "+from pkg.billing import Invoice, bill" in preview.diff
    assert '+    assert bill(Invoice()) == "$2"' in preview.diff
    assert preview.expires_at == 1600.0
    assert (repo / "pkg" / "billing.py").read_text().count("def charge") == 1

    stored = load_preview(store, preview.preview_id)
    assert stored == preview


def test_apply_writes_files_and_reindexes(repo: Path, store: GraphStore) -> None:
    preview = preview_rename(repo, store, "Invoice.rate", "unit_rate")
    applied = apply_refactor(repo, store, preview.preview_id)
    assert applied.files == ("pkg/billing.py",)
    assert applied.lines_changed == 2
    text = (repo / "pkg" / "billing.py").read_text()
    assert "def unit_rate(self)" in text and "self.unit_rate() * 2" in text
    assert load_preview(store, preview.preview_id) is None

    index_repository(repo, store)
    assert store.get_node("pkg.billing.Invoice.unit_rate") is not None
    with pytest.raises(RefactorError, match="no pending preview"):
        apply_refactor(repo, store, preview.preview_id)


def test_apply_refuses_changed_files(repo: Path, store: GraphStore) -> None:
    preview = preview_rename(repo, store, "pkg.billing.charge", "bill")
    test_file = repo / "tests" / "test_billing.py"
    test_file.write_text(test_file.read_text() + "\n# edited\n")
    with pytest.raises(RefactorError, match="tests/test_billing.py"):
        apply_refactor(repo, store, preview.preview_id)
    assert "def charge" in (repo / "pkg" / "billing.py").read_text()


def test_apply_refuses_expired_preview(repo: Path, store: GraphStore) -> None:
    preview = preview_rename(repo, store, "pkg.billing.charge", "bill", clock=lambda: 0.0)
    with pytest.raises(RefactorError, match="expired"):
        apply_refactor(repo, store, preview.preview_id, clock=lambda: 10_000.0)
    assert load_preview(store, preview.preview_id) is None


def test_preview_rejects_bad_input(repo: Path, store: GraphStore) -> None:
    with pytest.raises(RefactorError, match="no node"):
        preview_rename(repo, store, "nope.nothing", "x")
    with pytest.raises(RefactorError, match="cannot rename a module"):
        preview_rename(repo, store, "pkg.billing", "x")
    with pytest.raises(RefactorError, match="already called"):
        preview_rename(repo, store, "pkg.billing.charge", "charge")
    with pytest.raises(RefactorError, match="not a valid"):
        validate_new_name("class", "python")
    with pytest.raises(RefactorError, match="not a valid"):
        validate_new_name("9lives", "typescript")
    validate_new_name("$el", "typescript")


def test_word_boundaries_and_decorators(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "lib.py").write_text(
        "import functools\n\n\n@functools.cache\ndef size() -> int:\n    return 1\n\n\n"
        "def caller() -> int:\n    size_total = 3\n    return size() + size_total\n"
    )
    with GraphStore.open(root) as graph:
        index_repository(root, graph)
        preview = preview_rename(root, graph, "lib.size", "length")
    assert "+def length() -> int:" in preview.diff
    assert "+    return length() + size_total" in preview.diff
    assert "-    size_total = 3" not in preview.diff


def test_suggestions_cover_every_kind(repo: Path, store: GraphStore) -> None:
    (repo / "pkg" / "long.py").write_text(
        "def big() -> int:\n" + "".join(f"    x{i} = {i}\n" for i in range(30)) + "    return 0\n"
    )
    (repo / "web" / "consumers.py").write_text(
        "from pkg.util import fmt\n"
        + "".join(f"\n\ndef use_{n}() -> str:\n    return fmt({i})\n" for i, n in enumerate("abc"))
    )
    index_repository(repo, store)
    with store.transaction():
        store.add_edges(
            [
                Edge(
                    src="pkg.billing.Invoice.total",
                    dst="pkg.billing.Invoice.rate",
                    kind="similar_to",
                    path="pkg/billing.py",
                    line=10,
                    confidence=0.8,
                    detail="jaccard=0.80",
                )
            ]
        )
    found = suggest_refactorings(store, large_lines=20)
    kinds = {s.kind for s in found}
    assert kinds == {"dedupe", "move", "split", "remove"}
    move = next(s for s in found if s.kind == "move")
    assert move.qname == "pkg.util.fmt" and "3 of 4" in move.reason
    assert any(s.kind == "split" and s.qname == "pkg.long.big" for s in found)
    assert any(s.kind == "remove" and s.qname == "pkg.billing.unused_helper" for s in found)
    assert next(s for s in found if s.kind == "dedupe").score == 0.8
