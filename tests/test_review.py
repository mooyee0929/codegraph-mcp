from __future__ import annotations

import subprocess
from pathlib import Path

from codegraph.indexer import index_repository
from codegraph.review import changed_lines, detect_changes, review_context
from codegraph.store import GraphStore


def _edit_rate(repo: Path) -> None:
    billing = repo / "pkg" / "billing.py"
    text = billing.read_text().replace("        return 1\n", "        return 2\n")
    billing.write_text(text)


def test_changed_lines_working_tree(repo: Path) -> None:
    _edit_rate(repo)
    changed = changed_lines(repo)
    assert changed == {"pkg/billing.py": [14]}


def test_changed_lines_against_base(repo: Path) -> None:
    _edit_rate(repo)
    subprocess.run(["git", "commit", "-qam", "bump"], cwd=repo, check=True)
    assert changed_lines(repo, "HEAD~1") == {"pkg/billing.py": [14]}
    assert changed_lines(repo) == {}


def test_detect_changes_ranks_by_risk(repo: Path, store: GraphStore) -> None:
    _edit_rate(repo)
    index_repository(repo, store)
    changed = detect_changes(store, repo)
    assert [c.node.qname for c in changed] == ["pkg.billing.Invoice.rate"]
    only = changed[0]
    assert only.changed_lines == (14,)
    assert only.fan_in == 1
    assert only.covering_tests == ()
    assert "no test coverage" in only.reasons
    assert only.risk > 0


def test_review_context_includes_source_and_callers(repo: Path, store: GraphStore) -> None:
    snippets = review_context(store, repo, ["pkg.billing.charge"], context_lines=0)
    assert len(snippets) == 1
    snippet = snippets[0]
    assert "def charge(inv: Invoice) -> str:" in snippet.source
    assert snippet.callers == ("tests.test_billing.test_charge",)
    assert snippet.tests == ("tests.test_billing.test_charge",)
    assert review_context(store, repo, ["nope"]) == []
