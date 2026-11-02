from __future__ import annotations

import json
from pathlib import Path

import pytest

from codegraph.cli import main


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEGRAPH_HOME", str(tmp_path / "cg-home"))
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))


def run(repo: Path, *argv: str) -> int:
    return main(["--root", str(repo), *argv])


def test_registry_commands(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(repo, "repos") == 1
    assert run(repo, "register", "--name", "sample") == 0
    assert "registered sample" in capsys.readouterr().out
    assert run(repo, "repos") == 0
    assert "not indexed" in capsys.readouterr().out
    assert run(repo, "index") == 0
    assert run(repo, "repos", "--json") == 0
    out = capsys.readouterr().out
    assert json.loads(out[out.index("[") :])[0]["files"] == 8
    assert run(repo, "unregister", "sample") == 0
    assert run(repo, "unregister", "sample") == 1
    assert run(repo, "register", str(repo / "missing")) == 2


def test_watch_snapshot_and_refactor(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(repo, "watch", "--interval", "0", "--max-polls", "1") == 0
    assert "parsed 8" in capsys.readouterr().out
    assert run(repo, "watch", "--interval", "-1") == 2

    assert run(repo, "snapshot", "export") == 0
    assert "graph.db.gz" in capsys.readouterr().out
    assert run(repo, "snapshot", "import") == 0
    assert "restored" in capsys.readouterr().out

    assert run(repo, "rename", "pkg.billing.charge", "bill") == 0
    out = capsys.readouterr().out
    assert "+def bill(inv: Invoice) -> str:" in out
    preview_id = out.rsplit("codegraph apply ", 1)[1].strip()
    assert run(repo, "apply", preview_id) == 0
    assert "changed 3 line(s)" in capsys.readouterr().out
    assert run(repo, "apply", preview_id) == 2
    assert run(repo, "rename", "nope", "x") == 2
    assert run(repo, "suggest") == 0
    assert "remove" in capsys.readouterr().out


def test_document_commands(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(repo, "index") == 0
    capsys.readouterr()

    assert run(repo, "adr", "list") == 1
    assert run(repo, "adr", "create", "--title", "Use SQLite", "--status", "accepted") == 0
    assert "created docs/adr/0001-use-sqlite.md  [Accepted]" in capsys.readouterr().out
    assert run(repo, "adr", "supersede", "1", "--title", "Use DuckDB") == 0
    assert run(repo, "adr", "list") == 0
    listing = capsys.readouterr().out
    assert "Superseded" in listing and "Use DuckDB" in listing
    assert run(repo, "adr", "get", "1") == 0
    assert "Superseded by [2. Use DuckDB]" in capsys.readouterr().out
    assert run(repo, "adr", "get") == 2

    assert run(repo, "note", "save", "who calls charge", "--answer", "test_charge") == 0
    assert run(repo, "note", "search", "charge") == 0
    assert "who calls charge" in capsys.readouterr().out
    assert run(repo, "note", "search", "zzz") == 1
    assert run(repo, "note", "save", "missing answer") == 2

    assert run(repo, "wiki") == 0
    assert run(repo, "wiki", "--page", "pkg.billing") == 0
    assert "# pkg.billing" in capsys.readouterr().out
    assert run(repo, "wiki", "--page", "nope") == 1

    assert run(repo, "visualize", "--max-nodes", "3") == 0
    assert "(3 of" in capsys.readouterr().out
    assert run(repo, "export", "mermaid", "--stdout") == 0
    assert capsys.readouterr().out.startswith("flowchart LR")
    assert run(repo, "export", "graphml") == 0
    assert (repo / ".codegraph" / "exports" / "graph.graphml").is_file()
    assert run(repo, "export", "pdf") == 2
    assert run(repo, "ui", "--port", "70000") == 2
