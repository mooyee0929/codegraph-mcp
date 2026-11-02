from __future__ import annotations

import subprocess
from pathlib import Path

from codegraph.services.history import (
    Commit,
    FileChange,
    churn,
    co_changes,
    is_trackable,
    parse_log,
    read_history,
)


def _commit(sha: str, ts: int, author: str, *paths: str) -> Commit:
    return Commit(sha, ts, author, f"change {sha}", tuple(FileChange(p, 2, 1) for p in paths))


def test_parse_log_reads_numstat_and_binary_files() -> None:
    output = (
        "\x1eabc\x1f1700000000\x1fAda\x1fFix: a\x1fb\n\n3\t1\tsrc/a.py\n-\t-\tlogo.png\n"
        "\x1edef\x1f1700000100\x1fBob\x1fAdd b\n\n10\t0\tsrc/b.py\n"
    )
    first, second = parse_log(output)
    assert (first.sha, first.author, first.subject) == ("abc", "Ada", "Fix: a\x1fb")
    assert first.files == (FileChange("src/a.py", 3, 1), FileChange("logo.png", 0, 0))
    assert second.files == (FileChange("src/b.py", 10, 0),)
    assert parse_log("") == []


def test_trackable_filter() -> None:
    assert is_trackable("src/app.py")
    assert not is_trackable("package-lock.json")
    assert not is_trackable("web/node_modules/x/index.js")
    assert not is_trackable("static/logo.svg")


def test_churn_and_coupling_thresholds() -> None:
    commits = (
        _commit("1", 100, "ada", "a.py", "b.py"),
        _commit("2", 200, "bob", "a.py", "b.py"),
        _commit("3", 300, "ada", "a.py", "b.py", "c.py"),
        _commit("4", 400, "ada", "a.py", "c.py"),
        _commit("5", 500, "cy", "d.py"),
        _commit("6", 600, "cy", *(f"bulk{i}.py" for i in range(60))),
    )
    per_file = churn(commits)
    assert per_file["a.py"].commits == 4 and per_file["a.py"].authors == 2
    assert (per_file["a.py"].first_change, per_file["a.py"].last_change) == (100, 400)
    assert per_file["a.py"].added == 8 and per_file["a.py"].removed == 4
    assert "bulk0.py" not in per_file
    pairs = co_changes(commits)
    assert [(p.a, p.b, p.count, p.ratio) for p in pairs] == [("a.py", "b.py", 3, 1.0)]
    assert pairs[0].last_change == 300
    assert co_changes(commits, min_count=2)[1].b == "c.py"


def test_read_history_from_a_real_repository(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "ci@example.com")
    git("config", "user.name", "ci")
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", "-A")
    git("commit", "-q", "-m", "first")
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
    git("commit", "-qam", "second")
    history = read_history(tmp_path)
    assert history.status == "ok: 2 commits"
    assert [c.subject for c in history.commits] == ["second", "first"]
    assert history.commits[0].files == (FileChange("a.py", 1, 0),)
    assert read_history(tmp_path, max_commits=1).commits[0].subject == "second"


def test_read_history_without_git_is_empty(tmp_path: Path) -> None:
    history = read_history(tmp_path / "missing")
    assert history.commits == () and history.status.startswith("no history")
