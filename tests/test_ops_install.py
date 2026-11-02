from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from codegraph.cli import main
from codegraph.install import (
    HOOK_COMMAND,
    MANIFEST,
    InstallError,
    apply_plan,
    hook_entry,
    plan_install,
    plan_uninstall,
)


def no_claude(_name: str) -> str | None:
    return None


def fake_claude(_name: str) -> str | None:
    return "/opt/bin/claude"


class Recorder:
    def __init__(self, returncode: int = 0) -> None:
        self.calls: list[tuple[tuple[str, ...], Path]] = []
        self.returncode = returncode

    def __call__(self, argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        self.calls.append((tuple(argv), cwd))
        return subprocess.CompletedProcess(list(argv), self.returncode, "", "boom")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEGRAPH_HOME", str(tmp_path / "cg-home"))
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))


def _read(path: Path) -> dict[str, object]:
    data: object = json.loads(path.read_text())
    assert isinstance(data, dict)
    return data


def test_mcp_json_merge_and_exact_uninstall(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"other": {"command": "x"}}, "keep": True})
    )
    settings = root / ".claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"permissions": {"allow": ["Bash(ls)"]}}))

    plan = plan_install(root, hooks=True, which=no_claude)
    assert [s.target for s in plan.steps] == [".mcp.json", ".claude/settings.json", MANIFEST]
    apply_plan(plan, runner=Recorder())

    mcp = _read(root / ".mcp.json")
    assert mcp["keep"] is True
    assert mcp["mcpServers"] == {
        "other": {"command": "x"},
        "codegraph": {"command": "codegraph-mcp", "args": ["--root", "."]},
    }
    merged = _read(settings)
    assert merged["permissions"] == {"allow": ["Bash(ls)"]}
    assert merged["hooks"] == {"PostToolUse": [hook_entry()]}
    assert "$CLAUDE_PROJECT_DIR" in HOOK_COMMAND

    again = plan_install(root, hooks=True, which=no_claude)
    assert again.steps == () and any("already installed" in n for n in again.notes)

    apply_plan(plan_uninstall(root, which=no_claude), runner=Recorder())
    assert _read(root / ".mcp.json") == {"mcpServers": {"other": {"command": "x"}}, "keep": True}
    assert _read(settings) == {"permissions": {"allow": ["Bash(ls)"]}}
    assert not (root / MANIFEST).exists()


def test_files_created_by_install_are_removed(tmp_path: Path) -> None:
    apply_plan(plan_install(tmp_path, hooks=True, which=no_claude), runner=Recorder())
    assert (tmp_path / ".mcp.json").is_file()
    apply_plan(plan_uninstall(tmp_path, which=no_claude), runner=Recorder())
    assert not (tmp_path / ".mcp.json").exists()
    assert not (tmp_path / ".claude" / "settings.json").exists()


def test_user_edits_are_left_alone(tmp_path: Path) -> None:
    apply_plan(plan_install(tmp_path, which=no_claude), runner=Recorder())
    edited = {"mcpServers": {"codegraph": {"command": "my-own"}}}
    (tmp_path / ".mcp.json").write_text(json.dumps(edited))
    plan = plan_uninstall(tmp_path, which=no_claude)
    assert any("left it alone" in n for n in plan.notes)
    apply_plan(plan, runner=Recorder())
    assert _read(tmp_path / ".mcp.json") == edited


def test_conflicts_and_bad_json_are_errors(tmp_path: Path) -> None:
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"codegraph": {"command": "z"}}}))
    with pytest.raises(InstallError, match="different 'codegraph'"):
        plan_install(tmp_path, which=no_claude)
    (tmp_path / ".mcp.json").write_text("{oops")
    with pytest.raises(InstallError, match="not valid JSON"):
        plan_install(tmp_path, which=no_claude)
    with pytest.raises(InstallError, match="unknown scope"):
        plan_install(tmp_path, scope="galaxy", which=no_claude)


def test_claude_cli_path(tmp_path: Path) -> None:
    runner = Recorder()
    plan = plan_install(tmp_path, scope="project", which=fake_claude)
    apply_plan(plan, runner=runner)
    argv, cwd = runner.calls[0]
    assert argv == (
        "/opt/bin/claude",
        "mcp",
        "add",
        "--scope",
        "project",
        "codegraph",
        "--",
        "codegraph-mcp",
        "--root",
        str(tmp_path.resolve()),
    )
    assert cwd == tmp_path.resolve()
    assert not (tmp_path / ".mcp.json").exists()

    undo = Recorder()
    apply_plan(plan_uninstall(tmp_path, which=fake_claude), runner=undo)
    assert undo.calls[0][0] == (
        "/opt/bin/claude",
        "mcp",
        "remove",
        "--scope",
        "project",
        "codegraph",
    )
    assert not (tmp_path / MANIFEST).exists()


def test_failing_claude_command_is_reported(tmp_path: Path) -> None:
    plan = plan_install(tmp_path, which=fake_claude)
    with pytest.raises(InstallError, match="boom"):
        apply_plan(plan, runner=Recorder(returncode=1))
    assert not (tmp_path / MANIFEST).exists()


def test_uninstall_without_manifest(tmp_path: Path) -> None:
    plan = plan_uninstall(tmp_path, which=no_claude)
    assert plan.steps == () and "nothing to undo" in plan.notes[0]


def test_cli_dry_run_changes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--root", str(tmp_path), "install", "--hooks", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "would write  .mcp.json" in out and "would write  .claude/settings.json" in out
    assert not (tmp_path / ".mcp.json").exists()

    assert main(["--root", str(tmp_path), "install", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is False and (tmp_path / ".mcp.json").exists()
    assert main(["--root", str(tmp_path), "uninstall"]) == 0
    assert not (tmp_path / ".mcp.json").exists()
