"""Register the MCP server with Claude Code, and take it out again exactly.

``codegraph install`` uses ``claude mcp add`` when the ``claude`` CLI is on PATH (so
Claude Code owns the config format) and otherwise merges a ``codegraph`` entry into
the project's ``.mcp.json``. ``--hooks`` additionally adds a project-level PostToolUse
hook on ``Edit|Write`` to ``.claude/settings.json`` that re-indexes after every edit;
the hook resolves the repo through ``$CLAUDE_PROJECT_DIR`` at run time so the file is
shareable, and exits quietly when ``codegraph`` is not on PATH.

Every change is planned first (``--dry-run`` prints the plan and stops), JSON files
are merged without touching other keys, and what was actually added is recorded in
``.codegraph/install.json``. ``codegraph uninstall`` reverses only those records: an
entry the user edited since is left alone and reported, and a file is deleted only if
install created it and nothing else has been put in it since.

Both the ``claude`` lookup and the subprocess runner are injected so tests never touch
the real Claude configuration.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from codegraph.store import DB_DIRNAME

SERVER_NAME = "codegraph"
MCP_JSON = ".mcp.json"
SETTINGS_JSON = ".claude/settings.json"
MANIFEST = f"{DB_DIRNAME}/install.json"
SCOPES: tuple[str, ...] = ("local", "project", "user")
HOOK_MATCHER = "Edit|Write"
HOOK_COMMAND = (
    "command -v codegraph >/dev/null 2>&1 || exit 0; "
    'codegraph --root "$CLAUDE_PROJECT_DIR" index >/dev/null'
)
HOOK_TIMEOUT_SECONDS = 120

Runner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]
Which = Callable[[str], str | None]
StepAction = Literal["run", "write", "delete"]


class InstallError(RuntimeError):
    """A conflicting existing entry, unreadable JSON, or a failing ``claude`` command."""


@dataclass(frozen=True, slots=True)
class Step:
    action: StepAction
    target: str
    summary: str
    argv: tuple[str, ...] = ()
    content: str | None = None


@dataclass(frozen=True, slots=True)
class Plan:
    root: Path
    steps: tuple[Step, ...]
    notes: tuple[str, ...]

    def describe(self) -> list[str]:
        lines = [f"{s.action:<6} {s.target}  ({s.summary})" for s in self.steps]
        return lines + [f"note   {n}" for n in self.notes]


def run_command(argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), cwd=cwd, capture_output=True, text=True, check=False)


# -- JSON helpers -----------------------------------------------------------------


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        data: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InstallError(f"{path} is not valid JSON ({exc}); fix it and retry") from exc
    if not isinstance(data, dict):
        raise InstallError(f"{path} does not hold a JSON object")
    return data


def _child(parent: dict[str, object], key: str, path: Path) -> dict[str, object]:
    value = parent.setdefault(key, {})
    if not isinstance(value, dict):
        raise InstallError(f"{path}: {key!r} is not an object")
    return value


def _child_list(parent: dict[str, object], key: str, path: Path) -> list[object]:
    value = parent.setdefault(key, [])
    if not isinstance(value, list):
        raise InstallError(f"{path}: {key!r} is not a list")
    return value


def _dumps(data: dict[str, object]) -> str:
    return json.dumps(data, indent=2) + "\n"


def server_entry(command: Sequence[str]) -> dict[str, object]:
    return {"command": command[0], "args": [*command[1:], "--root", "."]}


def hook_entry() -> dict[str, object]:
    return {
        "matcher": HOOK_MATCHER,
        "hooks": [{"type": "command", "command": HOOK_COMMAND, "timeout": HOOK_TIMEOUT_SECONDS}],
    }


# -- install ------------------------------------------------------------------------


def plan_install(
    root: Path,
    *,
    hooks: bool = False,
    scope: str = "local",
    command: Sequence[str] = ("codegraph-mcp",),
    which: Which = shutil.which,
) -> Plan:
    root = root.resolve()
    if scope not in SCOPES:
        raise InstallError(f"unknown scope {scope!r}; expected one of {', '.join(SCOPES)}")
    if not command:
        raise InstallError("the server command must not be empty")
    manifest = _load_json(root / MANIFEST)
    steps: list[Step] = []
    notes: list[str] = []

    claude = which("claude")
    if "mcp" in manifest:
        notes.append(f"the server is already installed (see {MANIFEST}); uninstall to redo it")
    elif claude is not None:
        argv = (claude, "mcp", "add", "--scope", scope, SERVER_NAME, "--", *command)
        argv = (*argv, "--root", str(root))
        steps.append(Step("run", " ".join(argv), f"register with Claude Code ({scope})", argv))
        manifest["mcp"] = {"method": "claude", "scope": scope, "name": SERVER_NAME}
    else:
        mcp_path = root / MCP_JSON
        data = _load_json(mcp_path)
        servers = _child(data, "mcpServers", mcp_path)
        entry = server_entry(command)
        if servers.get(SERVER_NAME) == entry:
            notes.append(f"{MCP_JSON} already has this {SERVER_NAME!r} server")
        elif SERVER_NAME in servers:
            raise InstallError(
                f"{mcp_path} already has a different {SERVER_NAME!r} server; remove it first"
            )
        else:
            servers[SERVER_NAME] = entry
            steps.append(Step("write", MCP_JSON, f"add mcpServers.{SERVER_NAME}", (), _dumps(data)))
            manifest["mcp"] = {
                "method": "mcp.json",
                "entry": entry,
                "created_file": not mcp_path.exists(),
            }
        notes.append("the claude CLI was not found, so the server goes into .mcp.json")

    if hooks and "hooks" in manifest:
        notes.append(f"the re-index hook is already installed (see {MANIFEST})")
    elif hooks:
        settings_path = root / SETTINGS_JSON
        settings = _load_json(settings_path)
        hooks_obj = _child(settings, "hooks", settings_path)
        post = _child_list(hooks_obj, "PostToolUse", settings_path)
        entry = hook_entry()
        if entry in post:
            notes.append(f"{SETTINGS_JSON} already has the re-index hook")
        else:
            post.append(entry)
            steps.append(
                Step(
                    "write",
                    SETTINGS_JSON,
                    "add PostToolUse Edit|Write re-index hook",
                    (),
                    _dumps(settings),
                )
            )
            manifest["hooks"] = {"entry": entry, "created_file": not settings_path.exists()}

    if steps:
        steps.append(Step("write", MANIFEST, "record what install added", (), _dumps(manifest)))
    return Plan(root=root, steps=tuple(steps), notes=tuple(notes))


# -- uninstall ----------------------------------------------------------------------


def plan_uninstall(root: Path, *, which: Which = shutil.which) -> Plan:
    root = root.resolve()
    manifest_path = root / MANIFEST
    manifest = _load_json(manifest_path)
    if not manifest:
        return Plan(root, (), (f"nothing to undo: {MANIFEST} does not exist",))
    steps: list[Step] = []
    notes: list[str] = []

    mcp = manifest.get("mcp")
    if isinstance(mcp, dict) and mcp.get("method") == "claude":
        claude = which("claude")
        name, scope = str(mcp.get("name", SERVER_NAME)), str(mcp.get("scope", "local"))
        argv = ("claude" if claude is None else claude, "mcp", "remove", "--scope", scope, name)
        if claude is None:
            notes.append(f"claude CLI not found; run `{' '.join(argv)}` yourself")
        else:
            steps.append(Step("run", " ".join(argv), "unregister from Claude Code", argv))
    elif isinstance(mcp, dict) and mcp.get("method") == "mcp.json":
        steps += _undo_mcp_json(root, mcp, notes)

    hooks = manifest.get("hooks")
    if isinstance(hooks, dict):
        steps += _undo_hook(root, hooks, notes)

    steps.append(Step("delete", MANIFEST, "forget the install record"))
    return Plan(root=root, steps=tuple(steps), notes=tuple(notes))


def _undo_mcp_json(root: Path, record: dict[str, object], notes: list[str]) -> list[Step]:
    path = root / MCP_JSON
    data = _load_json(path)
    servers = data.get("mcpServers")
    if not isinstance(servers, dict) or servers.get(SERVER_NAME) != record.get("entry"):
        notes.append(f"{MCP_JSON} no longer has the entry install wrote; left it alone")
        return []
    del servers[SERVER_NAME]
    if not servers:
        del data["mcpServers"]
    if record.get("created_file") and not data:
        return [Step("delete", MCP_JSON, "install created it and it is now empty")]
    return [Step("write", MCP_JSON, f"remove mcpServers.{SERVER_NAME}", (), _dumps(data))]


def _undo_hook(root: Path, record: dict[str, object], notes: list[str]) -> list[Step]:
    path = root / SETTINGS_JSON
    settings = _load_json(path)
    hooks = settings.get("hooks")
    post = hooks.get("PostToolUse") if isinstance(hooks, dict) else None
    entry = record.get("entry")
    if not isinstance(hooks, dict) or not isinstance(post, list) or entry not in post:
        notes.append(f"{SETTINGS_JSON} no longer has the hook install wrote; left it alone")
        return []
    post.remove(entry)
    if not post:
        del hooks["PostToolUse"]
    if not hooks:
        del settings["hooks"]
    if record.get("created_file") and not settings:
        return [Step("delete", SETTINGS_JSON, "install created it and it is now empty")]
    return [Step("write", SETTINGS_JSON, "remove the re-index hook", (), _dumps(settings))]


# -- execution ----------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def apply_plan(plan: Plan, *, runner: Runner = run_command) -> list[str]:
    """Carry the steps out in order; stop at the first failing command."""
    done: list[str] = []
    for step in plan.steps:
        if step.action == "run":
            result = runner(step.argv, plan.root)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                raise InstallError(f"`{step.target}` failed: {detail or result.returncode}")
        elif step.action == "write":
            if step.content is None:
                raise InstallError(f"write step for {step.target} has no content")
            _atomic_write(plan.root / step.target, step.content)
        else:
            (plan.root / step.target).unlink(missing_ok=True)
        done.append(f"{step.action} {step.target}")
    return done


__all__ = [
    "HOOK_COMMAND",
    "MANIFEST",
    "SERVER_NAME",
    "InstallError",
    "Plan",
    "Step",
    "apply_plan",
    "hook_entry",
    "plan_install",
    "plan_uninstall",
    "run_command",
    "server_entry",
]
