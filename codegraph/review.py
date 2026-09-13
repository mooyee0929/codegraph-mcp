"""Turn a git diff into a risk-ranked list of touched symbols plus review context."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from codegraph.linker import is_test_path
from codegraph.models import ChangedSymbol, Node
from codegraph.queries import impact_radius
from codegraph.store import GraphStore

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


class GitError(RuntimeError):
    """``git`` could not be run or returned a failure."""


def _git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitError("git executable not found on PATH") from exc
    if completed.returncode != 0:
        raise GitError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout


def changed_lines(root: Path, base: str | None = None) -> dict[str, list[int]]:
    """Map changed file path -> added/modified line numbers on the new side.

    ``base=None`` diffs the working tree (staged + unstaged) against HEAD; otherwise
    ``base`` is any git revision (``main``, ``HEAD~3``, a SHA) to compare HEAD against.
    """
    args = ["diff", "--unified=0", "--no-color", "--diff-filter=AM"]
    if base is not None:
        args.append(f"{base}...HEAD" if base != "HEAD" else "HEAD")
    else:
        args.append("HEAD")
    output = _git(root, *args)

    lines_by_path: dict[str, list[int]] = {}
    current: str | None = None
    for line in output.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            current = None if target == "/dev/null" else target.removeprefix("b/")
            if current is not None:
                lines_by_path.setdefault(current, [])
            continue
        if current is None:
            continue
        match = _HUNK.match(line)
        if match:
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) is not None else 1
            lines_by_path[current].extend(range(start, start + max(count, 1)))
    return lines_by_path


def _risk(
    node: Node, fan_in: int, fan_out: int, tests: int, touched: int
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    if fan_in >= 5:
        score += 3.0
        reasons.append(f"{fan_in} callers")
    elif fan_in > 0:
        score += 1.0 + fan_in * 0.3
    if tests == 0 and node.kind in {"function", "method"} and not is_test_path(node.path):
        score += 2.0
        reasons.append("no test coverage")
    if fan_out >= 8:
        score += 1.5
        reasons.append(f"calls {fan_out} things")
    if touched >= 20:
        score += 1.0
        reasons.append(f"{touched} lines changed")
    if node.kind == "class":
        score += 0.5
    return round(score, 2), reasons


def detect_changes(store: GraphStore, root: Path, base: str | None = None) -> list[ChangedSymbol]:
    """Symbols overlapping the diff, ranked by how scary they are to change."""
    found: dict[str, ChangedSymbol] = {}
    for path, lines in changed_lines(root, base).items():
        nodes = [n for n in store.nodes_in_file(path) if n.kind != "module"]
        if not nodes:
            continue
        touched: dict[str, list[int]] = {}
        for line in lines:
            candidates = [n for n in nodes if n.line_start <= line <= n.line_end]
            if not candidates:
                continue
            innermost = min(candidates, key=lambda n: n.span)
            touched.setdefault(innermost.qname, []).append(line)
        for qname, hit_lines in touched.items():
            node = store.get_node(qname)
            if node is None:
                continue
            fan_in = len({e.src for e in store.edges_to(qname, "calls")})
            fan_out = len({e.dst for e in store.edges_from(qname, "calls")})
            tests = tuple(sorted({e.src for e in store.edges_to(qname, "tests")}))
            risk, reasons = _risk(node, fan_in, fan_out, len(tests), len(hit_lines))
            found[qname] = ChangedSymbol(
                node=node,
                changed_lines=tuple(sorted(hit_lines)),
                fan_in=fan_in,
                fan_out=fan_out,
                covering_tests=tests,
                risk=risk,
                reasons=tuple(reasons),
            )
    return sorted(found.values(), key=lambda c: (-c.risk, c.node.path, c.node.line_start))


@dataclass(frozen=True, slots=True)
class ReviewSnippet:
    qname: str
    path: str
    line_start: int
    line_end: int
    source: str
    callers: tuple[str, ...]
    tests: tuple[str, ...]


def review_context(
    store: GraphStore,
    root: Path,
    qnames: list[str],
    *,
    context_lines: int = 2,
    max_lines: int = 120,
) -> list[ReviewSnippet]:
    """Source for each symbol plus who depends on it — the minimum a reviewer needs."""
    snippets: list[ReviewSnippet] = []
    for qname in qnames:
        node = store.get_node(qname)
        if node is None:
            continue
        file_path = root / node.path
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        start = max(1, node.line_start - context_lines)
        end = min(len(lines), node.line_end + context_lines, start + max_lines - 1)
        body = "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(start, end + 1))
        if end < node.line_end + context_lines:
            body += f"\n       ... ({node.line_end - end} more lines)"
        report = impact_radius(store, qname, depth=1)
        snippets.append(
            ReviewSnippet(
                qname=qname,
                path=node.path,
                line_start=start,
                line_end=end,
                source=body,
                callers=report.direct_callers if report else (),
                tests=report.covering_tests if report else (),
            )
        )
    return snippets
