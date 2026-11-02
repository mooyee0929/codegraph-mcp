"""Canned workflows: which codegraph tools to call, in which order, for common jobs.

Each function returns the text of one MCP prompt. They are deliberately short — an
ordered tool plan plus what to report — because the value is in steering the agent to
the cheap structural answer before it starts reading files. Tools contributed by
optional modules may be missing (``--tools`` filtering, or a feature not installed),
so every prompt tells the agent to skip a step whose tool is not listed.
"""

from __future__ import annotations

_PREAMBLE = (
    "Use the codegraph MCP tools before reading files; they return exact path:line "
    "locations for a few hundred tokens. If a tool named below is not in your tool "
    "list, skip that step and continue.\n\n"
)


def _steps(title: str, steps: list[str], outcome: str) -> str:
    body = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))
    return f"{_PREAMBLE}## {title}\n\n{body}\n\n{outcome}"


def review_changes(base: str = "") -> str:
    scope = f"base={base!r}" if base else "no base (working tree against HEAD)"
    return _steps(
        "Review the current changes",
        [
            f"`detect_changes_tool` with {scope}: the risk-ranked symbols the diff touches.",
            "`get_change_impact` for the same diff, or `get_impact_radius` on each of the "
            "top three symbols by risk, to see who depends on them.",
            "`get_affected_flows` to list the execution flows the change passes through.",
            "`get_review_context` on the highest-risk qnames for source, callers and tests.",
            "`search_notes` with the changed symbol names, in case an earlier answer applies.",
        ],
        "Report: what changed, the riskiest symbols and why, callers or flows that may "
        "break, and the test gaps (changed callables with no covering tests).",
    )


def architecture_map() -> str:
    return _steps(
        "Map the architecture",
        [
            "`get_architecture` (or `get_architecture_overview`) for languages, packages, "
            "hubs and entry points.",
            "`list_communities` to see how the code clusters into subsystems.",
            "`trace_path` between the main entry points and the hubs they reach.",
            "`manage_adr` with action=list to read the recorded decisions.",
            "`generate_wiki` if a browsable per-module write-up would help.",
        ],
        "Produce a short overview: the subsystems, how they depend on each other, the "
        "entry points, and a Mermaid diagram of communities with the main dependencies.",
    )


def debug_issue(description: str = "") -> str:
    issue = description.strip() or "<describe the issue>"
    return _steps(
        f"Debug: {issue}",
        [
            "`search_notes` with the symptom, in case this was answered before.",
            "`semantic_search_nodes` (or `search_graph`) with keywords from the description "
            "to find candidate symbols.",
            "`trace_path` from the entry point that shows the symptom to each candidate.",
            "`get_review_context` on the two or three most likely culprits.",
            "`get_impact_radius` on the culprit before proposing a fix.",
            "`save_note` with the question, the root cause and the qnames involved.",
        ],
        "Report the most likely root cause with path:line evidence, the fix, and what "
        "else the fix could affect.",
    )


def onboard_developer() -> str:
    return _steps(
        "Onboard a new developer",
        [
            "`graph_status` for the size of the codebase.",
            "`get_architecture` (or `get_architecture_overview`) for the 30-second model.",
            "`list_communities` and present them as a table of subsystem, size, key modules.",
            "`manage_adr` with action=list for the decisions behind the current shape.",
            "`run_cypher` or `search_graph` to answer follow-up questions precisely.",
        ],
        "Give a guided tour: where to start reading, the three most important flows, "
        "the hubs to be careful with, and where the tests live.",
    )


def pre_merge_check(base: str = "main") -> str:
    return _steps(
        f"Pre-merge check against {base}",
        [
            f"`detect_changes_tool` with base={base!r} for the risk-ranked changed symbols.",
            "`get_change_impact` for the blast radius of the whole change.",
            "`get_affected_flows` for the execution flows the change touches.",
            "`get_impact_radius` on any changed symbol with five or more callers.",
            "`suggest_refactorings` to spot new dead code, duplicates or oversized functions.",
        ],
        "Answer GO or NO-GO with a one-sentence reason, then list the required "
        "follow-ups (missing tests, unreviewed high-fan-in symbols, broken flows).",
    )


PROMPT_NAMES: tuple[str, ...] = (
    "review_changes",
    "architecture_map",
    "debug_issue",
    "onboard_developer",
    "pre_merge_check",
)

__all__ = [
    "PROMPT_NAMES",
    "architecture_map",
    "debug_issue",
    "onboard_developer",
    "pre_merge_check",
    "review_changes",
]
