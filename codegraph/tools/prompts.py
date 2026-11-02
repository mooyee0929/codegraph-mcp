"""MCP prompts: canned review, architecture, debugging, onboarding and pre-merge workflows."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from codegraph import prompts
from codegraph.tools import ToolContext


def register(server: MCPServer[None], ctx: ToolContext) -> None:
    @server.prompt(name="review_changes", description="Review the working-tree or branch diff.")
    def review_changes(base: str = "") -> str:
        return prompts.review_changes(base)

    @server.prompt(name="architecture_map", description="Map subsystems and their dependencies.")
    def architecture_map() -> str:
        return prompts.architecture_map()

    @server.prompt(name="debug_issue", description="Find the root cause of a described issue.")
    def debug_issue(description: str = "") -> str:
        return prompts.debug_issue(description)

    @server.prompt(name="onboard_developer", description="A guided tour of the codebase.")
    def onboard_developer() -> str:
        return prompts.onboard_developer()

    @server.prompt(name="pre_merge_check", description="GO / NO-GO readiness check for a merge.")
    def pre_merge_check(base: str = "main") -> str:
        return prompts.pre_merge_check(base)


__all__ = ["register"]
