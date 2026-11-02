"""MCP tool modules.

Every module in this package that defines ``register(server, ctx)`` contributes tools
(and optionally prompts or resources) to the server built by
:func:`codegraph.server.build_server`. Modules are imported in name order, so a module
never depends on another module's registration having happened.

Conventions every tool follows:

- Return a JSON string built with :func:`dump`.
- Open the store per call with ``with ctx.open_store() as store:`` — the server is
  long-lived and other processes (the CLI, the watcher) write the same database.
- Report bad input as ``{"error": "..."}`` instead of raising.
- Name tools in ``snake_case``; prefer a verb (``find_``, ``get_``, ``list_``).
"""

from __future__ import annotations

import importlib
import json
import pkgutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import ModuleType

from mcp.server.mcpserver import MCPServer

from codegraph.models import Node
from codegraph.store import GraphStore, default_db_path


@dataclass(frozen=True, slots=True)
class ToolContext:
    root: Path

    def open_store(self) -> GraphStore:
        return GraphStore(default_db_path(self.root))


Register = Callable[[MCPServer[None], ToolContext], None]


def dump(value: object) -> str:
    """JSON-encode tool output; dataclasses, tuples and sets become plain JSON."""

    def default(obj: object) -> object:
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        if isinstance(obj, tuple | set | frozenset):
            return sorted(obj, key=str) if isinstance(obj, set | frozenset) else list(obj)
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"cannot serialise {type(obj).__name__}")

    return json.dumps(value, default=default, indent=2)


def node_summary(node: Node) -> dict[str, object]:
    return {
        "qname": node.qname,
        "kind": node.kind,
        "path": node.path,
        "lines": [node.line_start, node.line_end],
        "signature": node.signature,
        "docstring": node.docstring,
    }


def tool_modules() -> list[ModuleType]:
    modules: list[ModuleType] = []
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        if callable(getattr(module, "register", None)):
            modules.append(module)
    return modules


def register_all(server: MCPServer[None], ctx: ToolContext) -> None:
    for module in tool_modules():
        register: Register = module.register
        register(server, ctx)


__all__ = ["ToolContext", "dump", "node_summary", "register_all", "tool_modules"]
