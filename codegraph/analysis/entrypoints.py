"""Where execution starts: route handlers, scripts, ``main``, framework hooks and roots.

Flows begin at entry points and dead-code detection must never report one, so both
share this module. The graph has no decorator column: the Python parser records each
decorator as a ``calls`` edge from the decorated definition whose line is *above* the
``def``/``class`` line, and the JS/TS parser keeps decorators at the start of the
signature text. :func:`decorators_of` reads both.

Decorator names come out as the call target the parser could see: ``click.command``
when the import is known, ``*.route`` when the receiver is a local object such as a
Flask ``app``. The patterns below therefore match on the attribute tail, not on the
receiver.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from codegraph.analysis.common import CALLABLE_KINDS, is_test_node, module_of_path
from codegraph.graph import GraphView
from codegraph.models import Edge, Node

ENTRY_KIND_ORDER: tuple[str, ...] = (
    "route",
    "cli",
    "main",
    "decorator",
    "conventional",
    "module",
    "root",
)
"""Strongest evidence first; an entry point reports the first kind that applies."""

FRAMEWORK_DECORATORS: tuple[re.Pattern[str], ...] = (
    # HTTP routes: Flask, FastAPI, Starlette, Sanic, aiohttp, Django Ninja.
    re.compile(r"\.(get|post|put|delete|patch|head|options|route|api_route|websocket|view)$"),
    re.compile(
        r"\.(on_event|middleware|exception_handler|errorhandler|before_request|after_request"
        r"|teardown_request|before_app_request|after_app_request|context_processor)$"
    ),
    # CLIs: click and typer groups and commands.
    re.compile(r"\.(command|group|callback|result_callback)$"),
    # Task queues and schedulers.
    re.compile(r"(^|\.)(shared_task|periodic_task|scheduled_job)$"),
    re.compile(r"\.(task|flow|job|cron)$"),
    # Django, SQLAlchemy and pydantic hooks.
    re.compile(r"(^|\.)(receiver|api_view|action|listens_for|validates)$"),
    re.compile(
        r"(^|\.)(field_validator|model_validator|validator|root_validator"
        r"|field_serializer|model_serializer|computed_field)$"
    ),
    # Plugin and agent registries: MCP servers, pluggy, pydantic-ai, atexit.
    re.compile(r"\.(tool|tool_plain|prompt|resource|system_prompt|hookimpl|subscribe)$"),
    re.compile(r"^atexit\.register$|(^|\.)fixture$"),
    # TypeScript / Java annotations: NestJS, Angular, Spring.
    re.compile(
        r"(^|\.)(Get|Post|Put|Delete|Patch|Options|Head|All|Controller|Injectable|Component"
        r"|Module|Resolver|Query|Mutation|Subscription|EventPattern|MessagePattern|Cron"
        r"|OnEvent|HostListener|Directive|Pipe|NgModule|RequestMapping|GetMapping"
        r"|PostMapping|PutMapping|DeleteMapping|PatchMapping|Scheduled|EventListener"
        r"|KafkaListener|Bean)$"
    ),
)
"""Decorators that register the definition with a framework that will call it."""

IMPLICIT_DECORATORS: re.Pattern[str] = re.compile(
    r"(^|\.)(property|cached_property|abstractmethod|abstractproperty|staticmethod"
    r"|classmethod|overload|override|setter|getter|deleter|register|singledispatchmethod)$"
)
"""Decorators after which the runtime, not a visible call, reaches the definition."""

CONVENTIONAL_NAMES: re.Pattern[str] = re.compile(
    r"^(main|handler|lambda_handler|upgrade|downgrade"
    r"|do_(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)|do(Get|Post|Put|Delete)"
    r"|componentDidMount|componentDidUpdate|componentWillUnmount|shouldComponentUpdate"
    r"|ng(OnInit|OnChanges|OnDestroy|DoCheck|AfterContentInit|AfterViewInit))$"
)
"""Names that frameworks or runtimes look up by convention."""

_SIGNATURE_DECORATOR = re.compile(r"@([A-Za-z_$][\w$.]*)")


@dataclass(frozen=True, slots=True)
class EntryPoint:
    qname: str
    kind: str
    """One of :data:`ENTRY_KIND_ORDER`."""
    reason: str


def is_decorator_edge(node: Node, edge: Edge) -> bool:
    """A ``calls`` edge that the Python parser emitted for a decorator of ``node``."""
    return edge.kind == "calls" and edge.path == node.path and 0 < edge.line < node.line_start


def decorators_of(view: GraphView, node: Node) -> tuple[str, ...]:
    """Decorator targets of ``node``: ``("click.command", "*.route")``."""
    found: list[str] = []
    for edge in view.edges_out(node.qname, ("calls",)):
        if is_decorator_edge(node, edge) and edge.dst not in found:
            found.append(edge.dst)
    if node.signature.lstrip().startswith("@"):
        for name in _SIGNATURE_DECORATOR.findall(node.signature):
            if name not in found:
                found.append(name)
    return tuple(found)


def framework_decorator(decorators: Iterable[str]) -> str | None:
    for decorator in decorators:
        if any(pattern.search(decorator) for pattern in FRAMEWORK_DECORATORS):
            return decorator
    return None


def has_implicit_decorator(decorators: Iterable[str]) -> bool:
    return any(IMPLICIT_DECORATORS.search(d) for d in decorators)


def is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def class_of(view: GraphView, method: Node) -> Node | None:
    """The class whose body defines ``method``."""
    for parent in view.predecessors(method.qname, ("defines",)):
        candidate = view.nodes[parent]
        if candidate.kind == "class":
            return candidate
    return None


def may_override(view: GraphView, method: Node) -> bool:
    """True when a base class defines the same name, or a base lies outside the repo."""
    owner = class_of(view, method)
    if owner is None:
        return False
    seen: set[str] = {owner.qname}
    pending = [owner.qname]
    while pending:
        current = pending.pop()
        for edge in view.edges_out(current, ("inherits",)):
            if edge.dst not in view.nodes:
                return True  # an external base: its methods are invisible to us
            if edge.dst in seen:
                continue
            seen.add(edge.dst)
            if f"{edge.dst}.{method.name}" in view.nodes:
                return True
            pending.append(edge.dst)
    return False


def cli_scripts(root: Path | None) -> dict[str, str]:
    """``{"pkg.cli.main": "mytool"}`` from ``pyproject.toml`` script tables.

    Covers ``[project.scripts]``, ``[project.gui-scripts]``, ``[project.entry-points.*]``
    and ``[tool.poetry.scripts]``. A missing or unreadable file means no script entry
    points: indexing must not fail because packaging metadata is broken.
    """
    if root is None:
        return {}
    path = root / "pyproject.toml"
    if not path.is_file():
        return {}
    try:
        data: dict[str, object] = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return {}
    project = _table(data.get("project"))
    tables = [_table(project.get("scripts")), _table(project.get("gui-scripts"))]
    tables.extend(_table(group) for group in _table(project.get("entry-points")).values())
    tables.append(_table(_table(_table(data.get("tool")).get("poetry")).get("scripts")))
    scripts: dict[str, str] = {}
    for table in tables:
        for script, target in table.items():
            if isinstance(target, str) and ":" in target:
                module, _, attr = target.partition(":")
                scripts.setdefault(f"{module.strip()}.{attr.split('[')[0].strip()}", script)
    return scripts


def _table(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    return {}


def resolve_scripts(view: GraphView, scripts: Mapping[str, str]) -> dict[str, str]:
    """Map script targets onto node qnames; ``src/`` layouts make qnames longer."""
    if not scripts:
        return {}
    by_name: dict[str, list[str]] = {}
    for node in view.of_kind(*CALLABLE_KINDS):
        by_name.setdefault(node.name, []).append(node.qname)
    resolved: dict[str, str] = {}
    for target, script in scripts.items():
        if target in view.nodes:
            resolved[target] = script
            continue
        for qname in by_name.get(target.rsplit(".", 1)[-1], []):
            if qname.endswith(f".{target}"):
                resolved.setdefault(qname, script)
    return resolved


def unresolved_call_names(view: GraphView) -> frozenset[str]:
    """Bare names of call targets the linker could not pin down (``*.save`` -> ``save``)."""
    names: set[str] = set()
    for edges in view.out_edges.values():
        for edge in edges:
            if edge.kind in {"calls", "inherits"} and edge.dst not in view.nodes:
                names.add(edge.dst.rsplit(".", 1)[-1])
    return frozenset(names)


def detect_entry_points(
    view: GraphView,
    scripts: Mapping[str, str] | None = None,
    *,
    include_tests: bool = False,
) -> list[EntryPoint]:
    """Every callable where execution plausibly begins, strongest evidence first.

    ``root`` entries (nothing in production code calls them, but they call something)
    skip names that also appear as unresolved call targets, since an ambiguous
    ``obj.process()`` may well be reaching them.
    """
    handled: dict[str, str] = {}
    for edge in view.edges(("handles",)):
        route = view.nodes[edge.src]
        handled.setdefault(edge.dst, route.qname.removeprefix("route:"))
    script_targets = resolve_scripts(view, scripts or {})
    maybe_called = unresolved_call_names(view)

    found: list[EntryPoint] = []
    for node in sorted(view.of_kind(*CALLABLE_KINDS), key=lambda n: n.qname):
        if not include_tests and is_test_node(node):
            continue
        entry = _classify(view, node, handled, script_targets, maybe_called)
        if entry is not None:
            found.append(entry)
    rank = {kind: i for i, kind in enumerate(ENTRY_KIND_ORDER)}
    return sorted(found, key=lambda e: (rank[e.kind], e.qname))


def _classify(
    view: GraphView,
    node: Node,
    handled: Mapping[str, str],
    scripts: Mapping[str, str],
    maybe_called: frozenset[str],
) -> EntryPoint | None:
    if node.qname in handled:
        return EntryPoint(node.qname, "route", f"serves {handled[node.qname]}")
    if node.qname in scripts:
        return EntryPoint(node.qname, "cli", f"console script {scripts[node.qname]!r}")
    if node.name == "main" or module_of_path(node.path).endswith("__main__"):
        return EntryPoint(node.qname, "main", "main entry point")
    decorator = framework_decorator(decorators_of(view, node))
    if decorator is not None:
        return EntryPoint(node.qname, "decorator", f"registered by @{decorator}")
    if CONVENTIONAL_NAMES.match(node.name):
        return EntryPoint(node.qname, "conventional", f"conventional name {node.name!r}")

    production_callers: list[str] = []
    module_callers: list[str] = []
    for caller in view.predecessors(node.qname, ("calls",)):
        caller_node = view.nodes[caller]
        if caller == node.qname or is_test_node(caller_node):
            continue
        if caller_node.kind == "module":
            module_callers.append(caller)
        else:
            production_callers.append(caller)
    if production_callers:
        return None
    if module_callers:
        return EntryPoint(node.qname, "module", f"called at import time by {module_callers[0]}")
    has_callees = any(
        not is_decorator_edge(node, e) and e.dst != node.qname
        for e in view.edges_out(node.qname, ("calls",))
        if e.dst in view.nodes
    )
    if has_callees and not is_dunder(node.name) and node.name not in maybe_called:
        return EntryPoint(node.qname, "root", "no production callers")
    return None


__all__ = [
    "CONVENTIONAL_NAMES",
    "ENTRY_KIND_ORDER",
    "FRAMEWORK_DECORATORS",
    "IMPLICIT_DECORATORS",
    "EntryPoint",
    "class_of",
    "cli_scripts",
    "decorators_of",
    "detect_entry_points",
    "framework_decorator",
    "has_implicit_decorator",
    "is_decorator_edge",
    "is_dunder",
    "may_override",
    "resolve_scripts",
    "unresolved_call_names",
]
