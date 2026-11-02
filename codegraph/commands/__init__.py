"""CLI command modules.

Every module in this package that defines ``register(add)`` contributes subcommands to
``codegraph``. ``add(name, handler, help)`` creates the subparser (already carrying
``--json``) and returns it so the module can add its own arguments. A handler receives
the parsed namespace — ``args.root`` is the resolved repository root — and returns the
process exit code: 0 for success, 1 for "ran fine, found nothing", 2 for bad input.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
from collections.abc import Callable
from types import ModuleType

Handler = Callable[[argparse.Namespace], int]
AddCommand = Callable[[str, Handler, str], argparse.ArgumentParser]
Register = Callable[[AddCommand], None]


def command_modules() -> list[ModuleType]:
    modules: list[ModuleType] = []
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        if callable(getattr(module, "register", None)):
            modules.append(module)
    return modules


def register_all(add: AddCommand) -> None:
    for module in command_modules():
        register: Register = module.register
        register(add)


__all__ = ["AddCommand", "Handler", "command_modules", "register_all"]
