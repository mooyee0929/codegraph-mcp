"""Cross-file resolution for the tree-sitter languages and Kubernetes selectors.

The pass itself lives next to the parsers it serves (:mod:`codegraph.parsers.resolve`);
this module only exposes it to :func:`codegraph.passes.discover_passes`.
"""

from __future__ import annotations

from codegraph.parsers.resolve import PASS

__all__ = ["PASS"]
