"""Whole-graph analysis: communities, execution flows, centrality, navigation, architecture.

Everything here reads a :class:`codegraph.graph.GraphView` (or an open store) and returns
frozen dataclasses. Two post-index passes (:mod:`codegraph.passes.communities` and
:mod:`codegraph.passes.flows`) persist the expensive results into ``analysis_*`` tables;
readers fall back to computing in memory when those tables are missing or empty, so
every tool works on a database indexed before the passes existed.

Derived node and edge kinds produced by other passes (routes, channels, ``handles``,
``http_calls``, ``co_changes``) are used when present and never required. The package
deliberately imports nothing from those passes: it only looks at kinds in the graph.

This package must not import :mod:`codegraph.queries` from its ``__init__``:
``queries.dead_code`` imports :mod:`codegraph.analysis.deadcode` lazily.
"""
