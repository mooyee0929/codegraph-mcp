"""Post-parse pass that upgrades dangling edges into real ones.

Parsers only know one file at a time, so ``helper()`` with no import behind it is
stored as an edge to the bare string ``helper``. Once every file is in the store we
can look the name up: if exactly one definition in the whole repository is called
``helper``, that is the target. Ambiguous names stay unresolved rather than guessed.

The same pass infers ``tests`` edges: a function living in a test file that calls
``X`` (directly or by name) is recorded as testing ``X``.
"""

from __future__ import annotations

import re
from collections import defaultdict

from codegraph.models import Edge
from codegraph.store import GraphStore

_TEST_PATH = re.compile(
    r"(^|/)(tests?|__tests__|spec)(/|$)"
    r"|(^|/)test_[^/]*\.py$"
    r"|[._-](test|spec)\.[cm]?[jt]sx?$"
    r"|_test\.(go|rs|dart|exs|rb|lua|c|cc|cpp|cxx)$"
    r"|_(spec|unittest)\.(rb|lua|cc|cpp)$"
    r"|(Test|Tests|IT|Spec|Suite)\.(java|kt|kts|groovy|scala|cs|php|swift|m)$"
    r"|Spec\.hs$"
    r"|(^|/)test_[^/]*\.(c|cc|cpp|cxx|rb|sh|lua)$"
    r"|_(SUITE|tests)\.erl$"
    r"|\.t\.sol$"
)


def is_test_path(rel_path: str) -> bool:
    return bool(_TEST_PATH.search(rel_path))


# Method names so common on builtins and library objects that a unique project
# definition with the same name is almost certainly not the callee.
COMMON_METHOD_NAMES: frozenset[str] = frozenset(
    {
        "Add",
        "add",
        "alloc",
        "Any",
        "append",
        "apply",
        "as_ref",
        "as_str",
        "bind",
        "borrow",
        "borrow_mut",
        "call",
        "catch",
        "clear",
        "clone",
        "Close",
        "close",
        "collect",
        "Contains",
        "contains",
        "containsKey",
        "copy",
        "Count",
        "count",
        "dealloc",
        "debug",
        "decode",
        "delete",
        "describe",
        "Do",
        "done",
        "each",
        "emit",
        "encode",
        "endswith",
        "Equals",
        "equals",
        "Error",
        "error",
        "Errorf",
        "exists",
        "expect",
        "extend",
        "Fatal",
        "Fatalf",
        "filter",
        "find",
        "First",
        "forEach",
        "format",
        "get",
        "GetHashCode",
        "getOrDefault",
        "group",
        "has",
        "hashCode",
        "includes",
        "index",
        "info",
        "init",
        "insert",
        "into",
        "is_empty",
        "isEmpty",
        "it",
        "items",
        "iter",
        "join",
        "json",
        "keys",
        "Len",
        "len",
        "length",
        "Lock",
        "lock",
        "log",
        "lower",
        "map",
        "map_err",
        "match",
        "new",
        "next",
        "off",
        "ok_or",
        "on",
        "once",
        "open",
        "pop",
        "Print",
        "print",
        "Printf",
        "printf",
        "Println",
        "println",
        "push",
        "put",
        "puts",
        "Read",
        "read",
        "reduce",
        "reject",
        "release",
        "Remove",
        "remove",
        "replace",
        "require",
        "resolve",
        "retain",
        "run",
        "search",
        "Select",
        "send",
        "set",
        "size",
        "slice",
        "sort",
        "splice",
        "split",
        "Sprintf",
        "startswith",
        "status",
        "stream",
        "String",
        "strip",
        "sub",
        "test",
        "text",
        "then",
        "to_string",
        "ToList",
        "ToString",
        "toString",
        "Unlock",
        "unwrap",
        "unwrap_or",
        "update",
        "upper",
        "values",
        "Wait",
        "warn",
        "Where",
        "Write",
        "write",
    }
)


def link(store: GraphStore) -> int:
    """Resolve unique-name edges and infer test coverage. Returns edges upgraded."""
    nested: set[str] = set()
    for node in store.all_nodes(kinds=("function", "method")):
        for edge in store.edges_to(node.qname, kind="defines"):
            parent = store.get_node(edge.src)
            if parent is not None and parent.kind in {"function", "method"}:
                nested.add(node.qname)

    by_name: dict[str, list[str]] = defaultdict(list)
    for node in store.all_nodes(kinds=("function", "method", "class")):
        if node.qname not in nested:  # closures are not reachable by name from elsewhere
            by_name[node.name].append(node.qname)

    upgraded = 0
    for edge_id, edge in store.edge_ids_unresolved():
        is_method_call = edge.dst.startswith("*.")
        name = edge.dst.rsplit(".", 1)[-1]
        if is_method_call and name in COMMON_METHOD_NAMES:
            continue
        candidates = by_name.get(name, [])
        if len(candidates) == 1 and candidates[0] != edge.src:
            store.resolve_edge(edge_id, candidates[0])
            upgraded += 1
    store.drop_unresolved_duplicates()

    test_edges: list[Edge] = []
    for path in store.indexed_paths():
        if not is_test_path(path):
            continue
        for test_node in store.nodes_in_file(path):
            if test_node.kind == "module":
                continue
            for call in store.edges_from(test_node.qname, kind="calls"):
                if not call.resolved or is_test_path(_path_of(store, call.dst) or path):
                    continue
                test_edges.append(
                    Edge(
                        src=test_node.qname,
                        dst=call.dst,
                        kind="tests",
                        path=path,
                        line=call.line,
                    )
                )
    store.add_edges(test_edges)
    return upgraded


def _path_of(store: GraphStore, qname: str) -> str | None:
    node = store.get_node(qname)
    return node.path if node else None
