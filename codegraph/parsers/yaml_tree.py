"""YAML documents as small typed trees with line numbers, read through tree-sitter.

The project has no YAML library dependency, and the infrastructure parsers need line
numbers anyway (a resource node must point at its manifest), so documents are read
from the tree-sitter YAML grammar into :class:`YamlNode`. Only the data model matters
here: mappings, sequences and scalars. Anchors and tags are skipped, aliases become
their literal text, and malformed input yields whatever was recovered.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from typing import Literal

from tree_sitter import Language as TSLanguage
from tree_sitter import Node as TSNode
from tree_sitter import Parser as TSParser
from tree_sitter_language_pack import get_language

YamlKind = Literal["map", "seq", "scalar"]


@dataclass(frozen=True, slots=True)
class YamlNode:
    kind: YamlKind
    line: int
    end_line: int
    scalar: str | None = None
    items: tuple[YamlNode, ...] = ()
    pairs: tuple[tuple[str, YamlNode], ...] = ()

    def get(self, key: str) -> YamlNode | None:
        for name, value in self.pairs:
            if name == key:
                return value
        return None

    def path(self, *keys: str) -> YamlNode | None:
        cursor: YamlNode | None = self
        for key in keys:
            cursor = cursor.get(key) if cursor is not None else None
        return cursor

    def text(self, *keys: str) -> str | None:
        found = self.path(*keys)
        return found.scalar if found is not None and found.kind == "scalar" else None

    def entries(self, *keys: str) -> tuple[YamlNode, ...]:
        found = self.path(*keys)
        return found.items if found is not None and found.kind == "seq" else ()

    def mapping(self, *keys: str) -> dict[str, str]:
        """String values of a flat mapping (labels, selectors)."""
        found = self.path(*keys)
        if found is None or found.kind != "map":
            return {}
        return {k: v.scalar for k, v in found.pairs if v.kind == "scalar" and v.scalar is not None}


_EMPTY = YamlNode(kind="scalar", line=0, end_line=0)


@cache
def _language() -> TSLanguage:
    return get_language("yaml")


def load_documents(source: str) -> list[YamlNode]:
    """Every document in ``source``; empty documents are dropped."""
    raw = source.encode("utf-8")
    tree = TSParser(_language()).parse(raw)
    documents: list[YamlNode] = []
    for document in tree.root_node.named_children:
        if document.type != "document":
            continue
        content = next((c for c in document.named_children if c.type != "comment"), None)
        if content is None:
            continue
        converted = _convert(raw, content)
        if converted.kind != "scalar" or converted.scalar:
            documents.append(converted)
    return documents


def _text(raw: bytes, node: TSNode) -> str:
    return raw[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _lines(node: TSNode) -> tuple[int, int]:
    return node.start_point[0] + 1, node.end_point[0] + 1


def _content(node: TSNode) -> TSNode | None:
    for child in node.named_children:
        if child.type not in {"anchor", "tag", "comment"}:
            return child
    return None


def _convert(raw: bytes, node: TSNode | None) -> YamlNode:
    if node is None:
        return _EMPTY
    line, end = _lines(node)
    kind = node.type
    if kind in {"block_node", "flow_node"}:
        inner = _content(node)
        return _convert(raw, inner) if inner is not None else YamlNode("scalar", line, end)
    if kind in {"block_mapping", "flow_mapping"}:
        return YamlNode("map", line, end, pairs=tuple(_pairs(raw, node.named_children)))
    if kind in {"block_sequence", "flow_sequence"}:
        items: list[YamlNode] = []
        for child in node.named_children:
            if child.type == "comment":
                continue
            target = _content(child) if child.type == "block_sequence_item" else child
            if child.type == "flow_pair":
                start, stop = _lines(child)
                items.append(YamlNode("map", start, stop, pairs=tuple(_pairs(raw, [child]))))
            elif target is not None:
                items.append(_convert(raw, target))
        return YamlNode("seq", line, end, items=tuple(items))
    return YamlNode("scalar", line, end, scalar=_scalar(raw, node))


def _pairs(raw: bytes, pairs: list[TSNode]) -> Iterator[tuple[str, YamlNode]]:
    for pair in pairs:
        if pair.type not in {"block_mapping_pair", "flow_pair"}:
            continue
        key = pair.child_by_field_name("key")
        value = pair.child_by_field_name("value")
        key_text = _scalar(raw, key) if key is not None else None
        if key_text is None or key_text == "<<":
            continue
        converted = _convert(raw, value) if value is not None else None
        if converted is None:
            line, end = _lines(pair)
            converted = YamlNode("scalar", line, end)
        yield key_text, converted


def _scalar(raw: bytes, node: TSNode) -> str | None:
    inner = node
    while inner.type in {"flow_node", "block_node"}:
        found = _content(inner)
        if found is None:
            return None
        inner = found
    text = _text(raw, inner).strip()
    kind = inner.type
    if kind == "double_quote_scalar":
        return text[1:-1].replace('\\"', '"').replace("\\n", "\n")
    if kind == "single_quote_scalar":
        return text[1:-1].replace("''", "'")
    if kind == "block_scalar":
        body = text.split("\n", 1)[1] if "\n" in text else ""
        rows = body.split("\n")
        indent = min((len(r) - len(r.lstrip()) for r in rows if r.strip()), default=0)
        return "\n".join(r[indent:] for r in rows).strip("\n")
    if kind in {"block_mapping", "flow_mapping", "block_sequence", "flow_sequence"}:
        return None
    return text


__all__ = ["YamlNode", "load_documents"]
