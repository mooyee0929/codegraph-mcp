"""Terraform / HCL blocks as resources, with ``depends_on`` edges from their references.

Block naming follows Terraform's own address syntax so a reference in an expression
maps straight to a qname:

- ``resource "aws_s3_bucket" "logs"`` -> ``resource:terraform/aws_s3_bucket.logs``
- ``data "aws_ami" "ubuntu"``         -> ``resource:terraform/data.aws_ami.ubuntu``
- ``module "vpc"``                    -> ``resource:terraform/module.vpc``
- ``variable "region"``               -> ``resource:terraform/var.region``
- ``output "id"`` / ``locals { x }``  -> ``output.id`` / ``local.x``

Every ``a.b.c`` traversal inside a block body (including ``depends_on`` lists) that
starts with ``var``, ``local``, ``module``, ``data`` or a resource type becomes an edge
from that block. A module block with a relative ``source`` also imports the module
directory's ``main.tf``.
"""

from __future__ import annotations

from functools import cache

from tree_sitter import Language as TSLanguage
from tree_sitter import Node as TSNode
from tree_sitter import Parser as TSParser
from tree_sitter_language_pack import get_language

from codegraph.models import Language, ParseResult
from codegraph.parsers.paths import relative_module
from codegraph.parsers.resources import ResourceBuilder, resource_qname

_NOT_RESOURCES = frozenset({"each", "count", "path", "terraform", "self", "provider"})
_PREFIXES = {"var": 1, "local": 1, "module": 1, "data": 2}


@cache
def _language() -> TSLanguage:
    return get_language("hcl")


def _text(raw: bytes, node: TSNode) -> str:
    return raw[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _label(raw: bytes, node: TSNode) -> str:
    return _text(raw, node).strip().strip('"')


def address(block_type: str, labels: list[str]) -> str | None:
    """Terraform address of a block, or None for blocks that are not addressable."""
    if block_type == "resource" and len(labels) >= 2:
        return f"{labels[0]}.{labels[1]}"
    if block_type == "data" and len(labels) >= 2:
        return f"data.{labels[0]}.{labels[1]}"
    if block_type == "module" and labels:
        return f"module.{labels[0]}"
    if block_type == "variable" and labels:
        return f"var.{labels[0]}"
    if block_type == "output" and labels:
        return f"output.{labels[0]}"
    if block_type == "provider" and labels:
        return f"provider.{labels[0]}"
    return None


def reference_address(parts: list[str]) -> str | None:
    """``["aws_s3_bucket", "logs", "arn"]`` -> ``aws_s3_bucket.logs``; None for non-refs."""
    if not parts:
        return None
    head = parts[0]
    width = _PREFIXES.get(head)
    if width is not None:
        return ".".join(parts[: width + 1]) if len(parts) > width else None
    if head in _NOT_RESOURCES or len(parts) < 2 or "_" not in head:
        return None
    return f"{head}.{parts[1]}"


def _traversals(raw: bytes, node: TSNode) -> list[tuple[list[str], int]]:
    found: list[tuple[list[str], int]] = []
    stack = [node]
    while stack:
        current = stack.pop()
        children = current.named_children
        if current.type == "expression" and children and children[0].type == "variable_expr":
            parts = [_text(raw, children[0])]
            for child in children[1:]:
                if child.type != "get_attr":
                    break
                parts.append(_text(raw, child).lstrip(".").strip())
            found.append((parts, current.start_point[0] + 1))
        stack.extend(children)
    return found


class TerraformParser:
    """Parses ``.tf`` (and generic ``.hcl``) files."""

    def __init__(self, language: Language = "terraform") -> None:
        self.language: Language = language

    def parse(self, rel_path: str, source: str, module_qname: str) -> ParseResult:
        builder = ResourceBuilder(rel_path, module_qname, source, self.language)
        raw = source.encode("utf-8")
        root = TSParser(_language()).parse(raw).root_node
        body = next((c for c in root.named_children if c.type == "body"), None)
        if body is None:
            return builder.result()
        for block in body.named_children:
            if block.type == "block":
                self._block(builder, rel_path, raw, block)
        return builder.result()

    def _block(self, builder: ResourceBuilder, rel_path: str, raw: bytes, block: TSNode) -> None:
        children = block.named_children
        if not children or children[0].type != "identifier":
            return
        block_type = _text(raw, children[0])
        labels = [_label(raw, c) for c in children[1:] if c.type in {"string_lit", "identifier"}]
        inner = next((c for c in children if c.type == "body"), None)
        line = block.start_point[0] + 1
        end = block.end_point[0] + 1
        if block_type == "locals":
            if inner is None:
                return
            for attribute in inner.named_children:
                if attribute.type != "attribute" or not attribute.named_children:
                    continue
                name = _text(raw, attribute.named_children[0])
                qname = builder.add(
                    resource_qname("terraform", f"local.{name}"),
                    name,
                    attribute.start_point[0] + 1,
                    attribute.end_point[0] + 1,
                    f"local.{name}",
                )
                self._references(builder, raw, qname, attribute)
            return
        found = address(block_type, labels)
        if found is None:
            if not labels:
                return
            found = ".".join([block_type, *labels])
        qname = builder.add(
            resource_qname("terraform", found),
            found.rsplit(".", 1)[-1],
            line,
            end,
            " ".join([block_type, *(f'"{label}"' for label in labels)]),
        )
        if inner is None:
            return
        self._references(builder, raw, qname, inner)
        if block_type == "module":
            self._module_source(builder, rel_path, raw, qname, inner)

    def _references(self, builder: ResourceBuilder, raw: bytes, qname: str, node: TSNode) -> None:
        for parts, line in _traversals(raw, node):
            target = reference_address(parts)
            if target:
                builder.depends(qname, resource_qname("terraform", target), line)

    def _module_source(
        self, builder: ResourceBuilder, rel_path: str, raw: bytes, qname: str, inner: TSNode
    ) -> None:
        for attribute in inner.named_children:
            if attribute.type != "attribute" or not attribute.named_children:
                continue
            if _text(raw, attribute.named_children[0]) != "source":
                continue
            value = _label(raw, attribute.named_children[-1])
            if not value.startswith("."):
                return
            target = relative_module(rel_path, f"{value}/main.tf")
            if target:
                line = attribute.start_point[0] + 1
                builder.imports(target, line, value)
                builder.depends(qname, target, line, detail="source")
            return


__all__ = ["TerraformParser", "address", "reference_address"]
