"""Building blocks shared by the infrastructure parsers.

Infrastructure files contribute ``resource`` nodes (``resource:<type>/<name>``) and
``depends_on`` edges instead of classes and calls. Resource qnames are derived from
names, not file paths, so a reference in one manifest (``configMapRef: web-config``)
can name a resource defined in another without any cross-file lookup: the edge is
written with the target's qname directly and becomes live once that file is indexed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from codegraph.models import Edge, Language, Node, ParseResult
from codegraph.parsers.base import clip

RESOURCE_PREFIX = "resource:"


def resource_qname(kind: str, name: str) -> str:
    """``resource_qname("k8s/Deployment", "web")`` -> ``resource:k8s/Deployment/web``."""
    return f"{RESOURCE_PREFIX}{kind}/{name}"


def image_qname(image: str) -> str:
    return resource_qname("docker/image", image)


def module_node(rel_path: str, module_qname: str, source: str, language: Language) -> Node:
    return Node(
        qname=module_qname,
        kind="module",
        path=rel_path,
        line_start=1,
        line_end=max(1, source.count("\n") + 1),
        language=language,
        name=module_qname.rsplit(".", 1)[-1],
    )


@dataclass(slots=True)
class ResourceBuilder:
    """Accumulates resource nodes and their edges for one file."""

    rel_path: str
    module_qname: str
    source: str
    language: Language
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    _seen: set[tuple[str, str, int]] = field(default_factory=set)

    def add(
        self,
        qname: str,
        name: str,
        line: int,
        end_line: int,
        signature: str,
        docstring: str | None = None,
    ) -> str:
        if qname not in self.nodes:
            self.nodes[qname] = Node(
                qname=qname,
                kind="resource",
                path=self.rel_path,
                line_start=line,
                line_end=max(line, end_line),
                language=self.language,
                name=name,
                signature=clip(signature),
                docstring=docstring,
            )
            self.edges.append(
                Edge(
                    src=self.module_qname,
                    dst=qname,
                    kind="defines",
                    path=self.rel_path,
                    line=line,
                )
            )
        return qname

    def depends(
        self, src: str, dst: str, line: int, *, detail: str = "", resolved: bool = True
    ) -> None:
        key = (src, dst, line)
        if not dst or src == dst or key in self._seen:
            return
        self._seen.add(key)
        self.edges.append(
            Edge(
                src=src,
                dst=dst,
                kind="depends_on",
                path=self.rel_path,
                line=line,
                resolved=resolved,
                detail=detail,
            )
        )

    def imports(self, dst: str, line: int, spec: str) -> None:
        self.edges.append(
            Edge(
                src=self.module_qname,
                dst=dst,
                kind="imports",
                path=self.rel_path,
                line=line,
                detail=spec,
            )
        )

    def result(self) -> ParseResult:
        module = module_node(self.rel_path, self.module_qname, self.source, self.language)
        return ParseResult(nodes=(module, *self.nodes.values()), edges=tuple(self.edges))


__all__ = ["RESOURCE_PREFIX", "ResourceBuilder", "image_qname", "module_node", "resource_qname"]
