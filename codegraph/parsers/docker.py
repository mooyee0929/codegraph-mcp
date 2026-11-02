"""Dockerfiles and docker-compose files as resource graphs.

A Dockerfile becomes one ``resource:docker/stage/<name>`` node per ``FROM``: the name
is the ``AS`` alias, or the file's module qname for an unnamed final stage
(``#<index>`` for earlier unnamed ones). Each stage ``depends_on`` its base, which is
either an earlier stage or ``resource:docker/image/<image>``, and on every stage or
image it copies from (``COPY --from``, ``RUN --mount=from=``).

A compose file becomes ``resource:compose/service/<name>`` per service, with edges to
the services it depends on or links, the image it runs, the Dockerfile it builds
(by module qname) and the named volumes and networks it uses.

Dockerfile instructions are read line by line (with ``\\`` continuations) rather than
through a grammar: the format is line-oriented and this keeps unusual syntax
(heredocs, parser directives) from derailing the parse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from codegraph.models import Language, ParseResult
from codegraph.parsers.base import module_qname_for
from codegraph.parsers.paths import relative_path
from codegraph.parsers.resources import ResourceBuilder, image_qname, resource_qname
from codegraph.parsers.yaml_tree import YamlNode, load_documents

_FROM_FLAG = re.compile(r"--[\w-]+=\S+")
_MOUNT_FROM = re.compile(r"--mount=\S*?from=([^,\s]+)")
_COPY_FROM = re.compile(r"--from=(\S+)")


@dataclass(frozen=True, slots=True)
class Instruction:
    keyword: str
    arguments: str
    line: int


def instructions(source: str) -> list[Instruction]:
    """Logical Dockerfile instructions, continuation lines joined, comments dropped."""
    found: list[Instruction] = []
    pending: list[str] = []
    start = 0
    for number, raw in enumerate(source.splitlines(), start=1):
        stripped = raw.strip()
        if not pending and (not stripped or stripped.startswith("#")):
            continue
        if pending and stripped.startswith("#"):
            continue
        if not pending:
            start = number
        if stripped.endswith("\\"):
            pending.append(stripped[:-1])
            continue
        pending.append(stripped)
        text = " ".join(part.strip() for part in pending if part.strip())
        pending = []
        keyword, _, arguments = text.partition(" ")
        found.append(Instruction(keyword.upper(), arguments.strip(), start))
    if pending:
        text = " ".join(pending)
        keyword, _, arguments = text.partition(" ")
        found.append(Instruction(keyword.upper(), arguments.strip(), start))
    return found


def _comment_above(lines: list[str], line: int) -> str | None:
    collected: list[str] = []
    row = line - 2
    while row >= 0 and lines[row].strip().startswith("#"):
        text = lines[row].strip().lstrip("#").strip()
        if text and not text.lower().startswith(("syntax=", "escape=", "check=")):
            collected.append(text)
        row -= 1
    return " ".join(reversed(collected)) or None


class DockerfileParser:
    """Parses ``Dockerfile`` / ``Containerfile`` / ``*.dockerfile`` into stage resources."""

    language: Language = "dockerfile"

    def parse(self, rel_path: str, source: str, module_qname: str) -> ParseResult:
        builder = ResourceBuilder(rel_path, module_qname, source, self.language)
        steps = instructions(source)
        lines = source.splitlines()
        froms = [i for i, step in enumerate(steps) if step.keyword == "FROM"]
        stages: dict[str, str] = {}
        current: str | None = None
        for index, step in enumerate(steps):
            if step.keyword == "FROM":
                position = froms.index(index)
                words = _FROM_FLAG.sub("", step.arguments).split()
                if not words:
                    continue
                image = words[0]
                alias = words[2] if len(words) >= 3 and words[1].lower() == "as" else None
                is_last = position == len(froms) - 1
                name = alias or (module_qname if is_last else f"{module_qname}#{position}")
                following = froms[position + 1] if not is_last else None
                end = steps[following].line - 1 if following is not None else len(lines)
                current = builder.add(
                    resource_qname("docker/stage", name),
                    alias or f"stage{position}",
                    step.line,
                    end,
                    f"FROM {step.arguments}",
                    _comment_above(lines, step.line),
                )
                base = stages.get(image.lower())
                if base is not None:
                    builder.depends(current, base, step.line, detail="FROM")
                elif image.lower() != "scratch":
                    builder.depends(current, image_qname(image), step.line, detail="FROM")
                stages[str(position)] = current
                if alias:
                    stages[alias.lower()] = current
                continue
            if current is None:
                continue
            sources = _COPY_FROM.findall(step.arguments) if step.keyword in {"COPY", "ADD"} else []
            if step.keyword == "RUN":
                sources = _MOUNT_FROM.findall(step.arguments)
            for reference in sources:
                target = stages.get(reference.lower()) or image_qname(reference)
                builder.depends(current, target, step.line, detail=step.keyword)
        return builder.result()


# -- compose --------------------------------------------------------------------------


def _strings(node: YamlNode | None) -> list[str]:
    """A YAML list of strings, or the keys of a mapping (compose allows both forms)."""
    if node is None:
        return []
    if node.kind == "seq":
        return [item.scalar for item in node.items if item.kind == "scalar" and item.scalar]
    if node.kind == "map":
        return [key for key, _ in node.pairs]
    return [node.scalar] if node.scalar else []


def is_compose(document: YamlNode) -> bool:
    services = document.get("services")
    if services is None or services.kind != "map" or not services.pairs:
        return False
    markers = {"image", "build", "command", "ports", "depends_on", "environment"}
    return any(
        value.kind == "map" and markers & {key for key, _ in value.pairs}
        for _, value in services.pairs
    )


def _build_target(rel_path: str, build: YamlNode) -> str | None:
    if build.kind == "scalar":
        context, dockerfile = build.scalar or ".", "Dockerfile"
    else:
        context = build.text("context") or "."
        dockerfile = build.text("dockerfile") or "Dockerfile"
    if "://" in context:
        return None
    path = relative_path(rel_path, f"{context}/{dockerfile}")
    return module_qname_for(path) if path else None


def parse_compose(rel_path: str, source: str, module_qname: str, language: Language) -> ParseResult:
    builder = ResourceBuilder(rel_path, module_qname, source, language)
    for document in load_documents(source):
        if document.kind != "map" or document.get("services") is None:
            continue
        _compose_document(builder, rel_path, document)
    return builder.result()


def _compose_document(builder: ResourceBuilder, rel_path: str, document: YamlNode) -> None:
    named_volumes: set[str] = set()
    for kind in ("volumes", "networks"):
        section = document.get(kind)
        if section is None or section.kind != "map":
            continue
        for name, value in section.pairs:
            builder.add(
                resource_qname(f"compose/{kind[:-1]}", name),
                name,
                value.line or section.line,
                value.end_line or section.line,
                f"{kind[:-1]} {name}",
            )
            if kind == "volumes":
                named_volumes.add(name)
    services = document.get("services")
    if services is None or services.kind != "map":
        return
    for name, service in services.pairs:
        if service.kind != "map":
            continue
        image = service.text("image")
        signature = f"service {name}" + (f": image {image}" if image else "")
        line = max(1, service.line - 1) if service.line else services.line
        qname = builder.add(
            resource_qname("compose/service", name), name, line, service.end_line, signature
        )
        for dependency in _strings(service.get("depends_on")) + _strings(service.get("links")):
            target = dependency.split(":")[0]
            builder.depends(
                qname, resource_qname("compose/service", target), line, detail="service"
            )
        if image:
            builder.depends(qname, image_qname(image), line, detail="image")
        build = service.get("build")
        if build is not None:
            built = _build_target(rel_path, build)
            if built:
                builder.depends(qname, built, build.line or line, detail="build")
        for volume in _volume_names(service.get("volumes")):
            if volume in named_volumes:
                builder.depends(
                    qname, resource_qname("compose/volume", volume), line, detail="volume"
                )
        for network in _strings(service.get("networks")):
            builder.depends(
                qname, resource_qname("compose/network", network), line, detail="network"
            )
        extends = service.get("extends")
        if extends is not None:
            parent = extends.scalar if extends.kind == "scalar" else extends.text("service")
            if parent:
                builder.depends(
                    qname, resource_qname("compose/service", parent), line, detail="extends"
                )


def _volume_names(node: YamlNode | None) -> list[str]:
    names: list[str] = []
    if node is None:
        return names
    for item in node.items:
        if item.kind == "scalar" and item.scalar:
            names.append(item.scalar.split(":")[0])
        elif item.kind == "map":
            source = item.text("source")
            if source:
                names.append(source)
    return names


class ComposeParser:
    """Parses ``docker-compose.yml`` / ``compose.yaml`` files."""

    language: Language = "compose"

    def parse(self, rel_path: str, source: str, module_qname: str) -> ParseResult:
        return parse_compose(rel_path, source, module_qname, self.language)


__all__ = [
    "ComposeParser",
    "DockerfileParser",
    "Instruction",
    "instructions",
    "is_compose",
    "parse_compose",
]
