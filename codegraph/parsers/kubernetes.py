"""Kubernetes manifests, Kustomize overlays, and the dispatcher for plain ``.yaml`` files.

Every document with ``apiVersion`` + ``kind`` + ``metadata.name`` becomes
``resource:k8s/<Kind>/<name>``. Edges follow the references Kubernetes itself resolves
by name: config maps and secrets a pod mounts or reads, the PVCs it claims, its service
account and pull secrets, the image each container runs, an Ingress backend's Service,
an HPA's scale target, a binding's role and subjects.

Service (and NetworkPolicy, PodDisruptionBudget) selectors match pod-template labels.
When the matching workload is in the same file the edge is written directly; otherwise
an unresolved edge to ``resource:k8s/selector/<k=v,...>`` records the selector and
:mod:`codegraph.parsers.resolve` matches it against the rest of the repository.

A ``.yaml`` file that is neither Kubernetes nor compose (CI configs, app settings)
yields :meth:`ParseResult.empty`, so it adds nothing to the graph.
"""

from __future__ import annotations

from dataclasses import dataclass

from codegraph.models import Language, ParseResult
from codegraph.parsers.base import module_qname_for
from codegraph.parsers.docker import is_compose, parse_compose
from codegraph.parsers.paths import normalize, relative_path
from codegraph.parsers.resources import ResourceBuilder, image_qname, resource_qname
from codegraph.parsers.yaml_tree import YamlNode, load_documents

WORKLOAD_KINDS = frozenset(
    {
        "Deployment",
        "StatefulSet",
        "DaemonSet",
        "ReplicaSet",
        "ReplicationController",
        "Job",
        "CronJob",
        "Pod",
    }
)
SELECTOR_PREFIX = resource_qname("k8s/selector", "")


def k8s_qname(kind: str, name: str) -> str:
    return resource_qname(f"k8s/{kind}", name)


def selector_qname(selector: dict[str, str]) -> str:
    return SELECTOR_PREFIX + ",".join(f"{k}={v}" for k, v in sorted(selector.items()))


def parse_selector(qname: str) -> dict[str, str]:
    """Inverse of :func:`selector_qname`."""
    body = qname.removeprefix(SELECTOR_PREFIX)
    pairs = (item.partition("=") for item in body.split(",") if item)
    return {key: value for key, _, value in pairs}


def is_manifest(document: YamlNode) -> bool:
    return (
        document.kind == "map" and bool(document.text("apiVersion")) and bool(document.text("kind"))
    )


def _objects(documents: list[YamlNode]) -> list[YamlNode]:
    found: list[YamlNode] = []
    for document in documents:
        if not is_manifest(document):
            continue
        if document.text("kind") == "List":
            found.extend(item for item in document.entries("items") if is_manifest(item))
        else:
            found.append(document)
    return found


def _pod_template(document: YamlNode) -> YamlNode | None:
    kind = document.text("kind")
    if kind == "Pod":
        return document
    if kind == "CronJob":
        return document.path("spec", "jobTemplate", "spec", "template")
    return document.path("spec", "template")


def pod_labels(document: YamlNode) -> dict[str, str]:
    """Labels a workload stamps on its pods (what Service selectors match)."""
    if document.text("kind") not in WORKLOAD_KINDS:
        return {}
    template = _pod_template(document)
    return template.mapping("metadata", "labels") if template is not None else {}


@dataclass(frozen=True, slots=True)
class Workload:
    qname: str
    labels: dict[str, str]


def workloads(source: str) -> list[Workload]:
    """Workloads in a manifest file with their pod labels, for cross-file selector matching."""
    found: list[Workload] = []
    for document in _objects(load_documents(source)):
        name = document.text("metadata", "name")
        kind = document.text("kind")
        labels = pod_labels(document)
        if name and kind and labels:
            found.append(Workload(k8s_qname(kind, name), labels))
    return found


def _pod_references(spec: YamlNode) -> list[tuple[str, str]]:
    """(target qname, detail) for everything a pod spec names."""
    refs: list[tuple[str, str]] = []
    for key in ("serviceAccountName", "serviceAccount"):
        account = spec.text(key)
        if account:
            refs.append((k8s_qname("ServiceAccount", account), key))
    for secret in spec.entries("imagePullSecrets"):
        name = secret.text("name")
        if name:
            refs.append((k8s_qname("Secret", name), "imagePullSecrets"))
    for key in ("containers", "initContainers", "ephemeralContainers"):
        for container in spec.entries(key):
            image = container.text("image")
            if image:
                refs.append((image_qname(image), "image"))
            for source in container.entries("envFrom"):
                for ref_key, kind in (("configMapRef", "ConfigMap"), ("secretRef", "Secret")):
                    name = source.text(ref_key, "name")
                    if name:
                        refs.append((k8s_qname(kind, name), ref_key))
            for variable in container.entries("env"):
                for ref_key, kind in (("configMapKeyRef", "ConfigMap"), ("secretKeyRef", "Secret")):
                    name = variable.text("valueFrom", ref_key, "name")
                    if name:
                        refs.append((k8s_qname(kind, name), ref_key))
    for volume in spec.entries("volumes"):
        refs.extend(_volume_references(volume))
    return refs


def _volume_references(volume: YamlNode) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    name = volume.text("configMap", "name")
    if name:
        refs.append((k8s_qname("ConfigMap", name), "volume"))
    name = volume.text("secret", "secretName")
    if name:
        refs.append((k8s_qname("Secret", name), "volume"))
    name = volume.text("persistentVolumeClaim", "claimName")
    if name:
        refs.append((k8s_qname("PersistentVolumeClaim", name), "volume"))
    for source in volume.entries("projected", "sources"):
        for key, kind in (("configMap", "ConfigMap"), ("secret", "Secret")):
            name = source.text(key, "name")
            if name:
                refs.append((k8s_qname(kind, name), "projected"))
    return refs


def _ingress_references(spec: YamlNode) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    backends: list[YamlNode] = []
    for key in ("defaultBackend", "backend"):
        backend = spec.get(key)
        if backend is not None:
            backends.append(backend)
    for rule in spec.entries("rules"):
        for path in rule.entries("http", "paths"):
            backend = path.get("backend")
            if backend is not None:
                backends.append(backend)
    for backend in backends:
        service = backend.text("service", "name") or backend.text("serviceName")
        if service:
            refs.append((k8s_qname("Service", service), "backend"))
    for tls in spec.entries("tls"):
        secret = tls.text("secretName")
        if secret:
            refs.append((k8s_qname("Secret", secret), "tls"))
    return refs


def _object_references(document: YamlNode) -> list[tuple[str, str]]:
    kind = document.text("kind") or ""
    spec = document.get("spec")
    refs: list[tuple[str, str]] = []
    if kind in WORKLOAD_KINDS:
        template = _pod_template(document)
        pod_spec = template.get("spec") if template is not None else None
        if pod_spec is not None:
            refs.extend(_pod_references(pod_spec))
        service = spec.text("serviceName") if spec is not None else None
        if kind == "StatefulSet" and service:
            refs.append((k8s_qname("Service", service), "serviceName"))
    elif kind == "Ingress" and spec is not None:
        refs.extend(_ingress_references(spec))
    elif kind == "HorizontalPodAutoscaler" and spec is not None:
        target_kind = spec.text("scaleTargetRef", "kind")
        target_name = spec.text("scaleTargetRef", "name")
        if target_kind and target_name:
            refs.append((k8s_qname(target_kind, target_name), "scaleTargetRef"))
    elif kind in {"RoleBinding", "ClusterRoleBinding"}:
        role_kind = document.text("roleRef", "kind")
        role_name = document.text("roleRef", "name")
        if role_kind and role_name:
            refs.append((k8s_qname(role_kind, role_name), "roleRef"))
        for subject in document.entries("subjects"):
            subject_kind = subject.text("kind")
            subject_name = subject.text("name")
            if subject_kind == "ServiceAccount" and subject_name:
                refs.append((k8s_qname(subject_kind, subject_name), "subject"))
    return refs


def _selector(document: YamlNode) -> dict[str, str]:
    kind = document.text("kind")
    if kind == "Service":
        return document.mapping("spec", "selector")
    if kind == "NetworkPolicy":
        return document.mapping("spec", "podSelector", "matchLabels")
    if kind == "PodDisruptionBudget":
        return document.mapping("spec", "selector", "matchLabels")
    return {}


def parse_manifests(
    rel_path: str, source: str, module_qname: str, language: Language
) -> ParseResult:
    builder = ResourceBuilder(rel_path, module_qname, source, language)
    objects = _objects(load_documents(source))
    labelled: list[tuple[str, dict[str, str]]] = []
    for document in objects:
        kind = document.text("kind") or ""
        name = document.text("metadata", "name")
        if not name:
            continue
        api = document.text("apiVersion") or ""
        qname = builder.add(
            k8s_qname(kind, name),
            name,
            document.line,
            document.end_line,
            f"{kind} {name} ({api})",
        )
        labels = pod_labels(document)
        if labels:
            labelled.append((qname, labels))
    for document in objects:
        kind = document.text("kind") or ""
        name = document.text("metadata", "name")
        if not name:
            continue
        qname = k8s_qname(kind, name)
        for target, detail in _object_references(document):
            builder.depends(qname, target, document.line, detail=detail)
        selector = _selector(document)
        if not selector:
            continue
        matches = [
            other
            for other, labels in labelled
            if all(labels.get(k) == v for k, v in selector.items())
        ]
        for other in matches:
            builder.depends(qname, other, document.line, detail="selector")
        if not matches:
            builder.depends(
                qname, selector_qname(selector), document.line, detail="selector", resolved=False
            )
    return builder.result()


# -- kustomize ------------------------------------------------------------------------


def kustomization_qname(rel_path: str) -> str:
    directory = rel_path.rsplit("/", 1)[0] if "/" in rel_path else "."
    return resource_qname("kustomize", directory)


def _entry_target(rel_path: str, entry: str) -> str | None:
    if "://" in entry or entry.startswith(("github.com/", "git@")):
        return None
    path = relative_path(rel_path, entry)
    if path is None:
        return None
    if entry.endswith((".yaml", ".yml", ".json")):
        return module_qname_for(path)
    return resource_qname("kustomize", normalize(path) or ".")


def parse_kustomization(
    rel_path: str, source: str, module_qname: str, language: Language
) -> ParseResult:
    builder = ResourceBuilder(rel_path, module_qname, source, language)
    documents = [d for d in load_documents(source) if d.kind == "map"]
    if not documents:
        return builder.result()
    document = documents[0]
    qname = builder.add(
        kustomization_qname(rel_path),
        rel_path.rsplit("/", 1)[0] if "/" in rel_path else ".",
        document.line,
        document.end_line,
        f"kustomization {rel_path}",
    )
    for key in ("resources", "bases", "components", "patchesStrategicMerge", "crds"):
        for entry in document.entries(key):
            target = _entry_target(rel_path, entry.scalar or "")
            if target:
                builder.depends(qname, target, entry.line, detail=key)
    for patch in document.entries("patches") + document.entries("patchesJson6902"):
        path = patch.text("path") if patch.kind == "map" else patch.scalar
        target = _entry_target(rel_path, path or "")
        if target:
            builder.depends(qname, target, patch.line, detail="patch")
    for key, kind in (("configMapGenerator", "ConfigMap"), ("secretGenerator", "Secret")):
        for generator in document.entries(key):
            name = generator.text("name")
            if name:
                generated = builder.add(
                    k8s_qname(kind, name),
                    name,
                    generator.line,
                    generator.end_line,
                    f"{kind} {name} (generated by {key})",
                )
                builder.depends(qname, generated, generator.line, detail=key)
    return builder.result()


class KustomizeParser:
    language: Language = "kustomize"

    def parse(self, rel_path: str, source: str, module_qname: str) -> ParseResult:
        return parse_kustomization(rel_path, source, module_qname, self.language)


class YamlParser:
    """Plain ``.yaml``: Kubernetes manifests or compose files by content, else nothing."""

    language: Language = "yaml"

    def parse(self, rel_path: str, source: str, module_qname: str) -> ParseResult:
        documents = load_documents(source)
        if any(is_manifest(d) for d in documents):
            return parse_manifests(rel_path, source, module_qname, self.language)
        if any(d.kind == "map" and is_compose(d) for d in documents):
            return parse_compose(rel_path, source, module_qname, self.language)
        return ParseResult.empty()


__all__ = [
    "SELECTOR_PREFIX",
    "WORKLOAD_KINDS",
    "KustomizeParser",
    "Workload",
    "YamlParser",
    "is_manifest",
    "k8s_qname",
    "kustomization_qname",
    "parse_kustomization",
    "parse_manifests",
    "parse_selector",
    "pod_labels",
    "selector_qname",
    "workloads",
]
