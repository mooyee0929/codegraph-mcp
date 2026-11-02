"""Infrastructure as code: Dockerfile, compose, Kubernetes, Kustomize, Terraform, Make."""

from __future__ import annotations

from codegraph.models import ParseResult
from codegraph.parsers import module_qname_for, parser_for
from codegraph.parsers.docker import instructions, is_compose
from codegraph.parsers.inventory import resource_type
from codegraph.parsers.kubernetes import parse_selector, pod_labels, selector_qname, workloads
from codegraph.parsers.terraform import address, reference_address
from codegraph.parsers.yaml_tree import load_documents


def parse(language: str, path: str, source: str) -> ParseResult:
    return parser_for(language).parse(path, source, module_qname_for(path))


def depends(result: ParseResult) -> set[tuple[str, str, bool]]:
    return {(e.src, e.dst, e.resolved) for e in result.edges if e.kind == "depends_on"}


def resources(result: ParseResult) -> dict[str, tuple[int, int]]:
    return {n.qname: (n.line_start, n.line_end) for n in result.nodes if n.kind == "resource"}


DOCKERFILE = """# syntax=docker/dockerfile:1
ARG PY=3.12
# Build wheels.
FROM python:${PY}-slim AS builder
RUN pip wheel -w /wheels . \\
    --no-deps
FROM builder AS test
RUN --mount=type=cache,from=builder,target=/root pytest
FROM gcr.io/distroless/python3
COPY --from=builder /wheels /wheels
COPY --from=nginx:1.25 /etc/nginx /etc/nginx
"""


def test_dockerfile_stages_and_edges() -> None:
    result = parse("dockerfile", "api/Dockerfile", DOCKERFILE)
    found = resources(result)
    builder = "resource:docker/stage/builder"
    final = "resource:docker/stage/api.Dockerfile"
    assert found == {
        builder: (4, 6),
        "resource:docker/stage/test": (7, 8),
        final: (9, 11),
    }
    edges = depends(result)
    assert (builder, "resource:docker/image/python:${PY}-slim", True) in edges
    assert ("resource:docker/stage/test", builder, True) in edges
    assert (final, "resource:docker/image/gcr.io/distroless/python3", True) in edges
    assert (final, builder, True) in edges
    assert (final, "resource:docker/image/nginx:1.25", True) in edges
    stage = next(n for n in result.nodes if n.qname == builder)
    assert stage.docstring == "Build wheels."
    assert stage.signature.startswith("FROM python")
    assert result.nodes[0].kind == "module"


def test_dockerfile_instruction_continuations() -> None:
    steps = instructions("FROM a\nRUN x \\\n  # note\n  y\n")
    assert [(s.keyword, s.arguments, s.line) for s in steps] == [
        ("FROM", "a", 1),
        ("RUN", "x y", 2),
    ]


def test_dockerfile_scratch_and_unnamed_stages() -> None:
    result = parse("dockerfile", "Dockerfile", "FROM golang AS b\nFROM scratch\nFROM alpine\n")
    assert set(resources(result)) == {
        "resource:docker/stage/b",
        "resource:docker/stage/Dockerfile#1",
        "resource:docker/stage/Dockerfile",
    }
    assert ("resource:docker/stage/Dockerfile#1", "resource:docker/image/scratch", True) not in (
        depends(result)
    )


COMPOSE = """services:
  api:
    build:
      context: ./api
    depends_on:
      db:
        condition: service_healthy
    volumes:
      - data:/var/data
      - ./src:/src
    networks: [backend]
  db:
    image: postgres:16
    links: ["cache:redis"]
  cache:
    image: redis:7
volumes:
  data: {}
networks:
  backend: {}
"""


def test_compose_services() -> None:
    result = parse("compose", "docker-compose.yml", COMPOSE)
    api = "resource:compose/service/api"
    db = "resource:compose/service/db"
    assert set(resources(result)) >= {
        api,
        db,
        "resource:compose/service/cache",
        "resource:compose/volume/data",
        "resource:compose/network/backend",
    }
    edges = depends(result)
    assert (api, db, True) in edges
    assert (api, "api.Dockerfile", True) in edges
    assert (api, "resource:compose/volume/data", True) in edges
    assert (api, "resource:compose/network/backend", True) in edges
    assert (db, "resource:docker/image/postgres:16", True) in edges
    assert (db, "resource:compose/service/cache", True) in edges
    assert not any(dst.endswith("/src") for _, dst, _ in edges)


MANIFESTS = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  template:
    metadata:
      labels: {app: web, tier: front}
    spec:
      serviceAccountName: web-sa
      containers:
        - name: web
          image: "nginx:1.25"
          envFrom:
            - configMapRef: {name: web-config}
          env:
            - name: TOKEN
              valueFrom:
                secretKeyRef: {name: web-secret, key: token}
      volumes:
        - name: data
          persistentVolumeClaim: {claimName: web-data}
---
apiVersion: v1
kind: Service
metadata:
  name: web
spec:
  selector: {app: web}
---
apiVersion: v1
kind: Service
metadata:
  name: api
spec:
  selector: {app: api}
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web
spec:
  tls: [{secretName: web-tls}]
  rules:
    - http:
        paths:
          - path: /
            backend:
              service: {name: web, port: {number: 80}}
"""


def test_kubernetes_resources_and_references() -> None:
    result = parse("yaml", "k8s/web.yaml", MANIFESTS)
    deployment = "resource:k8s/Deployment/web"
    service = "resource:k8s/Service/web"
    found = resources(result)
    assert found[deployment] == (1, 22)
    assert set(found) == {
        deployment,
        service,
        "resource:k8s/Service/api",
        "resource:k8s/Ingress/web",
    }
    edges = depends(result)
    for target in (
        "resource:docker/image/nginx:1.25",
        "resource:k8s/ConfigMap/web-config",
        "resource:k8s/Secret/web-secret",
        "resource:k8s/PersistentVolumeClaim/web-data",
        "resource:k8s/ServiceAccount/web-sa",
    ):
        assert (deployment, target, True) in edges
    assert (service, deployment, True) in edges
    assert ("resource:k8s/Service/api", "resource:k8s/selector/app=api", False) in edges
    assert ("resource:k8s/Ingress/web", service, True) in edges
    assert ("resource:k8s/Ingress/web", "resource:k8s/Secret/web-tls", True) in edges
    assert all(n.language == "yaml" for n in result.nodes)


def test_workload_labels_and_selectors() -> None:
    documents = load_documents(MANIFESTS)
    assert pod_labels(documents[0]) == {"app": "web", "tier": "front"}
    assert pod_labels(documents[1]) == {}
    [workload] = workloads(MANIFESTS)
    assert workload.qname == "resource:k8s/Deployment/web"
    qname = selector_qname({"tier": "front", "app": "web"})
    assert qname == "resource:k8s/selector/app=web,tier=front"
    assert parse_selector(qname) == {"app": "web", "tier": "front"}


def test_cronjob_and_list_documents() -> None:
    source = """apiVersion: v1
kind: List
items:
  - apiVersion: batch/v1
    kind: CronJob
    metadata: {name: nightly}
    spec:
      jobTemplate:
        spec:
          template:
            spec:
              containers: [{name: j, image: "busybox"}]
"""
    result = parse("yaml", "jobs.yaml", source)
    assert ("resource:k8s/CronJob/nightly", "resource:docker/image/busybox", True) in depends(
        result
    )


def test_non_infrastructure_yaml_is_empty() -> None:
    ci = "name: ci\non: [push]\njobs:\n  test:\n    runs-on: ubuntu-latest\n"
    assert parse("yaml", ".github/workflows/ci.yml", ci) == ParseResult.empty()
    assert parse("yaml", "broken.yaml", "a: [unclosed\n  - : :") == ParseResult.empty()


def test_compose_detected_by_content() -> None:
    result = parse("yaml", "deploy/stack.yml", "services:\n  web:\n    image: nginx\n")
    assert "resource:compose/service/web" in resources(result)
    assert is_compose(load_documents("services:\n  web:\n    image: x\n")[0])
    assert not is_compose(load_documents("services: [a, b]\n")[0])


KUSTOMIZATION = """resources:
  - deployment.yaml
  - ../../base
  - https://github.com/org/repo//manifests
patches:
  - path: patch.yaml
configMapGenerator:
  - name: app-config
    literals: [A=1]
"""


def test_kustomization() -> None:
    result = parse("kustomize", "overlays/prod/kustomization.yaml", KUSTOMIZATION)
    me = "resource:kustomize/overlays/prod"
    assert set(resources(result)) == {me, "resource:k8s/ConfigMap/app-config"}
    edges = depends(result)
    assert (me, "overlays.prod.deployment", True) in edges
    assert (me, "resource:kustomize/base", True) in edges
    assert (me, "overlays.prod.patch", True) in edges
    assert (me, "resource:k8s/ConfigMap/app-config", True) in edges
    assert not any("github" in dst for _, dst, _ in edges)


TERRAFORM = """variable "bucket_name" { type = string }

locals {
  env = "prod"
  tags = { env = local.env }
}

resource "aws_s3_bucket" "logs" {
  bucket = var.bucket_name
  tags   = local.tags
}

data "aws_ami" "ubuntu" { most_recent = true }

module "vpc" {
  source = "./modules/vpc"
  ami    = data.aws_ami.ubuntu.id
  depends_on = [aws_s3_bucket.logs]
}

output "bucket" { value = aws_s3_bucket.logs.arn }
"""


def test_terraform_blocks_and_references() -> None:
    result = parse("terraform", "infra/main.tf", TERRAFORM)
    prefix = "resource:terraform/"
    assert {q.removeprefix(prefix) for q in resources(result)} == {
        "var.bucket_name",
        "local.env",
        "local.tags",
        "aws_s3_bucket.logs",
        "data.aws_ami.ubuntu",
        "module.vpc",
        "output.bucket",
    }
    edges = {(s.removeprefix(prefix), d.removeprefix(prefix)) for s, d, _ in depends(result)}
    assert ("aws_s3_bucket.logs", "var.bucket_name") in edges
    assert ("aws_s3_bucket.logs", "local.tags") in edges
    assert ("local.tags", "local.env") in edges
    assert ("module.vpc", "data.aws_ami.ubuntu") in edges
    assert ("module.vpc", "aws_s3_bucket.logs") in edges
    assert ("module.vpc", "infra.modules.vpc.main") in edges
    assert ("output.bucket", "aws_s3_bucket.logs") in edges
    imports = {e.dst for e in result.edges if e.kind == "imports"}
    assert imports == {"infra.modules.vpc.main"}


def test_terraform_addresses() -> None:
    assert address("resource", ["aws_s3_bucket", "logs"]) == "aws_s3_bucket.logs"
    assert address("data", ["aws_ami", "u"]) == "data.aws_ami.u"
    assert address("terraform", []) is None
    assert reference_address(["var", "x"]) == "var.x"
    assert reference_address(["data", "aws_ami", "u", "id"]) == "data.aws_ami.u"
    assert reference_address(["each", "value"]) is None
    assert reference_address(["count", "index"]) is None
    assert reference_address(["aws_vpc", "main", "id"]) == "aws_vpc.main"


def test_resource_types() -> None:
    assert resource_type("resource:k8s/Deployment/web") == "k8s/Deployment"
    assert resource_type("resource:terraform/aws_s3_bucket.logs") == "terraform/aws_s3_bucket"
    assert resource_type("resource:terraform/var.region") == "terraform/var"
    assert resource_type("resource:docker/stage/builder") == "docker/stage"
    assert resource_type("resource:kustomize/overlays/prod") == "kustomize"


MAKEFILE = """CC := gcc
.PHONY: all test

## Build and test everything.
all: build test

build: main.o ## Compile the binary.
\t$(CC) -o app main.o

test: build
\t./run_tests.sh
\t$(MAKE) lint

lint:
\truff check .
include common.mk
"""


def test_makefile_targets() -> None:
    result = parse("make", "Makefile", MAKEFILE)
    nodes = {n.qname: n for n in result.nodes}
    assert {"Makefile.all", "Makefile.build", "Makefile.test", "Makefile.lint"} <= set(nodes)
    assert nodes["Makefile.all"].docstring == "Build and test everything."
    assert nodes["Makefile.build"].docstring == "Compile the binary."
    calls = {(e.src, e.dst) for e in result.edges if e.kind == "calls"}
    assert calls == {
        ("Makefile.all", "Makefile.build"),
        ("Makefile.all", "Makefile.test"),
        ("Makefile.test", "Makefile.build"),
        ("Makefile.test", "Makefile.lint"),
    }
    assert {e.dst for e in result.edges if e.kind == "imports"} == {"common"}
