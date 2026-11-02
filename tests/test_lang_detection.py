"""Language detection, qname rules for new file kinds, test-path detection, dispatch."""

from __future__ import annotations

import pytest

from codegraph.linker import COMMON_METHOD_NAMES, is_test_path
from codegraph.parsers import language_of, module_qname_for, parser_for, supported_languages
from codegraph.parsers.languages import LANGUAGE_BY_FILENAME


@pytest.mark.parametrize(
    ("path", "language"),
    [
        ("cmd/main.go", "go"),
        ("src/lib.rs", "rust"),
        ("A.java", "java"),
        ("build.gradle.kts", "kotlin"),
        ("App.scala", "scala"),
        ("x.c", "c"),
        ("x.h", "c"),
        ("x.hpp", "cpp"),
        ("x.cc", "cpp"),
        ("A.cs", "csharp"),
        ("a.rb", "ruby"),
        ("Rakefile", "ruby"),
        ("a.php", "php"),
        ("A.swift", "swift"),
        ("a.dart", "dart"),
        ("a.lua", "lua"),
        ("deploy.sh", "bash"),
        ("a.ex", "elixir"),
        ("a_test.exs", "elixir"),
        ("Token.sol", "solidity"),
        ("A.m", "objc"),
        ("a.zig", "zig"),
        ("a.jl", "julia"),
        ("utils.R", "r"),
        ("Lib.pm", "perl"),
        ("Main.hs", "haskell"),
        ("build.gradle", "groovy"),
        ("Jenkinsfile", "groovy"),
        ("a.erl", "erlang"),
        ("App.vue", "vue"),
        ("App.svelte", "svelte"),
        ("analysis.ipynb", "notebook"),
        ("Dockerfile", "dockerfile"),
        ("Containerfile", "dockerfile"),
        ("Dockerfile.prod", "dockerfile"),
        ("api.dockerfile", "dockerfile"),
        ("docker-compose.yml", "compose"),
        ("compose.yaml", "compose"),
        ("docker-compose.override.yml", "compose"),
        ("kustomization.yaml", "kustomize"),
        ("Makefile", "make"),
        ("main.tf", "terraform"),
        ("terragrunt.hcl", "hcl"),
        ("k8s/deploy.yaml", "yaml"),
        (".github/workflows/ci.yml", "yaml"),
    ],
)
def test_language_of(path: str, language: str) -> None:
    assert language_of(path) == language


def test_unknown_files_are_not_indexed() -> None:
    assert language_of("LICENSE") is None
    assert language_of("notes.txt") is None
    assert language_of("dockerfile-notes.md") is None


def test_filename_table_covers_infrastructure_names() -> None:
    for name in ("Dockerfile", "Containerfile", "docker-compose.yml", "compose.yaml"):
        assert name in LANGUAGE_BY_FILENAME
    assert LANGUAGE_BY_FILENAME["kustomization.yaml"] == "kustomize"
    assert LANGUAGE_BY_FILENAME["Makefile"] == "make"


def test_every_supported_language_has_a_parser() -> None:
    infos = supported_languages()
    names = [info.language for info in infos]
    assert len(names) == len(set(names))
    for info in infos:
        assert parser_for(info.language).language == info.language
        assert info.suffixes or info.filenames
    families = {info.language: info.family for info in infos}
    assert families["dockerfile"] == "infrastructure"
    assert families["go"] == "code"


def test_parser_for_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="cobol"):
        parser_for("cobol")


def test_header_and_dockerfile_qnames_do_not_collide() -> None:
    assert module_qname_for("src/util.h") == "src.util_h"
    assert module_qname_for("src/util.c") == "src.util"
    assert module_qname_for("src/geo.hpp") == "src.geo_hpp"
    assert module_qname_for("api/Dockerfile") == "api.Dockerfile"
    assert module_qname_for("api/Dockerfile.prod") == "api.Dockerfile_prod"
    assert module_qname_for("src/net/mod.rs") == "src.net"
    assert module_qname_for("pkg/__init__.py") == "pkg"


@pytest.mark.parametrize(
    "path",
    [
        "pkg/server_test.go",
        "src/test/java/com/acme/ServiceTest.java",
        "app/src/main/java/ServiceTests.java",
        "ServiceIT.java",
        "Tests/AppTests.cs",
        "core/ParserSpec.scala",
        "spec/models/user_spec.rb",
        "lib/user_test.rb",
        "src/test_parser.c",
        "src/parser_unittest.cc",
        "test/billing_SUITE.erl",
        "contracts/Token.t.sol",
        "Sources/AppTests.swift",
        "lib/widget_test.dart",
        "src/ParserSpec.hs",
    ],
)
def test_is_test_path_new_languages(path: str) -> None:
    assert is_test_path(path)


@pytest.mark.parametrize(
    "path",
    ["src/contest.go", "src/main/java/Latest.java", "lib/testing_utils.rb", "src/attest.c"],
)
def test_is_test_path_rejects_lookalikes(path: str) -> None:
    assert not is_test_path(path)


def test_common_method_names_cover_new_standard_libraries() -> None:
    assert {"unwrap", "Println", "toString", "equals", "new", "len", "get"} <= COMMON_METHOD_NAMES
