"""Import resolution that needs only the importing file: paths, packages, use trees."""

from __future__ import annotations

import pytest

from codegraph.models import ParseResult
from codegraph.parsers import module_qname_for, parser_for
from codegraph.parsers.paths import normalize, package_root, relative_module, relative_path
from codegraph.parsers.treesitter import CommentStyle, leading_comment, path_parts, strip_generics
from codegraph.parsers.ts_imports import go_package_name
from codegraph.parsers.ts_languages import groovy_semicolons, header_grammar


def imports(language: str, path: str, source: str) -> dict[str, str]:
    """{import detail: target} for one file."""
    result: ParseResult = parser_for(language).parse(path, source, module_qname_for(path))
    return {e.detail: e.dst for e in result.edges if e.kind == "imports"}


def test_normalize_and_relative_paths() -> None:
    assert normalize("a/./b/../c") == "a/c"
    assert normalize("../x") is None
    assert relative_path("src/a/b.c", "../util.h") == "src/util.h"
    assert relative_path("main.c", "x/y.h") == "x/y.h"
    assert relative_path("a/b.c", "/abs.h") is None
    assert relative_module("app/x.rb", "lib/util", ".rb") == "app.lib.util"
    assert relative_module("a.c", "../../escape.h") is None


def test_package_root_infers_source_root() -> None:
    assert package_root("src/main/java/com/acme/app/Main.java", "com.acme.app") == "src.main.java"
    assert package_root("com/acme/Main.java", "com.acme") == ""
    assert package_root("flat/Main.kt", "com.acme") is None
    assert package_root("lib/x.kt", "") == "lib"


@pytest.mark.parametrize(
    ("path", "name"),
    [
        ("fmt", "fmt"),
        ("net/http", "http"),
        ("gopkg.in/yaml.v3", "yaml"),
        ("github.com/x/y/v2", "y"),
        ("github.com/mattn/go-sqlite3", "sqlite3"),
        ("github.com/a/b-c", "b_c"),
    ],
)
def test_go_package_name(path: str, name: str) -> None:
    assert go_package_name(path) == name


def test_go_dot_and_blank_imports_bind_nothing() -> None:
    source = (
        'package x\nimport (\n\t_ "github.com/lib/pq"\n\t. "strings"\n)\n'
        'func f() { ToUpper("a") }\n'
    )
    result = parser_for("go").parse("x.go", source, "x")
    calls = {(e.dst, e.resolved) for e in result.edges if e.kind == "calls"}
    assert calls == {("ToUpper", False)}


def test_rust_use_trees() -> None:
    source = """
use crate::net::{self, client::Client as C, *};
use super::super::models::Invoice;
use serde::Serialize;
extern crate log;
"""
    found = imports("rust", "src/billing/invoice/mod.rs", source)
    assert found["crate::net"] == "src.net"
    assert found["crate::net::client::Client"] == "src.net.client.Client"
    assert found["crate::net::*"] == "src.net"
    assert found["super::super::models::Invoice"] == "src.models.Invoice"
    assert found["serde::Serialize"] == "serde.Serialize"
    assert found["extern crate log"] == "log"


def test_rust_crate_root_files() -> None:
    found = imports("rust", "crates/core/src/lib.rs", "mod net;\nuse crate::net::Client;\n")
    assert found["mod net"] == "crates.core.src.net"
    assert found["crate::net::Client"] == "crates.core.src.net.Client"


def test_jvm_imports_split_local_from_external() -> None:
    source = (
        "package com.acme.app;\n"
        "import com.acme.util.Strings;\n"
        "import org.apache.commons.Lang;\n"
        "import java.util.*;\n"
    )
    found = imports("java", "src/main/java/com/acme/app/A.java", source)
    assert found["com.acme.util.Strings"] == "src.main.java.com.acme.util.Strings"
    assert found["org.apache.commons.Lang"] == "org.apache.commons.Lang"
    assert found["java.util.*"] == "java.util"


def test_kotlin_import_alias_and_scala_wildcard() -> None:
    kotlin = "package a.b\nimport a.b.util.Strings as S\n"
    assert imports("kotlin", "a/b/X.kt", kotlin)["a.b.util.Strings"] == "a.b.util.Strings"
    scala = "package a.b\nimport a.b.util._\n"
    assert imports("scala", "a/b/X.scala", scala)["a.b.util._"] == "a.b.util"


def test_c_includes_are_relative_to_the_includer() -> None:
    found = imports("c", "src/net/conn.c", '#include "../util.h"\n#include <stdlib.h>\n')
    assert found == {"../util.h": "src.util_h", "stdlib.h": "stdlib.h"}


def test_ruby_php_dart_solidity_specifiers() -> None:
    ruby = imports("ruby", "app/a.rb", 'require "net/http"\nrequire_relative "../lib/x"\n')
    assert ruby == {"net/http": "net.http", "../lib/x": "lib.x"}
    php = imports("php", "src/A.php", "<?php\nnamespace App;\nuse Vendor\\Pkg\\Thing;\n")
    assert php == {"Vendor.Pkg.Thing": "Vendor.Pkg.Thing"}
    dart = imports("dart", "lib/a/b.dart", "import 'dart:async';\nimport '../c.dart';\n")
    assert dart == {"dart:async": "dart:async", "../c.dart": "lib.c"}
    solidity = imports("solidity", "c/T.sol", 'import "@oz/token/ERC20.sol";\n')
    assert solidity == {"@oz/token/ERC20.sol": "@oz/token/ERC20.sol"}


def test_bash_source_strips_script_dir_prefix() -> None:
    source = 'source "${BASH_SOURCE%/*}/lib.sh"\n. "$HOME/.profile"\n'
    assert imports("bash", "bin/run.sh", source) == {"lib.sh": "bin.lib"}


def test_elixir_alias_as() -> None:
    source = "defmodule App.A do\n  alias App.Util.Strings, as: S\n  def f, do: S.pad(1)\nend\n"
    result = parser_for("elixir").parse("lib/a.ex", source, "lib.a")
    calls = {(e.dst, e.resolved) for e in result.edges if e.kind == "calls"}
    assert ("App.Util.Strings.pad", False) in calls


def test_path_parts_and_generics() -> None:
    assert path_parts("a::b.c") == ("a", "b", "c")
    assert path_parts("$this->repo") == ("this", "repo")
    assert path_parts("util:") == ("util",)
    assert path_parts("a().b") is None
    assert strip_generics("Vec::<u8>::new") == "Vec::new"
    assert strip_generics("Map<String, List<Int>>") == "Map"
    assert strip_generics("Stack[T]") == "Stack"


def test_leading_comment_styles() -> None:
    c_style = CommentStyle(("///", "//"), ("/*", "*/"), ("@",))
    lines = ["/**", " * Adds two.", " */", "@Inline", "int add();"]
    assert leading_comment(lines, 4, c_style) == "Adds two."
    lines = ["// one", "", "// two", "fn x()"]
    assert leading_comment(lines, 3, c_style) == "two"
    assert leading_comment(["x = 1", "fn y()"], 1, c_style) is None


def test_groovy_semicolons_preserve_lines() -> None:
    source = "def a() {\n    foo()\n        .bar()\n    x = 1\n}\n@Field\nint y\n"
    converted = groovy_semicolons(source)
    assert converted.count("\n") == source.count("\n")
    lines = converted.split("\n")
    assert lines[1] == "    foo()"
    assert lines[2] == "        .bar();"
    assert lines[5] == "@Field"


def test_header_grammar_sniffing() -> None:
    assert header_grammar("a.h", "int f(void);\n") == "c"
    assert header_grammar("a.h", "namespace x {}\n") == "cpp"
    assert header_grammar("a.h", "@interface A\n@end\n") == "objc"
    assert header_grammar("a.c", "namespace x {}\n") == "c"
