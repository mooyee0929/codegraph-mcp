from __future__ import annotations

from pathlib import Path

from codegraph.models import Node, NodeKind
from codegraph.services.resolve import NodeIndex
from codegraph.services.sources import is_test_file, iter_source_files, language_of
from codegraph.services.text import (
    CommentMask,
    keyword_arg,
    matching_close,
    module_constants,
    pick_arg,
    split_args,
    split_bare_args,
    string_literal,
    url_expression,
)
from codegraph.services.traces import parse_spans


def test_split_args_respects_nesting_strings_and_comments() -> None:
    text = 'f("a,b", g(1, 2), [3, 4], /* , */ x => { y, z })'
    args = split_args(text, 1, "js")
    assert args is not None
    assert [a.text for a in args] == ['"a,b"', "g(1, 2)", "[3, 4]", "/* , */ x => { y, z }"]
    assert matching_close(text, 1, "js") == len(text) - 1
    assert split_args("f(unclosed", 1, "js") is None


def test_keyword_and_positional_args() -> None:
    args = split_args('route("/x", methods=["GET"], name: "n")', 5, "python")
    assert args is not None
    first = pick_arg(args, 0)
    assert first is not None and first.text == '"/x"'
    method_arg = keyword_arg(args, "methods")
    assert method_arg is not None and method_arg.text == '["GET"]'
    assert pick_arg(args, 1) is None
    bare = split_bare_args("get '/a', to: 'x#y' do\n  more\n", 4, "ruby")
    assert [a.text for a in bare] == ["'/a'", "to: 'x#y'"]


def test_string_literals_per_family() -> None:
    assert string_literal("r'^a/$'", "python") is not None
    assert string_literal('@"C:\\path"', "csharp") is not None
    assert string_literal("`raw`", "go") is not None
    assert string_literal('r#"x"#', "rust") is not None
    assert string_literal('"a" + "b"', "jvm") is None
    interpolated = string_literal('"/u/#{id}"', "ruby")
    assert interpolated is not None and interpolated.interpolated


def test_url_expression_with_constants_and_bases() -> None:
    constants = {"BASE": "http://svc", "USERS": "/users"}
    url = url_expression("BASE + USERS + '/' + uid", "python", constants)
    assert url is not None and url.text == "http://svc/users/{param}" and not url.exact
    route = url_expression("prefix + '/x'", "js", {}, client=False)
    assert route is not None and route.text == "{param}/x"
    assert url_expression("someVariable", "js", {}) is None
    assert url_expression('"$base/users/$id"', "php", {"base": "/api"}) is not None


def test_module_constants_only_at_module_level_for_python() -> None:
    text = "TOPIC = 'orders'\ndef f():\n    LOCAL = 'no'\n"
    assert module_constants(text, "python") == {"TOPIC": "orders"}
    go = 'const (\n\tTopicA = "a"\n\tTopicB string = "b"\n)\n'
    assert module_constants(go, "go") == {"TopicA": "a", "TopicB": "b"}
    java = 'private static final String QUEUE = "jobs";'
    assert module_constants(java, "jvm") == {"QUEUE": "jobs"}


def test_comment_mask() -> None:
    text = 'x = "# not"  # yes\n"""doc"""\n'
    mask = CommentMask(text, "python")
    assert text.index("# not") not in mask
    assert text.index("# yes") in mask
    assert text.index("doc") in mask


def test_sources_walk_and_labels(tmp_path: Path) -> None:
    (tmp_path / "svc").mkdir()
    (tmp_path / "svc" / "main.go").write_text("package main\n")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.go").write_text("package lib\n")
    (tmp_path / "api.proto").write_text("syntax = 'proto3';\n")
    (tmp_path / "README.md").write_text("hi\n")
    found = {s.path: (s.language, s.family) for s in iter_source_files(tmp_path)}
    assert found == {"api.proto": ("protobuf", "proto"), "svc/main.go": ("go", "go")}
    assert language_of("x/Y.KT") == ("kotlin", "jvm")
    assert is_test_file("pkg/x_test.go") and is_test_file("src/test/java/A.java")
    assert not is_test_file("src/main/java/A.java")


def _node(qname: str, kind: NodeKind, start: int, end: int, path: str = "m.py") -> Node:
    return Node(
        qname=qname,
        kind=kind,
        path=path,
        line_start=start,
        line_end=end,
        language="python",
        name=qname.rsplit(".", 1)[-1],
    )


def test_node_index_owner_decorated_and_lookup() -> None:
    index = NodeIndex.from_nodes(
        [
            _node("m", "module", 1, 40),
            _node("m.App", "class", 1, 30),
            _node("m.App.get", "method", 3, 6),
            _node("m.helper", "function", 10, 12),
            _node("views.detail", "function", 1, 3, path="views.py"),
            _node("other.detail", "function", 1, 3, path="other.py"),
        ]
    )
    assert index.owner("m.py", 4) == "m.App.get"
    assert index.owner("m.py", 35) == "m"
    assert index.owner("nope/x.go", 1) == "nope.x"
    decorated = index.decorated("m.py", 2)
    assert decorated is not None and decorated.qname == "m.App.get"
    assert index.lookup("helper") is not None
    assert index.lookup("detail") is None
    found = index.lookup("views.detail")
    assert found is not None and found.qname == "views.detail"


def test_parse_spans_validation() -> None:
    spans, errors = parse_spans(
        [
            {"service": "a", "method": "post", "url": "https://h/x/1?y=2", "status": 201},
            {"path": "/y"},
            {"url": "/z", "status": "200", "caller": 3},
        ]
    )
    assert [(s.method, s.path, s.status) for s in spans] == [
        ("POST", "/x/1", 201),
        ("ANY", "/y", None),
    ]
    assert errors == ["span 2: 'status' must be an integer; 'caller' must be a string"]
