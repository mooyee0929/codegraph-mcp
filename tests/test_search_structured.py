from __future__ import annotations

from pathlib import Path

import pytest

from codegraph.indexer import index_repository
from codegraph.search.grep import compile_pattern, grep_code
from codegraph.search.snippet import SymbolNotFound, candidates_for, code_snippet
from codegraph.search.structured import GraphQuery, parse_kinds, search_graph
from codegraph.store import GraphStore

# -- parse_kinds -------------------------------------------------------------------


def test_parse_kinds_accepts_kinds_labels_and_lists() -> None:
    assert parse_kinds(None) == ()
    assert parse_kinds("function") == ("function",)
    assert parse_kinds("Callable") == ("function", "method")
    assert parse_kinds("class, Method,function") == ("class", "method", "function")
    assert parse_kinds(["route", "Channel"]) == ("route", "channel")
    assert parse_kinds(" , ") == ()
    with pytest.raises(ValueError, match="unknown kind 'widget'"):
        parse_kinds("widget")


# -- search_graph ------------------------------------------------------------------


def _qnames(store: GraphStore, **fields: object) -> list[str]:
    page = search_graph(store, GraphQuery(**fields))  # type: ignore[arg-type]
    return [hit.node.qname for hit in page.hits]


def test_filters_by_name_qname_kind_file_and_language(store: GraphStore) -> None:
    assert _qnames(store, name_pattern="^ch") == ["pkg.billing.charge"]
    assert _qnames(store, name_pattern="^CH") == ["pkg.billing.charge"]
    assert _qnames(store, name_pattern="^CH", case_sensitive=True) == []
    assert _qnames(store, qname_pattern=r"Invoice\.", sort="name") == [
        "pkg.billing.Invoice.rate",
        "pkg.billing.Invoice.total",
    ]
    assert set(_qnames(store, kinds=("class",))) == {
        "pkg.billing.Invoice",
        "pkg.models.Base",
        "web.app.Handler",
    }
    assert set(_qnames(store, file_glob="web/*", kinds=("function",))) == {
        "web.app.start",
        "web.util.helper",
    }
    assert set(_qnames(store, language="typescript", kinds=("method",))) == {
        "web.app.Handler.run",
        "web.app.Handler.compute",
    }


def test_fan_in_fan_out_and_tests(store: GraphStore) -> None:
    page = search_graph(store, GraphQuery(name_pattern="^charge$"))
    [hit] = page.hits
    assert (hit.fan_in, hit.fan_out, hit.tests) == (1, 3, 1)
    assert hit.as_dict()["span"] == hit.node.span
    assert _qnames(store, kinds=("function", "method"), min_fan_out=3) == ["pkg.billing.charge"]
    assert "pkg.billing.unused_helper" in _qnames(store, kinds=("function",), max_fan_in=0)
    assert "pkg.billing.charge" not in _qnames(store, kinds=("function",), max_fan_in=0)
    tested = set(_qnames(store, has_tests=True))
    assert "pkg.billing.charge" in tested
    assert "pkg.billing.unused_helper" in _qnames(store, has_tests=False, kinds=("function",))
    assert not tested & set(_qnames(store, has_tests=False))


def test_sorting_and_paging(store: GraphStore) -> None:
    by_fan_out = search_graph(store, GraphQuery(kinds=("function", "method"), sort="fan_out"))
    fan_outs = [hit.fan_out for hit in by_fan_out.hits]
    assert fan_outs == sorted(fan_outs, reverse=True)
    by_span = search_graph(store, GraphQuery(kinds=("module",), sort="span"))
    spans = [hit.node.span for hit in by_span.hits]
    assert spans == sorted(spans, reverse=True)
    by_name = [h.node.name.lower() for h in search_graph(store, GraphQuery()).hits]
    assert by_name == sorted(by_name)

    first = search_graph(store, GraphQuery(limit=5))
    second = search_graph(store, GraphQuery(limit=5, offset=5))
    assert first.total == second.total == store.counts()["nodes"]
    assert first.has_more and len(first.hits) == 5
    assert not {h.node.qname for h in first.hits} & {h.node.qname for h in second.hits}
    last = search_graph(store, GraphQuery(limit=50, offset=first.total - 1))
    assert len(last.hits) == 1 and not last.has_more
    payload = first.as_dict()
    assert payload["total"] == first.total and payload["has_more"] is True


def test_search_graph_rejects_bad_input(store: GraphStore) -> None:
    with pytest.raises(ValueError, match="invalid name_pattern"):
        search_graph(store, GraphQuery(name_pattern="("))
    with pytest.raises(ValueError, match="unknown sort"):
        search_graph(store, GraphQuery(sort="size"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="limit"):
        search_graph(store, GraphQuery(limit=0))
    with pytest.raises(ValueError, match="offset"):
        search_graph(store, GraphQuery(offset=-1))


def test_page_size_is_capped(store: GraphStore) -> None:
    assert search_graph(store, GraphQuery(limit=10_000)).limit == 500


# -- grep ----------------------------------------------------------------------------


def test_grep_literal_with_enclosing_symbols(repo: Path, store: GraphStore) -> None:
    result = grep_code(store, repo, "self.rate()")
    [match] = result.matches
    assert (match.path, match.line, match.column) == ("pkg/billing.py", 11, 16)
    assert match.symbol == "pkg.billing.Invoice.total" and match.symbol_kind == "method"
    assert result.files_searched == 8 and result.files_matched == 1
    assert not result.truncated


def test_grep_regex_case_and_glob(repo: Path, store: GraphStore) -> None:
    regex = grep_code(store, repo, r"def \w+\(self", regex=True)
    assert {m.symbol for m in regex.matches} == {
        "pkg.billing.Invoice.total",
        "pkg.billing.Invoice.rate",
        "pkg.models.Base.save",
    }
    assert grep_code(store, repo, "HANDLER").matches == ()
    insensitive = grep_code(store, repo, "HANDLER", case_sensitive=False, file_glob="web/*")
    assert {m.path for m in insensitive.matches} == {"web/app.ts"}
    only_ts = grep_code(store, repo, "return", file_glob="*.ts")
    assert only_ts.matches and all(m.path.endswith(".ts") for m in only_ts.matches)


def test_grep_context_and_caps(repo: Path, store: GraphStore) -> None:
    result = grep_code(store, repo, "return", context=1, max_per_file=1)
    assert result.truncated
    paths = [m.path for m in result.matches]
    assert len(paths) == len(set(paths))
    first = next(m for m in result.matches if m.path == "pkg/billing.py")
    assert first.before == ("    def total(self) -> int:",)
    assert first.after == ("",)
    assert "before" in first.as_dict()
    capped = grep_code(store, repo, "return", max_total=2)
    assert len(capped.matches) == 2 and capped.truncated
    module_level = grep_code(store, repo, "Billing logic")
    assert module_level.matches[0].symbol == "pkg.billing"


def test_grep_reports_unreadable_files(repo: Path, store: GraphStore) -> None:
    (repo / "pkg" / "util.py").unlink()
    result = grep_code(store, repo, "fmt")
    assert result.unreadable == ("pkg/util.py",)
    assert "pkg/util.py" in result.as_dict()["unreadable"]  # type: ignore[operator]


def test_grep_rejects_bad_input(repo: Path, store: GraphStore) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        grep_code(store, repo, "")
    with pytest.raises(ValueError, match="invalid regular expression"):
        grep_code(store, repo, "(", regex=True)
    with pytest.raises(ValueError, match="context"):
        grep_code(store, repo, "x", context=-1)
    assert compile_pattern("a.b", regex=False, case_sensitive=True).search("axb") is None


# -- snippets ------------------------------------------------------------------------


def test_code_snippet_by_name_path_and_suffix(repo: Path, store: GraphStore) -> None:
    by_name = code_snippet(store, repo, "charge")
    assert by_name.node.qname == "pkg.billing.charge"
    assert "def charge(inv: Invoice) -> str:" in by_name.snippet.source
    assert by_name.snippet.callers == ("tests.test_billing.test_charge",)
    by_line = code_snippet(store, repo, "pkg/billing.py:11")
    assert by_line.node.qname == "pkg.billing.Invoice.total"
    by_suffix = code_snippet(store, repo, "Invoice.rate", context=1)
    assert by_suffix.snippet.line_start == by_suffix.node.line_start - 1
    payload = by_suffix.as_dict()
    assert payload["symbol_lines"] == [13, 14]
    assert payload["shown_lines"] == [12, 15]


def test_code_snippet_errors(repo: Path, store: GraphStore) -> None:
    with pytest.raises(SymbolNotFound, match="matches no symbol") as missing:
        code_snippet(store, repo, "zzz_nothing")
    assert missing.value.candidates == ()
    (repo / "pkg" / "other.py").write_text("def charge() -> None:\n    return None\n")
    index_repository(repo, store)
    with pytest.raises(SymbolNotFound, match="is ambiguous") as ambiguous:
        code_snippet(store, repo, "charge")
    assert ambiguous.value.candidates == ("pkg.billing.charge", "pkg.other.charge")
    assert candidates_for(store, "charge") == ambiguous.value.candidates
    (repo / "pkg" / "util.py").unlink()
    with pytest.raises(SymbolNotFound, match="cannot read"):
        code_snippet(store, repo, "pkg.util.fmt")
    with pytest.raises(ValueError, match="context"):
        code_snippet(store, repo, "charge", context=-1)
