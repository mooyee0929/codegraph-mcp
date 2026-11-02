from __future__ import annotations

import csv
import io
import json
import threading
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

from codegraph import adr, notes
from codegraph.analysis.common import ensure_tables
from codegraph.analysis.communities import DDL as COMMUNITY_DDL
from codegraph.exports import (
    ExportError,
    community_membership,
    load_graph,
    normalise_format,
    render,
)
from codegraph.indexer import index_repository
from codegraph.models import Node
from codegraph.store import GraphStore
from codegraph.visualize import (
    UiApp,
    VisualizeError,
    graph_payload,
    host_allowed,
    make_ui_server,
    render_html,
    write_visualization,
)
from codegraph.wiki import generate_wiki, list_wiki_pages, read_wiki_page

TODAY = date(2026, 11, 2)
ANALYSIS_PASSES = frozenset({"communities", "flows"})


@pytest.fixture
def bare_store(repo: Path) -> Iterator[GraphStore]:
    """The sample repo indexed without the analysis passes, so no community tables."""
    with GraphStore.open(repo) as graph_store:
        index_repository(repo, graph_store, skip_passes=ANALYSIS_PASSES)
        yield graph_store


def _add_communities(store: GraphStore) -> None:
    """Replace whatever the communities pass stored with two known communities."""
    with store.transaction() as conn:
        ensure_tables(store, COMMUNITY_DDL)
        conn.execute("DELETE FROM analysis_community_members")
        conn.execute("DELETE FROM analysis_communities")
        conn.executemany(
            "INSERT INTO analysis_communities "
            "(id, name, size, cohesion, internal_weight, external_weight, files, prefix) "
            "VALUES (?, ?, ?, 0.5, 1.0, 1.0, 1, ?)",
            [(1, "billing", 3, "pkg"), (2, "web", 2, "web")],
        )
        conn.executemany(
            "INSERT INTO analysis_community_members (qname, community_id, rank) VALUES (?, ?, ?)",
            [
                ("pkg.billing", 1, 0),
                ("pkg.billing.charge", 1, 1),
                ("pkg.util.fmt", 1, 2),
                ("web.app", 2, 0),
                ("web.app.Handler", 2, 1),
            ],
        )


# -- ADRs -------------------------------------------------------------------------------


def test_adr_lifecycle(tmp_path: Path) -> None:
    folder = adr.resolve_adr_dir(tmp_path)
    assert folder == (tmp_path / "docs" / "adr").resolve()
    first = adr.create_adr(
        folder,
        "Use SQLite for the graph",
        context="We need one file.",
        decision="Store the graph in SQLite.",
        consequences="Easy to ship.",
        today=TODAY,
    )
    assert first.path.name == "0001-use-sqlite-for-the-graph.md"
    text = first.path.read_text()
    assert text.startswith("# 1. Use SQLite for the graph\n\nDate: 2026-11-02\n\n## Status")
    assert "## Context\n\nWe need one file." in text
    assert adr.parse_adr(text, first.path) == first

    accepted = adr.update_adr(folder, 1, status="accepted")
    assert accepted.status == "Accepted" and accepted.decision == first.decision

    new, old = adr.supersede_adr(folder, 1, "Use SQLite with WAL", decision="Turn on WAL.")
    assert new.number == 2 and new.status == "Accepted"
    assert new.links == (
        "Supersedes [1. Use SQLite for the graph](0001-use-sqlite-for-the-graph.md)",
    )
    reread = adr.get_adr(folder, 1)
    assert reread is not None and reread.status == "Superseded"
    assert reread.links == (f"Superseded by [2. Use SQLite with WAL]({new.path.name})",)
    with pytest.raises(adr.AdrError, match="already superseded"):
        adr.supersede_adr(folder, 1, "Again")
    assert [a.number for a in adr.list_adrs(folder)] == [1, 2]


def test_adr_validation_and_dir_file(tmp_path: Path) -> None:
    (tmp_path / ".adr-dir").write_text("doc/decisions\n")
    assert adr.resolve_adr_dir(tmp_path) == (tmp_path / "doc" / "decisions").resolve()
    with pytest.raises(adr.AdrError, match="outside"):
        adr.resolve_adr_dir(tmp_path, "../elsewhere")
    folder = adr.resolve_adr_dir(tmp_path)
    with pytest.raises(adr.AdrError, match="unknown status"):
        adr.create_adr(folder, "X", status="maybe")
    with pytest.raises(adr.AdrError, match="needs a title"):
        adr.create_adr(folder, "   ")
    with pytest.raises(adr.AdrError, match="no ADR number 7"):
        adr.update_adr(folder, 7, status="Accepted")
    assert adr.slugify("  Hello, World!  ") == "hello-world"


def test_manage_adr_dispatch(tmp_path: Path) -> None:
    created = adr.manage_adr(tmp_path, "create", title="Pick a queue", status="Accepted")
    assert isinstance(created["created"], dict)
    assert created["created"]["path"] == "docs/adr/0001-pick-a-queue.md"
    listed = adr.manage_adr(tmp_path, "list")
    assert listed["directory"] == "docs/adr"
    assert adr.manage_adr(tmp_path, "get", 1)["title"] == "Pick a queue"
    with pytest.raises(adr.AdrError, match="needs the ADR number"):
        adr.manage_adr(tmp_path, "get")
    with pytest.raises(adr.AdrError, match="unknown action"):
        adr.manage_adr(tmp_path, "delete", 1)


# -- notes ------------------------------------------------------------------------------


def test_notes_save_search_and_parse(tmp_path: Path) -> None:
    first = notes.save_note(
        tmp_path,
        "Where is  charge tested?",
        "In tests/test_billing.py::test_charge.",
        ("pkg.billing.charge", "pkg.billing.charge"),
        now=0.0,
    )
    assert first.path.name == "19700101-000000-where-is-charge-tested.md"
    assert first.qnames == ("pkg.billing.charge",)
    second = notes.save_note(tmp_path, "Where is charge tested", "Same answer.", now=0.0)
    assert second.path.name.endswith("-2.md")
    notes.save_note(tmp_path, "How does the web handler run", "Handler.run calls helper.", now=5)

    parsed = notes.parse_note(first.path, first.path.read_text())
    assert parsed == first

    hits = notes.search_notes(tmp_path, "pkg.billing.charge tests")
    assert hits[0].note.path == first.path
    assert all(h.note.question != "How does the web handler run" for h in hits)
    assert notes.search_notes(tmp_path, "zzz") == []
    assert len(notes.list_notes(tmp_path)) == 3

    loose = notes.notes_dir(tmp_path) / "hand-written.md"
    loose.write_text("# Why WAL\n\nConcurrent readers.\n")
    assert notes.search_notes(tmp_path, "wal")[0].note.question == "Why WAL"
    with pytest.raises(notes.NoteError):
        notes.save_note(tmp_path, "", "x")


# -- wiki -------------------------------------------------------------------------------


def test_wiki_pages_without_communities(repo: Path, bare_store: GraphStore) -> None:
    store = bare_store
    result = generate_wiki(repo, store)
    assert "index.md" in result.pages and "modules/pkg.billing.md" in result.pages
    index = read_wiki_page(repo, "index")
    assert index is not None and "[`pkg.billing`](modules/pkg.billing.md)" in index
    billing = read_wiki_page(repo, "pkg.billing")
    assert billing is not None
    assert "### pkg.billing.charge" in billing and "def charge(inv: Invoice) -> str" in billing
    assert "Tested by: [`tests.test_billing.test_charge`]" in billing
    assert "## Imported by" in billing
    assert not any(p.startswith("communities/") for p in result.pages)
    with pytest.raises(ValueError, match="outside"):
        read_wiki_page(repo, "../../etc/passwd")
    assert read_wiki_page(repo, "no-such-page") is None


def test_wiki_communities_and_stale_pages(repo: Path, store: GraphStore) -> None:
    stale = repo / ".codegraph" / "wiki" / "modules" / "gone.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("old")
    _add_communities(store)
    with store.transaction():
        store.add_nodes(
            [Node("route:GET /pay", "route", "pkg/billing.py", 17, 17, "python", "GET /pay")]
        )
    result = generate_wiki(repo, store)
    assert "modules/gone.md" in result.removed and not stale.exists()
    assert "communities/billing.md" in result.pages
    community = read_wiki_page(repo, "billing")
    assert community is not None and "pkg.billing.charge" in community
    index = read_wiki_page(repo, "index.md")
    assert index is not None and "## Route nodes" in index and "route:GET /pay" in index
    assert "communities/billing.md" in list_wiki_pages(repo)


# -- exports ----------------------------------------------------------------------------


def test_community_membership(bare_store: GraphStore) -> None:
    assert community_membership(bare_store) == {}
    _add_communities(bare_store)
    membership = community_membership(bare_store)
    assert membership["pkg.billing.charge"] == "billing"
    assert membership["web.app"] == "web"


def test_community_membership_from_the_pass(store: GraphStore) -> None:
    membership = community_membership(store)
    assert membership
    assert set(membership) <= {n.qname for n in store.all_nodes()}


def test_load_graph_caps_by_degree(store: GraphStore) -> None:
    full = load_graph(store)
    assert not full.truncated and full.total_nodes == len(full.nodes)
    capped = load_graph(store, max_nodes=3)
    assert len(capped.nodes) == 3 and capped.truncated
    kept = {n.qname for n in capped.nodes}
    assert all(e.src in kept for e in capped.edges)


def test_every_format_renders(store: GraphStore) -> None:
    _add_communities(store)
    data = load_graph(store)
    payload = json.loads(render(data, "json")["graph.json"])
    assert {n["qname"] for n in payload["nodes"]} >= {"pkg.billing.charge"}
    assert any(e["kind"] == "calls" and e["dst_known"] for e in payload["edges"])

    graphml = ET.fromstring(render(data, "graphml")["graph.graphml"])
    namespace = "{http://graphml.graphdrawing.org/xmlns}"
    assert len(graphml.findall(f".//{namespace}node")) == len(data.nodes)

    dot = render(data, "dot")["graph.dot"]
    assert dot.startswith("digraph codegraph {") and '"pkg.billing.charge" ->' in dot

    mermaid = render(data, "mermaid")["graph.mmd"]
    assert mermaid.startswith("flowchart LR") and "|calls|" in mermaid

    tables = render(data, "csv")
    nodes_rows = list(csv.reader(io.StringIO(tables["nodes.csv"])))
    assert nodes_rows[0][:2] == ["qname", "kind"] and len(nodes_rows) == len(data.nodes) + 1
    edge_rows = list(csv.reader(io.StringIO(tables["edges.csv"])))
    assert edge_rows[0][:3] == ["src", "dst", "kind"]

    cypher = render(data, "neo4j")["graph.cypher"]
    assert cypher.startswith("CREATE CONSTRAINT codenode_qname IF NOT EXISTS")
    assert "SET n:Function" in cypher and "MERGE (a)-[r:CALLS" in cypher
    assert "n.community = 'billing'" in cypher

    assert normalise_format("GV") == "dot"
    with pytest.raises(ExportError):
        normalise_format("pdf")


def test_exports_escape_hostile_text(store: GraphStore) -> None:
    hostile = Node(
        qname="pkg.evil",
        kind="function",
        path="pkg/evil.py",
        line_start=1,
        line_end=1,
        language="python",
        name="evil\"'<x>",
        signature="def evil() -> '</script>'",
    )
    with store.transaction():
        store.add_nodes([hostile])
    data = load_graph(store)
    ET.fromstring(render(data, "graphml")["graph.graphml"])
    assert "name = 'evil\"\\'<x>'" in render(data, "cypher")["graph.cypher"]
    assert 'label="evil\\"\'<x>"' in render(data, "dot")["graph.dot"]
    page = render_html(graph_payload(data, title="<t>"))
    assert "</script>'" not in page and "&lt;t&gt;" in page


# -- visualisation and UI ------------------------------------------------------------------


def test_visualization_is_self_contained(repo: Path, bare_store: GraphStore) -> None:
    path, data = write_visualization(repo, bare_store, max_nodes=5)
    assert path == repo / ".codegraph" / "graph.html"
    page = path.read_text()
    assert "<canvas" in page and "src=" not in page and "https://" not in page
    embedded = page.split('<script id="graph-data" type="application/json">', 1)[1]
    payload = json.loads(embedded.split("</script>", 1)[0])
    assert len(payload["nodes"]) == 5 and payload["total"] == data.total_nodes
    assert payload["colorBy"] == "package"
    with pytest.raises(VisualizeError):
        graph_payload(data, title="x", color_by="rainbow")


def test_payload_prefers_communities(store: GraphStore) -> None:
    _add_communities(store)
    payload = graph_payload(load_graph(store), title="t")
    assert payload["colorBy"] == "community" and payload["hasCommunities"] is True


def test_host_allowed() -> None:
    assert host_allowed("127.0.0.1:9749") and host_allowed("localhost") and host_allowed("[::1]:1")
    assert not host_allowed("evil.example:9749") and not host_allowed(None)


def test_ui_app_routes(repo: Path, store: GraphStore) -> None:
    app = UiApp(root=repo)
    page = app.handle("/")
    assert page.status == 200 and page.content_type.startswith("text/html")
    stats = json.loads(app.handle("/api/stats").body)
    assert stats["counts"]["files"] == 8
    node = json.loads(app.handle("/api/node?qname=pkg.billing.charge").body)
    assert node["qname"] == "pkg.billing.charge" and node["incoming"]
    assert app.handle("/api/node?qname=nope").status == 404
    hits = json.loads(app.handle("/api/search?q=charge").body)
    assert hits[0]["qname"] == "pkg.billing.charge"
    assert app.handle("/api/search").status == 400
    assert app.handle("/api/graph?max_nodes=abc").status == 400
    assert len(json.loads(app.handle("/api/graph?max_nodes=2").body)["nodes"]) == 2
    assert app.handle("/nowhere").status == 404
    assert UiApp(root=repo / "pkg").handle("/").status == 503


def test_ui_server_serves_loopback_only(repo: Path, store: GraphStore) -> None:
    server = make_ui_server(repo, port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with direct.open(f"http://127.0.0.1:{port}/api/stats", timeout=10) as reply:
            assert json.loads(reply.read())["counts"]["files"] == 8
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/", headers={"Host": "attacker.example"}
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            direct.open(request, timeout=10)
        assert caught.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(10)
