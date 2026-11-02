"""An interactive graph view: one self-contained HTML file, or a tiny local server.

:func:`render_html` produces a single page with the graph embedded as JSON and a
vanilla-JS force-directed layout on a ``<canvas>`` — no CDN, no network, so it opens
from disk and can be attached to a ticket. Large graphs keep only the
highest-degree nodes (``max_nodes``); the page says how many were left out. The page
filters by node and edge kind, colours by package, kind or community (when an
analysis pass stored communities), searches by name, and shows a node's signature,
docstring and neighbours on click. Every string from the graph reaches the DOM through
``textContent``, never ``innerHTML``, because docstrings are untrusted input.

``codegraph ui`` serves the same page, re-rendered on every load, plus a JSON API
(``/api/stats``, ``/api/graph``, ``/api/node``, ``/api/search``) from the stdlib
``http.server``. It binds to 127.0.0.1 and rejects requests whose ``Host`` header is
not a loopback name, so a web page cannot reach it through DNS rebinding.
"""

from __future__ import annotations

import html
import json
import socket
import socketserver
import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from codegraph.exports import GraphData, degrees, load_graph
from codegraph.models import Node
from codegraph.store import DB_DIRNAME, GraphStore, StoreError, default_db_path
from codegraph.tools import node_summary

DEFAULT_MAX_NODES = 500
DEFAULT_PORT = 9749
VISUALIZATION_FILENAME = "graph.html"
COLOR_MODES: tuple[str, ...] = ("package", "kind", "community")
_CODE_KINDS = frozenset({"module", "class", "function", "method"})
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class VisualizeError(ValueError):
    """Bad option, such as an unknown colour mode."""


def _group(node: Node) -> str:
    return node.qname.split(".")[0] if node.kind in _CODE_KINDS else node.kind


def _first_line(text: str | None) -> str:
    return text.strip().splitlines()[0].strip() if text and text.strip() else ""


def graph_payload(
    data: GraphData, *, title: str, color_by: str = "auto", api: bool = False
) -> dict[str, object]:
    """The JSON the page embeds; edges are ``[src index, dst index, kind index, unresolved]``."""
    if color_by not in (*COLOR_MODES, "auto"):
        raise VisualizeError(f"unknown colour mode {color_by!r}; expected {COLOR_MODES}")
    mode = ("community" if data.communities else "package") if color_by == "auto" else color_by
    index = {node.qname: i for i, node in enumerate(data.nodes)}
    linked = [e for e in data.edges if e.dst in index and e.src != e.dst]
    edge_kinds = sorted({e.kind for e in linked})
    kind_index = {kind: i for i, kind in enumerate(edge_kinds)}
    degree = degrees(index, linked)
    return {
        "title": title,
        "nodes": [
            {
                "q": n.qname,
                "n": n.name,
                "k": n.kind,
                "p": n.path,
                "l": n.line_start,
                "s": n.signature,
                "d": _first_line(n.docstring),
                "c": data.communities.get(n.qname, ""),
                "g": _group(n),
                "deg": degree[n.qname],
            }
            for n in data.nodes
        ],
        "edges": [
            [index[e.src], index[e.dst], kind_index[e.kind], 0 if e.resolved else 1] for e in linked
        ],
        "nodeKinds": sorted({n.kind for n in data.nodes}),
        "edgeKinds": edge_kinds,
        "hasCommunities": bool(data.communities),
        "colorBy": mode,
        "total": data.total_nodes,
        "api": api,
    }


def _script_json(value: object) -> str:
    """JSON safe to place inside ``<script>``: no ``<``, ``>`` or ``&`` survive raw."""
    text = json.dumps(value, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render_html(payload: dict[str, object]) -> str:
    title = html.escape(str(payload.get("title", "codegraph")))
    return _PAGE.replace("__TITLE__", title).replace("__DATA__", _script_json(payload))


def write_visualization(
    root: Path,
    store: GraphStore,
    *,
    out: Path | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    color_by: str = "auto",
) -> tuple[Path, GraphData]:
    data = load_graph(store, max_nodes=max_nodes if max_nodes > 0 else None)
    target = out if out is not None else root / DB_DIRNAME / VISUALIZATION_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = graph_payload(data, title=f"{root.name} code graph", color_by=color_by)
    target.write_text(render_html(payload), encoding="utf-8")
    return target, data


# -- local UI server --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Response:
    status: int
    content_type: str
    body: bytes


def _json(status: int, value: object) -> Response:
    return Response(status, "application/json", json.dumps(value, indent=2).encode())


def _int_param(params: dict[str, list[str]], name: str, default: int) -> int:
    raw = params.get(name, [str(default)])[0]
    try:
        return int(raw)
    except ValueError as exc:
        raise VisualizeError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True, slots=True)
class UiApp:
    """Request routing for ``codegraph ui``, kept apart from the socket plumbing."""

    root: Path
    max_nodes: int = DEFAULT_MAX_NODES

    def handle(self, target: str) -> Response:
        parsed = urlsplit(target)
        params = parse_qs(parsed.query)
        if not default_db_path(self.root).is_file():
            return _json(503, {"error": f"{self.root} is not indexed; run `codegraph index`"})
        try:
            return self._route(parsed.path, params)
        except VisualizeError as exc:
            return _json(400, {"error": str(exc)})

    def _route(self, path: str, params: dict[str, list[str]]) -> Response:
        with GraphStore(default_db_path(self.root)) as store:
            if path in {"/", "/index.html"}:
                payload = self._payload(store, params, api=True)
                return Response(200, "text/html; charset=utf-8", render_html(payload).encode())
            if path == "/api/graph":
                return _json(200, self._payload(store, params, api=True))
            if path == "/api/stats":
                return _json(200, {"counts": store.counts(), **store.counts_by_kind()})
            if path == "/api/node":
                qname = params.get("qname", [""])[0]
                node = store.get_node(qname)
                if node is None:
                    return _json(404, {"error": f"no node {qname!r}"})
                return _json(
                    200,
                    {
                        **node_summary(node),
                        "outgoing": [asdict(e) for e in store.edges_from(qname)],
                        "incoming": [asdict(e) for e in store.edges_to(qname)],
                    },
                )
            if path == "/api/search":
                query = params.get("q", [""])[0]
                if not query.strip():
                    return _json(400, {"error": "q is required"})
                hits = store.search_nodes(query, limit=_int_param(params, "limit", 20))
                return _json(200, [node_summary(n) for n in hits])
        return _json(404, {"error": f"no route {path}"})

    def _payload(
        self, store: GraphStore, params: dict[str, list[str]], *, api: bool
    ) -> dict[str, object]:
        cap = _int_param(params, "max_nodes", self.max_nodes)
        data = load_graph(store, max_nodes=cap if cap > 0 else None)
        color = params.get("color", ["auto"])[0]
        return graph_payload(data, title=f"{self.root.name} code graph", color_by=color, api=api)


def host_allowed(host_header: str | None) -> bool:
    if not host_header:
        return False
    host = host_header.strip()
    if host.startswith("["):
        name = host[1:].split("]", 1)[0]
    else:
        name = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    return name.lower() in _LOOPBACK_HOSTS


class _Handler(BaseHTTPRequestHandler):
    server_version = "codegraph-ui"

    def __init__(
        self,
        request: socket.socket,
        client_address: tuple[str, int],
        server: socketserver.BaseServer,
        *,
        app: UiApp,
    ) -> None:
        self.app = app
        super().__init__(request, client_address, server)

    def do_GET(self) -> None:
        if not host_allowed(self.headers.get("Host")):
            self._send(Response(403, "text/plain; charset=utf-8", b"forbidden host\n"))
            return
        try:
            response = self.app.handle(self.path)
        except (StoreError, sqlite3.Error) as exc:
            response = _json(500, {"error": str(exc)})
        self._send(response)

    def _send(self, response: Response) -> None:
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(response.body)


def make_ui_server(
    root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> ThreadingHTTPServer:
    app = UiApp(root=root.resolve(), max_nodes=max_nodes)

    def handler(
        request: socket.socket, client_address: tuple[str, int], server: socketserver.BaseServer
    ) -> _Handler:
        return _Handler(request, client_address, server, app=app)

    return ThreadingHTTPServer((host, port), handler)


def serve_ui(
    root: Path,
    *,
    port: int = DEFAULT_PORT,
    max_nodes: int = DEFAULT_MAX_NODES,
    announce: Callable[[str], None] = print,
) -> None:
    """Serve until interrupted; the caller handles ``KeyboardInterrupt``."""
    server = make_ui_server(root, port=port, max_nodes=max_nodes)
    try:
        bound = server.server_address[1]
        announce(f"codegraph ui on http://127.0.0.1:{bound}/  (Ctrl-C to stop)")
        server.serve_forever()
    finally:
        server.server_close()


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="icon" href="data:,">
<style>
:root { --bg:#0f1117; --panel:#171a23; --line:#2a2f3c; --text:#e6e6e6; --muted:#9aa0aa;
        --accent:#5aa9ff; }
* { box-sizing: border-box; }
body { margin:0; display:flex; height:100vh; overflow:hidden; background:var(--bg);
       color:var(--text); font:13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI",
       sans-serif; }
#side { width:310px; flex:none; overflow-y:auto; padding:12px 14px; background:var(--panel);
        border-right:1px solid var(--line); }
#stage { flex:1; position:relative; }
canvas { display:block; width:100%; height:100%; cursor:grab; }
h1 { font-size:15px; margin:0 0 4px; }
h2 { font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted);
     margin:16px 0 6px; }
input[type=search], select { width:100%; padding:6px 8px; color:var(--text);
     background:var(--bg); border:1px solid var(--line); border-radius:4px; }
label { display:flex; align-items:center; gap:6px; cursor:pointer; }
.swatch { width:10px; height:10px; border-radius:50%; flex:none; }
.muted { color:var(--muted); }
.q { font-family:ui-monospace, Menlo, Consolas, monospace; color:var(--accent);
     word-break:break-all; }
.mono { font-family:ui-monospace, Menlo, Consolas, monospace; white-space:pre-wrap;
        word-break:break-word; margin:4px 0; }
.link { cursor:pointer; word-break:break-all; }
.link:hover { color:var(--accent); }
button { color:var(--text); background:#232838; border:1px solid var(--line);
         border-radius:4px; padding:4px 10px; cursor:pointer; }
</style>
</head>
<body>
<div id="side">
  <h1 id="title"></h1>
  <div id="summary" class="muted"></div>
  <h2>Search</h2>
  <input type="search" id="search" placeholder="name or qname; Enter for next">
  <div id="matches" class="muted"></div>
  <h2>Colour by</h2>
  <select id="colorBy"></select>
  <h2>Node kinds</h2>
  <div id="nodeKinds"></div>
  <h2>Edge kinds</h2>
  <div id="edgeKinds"></div>
  <h2>Details</h2>
  <div id="info" class="muted">Click a node.</div>
  <h2>Legend</h2>
  <div id="legend"></div>
  <p><button id="relayout">Re-run layout</button> <button id="fit">Fit</button></p>
</div>
<div id="stage"><canvas id="canvas"></canvas></div>
<script id="graph-data" type="application/json">__DATA__</script>
<script>
"use strict";
(function () {
  const DATA = JSON.parse(document.getElementById("graph-data").textContent);
  const PALETTE = ["#5aa9ff", "#ff7a59", "#5ad17a", "#d77bff", "#ffc94a", "#3fd0c9",
                   "#ff5c8a", "#a0a4ff", "#9bd65a", "#ff9f40", "#64c2ff"];
  const GREY = "#6b7080";
  const nodes = DATA.nodes.map(function (n, i) {
    return Object.assign({ i: i, x: 0, y: 0, vx: 0, vy: 0, fixed: false }, n);
  });
  const edges = DATA.edges.map(function (e) {
    return { s: e[0], t: e[1], k: DATA.edgeKinds[e[2]], u: e[3] === 1 };
  });
  const adj = nodes.map(function () { return []; });
  edges.forEach(function (e) { adj[e.s].push(e); adj[e.t].push(e); });

  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const state = { scale: 1, ox: 0, oy: 0, selected: -1, matches: [], matchAt: 0, alpha: 1,
                  running: false, fitted: false, colorBy: DATA.colorBy,
                  hiddenNodes: new Set(), hiddenEdges: new Set() };
  let colorOf = new Map();

  function el(tag, cls, value) {
    const node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (value !== undefined) { node.textContent = value; }
    return node;
  }
  function byId(id) { return document.getElementById(id); }

  byId("title").textContent = DATA.title;
  byId("summary").textContent = nodes.length + " nodes, " + edges.length + " edges" +
    (DATA.total > nodes.length ? " (top " + nodes.length + " of " + DATA.total +
     " nodes by degree)" : "");

  function groupKey(n) {
    if (state.colorBy === "community") { return n.c || "(none)"; }
    if (state.colorBy === "kind") { return n.k; }
    return n.g;
  }
  function recolor() {
    const counts = new Map();
    nodes.forEach(function (n) { const k = groupKey(n); counts.set(k, (counts.get(k) || 0) + 1); });
    const keys = Array.from(counts.keys()).sort(function (a, b) {
      return counts.get(b) - counts.get(a);
    });
    colorOf = new Map(keys.map(function (k, i) {
      return [k, i < PALETTE.length ? PALETTE[i] : GREY];
    }));
    const legend = byId("legend");
    legend.replaceChildren();
    keys.slice(0, PALETTE.length).forEach(function (k) {
      const row = el("label");
      const swatch = el("span", "swatch");
      swatch.style.background = colorOf.get(k);
      row.append(swatch, el("span", "", k + " (" + counts.get(k) + ")"));
      legend.append(row);
    });
    if (keys.length > PALETTE.length) {
      legend.append(el("div", "muted", "+" + (keys.length - PALETTE.length) + " more in grey"));
    }
    draw();
  }

  const select = byId("colorBy");
  ["package", "kind"].concat(DATA.hasCommunities ? ["community"] : []).forEach(function (mode) {
    const option = el("option", "", mode);
    option.value = mode;
    select.append(option);
  });
  select.value = state.colorBy;
  select.addEventListener("change", function () { state.colorBy = select.value; recolor(); });

  function checkboxes(containerId, kinds, hidden) {
    const box = byId(containerId);
    kinds.forEach(function (kind) {
      const row = el("label");
      const box2 = el("input");
      box2.type = "checkbox";
      box2.checked = true;
      box2.addEventListener("change", function () {
        if (box2.checked) { hidden.delete(kind); } else { hidden.add(kind); }
        draw();
      });
      row.append(box2, el("span", "", kind));
      box.append(row);
    });
  }
  checkboxes("nodeKinds", DATA.nodeKinds, state.hiddenNodes);
  checkboxes("edgeKinds", DATA.edgeKinds, state.hiddenEdges);
  function shown(n) { return !state.hiddenNodes.has(n.k); }
  function edgeShown(e) {
    return !state.hiddenEdges.has(e.k) && shown(nodes[e.s]) && shown(nodes[e.t]);
  }
  function radius(n) {
    return Math.max(3 + Math.min(9, Math.sqrt(n.deg)), 4 / state.scale);
  }
  const ALWAYS_LABEL = nodes.length <= 80;

  function seed() {
    const spread = 25 * Math.sqrt(nodes.length + 1);
    nodes.forEach(function (n, i) {
      const angle = i * 2.399963;
      const dist = spread * Math.sqrt((i + 0.5) / nodes.length);
      n.x = Math.cos(angle) * dist;
      n.y = Math.sin(angle) * dist;
      n.vx = 0;
      n.vy = 0;
    });
  }
  function clamp(v) { return Math.max(-25, Math.min(25, v)); }
  function tick() {
    const a = state.alpha;
    const ideal = 45;
    for (let i = 0; i < nodes.length; i++) {
      const p = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const q = nodes[j];
        let dx = p.x - q.x;
        let dy = p.y - q.y;
        let d2 = dx * dx + dy * dy;
        if (d2 > 360000) { continue; }
        if (d2 < 0.01) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; d2 = 0.25; }
        const f = ideal * ideal / d2 * a;
        p.vx += dx * f; p.vy += dy * f; q.vx -= dx * f; q.vy -= dy * f;
      }
    }
    edges.forEach(function (e) {
      const p = nodes[e.s];
      const q = nodes[e.t];
      const dx = q.x - p.x;
      const dy = q.y - p.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const f = (d - ideal) / d * 0.06 * a;
      p.vx += dx * f; p.vy += dy * f; q.vx -= dx * f; q.vy -= dy * f;
    });
    nodes.forEach(function (n) {
      if (n.fixed) { n.vx = 0; n.vy = 0; return; }
      n.vx = (n.vx - n.x * 0.003 * a) * 0.6;
      n.vy = (n.vy - n.y * 0.003 * a) * 0.6;
      n.x += clamp(n.vx);
      n.y += clamp(n.vy);
    });
    state.alpha *= 0.985;
  }
  function loop() {
    if (state.alpha > 0.02) {
      tick();
      draw();
      requestAnimationFrame(loop);
      return;
    }
    state.running = false;
    if (!state.fitted) { state.fitted = true; fit(); draw(); }
  }
  function reheat(alpha) {
    state.alpha = Math.max(state.alpha, alpha);
    if (!state.running) { state.running = true; requestAnimationFrame(loop); }
  }

  function toWorld(sx, sy) {
    return [(sx - state.ox - canvas.clientWidth / 2) / state.scale,
            (sy - state.oy - canvas.clientHeight / 2) / state.scale];
  }
  function trace(n, r) {
    ctx.beginPath();
    if (n.k === "module" || n.k === "class") {
      ctx.rect(n.x - r, n.y - r, r * 2, r * 2);
    } else if (n.k === "route") {
      ctx.moveTo(n.x, n.y - r * 1.3); ctx.lineTo(n.x + r * 1.3, n.y);
      ctx.lineTo(n.x, n.y + r * 1.3); ctx.lineTo(n.x - r * 1.3, n.y); ctx.closePath();
    } else if (n.k === "channel") {
      ctx.moveTo(n.x, n.y - r * 1.3); ctx.lineTo(n.x + r * 1.2, n.y + r);
      ctx.lineTo(n.x - r * 1.2, n.y + r); ctx.closePath();
    } else if (n.k === "resource") {
      for (let s = 0; s < 6; s++) {
        const ang = Math.PI / 3 * s;
        const px = n.x + Math.cos(ang) * r * 1.2;
        const py = n.y + Math.sin(ang) * r * 1.2;
        if (s === 0) { ctx.moveTo(px, py); } else { ctx.lineTo(px, py); }
      }
      ctx.closePath();
    } else {
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
    }
  }
  function arrowHead(p, q) {
    const dx = q.x - p.x;
    const dy = q.y - p.y;
    const d = Math.sqrt(dx * dx + dy * dy) || 1;
    const r = radius(q) + 2;
    const tx = q.x - dx / d * r;
    const ty = q.y - dy / d * r;
    const ux = dx / d * 6 / state.scale;
    const uy = dy / d * 6 / state.scale;
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx - ux - uy * 0.5, ty - uy + ux * 0.5);
    ctx.lineTo(tx - ux + uy * 0.5, ty - uy - ux * 0.5);
    ctx.closePath();
    ctx.fill();
  }
  function draw() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.translate(state.ox + w / 2, state.oy + h / 2);
    ctx.scale(state.scale, state.scale);
    const sel = state.selected;
    const near = new Set();
    if (sel >= 0) {
      near.add(sel);
      adj[sel].forEach(function (e) { near.add(e.s); near.add(e.t); });
    }
    ctx.lineWidth = 1 / state.scale;
    edges.forEach(function (e) {
      if (!edgeShown(e)) { return; }
      const p = nodes[e.s];
      const q = nodes[e.t];
      const hot = sel >= 0 && (e.s === sel || e.t === sel);
      ctx.strokeStyle = hot ? "rgba(90,169,255,0.9)" :
        (sel >= 0 ? "rgba(150,150,170,0.06)" : "rgba(150,150,170,0.25)");
      ctx.setLineDash(e.u ? [4 / state.scale, 3 / state.scale] : []);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(q.x, q.y);
      ctx.stroke();
      if (hot) { ctx.fillStyle = "rgba(90,169,255,0.9)"; arrowHead(p, q); }
    });
    ctx.setLineDash([]);
    const matched = new Set(state.matches);
    nodes.forEach(function (n) {
      if (!shown(n)) { return; }
      ctx.globalAlpha = sel >= 0 && !near.has(n.i) ? 0.25 : 1;
      ctx.fillStyle = colorOf.get(groupKey(n)) || GREY;
      trace(n, radius(n));
      ctx.fill();
      if (matched.has(n.i) || n.i === sel) {
        ctx.lineWidth = 2 / state.scale;
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();
        ctx.lineWidth = 1 / state.scale;
      }
    });
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#e6e6e6";
    ctx.font = (11 / state.scale) + "px sans-serif";
    nodes.forEach(function (n) {
      if (!shown(n)) { return; }
      if (ALWAYS_LABEL || state.scale > 1.6 || near.has(n.i) || matched.has(n.i)) {
        ctx.fillText(n.n, n.x + radius(n) + 2, n.y + 3);
      }
    });
    ctx.restore();
  }
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }
  function fit() {
    const list = nodes.filter(shown);
    if (!list.length) { return; }
    let minX = Infinity; let minY = Infinity; let maxX = -Infinity; let maxY = -Infinity;
    list.forEach(function (n) {
      minX = Math.min(minX, n.x); minY = Math.min(minY, n.y);
      maxX = Math.max(maxX, n.x); maxY = Math.max(maxY, n.y);
    });
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const wanted = Math.min((w - 120) / (maxX - minX + 1), (h - 80) / (maxY - minY + 1));
    state.scale = Math.max(0.05, Math.min(4, wanted));
    state.ox = -(minX + maxX) / 2 * state.scale;
    state.oy = -(minY + maxY) / 2 * state.scale;
  }
  function focusOn(i) {
    state.ox = -nodes[i].x * state.scale;
    state.oy = -nodes[i].y * state.scale;
  }
  function nodeAt(sx, sy) {
    const world = toWorld(sx, sy);
    let best = -1;
    let bestD = Infinity;
    nodes.forEach(function (n) {
      if (!shown(n)) { return; }
      const r = radius(n) + 3 / state.scale;
      const d = (n.x - world[0]) * (n.x - world[0]) + (n.y - world[1]) * (n.y - world[1]);
      if (d < r * r && d < bestD) { best = n.i; bestD = d; }
    });
    return best;
  }

  function edgeList(info, title, list, other, arrow) {
    if (!list.length) { return; }
    info.append(el("div", "muted", title + " (" + list.length + ")"));
    list.slice(0, 40).forEach(function (e) {
      const j = other(e);
      const label = arrow + " " + e.k + " " + nodes[j].q + (e.u ? " (unresolved)" : "");
      const row = el("div", "link", label);
      row.addEventListener("click", function () { focusOn(j); choose(j); });
      info.append(row);
    });
    if (list.length > 40) { info.append(el("div", "muted", "...")); }
  }
  function showInfo(i) {
    const info = byId("info");
    info.replaceChildren();
    info.className = "";
    if (i < 0) { info.className = "muted"; info.textContent = "Click a node."; return; }
    const n = nodes[i];
    info.append(el("div", "q", n.q), el("div", "muted", n.k + " - " + n.p + ":" + n.l));
    if (n.c) { info.append(el("div", "muted", "community: " + n.c)); }
    if (n.s) { info.append(el("div", "mono", n.s)); }
    if (n.d) { info.append(el("div", "", n.d)); }
    edgeList(info, "outgoing", adj[i].filter(function (e) { return e.s === i; }),
             function (e) { return e.t; }, "->");
    edgeList(info, "incoming", adj[i].filter(function (e) { return e.t === i; }),
             function (e) { return e.s; }, "<-");
    if (DATA.api) {
      fetch("/api/node?qname=" + encodeURIComponent(n.q))
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (full) {
          if (full && state.selected === i) {
            info.append(el("div", "muted", "in the full graph: " + full.incoming.length +
                           " incoming, " + full.outgoing.length + " outgoing edges"));
          }
        })
        .catch(function (err) { info.append(el("div", "muted", "api error: " + err)); });
    }
  }
  function choose(i) { state.selected = i; showInfo(i); draw(); }

  let drag = null;
  canvas.addEventListener("mousedown", function (ev) {
    const hit = nodeAt(ev.offsetX, ev.offsetY);
    drag = { node: hit, x: ev.offsetX, y: ev.offsetY, moved: false };
    if (hit >= 0) { nodes[hit].fixed = true; }
  });
  window.addEventListener("mousemove", function (ev) {
    if (!drag) { return; }
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left;
    const sy = ev.clientY - rect.top;
    const dx = sx - drag.x;
    const dy = sy - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 2) { drag.moved = true; }
    if (drag.node >= 0) {
      const world = toWorld(sx, sy);
      nodes[drag.node].x = world[0];
      nodes[drag.node].y = world[1];
      reheat(0.08);
    } else {
      state.ox += dx;
      state.oy += dy;
    }
    drag.x = sx;
    drag.y = sy;
    draw();
  });
  window.addEventListener("mouseup", function () {
    if (!drag) { return; }
    if (drag.node >= 0) { nodes[drag.node].fixed = false; }
    if (!drag.moved) { choose(drag.node); }
    drag = null;
  });
  canvas.addEventListener("wheel", function (ev) {
    ev.preventDefault();
    const world = toWorld(ev.offsetX, ev.offsetY);
    state.scale = Math.min(8, Math.max(0.05, state.scale * Math.exp(-ev.deltaY * 0.0015)));
    state.ox = ev.offsetX - canvas.clientWidth / 2 - world[0] * state.scale;
    state.oy = ev.offsetY - canvas.clientHeight / 2 - world[1] * state.scale;
    draw();
  }, { passive: false });

  const search = byId("search");
  search.addEventListener("input", function () {
    const needle = search.value.trim().toLowerCase();
    state.matches = needle ? nodes.filter(function (n) {
      return n.q.toLowerCase().indexOf(needle) >= 0;
    }).map(function (n) { return n.i; }) : [];
    state.matchAt = 0;
    byId("matches").textContent = needle ? state.matches.length + " match(es)" : "";
    if (state.matches.length) { focusOn(state.matches[0]); }
    draw();
  });
  search.addEventListener("keydown", function (ev) {
    if (ev.key !== "Enter" || !state.matches.length) { return; }
    const i = state.matches[state.matchAt % state.matches.length];
    state.matchAt += 1;
    focusOn(i);
    choose(i);
  });
  byId("relayout").addEventListener("click", function () {
    seed();
    state.fitted = false;
    reheat(1);
  });
  byId("fit").addEventListener("click", function () { fit(); draw(); });
  window.addEventListener("resize", resize);

  seed();
  recolor();
  resize();
  reheat(1);
})();
</script>
</body>
</html>
"""


__all__ = [
    "COLOR_MODES",
    "DEFAULT_MAX_NODES",
    "DEFAULT_PORT",
    "Response",
    "UiApp",
    "VisualizeError",
    "graph_payload",
    "host_allowed",
    "make_ui_server",
    "render_html",
    "serve_ui",
    "write_visualization",
]
