from __future__ import annotations

from codegraph.parsers import language_of, module_qname_for, parser_for


def test_module_qname_conventions() -> None:
    assert module_qname_for("pkg/billing.py") == "pkg.billing"
    assert module_qname_for("pkg/__init__.py") == "pkg"
    assert module_qname_for("web/routes/index.ts") == "web.routes"
    assert module_qname_for("app.d.ts") == "app"
    assert module_qname_for("main.py") == "main"


def test_language_of() -> None:
    assert language_of("a.py") == "python"
    assert language_of("a.TSX") == "typescript"
    assert language_of("a.mjs") == "javascript"
    assert language_of("Makefile") is None


def test_python_parser_nodes_and_edges() -> None:
    source = '''
"""Doc."""
from pkg import models
from .util import fmt as f

class Invoice(models.Base):
    def total(self) -> int:
        return self.rate() + f(1)
    def rate(self): return 1

@decorate
def charge(inv):
    inv.total()
    models.save(inv)
'''
    result = parser_for("python").parse("pkg/billing.py", source, "pkg.billing")
    kinds = {n.qname: n.kind for n in result.nodes}
    assert kinds == {
        "pkg.billing": "module",
        "pkg.billing.Invoice": "class",
        "pkg.billing.Invoice.total": "method",
        "pkg.billing.Invoice.rate": "method",
        "pkg.billing.charge": "function",
    }
    module = next(n for n in result.nodes if n.kind == "module")
    assert module.docstring == "Doc."
    edges = {(e.kind, e.src, e.dst, e.resolved) for e in result.edges}
    assert ("imports", "pkg.billing", "pkg.models", True) in edges
    assert ("imports", "pkg.billing", "pkg.util.fmt", True) in edges
    assert ("inherits", "pkg.billing.Invoice", "pkg.models.Base", True) in edges
    assert ("calls", "pkg.billing.Invoice.total", "pkg.billing.Invoice.rate", True) in edges
    assert ("calls", "pkg.billing.Invoice.total", "pkg.util.fmt", True) in edges
    assert ("calls", "pkg.billing.charge", "pkg.models.save", True) in edges
    assert ("calls", "pkg.billing.charge", "*.total", False) in edges
    assert ("calls", "pkg.billing.charge", "decorate", False) in edges


def test_python_parser_relative_import_levels() -> None:
    source = "from ..core import thing\nfrom . import sibling\n"
    result = parser_for("python").parse("a/b/c.py", source, "a.b.c")
    targets = {e.dst for e in result.edges if e.kind == "imports"}
    assert targets == {"a.core.thing", "a.b.sibling"}


def test_python_parser_survives_syntax_error() -> None:
    result = parser_for("python").parse("bad.py", "def (:\n", "bad")
    assert [n.kind for n in result.nodes] == ["module"]
    assert result.edges == ()


def test_typescript_parser_nodes_and_edges() -> None:
    source = """
import { helper, other as alias } from "./util";
import def from "../lib/thing";
import * as ns from "./ns";
/** Invoice class. */
export class Invoice extends Base {
  total(): number { return helper(this.rate()); }
  rate() { return 1; }
}
export function charge(x: number): void { new Invoice().total(); alias(); ns.go(); def.run(); }
const send = async (a) => { await charge(a); };
export interface Payable { pay(): void }
"""
    result = parser_for("typescript").parse("web/app.ts", source, "web.app")
    kinds = {n.qname: n.kind for n in result.nodes}
    assert kinds["web.app.Invoice"] == "class"
    assert kinds["web.app.Invoice.total"] == "method"
    assert kinds["web.app.charge"] == "function"
    assert kinds["web.app.send"] == "function"
    assert kinds["web.app.Payable"] == "class"
    assert kinds["web.app.Payable.pay"] == "method"
    invoice = next(n for n in result.nodes if n.qname == "web.app.Invoice")
    assert invoice.docstring == "Invoice class."
    edges = {(e.kind, e.src, e.dst, e.resolved) for e in result.edges}
    assert ("imports", "web.app", "web.util", True) in edges
    assert ("imports", "web.app", "lib.thing", True) in edges
    assert ("calls", "web.app.Invoice.total", "web.util.helper", True) in edges
    assert ("calls", "web.app.Invoice.total", "web.app.Invoice.rate", True) in edges
    assert ("calls", "web.app.charge", "web.app.Invoice", True) in edges
    assert ("calls", "web.app.charge", "web.util.other", True) in edges
    assert ("calls", "web.app.charge", "web.ns.go", True) in edges
    assert ("calls", "web.app.charge", "lib.thing.run", True) in edges
    assert ("calls", "web.app.send", "web.app.charge", True) in edges
    assert ("inherits", "web.app.Invoice", "Base", False) in edges


def test_javascript_parser_commonjs_style() -> None:
    source = """
function outer() { inner(); }
function inner() {}
class Thing { go() { this.stop(); } stop() {} }
"""
    result = parser_for("javascript").parse("lib/x.js", source, "lib.x")
    edges = {(e.kind, e.src, e.dst, e.resolved) for e in result.edges}
    assert ("calls", "lib.x.outer", "lib.x.inner", True) in edges
    assert ("calls", "lib.x.Thing.go", "lib.x.Thing.stop", True) in edges
