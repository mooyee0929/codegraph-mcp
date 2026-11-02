"""One real-looking snippet per tree-sitter language: definitions, calls and imports."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from codegraph.models import ParseResult
from codegraph.parsers import module_qname_for, parser_for

EdgeKey = tuple[str, str, str, bool]


@dataclass(frozen=True)
class Case:
    language: str
    path: str
    source: str
    nodes: dict[str, str]
    edges: frozenset[EdgeKey]
    docs: tuple[tuple[str, str], ...] = ()


def parse(language: str, path: str, source: str) -> ParseResult:
    return parser_for(language).parse(path, source, module_qname_for(path))


def edge_keys(result: ParseResult) -> set[EdgeKey]:
    return {(e.kind, e.src, e.dst, e.resolved) for e in result.edges}


GO = """package main

import (
\t"fmt"
\tu "example.com/app/internal/util"
)

// Server serves.
type Server struct {
\tBase
}

// Start starts it.
func (s *Server) Start(port int) error {
\ts.helper()
\tu.Pad("x")
\tfmt.Println("ok")
\treturn nil
}

func main() { local() }
"""

RUST = """use crate::util::{pad, Strings as S};
use std::collections::HashMap;
mod net;

/// A point.
pub struct Point { x: i32 }

pub trait Shape { fn area(&self) -> f64; }

impl Shape for Point {
    fn area(&self) -> f64 { self.helper(); pad(1); net::connect(); 0.0 }
}

impl Point {
    pub fn new() -> Self { Self::origin() }
}

fn main() { let p = Point::new(); p.area(); }
"""

JAVA = """package com.example.app;

import com.example.util.Strings;
import java.util.List;
import static com.example.util.Math.max;

/** The service. */
public class Service extends Base implements Runnable {
    public void run() {
        Strings.pad("x");
        repo.save(1);
        this.helper();
        max(1, 2);
        Math.abs(-1);
    }
    void helper() {}
    interface Inner { void go(); }
}
"""

KOTLIN = """package com.example.app

import com.example.util.Strings
import com.example.util.pad as padIt

/** A service. */
class Service(val repo: Repo) : Base(), Runnable {
    override fun run() { Strings.pad("x"); helper(); padIt("y") }
    fun helper() {}
    companion object { fun create() = Service(Repo()) }
}

object Registry { fun get() = 1 }
"""

SCALA = """package com.example.app

import com.example.util.Strings
import com.example.util.{Pad, Trim => T}

/** A service. */
class Service(repo: Repo) extends Base with Runnable {
  def run(): Unit = { Strings.pad("x"); helper() }
  def helper(): Int = 1
}

object Main { def main(args: Array[String]): Unit = new Service(new Repo).run() }
"""

GROOVY = """package com.example

import com.example.util.Strings

/** A service. */
class Service extends Base {
    void run() {
        Strings.pad("x")
        helper()
    }
    def helper() { 1 }
}
"""

C = """#include <stdio.h>
#include "util/strings.h"

/** Adds. */
static int add(int a, int b) {
    helper(a);
    return pad(a) + b;
}

int helper(int a);
typedef struct { int y; } vec_t;
int main(void) { printf("%d", add(1, 2)); return 0; }
"""

CPP = """#include "shapes/base.hpp"

namespace geo {
/// A circle.
class Circle : public Shape {
public:
    double area() const { return helper() * 2; }
    void draw();
private:
    int helper() const { return 1; }
};

void Circle::draw() { this->area(); std::sort(); }
}
"""

CSHARP = """using System;
using Str = Company.Util.Strings;

namespace Company.App
{
    /// <summary>The service.</summary>
    public class Service : BaseService, IRunnable
    {
        public void Run() { Str.Pad("x"); Helper(); this.Helper(); var r = new Repo(); }
        private void Helper() {}
    }
    public interface IRunnable { void Run(); }
}
"""

RUBY = """require "json"
require_relative "lib/util"

module Billing
  # An invoice.
  class Invoice < Base
    include Comparable
    def total
      helper(1)
      self.rate
    end

    def self.build
      new
    end
  end
end
"""

PHP = """<?php
namespace App\\Services;

use App\\Util\\Strings;

/** The service. */
class Service extends BaseService implements Runnable
{
    public function run(): void
    {
        Strings::pad("x");
        $this->helper();
        $this->repo->save(1);
        helper_fn();
    }
    private function helper() {}
}

function helper_fn() { return 1; }
"""

SWIFT = """import Foundation

/// A circle.
class Circle: Base {
    func area() -> Double { return helper() * 2 }
    private func helper() -> Double { self.draw(); return 1 }
}

extension Circle { func draw() {} }
func main() { let c = Circle(); c.area() }
"""

DART = """import 'models.dart' as m;

/// A service.
class Service extends Base with Logger {
  void run() {
    m.save(1);
    helper();
  }
  int helper() => 1;
}

mixin Logger { void log() {} }
void main() { Service().run(); }
"""

LUA = """local util = require("app.util")
local M = {}

--- Adds things.
function M.add(a, b)
  return util.pad(a) + b
end

local function helper(x) return x end

function M:run() self:other(); helper(1); M.add(1, 2) end
return M
"""

BASH = """#!/usr/bin/env bash
source ./lib/common.sh

# Deploys things.
deploy() {
  log_info "deploying"
  build_image "$1"
  echo done
}

function build_image {
  docker build -t "$1" .
}

deploy "$@"
"""

ELIXIR = """defmodule MyApp.Billing do
  alias MyApp.Util.Strings

  # Charges.
  def charge(inv) do
    Strings.pad("x")
    Enum.map([], &(&1))
    total(inv)
  end

  defp total(inv), do: inv.amount
end
"""

SOLIDITY = """pragma solidity ^0.8.0;
import {SafeMath} from "./lib/SafeMath.sol";

/// A token.
contract Token is Base {
    function transfer(address to, uint amount) public returns (bool) {
        _move(msg.sender, to, amount);
        SafeMath.add(1, 2);
        return true;
    }
    function _move(address a, address b, uint c) internal {}
}
"""

OBJC = """#import <Foundation/Foundation.h>
#import "Util.h"

/// A shape.
@interface Circle : Shape
- (double)area;
@end

@implementation Circle
- (double)area {
    [self helper];
    return compute(2);
}
- (void)helper {}
@end

int compute(int x) { return x; }
"""

ZIG = """const std = @import("std");
const util = @import("util.zig");

/// A point.
const Point = struct {
    x: i32,
    pub fn len(self: Point) i32 {
        return util.abs(self.x) + helper();
    }
};

fn helper() i32 { return 1; }
pub fn main() void { std.debug.print("hi", .{}); }
"""

JULIA = """module Geo
include("util.jl")

\"\"\"A point.\"\"\"
struct Point <: AbstractPoint
    x::Float64
end

function norm2(p::Point)
    helper(p.x)
end

helper(x) = x * x
end
"""

R = """library(dplyr)
source("utils.R")

#' Adds.
add <- function(a, b) {
  helper(a)
  dplyr::filter(a)
}

helper = function(x) x
"""

PERL = """package My::Billing;
use strict;
use My::Util qw(pad);
use parent -norequire, 'My::Base';

# Charges.
sub charge {
    my ($self, $inv) = @_;
    pad("x");
    $self->total($inv);
}

sub total { return 1; }
1;
"""

HASKELL = """module Billing.Invoice (charge) where

import qualified Data.Map as M
import Billing.Util (pad)

data Invoice = Invoice { amount :: Int }

-- | Charges.
charge :: Invoice -> Int
charge inv = pad (amount inv) + helper inv

helper :: Invoice -> Int
helper _ = M.size M.empty
"""

ERLANG = """-module(billing).
-export([charge/1]).
-import(util, [pad/1]).
-include("records.hrl").

%% Charges.
charge(Inv) ->
    pad(Inv),
    lists:map(fun(X) -> X end, []),
    helper(Inv).

helper(_) -> ok.
"""

CASES = [
    Case(
        "go",
        "cmd/app/main.go",
        GO,
        {
            "cmd.app.main.Server": "class",
            "cmd.app.main.Server.Start": "method",
            "cmd.app.main.main": "function",
        },
        frozenset(
            {
                ("inherits", "cmd.app.main.Server", "Base", False),
                ("calls", "cmd.app.main.Server.Start", "cmd.app.main.Server.helper", True),
                (
                    "calls",
                    "cmd.app.main.Server.Start",
                    "example.com/app/internal/util.Pad",
                    False,
                ),
                ("calls", "cmd.app.main.Server.Start", "fmt.Println", True),
                ("calls", "cmd.app.main.main", "local", False),
                ("imports", "cmd.app.main", "example.com/app/internal/util", True),
            }
        ),
        (
            ("cmd.app.main.Server", "Server serves."),
            ("cmd.app.main.Server.Start", "Start starts it."),
        ),
    ),
    Case(
        "rust",
        "src/net/client.rs",
        RUST,
        {
            "src.net.client.Point": "class",
            "src.net.client.Shape": "class",
            "src.net.client.Point.area": "method",
            "src.net.client.Point.new": "method",
            "src.net.client.main": "function",
        },
        frozenset(
            {
                ("inherits", "src.net.client.Point", "src.net.client.Shape", True),
                ("calls", "src.net.client.Point.area", "src.net.client.Point.helper", True),
                ("calls", "src.net.client.Point.area", "src.util.pad", False),
                ("calls", "src.net.client.Point.area", "src.net.client.net.connect", True),
                ("calls", "src.net.client.Point.new", "src.net.client.Point.origin", True),
                ("calls", "src.net.client.main", "src.net.client.Point.new", True),
                ("imports", "src.net.client", "src.util.Strings", True),
                ("imports", "src.net.client", "std.collections.HashMap", True),
                ("imports", "src.net.client", "src.net.client.net", True),
            }
        ),
        (("src.net.client.Point", "A point."),),
    ),
    Case(
        "java",
        "src/main/java/com/example/app/Service.java",
        JAVA,
        {
            "src.main.java.com.example.app.Service.Service": "class",
            "src.main.java.com.example.app.Service.Service.run": "method",
            "src.main.java.com.example.app.Service.Service.Inner": "class",
        },
        frozenset(
            {
                (
                    "calls",
                    "src.main.java.com.example.app.Service.Service.run",
                    "src.main.java.com.example.util.Strings.Strings.pad",
                    False,
                ),
                ("calls", "src.main.java.com.example.app.Service.Service.run", "*.save", False),
                (
                    "calls",
                    "src.main.java.com.example.app.Service.Service.run",
                    "src.main.java.com.example.app.Service.Service.helper",
                    True,
                ),
                (
                    "calls",
                    "src.main.java.com.example.app.Service.Service.run",
                    "src.main.java.com.example.util.Math.Math.max",
                    False,
                ),
                ("calls", "src.main.java.com.example.app.Service.Service.run", "Math.abs", True),
                (
                    "imports",
                    "src.main.java.com.example.app.Service",
                    "src.main.java.com.example.util.Strings",
                    True,
                ),
                ("imports", "src.main.java.com.example.app.Service", "java.util.List", True),
            }
        ),
        (("src.main.java.com.example.app.Service.Service", "The service."),),
    ),
    Case(
        "kotlin",
        "src/main/kotlin/com/example/app/Service.kt",
        KOTLIN,
        {
            "src.main.kotlin.com.example.app.Service.Service": "class",
            "src.main.kotlin.com.example.app.Service.Service.create": "method",
            "src.main.kotlin.com.example.app.Service.Registry.get": "method",
        },
        frozenset(
            {
                ("inherits", "src.main.kotlin.com.example.app.Service.Service", "Base", False),
                (
                    "calls",
                    "src.main.kotlin.com.example.app.Service.Service.run",
                    "src.main.kotlin.com.example.util.pad",
                    False,
                ),
                (
                    "calls",
                    "src.main.kotlin.com.example.app.Service.Service.create",
                    "src.main.kotlin.com.example.app.Service.Service",
                    True,
                ),
            }
        ),
        (("src.main.kotlin.com.example.app.Service.Service", "A service."),),
    ),
    Case(
        "scala",
        "src/main/scala/com/example/app/Service.scala",
        SCALA,
        {
            "src.main.scala.com.example.app.Service.Service": "class",
            "src.main.scala.com.example.app.Service.Main.main": "method",
        },
        frozenset(
            {
                (
                    "calls",
                    "src.main.scala.com.example.app.Service.Service.run",
                    "src.main.scala.com.example.util.Strings.Strings.pad",
                    False,
                ),
                (
                    "calls",
                    "src.main.scala.com.example.app.Service.Main.main",
                    "src.main.scala.com.example.app.Service.Service",
                    True,
                ),
                (
                    "imports",
                    "src.main.scala.com.example.app.Service",
                    "src.main.scala.com.example.util.Trim",
                    True,
                ),
            }
        ),
    ),
    Case(
        "groovy",
        "src/main/groovy/com/example/Service.groovy",
        GROOVY,
        {
            "src.main.groovy.com.example.Service.Service": "class",
            "src.main.groovy.com.example.Service.Service.run": "method",
            "src.main.groovy.com.example.Service.Service.helper": "method",
        },
        frozenset(
            {
                (
                    "calls",
                    "src.main.groovy.com.example.Service.Service.run",
                    "src.main.groovy.com.example.Service.Service.helper",
                    True,
                ),
                (
                    "calls",
                    "src.main.groovy.com.example.Service.Service.run",
                    "src.main.groovy.com.example.util.Strings.Strings.pad",
                    False,
                ),
            }
        ),
    ),
    Case(
        "c",
        "src/add.c",
        C,
        {"src.add.add": "function", "src.add.main": "function", "src.add.vec_t": "class"},
        frozenset(
            {
                ("calls", "src.add.add", "helper", False),
                ("calls", "src.add.main", "src.add.add", True),
                ("imports", "src.add", "src.util.strings_h", True),
                ("imports", "src.add", "stdio.h", True),
            }
        ),
        (("src.add.add", "Adds."),),
    ),
    Case(
        "cpp",
        "src/circle.cpp",
        CPP,
        {
            "src.circle.Circle": "class",
            "src.circle.Circle.area": "method",
            "src.circle.Circle.draw": "method",
        },
        frozenset(
            {
                ("inherits", "src.circle.Circle", "Shape", False),
                ("calls", "src.circle.Circle.area", "src.circle.Circle.helper", True),
                ("calls", "src.circle.Circle.draw", "src.circle.Circle.area", True),
                ("calls", "src.circle.Circle.draw", "std.sort", True),
                ("imports", "src.circle", "src.shapes.base_hpp", True),
            }
        ),
        (("src.circle.Circle", "A circle."),),
    ),
    Case(
        "csharp",
        "src/Service.cs",
        CSHARP,
        {"src.Service.Service": "class", "src.Service.Service.Run": "method"},
        frozenset(
            {
                ("inherits", "src.Service.Service", "src.Service.IRunnable", True),
                ("calls", "src.Service.Service.Run", "Company.Util.Strings.Pad", False),
                ("calls", "src.Service.Service.Run", "src.Service.Service.Helper", True),
                ("calls", "src.Service.Service.Run", "Repo", False),
                ("imports", "src.Service", "System", True),
            }
        ),
        (("src.Service.Service", "The service."),),
    ),
    Case(
        "ruby",
        "app/billing.rb",
        RUBY,
        {
            "app.billing.Billing": "class",
            "app.billing.Billing.Invoice": "class",
            "app.billing.Billing.Invoice.build": "method",
        },
        frozenset(
            {
                ("inherits", "app.billing.Billing.Invoice", "Base", False),
                ("inherits", "app.billing.Billing.Invoice", "Comparable", False),
                (
                    "calls",
                    "app.billing.Billing.Invoice.total",
                    "app.billing.Billing.Invoice.rate",
                    True,
                ),
                ("imports", "app.billing", "app.lib.util", True),
                ("imports", "app.billing", "json", True),
            }
        ),
        (("app.billing.Billing.Invoice", "An invoice."),),
    ),
    Case(
        "php",
        "src/Services/Service.php",
        PHP,
        {"src.Services.Service.Service": "class", "src.Services.Service.helper_fn": "function"},
        frozenset(
            {
                (
                    "calls",
                    "src.Services.Service.Service.run",
                    "App.Util.Strings.Strings.pad",
                    False,
                ),
                (
                    "calls",
                    "src.Services.Service.Service.run",
                    "src.Services.Service.Service.helper",
                    True,
                ),
                ("calls", "src.Services.Service.Service.run", "*.save", False),
                (
                    "calls",
                    "src.Services.Service.Service.run",
                    "src.Services.Service.helper_fn",
                    True,
                ),
            }
        ),
    ),
    Case(
        "swift",
        "Sources/App/Circle.swift",
        SWIFT,
        {"Sources.App.Circle.Circle": "class", "Sources.App.Circle.Circle.draw": "method"},
        frozenset(
            {
                (
                    "calls",
                    "Sources.App.Circle.Circle.helper",
                    "Sources.App.Circle.Circle.draw",
                    True,
                ),
                ("calls", "Sources.App.Circle.main", "Sources.App.Circle.Circle", True),
                ("imports", "Sources.App.Circle", "Foundation", True),
            }
        ),
    ),
    Case(
        "dart",
        "lib/service.dart",
        DART,
        {"lib.service.Service": "class", "lib.service.Service.run": "method"},
        frozenset(
            {
                ("inherits", "lib.service.Service", "lib.service.Logger", True),
                ("calls", "lib.service.Service.run", "lib.models.save", True),
                ("calls", "lib.service.Service.run", "lib.service.Service.helper", True),
                ("calls", "lib.service.main", "lib.service.Service", True),
                ("imports", "lib.service", "lib.models", True),
            }
        ),
    ),
    Case(
        "lua",
        "app/m.lua",
        LUA,
        {"app.m.M.add": "method", "app.m.helper": "function", "app.m.M.run": "method"},
        frozenset(
            {
                ("calls", "app.m.M.add", "app.util.pad", False),
                ("calls", "app.m.M.run", "app.m.M.other", True),
                ("calls", "app.m.M.run", "app.m.helper", True),
                ("calls", "app.m.M.run", "app.m.M.add", True),
                ("imports", "app.m", "app.util", True),
            }
        ),
        (("app.m.M.add", "Adds things."),),
    ),
    Case(
        "bash",
        "scripts/deploy.sh",
        BASH,
        {"scripts.deploy.deploy": "function", "scripts.deploy.build_image": "function"},
        frozenset(
            {
                ("calls", "scripts.deploy.deploy", "log_info", False),
                ("calls", "scripts.deploy.deploy", "scripts.deploy.build_image", True),
                ("calls", "scripts.deploy", "scripts.deploy.deploy", True),
                ("imports", "scripts.deploy", "scripts.lib.common", True),
            }
        ),
        (("scripts.deploy.deploy", "Deploys things."),),
    ),
    Case(
        "elixir",
        "lib/my_app/billing.ex",
        ELIXIR,
        {
            "lib.my_app.billing.MyApp.Billing": "class",
            "lib.my_app.billing.MyApp.Billing.charge": "method",
            "lib.my_app.billing.MyApp.Billing.total": "method",
        },
        frozenset(
            {
                (
                    "calls",
                    "lib.my_app.billing.MyApp.Billing.charge",
                    "MyApp.Util.Strings.pad",
                    False,
                ),
                ("calls", "lib.my_app.billing.MyApp.Billing.charge", "Enum.map", True),
                (
                    "calls",
                    "lib.my_app.billing.MyApp.Billing.charge",
                    "lib.my_app.billing.MyApp.Billing.total",
                    True,
                ),
                ("imports", "lib.my_app.billing", "MyApp.Util.Strings", True),
            }
        ),
        (("lib.my_app.billing.MyApp.Billing.charge", "Charges."),),
    ),
    Case(
        "solidity",
        "contracts/Token.sol",
        SOLIDITY,
        {"contracts.Token.Token": "class", "contracts.Token.Token.transfer": "method"},
        frozenset(
            {
                ("inherits", "contracts.Token.Token", "Base", False),
                ("calls", "contracts.Token.Token.transfer", "contracts.Token.Token._move", True),
                (
                    "calls",
                    "contracts.Token.Token.transfer",
                    "contracts.lib.SafeMath.SafeMath.add",
                    True,
                ),
                ("imports", "contracts.Token", "contracts.lib.SafeMath", True),
            }
        ),
    ),
    Case(
        "objc",
        "Circle.m",
        OBJC,
        {"Circle.Circle": "class", "Circle.Circle.area": "method", "Circle.compute": "function"},
        frozenset(
            {
                ("inherits", "Circle.Circle", "Shape", False),
                ("calls", "Circle.Circle.area", "Circle.Circle.helper", True),
                ("calls", "Circle.Circle.area", "Circle.compute", True),
                ("imports", "Circle", "Util_h", True),
            }
        ),
    ),
    Case(
        "zig",
        "src/point.zig",
        ZIG,
        {"src.point.Point": "class", "src.point.Point.len": "method", "src.point.main": "function"},
        frozenset(
            {
                ("calls", "src.point.Point.len", "src.util.abs", True),
                ("calls", "src.point.Point.len", "src.point.helper", True),
                ("calls", "src.point.main", "std.debug.print", True),
                ("imports", "src.point", "src.util", True),
            }
        ),
        (("src.point.Point", "A point."),),
    ),
    Case(
        "julia",
        "src/Geo.jl",
        JULIA,
        {
            "src.Geo.Geo.Point": "class",
            "src.Geo.Geo.norm2": "function",
            "src.Geo.Geo.helper": "function",
        },
        frozenset(
            {
                ("inherits", "src.Geo.Geo.Point", "AbstractPoint", False),
                ("calls", "src.Geo.Geo.norm2", "src.Geo.Geo.helper", True),
                ("imports", "src.Geo", "src.util", True),
            }
        ),
        (("src.Geo.Geo.Point", "A point."),),
    ),
    Case(
        "r",
        "R/add.R",
        R,
        {"R.add.add": "function", "R.add.helper": "function"},
        frozenset(
            {
                ("calls", "R.add.add", "R.add.helper", True),
                ("calls", "R.add.add", "dplyr.filter", True),
                ("imports", "R.add", "R.utils", True),
                ("imports", "R.add", "dplyr", True),
            }
        ),
        (("R.add.add", "Adds."),),
    ),
    Case(
        "perl",
        "lib/My/Billing.pm",
        PERL,
        {"lib.My.Billing.My.Billing": "class", "lib.My.Billing.My.Billing.charge": "method"},
        frozenset(
            {
                ("inherits", "lib.My.Billing.My.Billing", "*.Base", False),
                ("calls", "lib.My.Billing.My.Billing.charge", "My.Util.pad", False),
                (
                    "calls",
                    "lib.My.Billing.My.Billing.charge",
                    "lib.My.Billing.My.Billing.total",
                    True,
                ),
                ("imports", "lib.My.Billing", "My.Util", True),
            }
        ),
        (("lib.My.Billing.My.Billing.charge", "Charges."),),
    ),
    Case(
        "haskell",
        "src/Billing/Invoice.hs",
        HASKELL,
        {"src.Billing.Invoice.Invoice": "class", "src.Billing.Invoice.charge": "function"},
        frozenset(
            {
                ("calls", "src.Billing.Invoice.charge", "src.Billing.Util.pad", False),
                ("calls", "src.Billing.Invoice.charge", "src.Billing.Invoice.helper", True),
                ("calls", "src.Billing.Invoice.helper", "Data.Map.size", True),
                ("imports", "src.Billing.Invoice", "src.Billing.Util", True),
            }
        ),
        (("src.Billing.Invoice.charge", "Charges."),),
    ),
    Case(
        "erlang",
        "src/billing.erl",
        ERLANG,
        {"src.billing.charge": "function", "src.billing.helper": "function"},
        frozenset(
            {
                ("calls", "src.billing.charge", "util.pad", False),
                ("calls", "src.billing.charge", "lists.map", True),
                ("calls", "src.billing.charge", "src.billing.helper", True),
                ("imports", "src.billing", "src.records", True),
            }
        ),
        (("src.billing.charge", "Charges."),),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.language for c in CASES])
def test_language_snippet(case: Case) -> None:
    result = parse(case.language, case.path, case.source)
    kinds = {n.qname: n.kind for n in result.nodes}
    module = module_qname_for(case.path)
    assert kinds[module] == "module"
    for qname, kind in case.nodes.items():
        assert kinds.get(qname) == kind, f"{qname}: {sorted(kinds)}"
    found = edge_keys(result)
    missing = case.edges - found
    assert not missing, f"missing {sorted(missing)}\nfound {sorted(found)}"
    docs = {n.qname: n.docstring for n in result.nodes}
    for qname, doc in case.docs:
        assert docs[qname] == doc
    assert all(n.language == case.language for n in result.nodes)
    defined = {e.dst for e in result.edges if e.kind == "defines"}
    assert defined == set(kinds) - {module}


@pytest.mark.parametrize("case", CASES, ids=[c.language for c in CASES])
def test_malformed_input_never_raises(case: Case) -> None:
    broken = case.source[: len(case.source) // 2] + "\n}}} ((( @@@ \x00 end"
    result = parse(case.language, case.path, broken)
    assert result.nodes[0].kind == "module"


def test_bodiless_declarations_are_not_nodes() -> None:
    java = parse("java", "A.java", JAVA)
    assert "A.Service.Inner.go" not in {n.qname for n in java.nodes}
    c = parse("c", "a.c", C)
    assert [n.qname for n in c.nodes if n.name == "helper"] == []
    rust = parse("rust", "src/lib.rs", RUST)
    assert "src.lib.Shape.area" not in {n.qname for n in rust.nodes}


def test_signatures_stop_at_the_body() -> None:
    go = {n.qname: n.signature for n in parse("go", "a.go", GO).nodes}
    assert go["a.Server.Start"] == "func (s *Server) Start(port int) error"
    haskell = {n.qname: n for n in parse("haskell", "src/Billing/Invoice.hs", HASKELL).nodes}
    charge = haskell["src.Billing.Invoice.charge"]
    assert charge.signature == "charge :: Invoice -> Int"
    assert charge.line_start == 9


def test_go_receiver_name_resolves_to_its_type() -> None:
    edges = edge_keys(parse("go", "a.go", GO))
    assert ("calls", "a.Server.Start", "a.Server.helper", True) in edges


def test_rust_self_path_is_the_current_module() -> None:
    source = "fn a() { self::b(); }\nfn b() {}\n"
    edges = edge_keys(parse("rust", "src/x.rs", source))
    assert ("calls", "src.x.a", "src.x.b", True) in edges


def test_receiver_chains_through_self_are_not_trusted() -> None:
    edges = edge_keys(parse("php", "a.php", PHP))
    assert ("calls", "a.Service.run", "a.Service.repo.save", True) not in edges


def test_unknown_static_receiver_keeps_its_path_in_detail() -> None:
    result = parse("java", "A.java", "class A { void f() { Helpers.go(); } }")
    [call] = [e for e in result.edges if e.kind == "calls"]
    assert (call.dst, call.detail, call.resolved) == ("*.go", "Helpers.go", False)


def test_alias_guess_is_copied_into_detail() -> None:
    result = parse("rust", "src/main.rs", "use crate::util::pad;\nfn main() { pad(); }\n")
    [call] = [e for e in result.edges if e.kind == "calls"]
    assert call.dst == "src.util.pad"
    assert call.detail == "src.util.pad"
    assert not call.resolved


def test_header_files_pick_their_grammar() -> None:
    header = "namespace geo { class Box { public: int w() { return 1; } }; }\n"
    result = parse("c", "include/box.h", header)
    assert {n.qname for n in result.nodes} >= {"include.box_h.Box", "include.box_h.Box.w"}
    objc = "#import <Foundation/Foundation.h>\n@interface Thing : NSObject\n@end\n"
    assert "include.thing_h.Thing" in {n.qname for n in parse("c", "include/thing.h", objc).nodes}
