from __future__ import annotations

import random
import time

from codegraph.services.clones import (
    ShingleHasher,
    find_clones,
    fingerprint,
    jaccard,
    normalise_tokens,
    shingle_set,
    signature,
)

ORIGINAL = """
def total(items, rate):
    # running sum
    subtotal = 0
    for item in items:
        if item.qty <= 0:
            continue
        subtotal += item.price * item.qty
    if subtotal > 100:
        subtotal = subtotal - 5
    return round(subtotal * (1 + rate), 2)
"""

RENAMED = """
def amount(lines, vat):
    acc = 0
    for line in lines:
        if line.count <= 0:
            continue
        acc += line.unit * line.count
    if acc > 250:
        acc = acc - 9
    return round(acc * (1 + vat), 2)
"""

DIFFERENT = """
def describe(order, status):
    parts = []
    while len(parts) < 3:
        try:
            parts.append(status.upper())
        except AttributeError:
            break
    with open(order.path) as handle:
        handle.write(", ".join(parts))
    return {"order": order.id, "parts": parts, "ok": True}
"""


def _print(name: str, body: str, path: str = "a.py", start: int = 1) -> object:
    item = fingerprint(name, path, start, start + body.count("\n"), body, "python", ShingleHasher())
    assert item is not None
    return item


def test_normalise_tokens_strips_comments_and_names() -> None:
    tokens = normalise_tokens("x = foo(1, 'a')  # note\n", "python")
    assert tokens == ["I", "=", "I", "(", "N", ",", "S", ")"]
    assert normalise_tokens("if (a) { return b; } // c", "js") == [
        "if",
        "(",
        "I",
        ")",
        "{",
        "return",
        "I",
        ";",
        "}",
    ]


def test_signature_is_deterministic_and_dense() -> None:
    hasher = ShingleHasher()
    shingles = shingle_set(normalise_tokens(ORIGINAL, "python"), hasher)
    first, second = signature(shingles), signature(set(shingles))
    assert first == second and len(first) == 64
    assert all(value != (1 << 64) - 1 for value in first)
    assert jaccard(shingles, shingles) == 1.0


def test_renamed_copy_is_found_and_unrelated_code_is_not() -> None:
    hasher = ShingleHasher()
    prints = [
        fingerprint("m.total", "m.py", 1, 11, ORIGINAL, "python", hasher),
        fingerprint("n.amount", "n.py", 1, 10, RENAMED, "python", hasher),
        fingerprint("o.describe", "o.py", 1, 11, DIFFERENT, "python", hasher),
    ]
    assert all(p is not None for p in prints)
    pairs = find_clones([p for p in prints if p is not None])
    assert [(p.a, p.b) for p in pairs] == [("m.total", "n.amount")]
    assert pairs[0].score >= 0.8


def test_tiny_and_nested_functions_are_skipped() -> None:
    hasher = ShingleHasher()
    assert fingerprint("t", "t.py", 1, 2, "def t():\n    return 1\n", "python", hasher) is None
    outer = fingerprint("m.outer", "m.py", 1, 30, ORIGINAL, "python", hasher)
    inner = fingerprint("m.outer.inner", "m.py", 2, 12, ORIGINAL, "python", hasher)
    assert outer is not None and inner is not None
    assert find_clones([outer, inner]) == []


def test_find_clones_scales_to_thousands_of_functions() -> None:
    rng = random.Random(7)
    keywords = ["if", "for", "while", "return", "try", "else", "with", "yield"]
    hasher = ShingleHasher()
    prints = []
    for index in range(3000):
        lines = [
            f"    {rng.choice(keywords)} v{rng.randint(0, 9)} {rng.choice('+-*/<>')} "
            f"{rng.randint(0, 99)}: call{rng.randint(0, 5)}(x[{rng.randint(0, 3)}])"
            for _ in range(12)
        ]
        body = f"def f{index}(x):\n" + "\n".join(lines) + "\n"
        item = fingerprint(f"m.f{index}", f"m{index}.py", 1, 13, body, "python", hasher)
        if item is not None:
            prints.append(item)
    copy = fingerprint("copy.f0", "copy.py", 1, 13, ORIGINAL, "python", hasher)
    twin = fingerprint("copy.f1", "twin.py", 1, 13, RENAMED, "python", hasher)
    assert copy is not None and twin is not None
    prints += [copy, twin]
    started = time.perf_counter()
    pairs = find_clones(prints)
    assert time.perf_counter() - started < 10
    assert ("copy.f0", "copy.f1") in {(p.a, p.b) for p in pairs}
