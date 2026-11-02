"""Near-clone detection: MinHash over normalised token shingles, LSH, exact Jaccard.

Each function body is tokenised with comments and docstrings removed, then identifiers
become ``I``, numbers ``N`` and string literals ``S`` while keywords and punctuation stay
as written. Renaming variables or changing a constant therefore does not hide a copy,
but changing control flow does. Shingles are runs of :data:`SHINGLE` tokens; runs made
only of ``I``/``N``/``S`` carry no structure and are dropped, as in codebase-memory's
structural weighting.

Candidates come from banded locality-sensitive hashing over a one-permutation MinHash
signature (one hash per shingle instead of :data:`K`, with rotation densification for
empty bins), which keeps the pass linear in total body size. Every candidate pair is
then confirmed with the exact Jaccard similarity of the two shingle sets, so the
reported score is never an estimate.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from codegraph.services.text import comment_spans

SHINGLE = 4
K = 64
BANDS = 16
ROWS = K // BANDS
DEFAULT_THRESHOLD = 0.8
MIN_TOKENS = 40
MIN_SHINGLES = 20
MAX_BUCKET = 200
MAX_PER_NODE = 10
_EMPTY = (1 << 64) - 1
_ROTATION = 1 << 58

_TOKEN = re.compile(
    r"[A-Za-z_$][\w$]*|\d[\w.]*|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`"
    r"|==|!=|<=|>=|&&|\|\||->|=>|::|\+\+|--|[^\s\w]"
)
KEYWORDS = frozenset(
    [
        "if",
        "elif",
        "else",
        "for",
        "while",
        "do",
        "return",
        "def",
        "function",
        "func",
        "fn",
        "class",
        "struct",
        "interface",
        "enum",
        "trait",
        "impl",
        "try",
        "except",
        "catch",
        "finally",
        "raise",
        "throw",
        "throws",
        "new",
        "delete",
        "await",
        "async",
        "yield",
        "in",
        "of",
        "not",
        "and",
        "or",
        "is",
        "none",
        "null",
        "nil",
        "true",
        "false",
        "self",
        "this",
        "super",
        "const",
        "let",
        "var",
        "val",
        "mut",
        "static",
        "public",
        "private",
        "protected",
        "switch",
        "case",
        "default",
        "break",
        "continue",
        "goto",
        "with",
        "as",
        "import",
        "from",
        "package",
        "lambda",
        "go",
        "defer",
        "select",
        "chan",
        "map",
        "range",
        "match",
        "when",
        "where",
        "loop",
        "pass",
        "assert",
        "typeof",
        "instanceof",
        "void",
        "sizeof",
        "unless",
        "until",
        "begin",
        "end",
        "then",
        "module",
        "record",
        "sealed",
        "override",
        "virtual",
        "abstract",
        "final",
        "using",
        "namespace",
    ]
)
_COMMENT_FAMILY = {"python": "python", "ruby": "ruby", "php": "php"}


@dataclass(frozen=True, slots=True)
class Fingerprint:
    qname: str
    path: str
    line_start: int
    line_end: int
    shingles: frozenset[int]
    signature: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ClonePair:
    a: str
    b: str
    score: float


def normalise_tokens(body: str, family: str) -> list[str]:
    """Body text -> normalised token stream (comments and docstrings removed)."""
    spans = comment_spans(body, _COMMENT_FAMILY.get(family, family))
    if spans:
        pieces: list[str] = []
        cursor = 0
        for start, end in spans:
            pieces.append(body[cursor:start])
            pieces.append(" ")
            cursor = end
        pieces.append(body[cursor:])
        body = "".join(pieces)
    tokens: list[str] = []
    for match in _TOKEN.finditer(body):
        token = match.group(0)
        first = token[0]
        if first in "\"'`":
            tokens.append("S")
        elif first.isdigit():
            tokens.append("N")
        elif first.isalpha() or first in "_$":
            tokens.append(token if token.lower() in KEYWORDS else "I")
        else:
            tokens.append(token)
    return tokens


class ShingleHasher:
    """Stable 64-bit shingle hashes, cached because the normalised vocabulary is small."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, ...], int] = {}

    def __call__(self, shingle: tuple[str, ...]) -> int:
        value = self._cache.get(shingle)
        if value is None:
            digest = hashlib.blake2b("\x1f".join(shingle).encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            self._cache[shingle] = value
        return value


def shingle_set(tokens: list[str], hasher: ShingleHasher) -> frozenset[int]:
    found: set[int] = set()
    for index in range(len(tokens) - SHINGLE + 1):
        window = tuple(tokens[index : index + SHINGLE])
        if all(t in {"I", "N", "S"} for t in window):
            continue
        found.add(hasher(window))
    return frozenset(found)


def signature(shingles: Iterable[int]) -> tuple[int, ...]:
    """One-permutation MinHash with rotation densification."""
    bins = [_EMPTY] * K
    for value in shingles:
        slot = value % K
        rest = value // K
        if rest < bins[slot]:
            bins[slot] = rest
    if all(b == _EMPTY for b in bins):
        return tuple(bins)
    dense = list(bins)
    for index in range(K):
        if bins[index] != _EMPTY:
            continue
        distance = 1
        while bins[(index + distance) % K] == _EMPTY:
            distance += 1
        dense[index] = bins[(index + distance) % K] + distance * _ROTATION
    return tuple(dense)


def fingerprint(
    qname: str,
    path: str,
    line_start: int,
    line_end: int,
    body: str,
    family: str,
    hasher: ShingleHasher,
) -> Fingerprint | None:
    """None for bodies too small to say anything about."""
    tokens = normalise_tokens(body, family)
    if len(tokens) < MIN_TOKENS:
        return None
    shingles = shingle_set(tokens, hasher)
    if len(shingles) < MIN_SHINGLES:
        return None
    return Fingerprint(qname, path, line_start, line_end, shingles, signature(shingles))


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _nested(a: Fingerprint, b: Fingerprint) -> bool:
    if a.path != b.path:
        return False
    return (a.line_start <= b.line_start and b.line_end <= a.line_end) or (
        b.line_start <= a.line_start and a.line_end <= b.line_end
    )


def find_clones(
    prints: list[Fingerprint],
    threshold: float = DEFAULT_THRESHOLD,
    max_per_node: int = MAX_PER_NODE,
) -> list[ClonePair]:
    """Pairs with exact Jaccard >= ``threshold``; each function keeps its best few."""
    buckets: defaultdict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for index, item in enumerate(prints):
        for band in range(BANDS):
            key = (band, item.signature[band * ROWS : (band + 1) * ROWS])
            buckets[key].append(index)
    candidates: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2 or len(members) > MAX_BUCKET:
            continue
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                candidates.add((left, right) if left < right else (right, left))
    scored: list[ClonePair] = []
    for left, right in candidates:
        a, b = prints[left], prints[right]
        small, large = sorted((len(a.shingles), len(b.shingles)))
        if small < threshold * large or _nested(a, b):
            continue
        score = jaccard(a.shingles, b.shingles)
        if score >= threshold:
            first, second = sorted((a.qname, b.qname))
            scored.append(ClonePair(first, second, round(score, 3)))
    scored.sort(key=lambda p: (-p.score, p.a, p.b))
    kept: list[ClonePair] = []
    per_node: defaultdict[str, int] = defaultdict(int)
    for pair in scored:
        if per_node[pair.a] >= max_per_node or per_node[pair.b] >= max_per_node:
            continue
        per_node[pair.a] += 1
        per_node[pair.b] += 1
        kept.append(pair)
    return kept


__all__ = [
    "ClonePair",
    "Fingerprint",
    "ShingleHasher",
    "find_clones",
    "fingerprint",
    "jaccard",
    "normalise_tokens",
    "shingle_set",
    "signature",
]
