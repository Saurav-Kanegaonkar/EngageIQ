"""Probabilistic data-structure sketches, built from scratch (BAX-423 Lecture 2).

Two streaming/sketch algorithms the analytics batch (capability 5) uses to compute
trend insights over the full 20,995-record corpus in sub-linear memory:

  CountMinSketch  approximate frequency of every term in a stream, in fixed memory
                  -> "trending topics": the most frequent terms without a giant dict.
  HyperLogLog     approximate count of DISTINCT items in a stream, in ~m bytes
                  -> "reach": how many distinct authors/communities a domain touches.

Both are implemented here with only the standard library + numpy as a flat array
backing store (the algorithm, hashing, and estimators are hand-written, not a
library call). Each ships a benchmark against the exact answer (accuracy + memory),
which capability 5 prints into reports/phase5_analytics.txt.

References: Cormode & Muthukrishnan (Count-Min, 2005); Flajolet et al. (HyperLogLog,
2007). Hashing uses blake2b with the Kirsch-Mitzenmacher double-hashing trick so we
pay one real hash per item and derive the rest arithmetically.
"""
from __future__ import annotations

import hashlib
import math
import sys

import numpy as np

_MASK64 = (1 << 64) - 1


def _hash128(item) -> tuple[int, int]:
    """One blake2b digest -> two independent 64-bit hashes (h1, h2).

    Kirsch-Mitzenmacher: the i-th hash is (h1 + i*h2), so d hash functions cost one
    real hash. blake2b is fast, keyless here, and well distributed."""
    b = item if isinstance(item, bytes) else str(item).encode("utf-8")
    d = hashlib.blake2b(b, digest_size=16).digest()
    h1 = int.from_bytes(d[:8], "little")
    h2 = int.from_bytes(d[8:], "little")
    return h1, (h2 | 1)        # h2 odd -> better spread across the table width


# ── Count-Min Sketch ──────────────────────────────────────────────────────────
class CountMinSketch:
    """Approximate frequency counts in fixed memory (depth x width int32 grid).

    Guarantees, for total mass N: estimate >= true count always (never undercounts),
    and estimate <= true + eps*N with probability >= 1 - delta, when
    width = ceil(e/eps) and depth = ceil(ln(1/delta)). Collisions only ever ADD, so
    the per-row minimum is the tightest estimate. Memory is width*depth*4 bytes,
    independent of how many distinct keys are seen.
    """

    def __init__(self, width: int = 2048, depth: int = 5):
        self.width = int(width)
        self.depth = int(depth)
        self.table = np.zeros((self.depth, self.width), dtype=np.int64)
        self.total = 0

    @classmethod
    def from_error(cls, eps: float = 0.001, delta: float = 0.01) -> "CountMinSketch":
        """Size the sketch from target error eps and failure prob delta."""
        width = max(2, math.ceil(math.e / eps))
        depth = max(1, math.ceil(math.log(1.0 / delta)))
        return cls(width=width, depth=depth)

    def add(self, key, count: int = 1) -> None:
        h1, h2 = _hash128(key)
        for i in range(self.depth):
            self.table[i, (h1 + i * h2) % self.width] += count
        self.total += count

    def estimate(self, key) -> int:
        h1, h2 = _hash128(key)
        return int(min(self.table[i, (h1 + i * h2) % self.width] for i in range(self.depth)))

    def memory_bytes(self) -> int:
        return int(self.table.nbytes)


# ── HyperLogLog ───────────────────────────────────────────────────────────────
# Bias-correction constant alpha_m (Flajolet et al.), with the small-m specials.
def _alpha(m: int) -> float:
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / m)


class HyperLogLog:
    """Approximate distinct-count (cardinality) in m = 2^p bytes.

    Each item hashes to 64 bits: the top p bits pick a register, and we store the
    max position of the leftmost 1-bit in the remaining bits. The harmonic mean of
    2^register across all registers estimates the cardinality. Standard error is
    ~1.04/sqrt(m); p=14 -> 16384 registers -> ~0.8% error in ~16 KB, regardless of
    whether the true count is hundreds or millions.
    """

    def __init__(self, p: int = 14):
        if not (4 <= p <= 18):
            raise ValueError("p must be in [4, 18]")
        self.p = p
        self.m = 1 << p
        self.registers = np.zeros(self.m, dtype=np.uint8)
        self._bits = 64 - p

    def add(self, item) -> None:
        h = _hash128(item)[0] & _MASK64
        idx = h >> self._bits                     # top p bits -> register index
        w = (h << self.p) & _MASK64               # remaining bits, left-aligned in 64-bit field
        # rank = position of the leftmost 1-bit (1-indexed from the MSB) in the window;
        # if the window is all zeros, it is the deepest possible run, (64 - p) + 1.
        rank = (64 - w.bit_length() + 1) if w else (self._bits + 1)
        if rank > self.registers[idx]:
            self.registers[idx] = rank

    def count(self) -> int:
        m = self.m
        reg = self.registers.astype(np.float64)
        z = 1.0 / np.sum(np.power(2.0, -reg))
        est = _alpha(m) * m * m * z
        if est <= 2.5 * m:                        # small-range: linear counting
            zeros = int(np.count_nonzero(self.registers == 0))
            if zeros:
                est = m * math.log(m / zeros)
        return int(round(est))

    def memory_bytes(self) -> int:
        return int(self.registers.nbytes)


# ── benchmarks (accuracy + memory vs the exact answer) ────────────────────────
def benchmark_cms(items: list, eps: float = 0.001, delta: float = 0.01, top: int = 20) -> dict:
    """Count a stream both exactly (dict) and with a Count-Min Sketch, and report
    the mean/max absolute error on the true top-`top` terms plus the memory each used."""
    from collections import Counter

    exact = Counter(items)
    cms = CountMinSketch.from_error(eps=eps, delta=delta)
    for it in items:
        cms.add(it)
    top_terms = exact.most_common(top)
    errs = [abs(cms.estimate(t) - c) for t, c in top_terms]
    # exact dict cost: one (str key, int value, table slot) per DISTINCT key. A CPython
    # str is ~49 bytes + length; a small int ~28 bytes; an open-addressed slot ~16 bytes.
    avg_key = (sum(len(str(k)) for k in exact) / len(exact)) if exact else 0
    exact_mem = sys.getsizeof(exact) + int(len(exact) * (avg_key + 49 + 28 + 16))
    return {
        "distinct_keys": len(exact),
        "stream_len": len(items),
        "width": cms.width, "depth": cms.depth,
        "mean_abs_err": round(float(np.mean(errs)), 3) if errs else 0.0,
        "max_abs_err": int(max(errs)) if errs else 0,
        "max_rel_err": round(max((abs(cms.estimate(t) - c) / c) for t, c in top_terms), 4) if top_terms else 0.0,
        "cms_bytes": cms.memory_bytes(),
        "exact_bytes": int(exact_mem),
        "compression": round(exact_mem / max(1, cms.memory_bytes()), 1),
    }


def benchmark_hll(items: list, p: int = 14) -> dict:
    """Count distinct items both exactly (set) and with HyperLogLog, and report the
    relative error and the memory each used."""
    exact = set(items)
    hll = HyperLogLog(p=p)
    for it in items:
        hll.add(it)
    true_n = len(exact)
    est = hll.count()
    exact_mem = sys.getsizeof(exact) + sum(sys.getsizeof(x) for x in list(exact)[:5000])
    return {
        "true_distinct": true_n,
        "estimate": est,
        "rel_err": round(abs(est - true_n) / max(1, true_n), 4),
        "p": p, "registers": hll.m,
        "hll_bytes": hll.memory_bytes(),
        "exact_bytes": int(exact_mem),
        "compression": round(exact_mem / max(1, hll.memory_bytes()), 1),
    }
