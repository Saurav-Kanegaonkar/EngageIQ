"""Tests for capabilities 5 (batch analytics & trend detection) and 6 (dashboard +
engagement brief). Run directly (no pytest needed):

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=code .venv/bin/python code/test_capabilities_5_6.py

Covers: the L2 sketches (accuracy + the merge property the dashboard relies on),
the analytics artifacts (trends.json + coords schema + benchmark bounds), and, if the
API is running on :8000, the /api/trends and /api/brief endpoints.
"""
from __future__ import annotations

import json
import os
import random
import urllib.request
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from sketches import CountMinSketch, HyperLogLog, benchmark_cms, benchmark_hll

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


# ── L2 sketches ───────────────────────────────────────────────────────────────
def test_sketches() -> None:
    print("L2 sketches (Count-Min + HyperLogLog)")
    rng = random.Random(7)
    stream = [f"t{int(rng.paretovariate(1.25))}" for _ in range(80000)]
    from collections import Counter
    exact = Counter(stream)
    cms = CountMinSketch.from_error(eps=5e-4, delta=0.01)
    for w in stream:
        cms.add(w)
    # never undercounts, and stays within eps*N on the heavy hitters
    eps_bound = 5e-4 * len(stream)
    over = [cms.estimate(t) - c for t, c in exact.most_common(50)]
    check("CMS never undercounts", all(o >= 0 for o in over), f"min={min(over)}")
    check("CMS within eps*N bound", all(o <= eps_bound for o in over), f"max={max(over)} bound={eps_bound:.0f}")

    b = benchmark_cms(stream, eps=5e-4, delta=0.01, top=25)
    check("CMS benchmark reports memory + error", b["cms_bytes"] > 0 and b["max_rel_err"] < 0.1, str(b["max_rel_err"]))

    for n in (1000, 20000, 60000):
        items = [f"a{i}" for i in range(n) for _ in range(rng.randint(1, 4))]
        h = benchmark_hll(items, p=14)
        check(f"HLL within 3% at n={n}", h["rel_err"] < 0.03, f"rel_err={h['rel_err']}")

    # the merge property the dashboard uses: union of two domains' registers (max)
    h1, h2 = HyperLogLog(p=14), HyperLogLog(p=14)
    a = {f"u{i}" for i in range(6000)}
    b2 = {f"u{i}" for i in range(4000, 11000)}        # overlaps a
    for x in a:
        h1.add(x)
    for x in b2:
        h2.add(x)
    merged = HyperLogLog(p=14)
    merged.registers = np.maximum(h1.registers, h2.registers)
    true_union = len(a | b2)
    err = abs(merged.count() - true_union) / true_union
    check("HLL register-merge ~ true union", err < 0.03, f"est={merged.count()} true={true_union} err={err:.3f}")


# ── capability 5 artifacts ────────────────────────────────────────────────────
def test_analytics_artifacts() -> None:
    print("Capability 5 artifacts (trends.json + coords)")
    tp = DATA / "trends.json"
    check("trends.json exists", tp.exists())
    if not tp.exists():
        return
    t = json.loads(tp.read_text())
    for k in ("corpus", "global", "domains", "domain_similarity", "topic_map", "benchmarks"):
        check(f"trends.json has '{k}'", k in t)
    check("corpus total == 20995", t["corpus"]["total"] == 20995, str(t["corpus"]["total"]))
    check("all 15 domains present", len(t["domains"]) == 15, str(len(t["domains"])))

    sample = next(iter(t["domains"].values()))
    for k in ("count", "top_terms", "top_communities", "distinct_authors", "weekly_counts",
              "wow", "rising_score", "related"):
        check(f"domain has '{k}'", k in sample)
    check("wow has share momentum", "delta_pp" in sample["wow"] and "share_last" in sample["wow"])

    b = t["benchmarks"]
    check("CMS benchmark present + accurate", b["cms"]["max_rel_err"] < 0.05, str(b["cms"]["max_rel_err"]))
    check("HLL benchmark present + accurate", b["hll"]["rel_err"] < 0.03, str(b["hll"]["rel_err"]))
    check("CMS compresses vs exact", b["cms"]["compression"] > 1, str(b["cms"]["compression"]))
    check("HLL compresses vs exact", b["hll"]["compression"] > 1, str(b["hll"]["compression"]))

    cp = DATA / "trends_coords.npz"
    check("trends_coords.npz exists", cp.exists())
    if cp.exists():
        z = np.load(cp, allow_pickle=True)
        check("coords xy is (N,2)", z["xy"].ndim == 2 and z["xy"].shape[1] == 2, str(z["xy"].shape))
        check("per-domain HLL registers stored (15, m)", z["hll_regs"].shape[0] == 15, str(z["hll_regs"].shape))


# ── endpoints (capability 5 + 6), if the API is up ────────────────────────────
def _post(path: str, body: dict, timeout: int = 60):
    req = urllib.request.Request("http://127.0.0.1:8000" + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("content-type", "")


def test_endpoints() -> None:
    print("Endpoints (/api/trends, /api/brief) — needs the API on :8000")
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/api/personas", timeout=3)
    except Exception:  # noqa: BLE001
        print("  SKIP  API not reachable on :8000")
        return

    raw, _ = _post("/api/trends", {"persona_id": "david"})
    d = json.loads(raw)
    check("trends ok", d.get("ok"))
    check("trends persona-scoped (3 domains)", d["scope"]["n"] == 3, str(d["scope"]))
    for k in ("kpis", "domains", "map_img", "trending_terms", "communities", "weekly", "rising_opps"):
        check(f"trends response has '{k}'", k in d)
    check("trends distinct_reach via HLL > 0", d["kpis"]["distinct_reach"] > 0)
    check("trends map image path", d["map_img"].startswith("/trends_img/"))

    raw, _ = _post("/api/trends", {"persona_id": "lina"})
    dl = json.loads(raw)
    check("breadth persona => all topics", dl["scope"]["breadth"] is True)

    raw, _ = _post("/api/brief", {"persona_id": "david", "session_id": "test"})
    br = json.loads(raw)
    check("brief ok", br.get("ok"))
    check("brief has plan buckets", len(br["plan"]["buckets"]) > 0)
    check("brief has top opportunities", len(br["top"]) >= 5)
    check("brief top item has why", bool(br["top"][0].get("why")))
    has_action = any(o.get("suggested_action") and o["suggested_action"].get("text") for o in br["top"][:3])
    check("brief has >=1 LLM suggested action", has_action)
    check("brief includes trends", br.get("trends") is not None)

    raw, ctype = _post("/api/brief/html", {"persona_id": "david", "session_id": "test"})
    htmldoc = raw.decode("utf-8", "ignore")
    check("brief/html is an HTML doc", htmldoc.lstrip().lower().startswith("<!doctype html"))
    check("brief/html names the persona", "David" in htmldoc)
    check("brief/html is self-contained (inline style)", "<style>" in htmldoc)


if __name__ == "__main__":
    test_sketches()
    test_analytics_artifacts()
    test_endpoints()
    print(f"\n{'=' * 48}\n  {_passed} passed, {_failed} failed\n{'=' * 48}")
    raise SystemExit(1 if _failed else 0)
