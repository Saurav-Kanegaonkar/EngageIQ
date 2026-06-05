"""Per-opportunity signal engineering + cross-source normalization (Phase 3).

The spec scores engagement value on four signals: relevance, community-health,
visibility, and effort. Three of those are PERSONA-INDEPENDENT properties of the
opportunity itself, so we precompute them once for all records and cache them in
the `opportunity_features` table. Relevance (cosine) is persona-specific and is
added at rank time.

The hard problem is that raw counts are INCOMPARABLE across sources (HN points
max 1541, Bluesky likes 191, Dev.to reactions 67, GitHub has stars instead of a
post score). We solve it with PER-SOURCE PERCENTILE normalization: a value maps
to its empirical CDF within its own source, so "top 10% of HN by points" and
"top 10% of Bluesky by likes" both become ~0.9 and are directly comparable.

Signals produced (all in [0,1] unless noted):
  popularity  - per-source percentile of the primary engagement count
                (GitHub: repo stars; HN: points; Bluesky: likes; Dev.to: reactions)
  discussion  - per-source percentile of comment/reply count (live conversation)
  velocity    - per-source percentile of engagement-rate
                (GitHub: stars/day; else: score / age_days) -> Lina's signal
  recency     - exponential decay on age (half-life 30d)
  standout    - GitHub: (open_issues + comments) / contributors, percentiled
                -> David's "few contributors = room to stand out"; 0.5 elsewhere
  openness    - GitHub: 1 if issue open & repo not archived else 0; 1 elsewhere
  community_health - composite(discussion, recency, velocity) gated by openness
  visibility       - composite(popularity, velocity)
  effort_min       - raw minutes (effort.py); effort = normalized [0,1]
"""
from __future__ import annotations

import bisect
import json
import math
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import effort

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "engageiq.sqlite"
RECENCY_HALFLIFE_DAYS = 30.0
NOW = datetime.now(timezone.utc)

# the signals we percentile-normalize, per source
PCT_FIELDS = ("pop", "disc", "vel", "standout")


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def age_days(iso: str | None) -> float:
    if not iso:
        return 9999.0
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return max(0.0, (NOW - dt).total_seconds() / 86400.0)
    except Exception:  # noqa: BLE001
        return 9999.0


def recency_decay(age: float) -> float:
    return math.exp(-math.log(2) * age / RECENCY_HALFLIFE_DAYS)


def _repo_reputation(stars) -> float:
    """Sweet-spot on repo stars (research: popularity is recognition, not learnability,
    and hyper-popular repos preempt newcomers). ~0 for random/abandoned repos, rising
    for established projects, a mild crowding penalty for the giants. Not a popularity sort."""
    s = float(stars or 0)
    if s < 1:
        return 0.0
    base = min(1.0, math.log10(s + 1) / math.log10(5000.0))   # ~1.0 around 5k stars
    if s > 50000:
        base *= 0.85                                          # crowding / preemption penalty
    return _clamp(base)


def _issue_quality(body: str | None, source: str, typ: str | None) -> float:
    """Issue write-up quality, the research's STRONG newcomer-suitability signal
    (vs the weak 'good first issue' label). Rewards a well-specified issue."""
    if source != "github" or not body:
        return 0.5
    b = body.lower()
    q = min(0.35, len(body) / 1200.0 * 0.35)                  # enough detail
    if "```" in body:
        q += 0.15                                             # code block
    if re.search(r"(?m)^\s*(\d+\.|[-*])\s", body):
        q += 0.15                                             # numbered / bulleted steps
    if re.search(r"https?://", body):
        q += 0.10                                             # links / references
    if any(k in b for k in ("reproduce", "steps to", "expected", "actual", "screenshot", "example")):
        q += 0.15                                             # repro / expected-behaviour
    if any(k in b for k in ("good first issue", "beginner", "newcomer", "help wanted")):
        q += 0.10
    return _clamp(q)


def raw_signals(r: dict) -> dict:
    """The raw (un-normalized) inputs we percentile within each source."""
    src = r["source"]
    sig = r.get("_signals") or {}
    age = age_days(r.get("created_at"))
    score = r.get("score") or 0
    pop = (r.get("community_size") if src == "github" else score) or 0     # stars vs primary count
    if src == "github":
        vel = sig.get("star_velocity")
    else:
        vel = score / max(age, 1.0)
    disc = r.get("num_comments") or 0
    standout = None
    if src == "github":
        contrib = sig.get("contributors")
        openi = sig.get("open_issues") or 0
        if contrib:
            standout = (openi + disc) / max(contrib, 1)
    return {"pop": float(pop), "disc": float(disc),
            "vel": (float(vel) if vel is not None else None),
            "standout": standout}


class SourceNormalizer:
    """Empirical-CDF percentile lookup per (source, field). Fit once over the corpus."""

    def __init__(self) -> None:
        self.tables: dict[tuple[str, str], list[float]] = {}

    def fit(self, recs: list[dict]) -> "SourceNormalizer":
        buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in recs:
            for f, v in raw_signals(r).items():
                if v is not None:
                    buckets[(r["source"], f)].append(float(v))
        self.tables = {k: sorted(v) for k, v in buckets.items()}
        return self

    def pct(self, source: str, field: str, value: float | None) -> float:
        """Empirical-CDF percentile = fraction of the source's records this value
        STRICTLY beats. bisect_left (not _right) is deliberate: with these heavily
        zero-inflated signals, a 0 should map to ~0 (beats nobody), not to the top
        of the tied-zero block. So zero engagement -> ~0, real engagement climbs."""
        if value is None:
            return 0.0
        arr = self.tables.get((source, field))
        if not arr:
            return 0.0
        return bisect.bisect_left(arr, float(value)) / len(arr)


def compute_features(r: dict, norm: SourceNormalizer) -> dict:
    src = r["source"]
    sig = r.get("_signals") or {}
    age = age_days(r.get("created_at"))
    raw = raw_signals(r)

    popularity = norm.pct(src, "pop", raw["pop"])
    discussion = norm.pct(src, "disc", raw["disc"])
    velocity = norm.pct(src, "vel", raw["vel"])
    recency = recency_decay(age)

    if src == "github":
        standout = norm.pct(src, "standout", raw["standout"]) if raw["standout"] is not None else 0.5
        archived = bool(sig.get("archived"))
        state = (sig.get("state") or "open")
        openness = 0.0 if (archived or state != "open") else 1.0
    else:
        standout = 0.5            # neutral: no contributor data off GitHub
        openness = 1.0

    health = _clamp(0.45 * discussion + 0.30 * recency + 0.25 * velocity)
    health *= (1.0 if openness else 0.4)         # closed/archived GitHub = stale
    visibility = _clamp(0.6 * popularity + 0.4 * velocity)

    # contribution-worthiness (research-backed): is this a genuinely worthwhile repo to
    # contribute to, not a random/abandoned personal repo? Issue quality + a responsiveness
    # proxy (community_health) are weighted ABOVE raw reputation, per the evidence
    # (responsiveness > popularity; issue write-up quality > the 'good first issue' label).
    if src == "github":
        repo_rep = _repo_reputation(r.get("community_size"))
        issue_q = _issue_quality(r.get("body"), src, r.get("opportunity_type"))
        contrib_worth = _clamp(0.30 * repo_rep + 0.40 * issue_q + 0.30 * health) * (1.0 if openness else 0.5)
    else:
        repo_rep, issue_q, contrib_worth = 0.5, 0.5, 0.5      # neutral off GitHub

    eff_min = effort.heuristic_minutes(r)
    return {
        "popularity": round(popularity, 4), "discussion": round(discussion, 4),
        "velocity": round(velocity, 4), "recency": round(recency, 4),
        "standout": round(standout, 4), "openness": openness,
        "community_health": round(health, 4), "visibility": round(visibility, 4),
        "repo_reputation": round(repo_rep, 4), "issue_quality": round(issue_q, 4),
        "contribution_worth": round(contrib_worth, 4),
        "effort_min": round(eff_min, 1), "effort": round(effort.normalized_effort(eff_min), 4),
    }


# ── corpus IO ─────────────────────────────────────────────────────────────
_LOAD_COLS = ("opportunity_id, source, opportunity_type, domain, domains, language, "
              "created_at, last_activity_at, score, num_comments, community_size, "
              "tags, signals, body")


def _row_to_rec(row: tuple) -> dict:
    (oid, src, typ, dom, doms, lang, created, last_act, score, ncom, csize,
     tags, signals, body) = row
    try:
        _tags = json.loads(tags) if tags else []
    except Exception:  # noqa: BLE001
        _tags = []
    try:
        _sig = json.loads(signals) if signals else {}
    except Exception:  # noqa: BLE001
        _sig = {}
    try:
        _doms = json.loads(doms) if doms else [dom]
    except Exception:  # noqa: BLE001
        _doms = [dom]
    return {"opportunity_id": oid, "source": src, "opportunity_type": typ,
            "domain": dom, "domains": _doms, "language": lang,
            "created_at": created, "last_activity_at": last_act,
            "score": score, "num_comments": ncom, "community_size": csize,
            "body": body or "", "_tags": _tags, "_signals": _sig}


def load_all(conn: sqlite3.Connection) -> list[dict]:
    return [_row_to_rec(r) for r in conn.execute(f"SELECT {_LOAD_COLS} FROM opportunities")]


FEATURE_COLS = ("popularity", "discussion", "velocity", "recency", "standout",
                "openness", "community_health", "visibility", "repo_reputation",
                "issue_quality", "contribution_worth", "effort_min", "effort")


def build_and_store() -> None:
    conn = sqlite3.connect(str(DB_PATH))
    recs = load_all(conn)
    norm = SourceNormalizer().fit(recs)

    conn.execute("DROP TABLE IF EXISTS opportunity_features")
    conn.execute(
        "CREATE TABLE opportunity_features (opportunity_id TEXT PRIMARY KEY, "
        + ", ".join(f"{c} REAL" for c in FEATURE_COLS) + ")")
    rows = []
    for r in recs:
        f = compute_features(r, norm)
        rows.append((r["opportunity_id"], *[f[c] for c in FEATURE_COLS]))
    conn.executemany(
        f"INSERT OR REPLACE INTO opportunity_features (opportunity_id, {', '.join(FEATURE_COLS)}) "
        f"VALUES ({', '.join(['?'] * (len(FEATURE_COLS) + 1))})", rows)
    conn.commit()
    print(f"stored features for {len(rows)} records")

    print("\n-- mean signal by source --")
    print(f"{'source':12} " + " ".join(f"{c[:8]:>8}" for c in
          ("popularity", "discussion", "velocity", "recency", "comm_health", "visibility", "effort_min")))
    for src in [r[0] for r in conn.execute(
            "SELECT source FROM opportunities GROUP BY source ORDER BY COUNT(*) DESC")]:
        q = conn.execute(
            "SELECT AVG(f.popularity), AVG(f.discussion), AVG(f.velocity), AVG(f.recency), "
            "AVG(f.community_health), AVG(f.visibility), AVG(f.effort_min) "
            "FROM opportunity_features f JOIN opportunities o USING(opportunity_id) "
            "WHERE o.source=?", (src,)).fetchone()
        print(f"{src:12} " + " ".join(f"{v:8.3f}" if v is not None else f"{'-':>8}" for v in q))


def load_features(conn: sqlite3.Connection) -> dict[str, dict]:
    cols = ", ".join(FEATURE_COLS)
    out = {}
    for row in conn.execute(f"SELECT opportunity_id, {cols} FROM opportunity_features"):
        out[row[0]] = dict(zip(FEATURE_COLS, row[1:]))
    return out


if __name__ == "__main__":
    build_and_store()
