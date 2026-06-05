"""Phase-3 grounding: what raw signals does each source actually give us?

Engagement scoring (community-health, visibility, effort, velocity) can only use
fields that are actually populated. This audits per-source coverage + value
ranges of every scoring-relevant field, and enumerates the keys present in the
per-source `signals` JSON blob, so the Phase-3 feature engineering is grounded
in what exists rather than what we wish existed.
"""
from __future__ import annotations

import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "engageiq.sqlite"

CORE_FIELDS = [
    "score", "num_comments", "author_reputation", "community",
    "community_size", "last_activity_at", "language",
]


def pct(n: int, d: int) -> str:
    return f"{(100*n/d):5.1f}%" if d else "  n/a"


def main() -> None:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    cols = [r[1] for r in conn.execute("PRAGMA table_info(opportunities)")]
    print("=== columns ===")
    print(", ".join(cols))

    total = conn.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0]
    print(f"\n=== total records: {total} ===")

    sources = [r[0] for r in conn.execute(
        "SELECT source FROM opportunities GROUP BY source ORDER BY COUNT(*) DESC")]

    for src in sources:
        rows = conn.execute("SELECT * FROM opportunities WHERE source=?", (src,)).fetchall()
        n = len(rows)
        print(f"\n{'='*70}\n{src.upper()}  ({n} records)\n{'='*70}")

        # core-field coverage
        print("-- core-field coverage --")
        for f in CORE_FIELDS:
            present = sum(1 for r in rows if r[f] not in (None, "", 0))
            print(f"   {f:20} {pct(present, n)}  ({present})")
        # tags non-empty
        tag_present = sum(1 for r in rows if r["tags"] and r["tags"] not in ("[]", "null"))
        print(f"   {'tags(non-empty)':20} {pct(tag_present, n)}  ({tag_present})")

        # numeric ranges for the popularity signals
        for f in ("score", "num_comments", "community_size", "author_reputation"):
            vals = [r[f] for r in rows if isinstance(r[f], (int, float)) and r[f] is not None]
            if vals:
                vals_sorted = sorted(vals)
                p50 = statistics.median(vals)
                p90 = vals_sorted[int(0.9 * (len(vals_sorted) - 1))]
                print(f"   {f:20} min={min(vals):<8} p50={p50:<10} p90={p90:<10} max={max(vals)}")

        # signals JSON keys
        key_counts: Counter = Counter()
        sample_vals: dict = defaultdict(list)
        for r in rows:
            try:
                s = json.loads(r["signals"]) if r["signals"] else {}
            except Exception:  # noqa: BLE001
                s = {}
            for k, v in s.items():
                key_counts[k] += 1
                if len(sample_vals[k]) < 3:
                    sample_vals[k].append(v)
        if key_counts:
            print("-- signals[] keys (coverage, sample values) --")
            for k, c in key_counts.most_common():
                print(f"   {k:22} {pct(c, n)}  e.g. {sample_vals[k]}")

        # opportunity_type split
        types = Counter(r["opportunity_type"] for r in rows)
        print(f"-- opportunity_type: {dict(types)}")

        # created_at range
        dates = sorted(r["created_at"] for r in rows if r["created_at"])
        if dates:
            print(f"-- created_at: {dates[0][:10]} .. {dates[-1][:10]}")


if __name__ == "__main__":
    main()
