"""Merge the summarize-cards sub-agent outputs (data/summ_results/chunk_*.json, each
{oid: {summary, what, task, do, gain}}) into BOTH caches the API reads:
  - data/summaries.json         (the paragraph `summary`)
  - data/summary_sections.json  (the labeled {what, task, do, gain})
so cards render the full WHAT / THE TASK / YOU'D DO / YOU'D GAIN rows.

Run after the summarize-cards Workflow finishes:
  PYTHONPATH=code .venv/bin/python -m merge_card_summaries
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from engageiq.embed import DB_PATH
from engageiq.summarize import CACHE_PATH, SECTIONS_PATH

RESULTS = Path("data/summ_results")
_SEC_KEYS = ("what", "task", "do", "gain")


def clean(s: str, cap: int) -> str:
    """House style: no markdown, no dashes, single-spaced, capped."""
    s = re.sub(r"[*#`_]+", "", s or "")
    s = s.replace(" — ", ", ").replace("—", ", ").replace(" – ", ", ").replace("–", "-")
    return " ".join(s.split())[:cap]


def main() -> None:
    summ = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    sec = json.loads(SECTIONS_PATH.read_text()) if SECTIONS_PATH.exists() else {}
    before = len(summ)
    add_summ = add_sec = bad = 0

    for f in sorted(RESULTS.glob("chunk_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f.name}: {e}")
            bad += 1
            continue
        for oid, obj in (d.items() if isinstance(d, dict) else []):
            if not isinstance(obj, dict):
                continue
            para = clean(obj.get("summary", ""), 650)
            if para and oid not in summ:
                summ[oid] = para
                add_summ += 1
            secobj = {k: clean(obj.get(k, ""), 240) for k in _SEC_KEYS}
            if any(secobj.values()):
                sec[oid] = secobj
                add_sec += 1

    CACHE_PATH.write_text(json.dumps(summ))
    SECTIONS_PATH.write_text(json.dumps(sec))

    # per-source coverage report (on the sections cache = what the card shows)
    conn = sqlite3.connect(str(DB_PATH))
    bysrc = Counter()
    sec_ids = set(sec)
    for oid, src in conn.execute("SELECT opportunity_id, source FROM opportunities"):
        if oid in sec_ids:
            bysrc[src] += 1
    conn.close()

    print(f"merged: +{add_summ} summaries ({before} -> {len(summ)}), +{add_sec} section sets "
          f"(-> {len(sec)} total){f', {bad} bad files' if bad else ''}")
    print("cached SECTIONS by source: " + ", ".join(f"{s}={bysrc[s]}" for s in
          ["github", "devto", "bluesky", "reddit", "hackernews"]))


if __name__ == "__main__":
    main()
