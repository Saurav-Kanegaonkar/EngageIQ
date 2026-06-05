"""Dump EVERY opportunity not yet summarized into chunk files, so a Workflow of
Claude sub-agents can summarize the WHOLE corpus. Once every item has a summary,
coverage is 100% for any profile (no need to predict what a profile surfaces).

Chunk size is tuned so the agent count stays well under the Workflow's 1000-agent
cap. The data is static, so this is a one-time fill (re-run if we re-scrape).

Run: PYTHONPATH=code python -m dump_all_remaining
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import math
import shutil
import sqlite3
from pathlib import Path

from engageiq import summarize
from engageiq.embed import DB_PATH

CHUNK_SIZE = 20   # 20/agent keeps ~880 agents for ~17.6k items, under the 1000 cap


def main() -> None:
    summ = summarize.Summarizer()
    conn = sqlite3.connect(str(DB_PATH))
    items = []
    for oid, src, title, body in conn.execute(
            "SELECT opportunity_id, source, title, body FROM opportunities"):
        if oid in summ.cache:
            continue
        items.append({"oid": oid, "title": (title or "").strip(),
                      "source": src, "body": (body or "")[:1500]})
    conn.close()

    cdir, rdir = Path("data/summ_chunks"), Path("data/summ_results")
    for dd in (cdir, rdir):
        if dd.exists():
            shutil.rmtree(dd)
        dd.mkdir(parents=True, exist_ok=True)
    n = math.ceil(len(items) / CHUNK_SIZE) if items else 0
    for i in range(n):
        (cdir / f"chunk_{i}.json").write_text(
            json.dumps(items[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE], indent=2))

    print(json.dumps({"remaining": len(items), "chunks": n,
                      "chunk_size": CHUNK_SIZE, "already_cached": len(summ.cache)}))


if __name__ == "__main__":
    main()
