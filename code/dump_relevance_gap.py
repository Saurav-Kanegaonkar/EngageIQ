"""Final comprehensive coverage pass: dump the RELEVANCE-ELIGIBLE pool per domain
(top-N retrieved for each domain), not just what specific synthetic profiles surface.

Why this basis: a profile's surfaced cards are a subset of items RELEVANT to its
domains, so covering the top-N relevant items per domain covers most of what any
realistic (even rich-goal) profile can surface, which combo sweeps miss. The data is
static, so this is a one-time fill. Skips anything already summarized.

Run: PYTHONPATH=code python -m dump_relevance_gap
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
from engageiq.retrieve import Retriever

PER_DOMAIN = 150   # top-N relevant items per domain to ensure coverage
CHUNK_SIZE = 15


def main() -> None:
    summ = summarize.Summarizer()
    conn = sqlite3.connect(str(DB_PATH))
    domains = [d for (d,) in conn.execute(
        "SELECT DISTINCT domain FROM opportunities WHERE domain IS NOT NULL")]

    r = Retriever()
    pool: set[str] = set()
    for d in domains:
        for oid, _score in r.search(d.replace("_", " "), k=PER_DOMAIN):
            pool.add(oid)

    miss = [o for o in pool if o not in summ.cache]   # only what we haven't summarized

    allrows = {}
    for oid, src, title, body in conn.execute(
            "SELECT opportunity_id, source, title, body FROM opportunities"):
        allrows[oid] = {"oid": oid, "title": (title or "").strip(),
                        "source": src, "body": (body or "")[:1500]}
    conn.close()
    items = [allrows[o] for o in miss if o in allrows]

    cdir, rdir = Path("data/summ_chunks"), Path("data/summ_results")
    for dd in (cdir, rdir):
        if dd.exists():
            shutil.rmtree(dd)
        dd.mkdir(parents=True, exist_ok=True)
    n = math.ceil(len(items) / CHUNK_SIZE) if items else 0
    for i in range(n):
        (cdir / f"chunk_{i}.json").write_text(
            json.dumps(items[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE], indent=2))

    print(json.dumps({"pool": len(pool), "uncovered": len(items), "chunks": n}))


if __name__ == "__main__":
    main()
