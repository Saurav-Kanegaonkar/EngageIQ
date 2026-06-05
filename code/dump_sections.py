"""Dump the existing paragraph summaries (data/summaries.json) into chunks, so a
Workflow of Claude agents can split each into labeled sections (what / task / do /
gain) for the card's micro-section layout. Reuses the existing summaries (quality
preserved); the agents only re-shape them. Run: PYTHONPATH=code python -m dump_sections
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from engageiq.summarize import CACHE_PATH


def main() -> None:
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    items = [{"oid": k, "summary": v} for k, v in cache.items()
             if isinstance(v, str) and v.strip()]
    cdir, rdir = Path("data/sec_chunks"), Path("data/sec_results")
    for d in (cdir, rdir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    n = max(1, min(16, math.ceil(len(items) / 26))) if items else 0
    chunks = [items[i::n] for i in range(n)] if n else []
    for i, c in enumerate(chunks):
        (cdir / f"chunk_{i}.json").write_text(json.dumps(c, indent=2))
    print(json.dumps({"items": len(items), "chunks": n}))


if __name__ == "__main__":
    main()
