"""Demo: profile -> retrieval for each persona (Phase 2 end-to-end sanity check)."""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engageiq.personas import PERSONAS, profile_text  # noqa: E402
from engageiq.retrieve import Retriever                # noqa: E402

r = Retriever()
for _, p in PERSONAS.items():
    print(f"\n=== {p['name']} (budget {p['time_budget_hours']}h/wk) ===")
    for rec in r.recommend(profile_text(p), k=5):
        title = (rec["title"] or rec["body"] or "")[:60].replace("\n", " ")
        print(f"  sim={rec['similarity']:.2f} [{rec['source']:10}] {rec['domain']:16} {title}")
