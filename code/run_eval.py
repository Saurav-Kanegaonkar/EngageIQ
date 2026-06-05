"""Run the embedding/retrieval quality eval on the cached embeddings.

Usage: python code/run_eval.py   (after code/engageiq/embed.py has been run)
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `engageiq` importable

from engageiq.embed import DB_PATH, encode, load_cached  # noqa: E402
from engageiq.eval import embedding_eval as E            # noqa: E402


def main() -> None:
    ids, embs = load_cached()
    conn = sqlite3.connect(str(DB_PATH))
    dmap = dict(conn.execute("SELECT opportunity_id, domain FROM opportunities"))
    labels = np.array([dmap.get(i, "?") for i in ids])
    print(f"records: {len(ids)} | dims: {embs.shape[1]}")

    # silhouette is O(n^2) -> sample for the separation metric (statistical, sample is fine)
    rng = np.random.default_rng(0)
    samp = rng.choice(len(ids), size=min(4000, len(ids)), replace=False)
    print("\n== domain separation (sample 4000) ==")
    sep = E.domain_separation(embs[samp], list(labels[samp]))
    for k, v in sep.items():
        print(f"  {k}: {v:.4f}")

    print("\n== retrieval precision@10 (15 per-domain queries) ==")
    r = E.retrieval_precision_at_k(embs, list(labels), encode, k=10)
    print(f"  MEAN P@10: {r['mean_precision_at_k']:.3f}")
    for d, info in r["per_query"].items():
        print(f"    {d:22} P@10={info['precision_at_k']:.2f}")

    print("\n== nearest-neighbor coherence (sample 1000) ==")
    print(f"  {E.nearest_neighbor_coherence(embs, list(labels), sample=1000):.3f}")

    print("\n== near-duplicates ==")
    for th in (0.97, 0.95, 0.90):
        print(f"  cos>={th}: {len(E.find_near_duplicates(embs, ids, threshold=th))} pairs")


if __name__ == "__main__":
    main()
