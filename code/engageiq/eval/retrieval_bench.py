"""Matching-engine benchmark: dense vs sparse vs hybrid vs cross-encoder.

Phase-3 found a TF-IDF keyword baseline beating our MiniLM dense retriever on
NDCG — expected for keyword-rich queries over short jargon text. This script
tests the standard remedies on the SAME cached LLM judgments (no new API calls):

  dense   - MiniLM bi-encoder cosine (current)
  tfidf   - TF-IDF keyword baseline
  hybrid  - reciprocal-rank fusion of dense + tfidf (sparse+dense, the SOTA combo)
  cross   - cross-encoder re-rank (reads query+doc together) over the fused pool

Each method just RE-ORDERS each persona's already-judged pool, so it's a fair,
free, apples-to-apples re-ranking comparison.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .. import personas, rerank
from ..embed import encode
from ..rank import Recommender
from .judge import CACHE_DIR
from .ranking_eval import dcg


def _rrf(rank_a: dict, rank_b: dict, ids: list[str], k: int = 60) -> dict:
    return {i: 1.0 / (k + rank_a[i]) + 1.0 / (k + rank_b[i]) for i in ids}


def ndcg(order: list[str], grades: dict, k: int = 10) -> float:
    idcg = dcg(sorted(grades.values(), reverse=True)[:k])
    return (dcg([grades.get(i, 0) for i in order[:k]]) / idcg) if idcg > 0 else 0.0


def main() -> None:
    rec = Recommender()
    id2row = rec.id2row
    # tf-idf over the full corpus, aligned to embedding id order
    txt = {r[0]: f"{r[1] or ''} {r[2] or ''}" for r in
           rec.conn.execute("SELECT opportunity_id, title, body FROM opportunities")}
    texts = [txt.get(i, "") for i in rec.ret.ids]
    vec = TfidfVectorizer(max_features=40000, stop_words="english", ngram_range=(1, 2))
    tfidf = vec.fit_transform(texts)

    methods = ["dense", "tfidf", "hybrid", "cross"]
    totals = {m: 0.0 for m in methods}
    n = 0
    print(f"{'persona':8} " + " ".join(f"{m:>8}" for m in methods))
    for pid, p in personas.PERSONAS.items():
        cache_path = CACHE_DIR / f"{pid}.json"
        if not cache_path.exists():
            continue
        grades = {k: int(v) for k, v in json.loads(cache_path.read_text()).items()}
        pool = [i for i in grades if i in id2row]
        if not pool:
            continue

        # dense similarity (best over the persona's facets)
        qd = encode(personas.facets(p)).astype(np.float32)
        rows = np.array([id2row[i] for i in pool])
        dsim = {i: float(s) for i, s in zip(pool, (rec.ret.embs[rows] @ qd.T).max(axis=1))}
        # tfidf similarity to the single profile vector
        q = vec.transform([personas.profile_text(p)])
        tsim_full = linear_kernel(q, tfidf).ravel()
        tsim = {i: float(tsim_full[id2row[i]]) for i in pool}

        order_d = sorted(pool, key=lambda i: -dsim[i])
        order_t = sorted(pool, key=lambda i: -tsim[i])
        rank_d = {i: r for r, i in enumerate(order_d)}
        rank_t = {i: r for r, i in enumerate(order_t)}
        hyb = _rrf(rank_d, rank_t, pool)
        order_h = sorted(pool, key=lambda i: -hyb[i])

        # cross-encoder re-rank over the fused top-20
        topN = order_h[:20]
        cands = [{"id": i, "title": rec.meta[i]["disp"], "body": txt.get(i, "")[:400]} for i in topN]
        order_c = [c["id"] for c in rerank.cross_encoder_rerank(personas.profile_text(p), cands)]
        order_c += [i for i in pool if i not in set(order_c)]

        res = {"dense": ndcg(order_d, grades), "tfidf": ndcg(order_t, grades),
               "hybrid": ndcg(order_h, grades), "cross": ndcg(order_c, grades)}
        for m in methods:
            totals[m] += res[m]
        n += 1
        print(f"{pid:8} " + " ".join(f"{res[m]:8.3f}" for m in methods))

    print("-" * 44)
    print(f"{'MEAN':8} " + " ".join(f"{totals[m]/n:8.3f}" for m in methods))


if __name__ == "__main__":
    main()
