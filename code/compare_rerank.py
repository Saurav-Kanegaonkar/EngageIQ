"""Benchmark: cosine-only vs +cross-encoder vs +LLM re-rank.

For each of the 15 per-domain queries: retrieve top-30 by cosine, then measure
precision@10 (domain match) for cosine-only, cross-encoder-reranked, and
LLM-reranked. Shows the accuracy difference empirically.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engageiq.embed import DB_PATH                                  # noqa: E402
from engageiq.eval.embedding_eval import DEFAULT_QUERIES            # noqa: E402
from engageiq.rerank import cross_encoder_rerank, llm_rerank        # noqa: E402
from engageiq.retrieve import Retriever                            # noqa: E402

r = Retriever()
conn = sqlite3.connect(str(DB_PATH))
dmap = dict(conn.execute("SELECT opportunity_id, domain FROM opportunities"))


def p_at_10(items, expected):
    return sum(1 for it in items[:10] if dmap.get(it["id"]) == expected) / 10.0


cos, ce, lm = [], [], []
print(f"{'domain':22} {'cosine':>7} {'+CE':>7} {'+LLM':>7}")
for domain, q in DEFAULT_QUERIES.items():
    cands = r.recommend(q, k=30)
    a = p_at_10(cands, domain)
    b = p_at_10(cross_encoder_rerank(q, cands), domain)
    c = p_at_10(llm_rerank(q, cands), domain)
    cos.append(a); ce.append(b); lm.append(c)
    print(f"{domain:22} {a:>7.2f} {b:>7.2f} {c:>7.2f}")

print("-" * 46)
print(f"{'MEAN P@10':22} {statistics.mean(cos):>7.3f} {statistics.mean(ce):>7.3f} {statistics.mean(lm):>7.3f}")
