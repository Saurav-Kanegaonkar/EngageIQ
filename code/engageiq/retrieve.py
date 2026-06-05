"""FAISS retrieval over opportunity embeddings (Phase 2, Lecture 5).

IndexFlatIP on L2-normalized vectors == exact cosine-similarity search. At ~22k
records this is fast; swap to IndexIVFFlat if the corpus grows past ~1M.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid faiss/torch OpenMP segfault (macOS)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sqlite3

import numpy as np
import faiss

from .embed import DB_PATH, encode, load_cached


class Retriever:
    def __init__(self):
        self.ids, self.embs = load_cached()
        self.index = faiss.IndexFlatIP(self.embs.shape[1])
        self.index.add(self.embs)

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        return self.search_vec(encode([query])[0], k)

    def search_vec(self, qv: np.ndarray, k: int = 20) -> list[tuple[str, float]]:
        scores, idx = self.index.search(qv.reshape(1, -1).astype(np.float32), k)
        return [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i != -1]

    def recommend(self, profile_text: str, k: int = 20) -> list[dict]:
        """Profile text -> top-k candidate opportunities (full records + similarity).

        This is the candidate-generation entry point the downstream scoring/ranking
        phases build on (Phase 3+). Stage 2 (LLM relevance re-rank) plugs in on top.
        """
        return self._hydrate(self.search(profile_text, k))

    def _hydrate(self, hits: list[tuple[str, float]]) -> list[dict]:
        if not hits:
            return []
        ids = [i for i, _ in hits]
        sim = dict(hits)
        conn = sqlite3.connect(str(DB_PATH))
        ph = ",".join("?" * len(ids))
        cols = "opportunity_id, source, domain, title, body, url, score, num_comments, created_at"
        rows = {r[0]: r for r in conn.execute(
            f"SELECT {cols} FROM opportunities WHERE opportunity_id IN ({ph})", ids)}
        out = []
        for i, _ in hits:
            r = rows.get(i)
            if r:
                out.append({"id": i, "similarity": sim[i], "source": r[1], "domain": r[2],
                            "title": r[3], "body": r[4], "url": r[5], "engagement": r[6],
                            "num_comments": r[7], "created_at": r[8]})
        return out
