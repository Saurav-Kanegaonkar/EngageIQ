"""Stage-2 re-rankers — the "better than cosine" comparators (Lecture 5/7).

A cosine bi-encoder retrieves a shortlist fast; a CROSS-ENCODER or an LLM then
re-scores each (query, candidate) pair *together* for higher accuracy. Both are
provided so we can benchmark cosine-only vs +cross-encoder vs +LLM.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import re

from sentence_transformers import CrossEncoder

from . import llm

CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_ce: CrossEncoder | None = None


def _text(c: dict) -> str:
    return ((c.get("title") or "") + " " + (c.get("body") or "")).strip()[:400]


def _model() -> CrossEncoder:
    global _ce
    if _ce is None:
        _ce = CrossEncoder(CE_MODEL)
    return _ce


def cross_encoder_rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Re-rank candidates with a cross-encoder (reads query+doc together)."""
    scores = _model().predict([[query, _text(c)] for c in candidates])
    order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
    return [candidates[i] for i in order]


def cross_encoder_scores(query: str, texts: list[str]) -> list[float]:
    """Relevance scores for (query, text) pairs — used as the pipeline's Stage-2
    relevance signal (benchmarked best: NDCG 0.81 vs 0.62 dense / 0.77 TF-IDF)."""
    if not texts:
        return []
    return [float(s) for s in _model().predict([[query, t] for t in texts])]


def llm_rerank(query: str, candidates: list[dict], model: str | None = None,
               top_k: int = 10) -> list[dict]:
    """Re-rank candidates with the LLM. Falls back to input order if no key."""
    listing = "\n".join(f"[{i}] {_text(c)[:160]}" for i, c in enumerate(candidates))
    prompt = (f'A user is looking for: "{query}".\nCandidates:\n{listing}\n\n'
              f"Return ONLY a JSON array of the {top_k} most relevant item numbers, "
              f"best first (e.g. [3,0,7]). No other text.")
    out = llm.chat([{"role": "system", "content": "detailed thinking off. Output only JSON."},
                    {"role": "user", "content": prompt}],
                   model=model or llm.FAST_MODEL, max_tokens=120, temperature=0)
    order: list[int] = []
    if out:
        m = re.search(r"\[[\d,\s]+\]", out)
        if m:
            try:
                for i in json.loads(m.group(0)):
                    if isinstance(i, int) and 0 <= i < len(candidates) and i not in order:
                        order.append(i)
            except Exception:  # noqa: BLE001
                pass
    for i in range(len(candidates)):   # append any the LLM didn't list
        if i not in order:
            order.append(i)
    return [candidates[i] for i in order]
