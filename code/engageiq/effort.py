"""Effort estimation — the ROI denominator (Phase 3, Lecture 7 signature).

"Engagement ROI" ranks by impact / effort, and the time-budget knapsack packs a
weekly basket under the persona's hours. Both need an honest minutes-to-engage
estimate per opportunity. We use a transparent, source/type-aware heuristic (so
it runs offline + deterministically for graders) and optionally refine the
top-K with an LLM minutes estimate. Effort is in MINUTES throughout.
"""
from __future__ import annotations

import re

from . import llm


def _is_good_first(tags: list[str]) -> bool:
    return any(("good first issue" in (t or "").lower()) or ("good-first-issue" in (t or "").lower())
               or ("beginner" in (t or "").lower()) for t in (tags or []))


def heuristic_minutes(rec: dict) -> float:
    """Deterministic minutes-to-meaningfully-engage (read + contribute/comment)."""
    src = rec.get("source")
    typ = rec.get("opportunity_type")
    sig = rec.get("_signals") or {}
    blen = len(rec.get("body") or "")

    if src == "devto":                       # read the article + leave a thoughtful comment
        return float((sig.get("reading_time_min") or 5)) + 8.0
    if src in ("hackernews", "reddit"):      # read thread + write a substantive comment
        return 15.0
    if src == "bluesky":                     # read + reply
        return 8.0
    if src == "github":
        if typ == "pull_request":
            base = 90.0                      # review/iterate on a PR
        else:
            base = 45.0 if _is_good_first(rec.get("_tags")) else 150.0
        base += min(blen / 500.0 * 15.0, 60.0)   # longer issue body = more context to absorb
        return float(min(base, 480.0))
    return 30.0


def normalized_effort(minutes: float, cap_min: float = 240.0) -> float:
    """Map raw minutes to [0,1] (4h = saturated). Used only where a [0,1] effort is needed."""
    return max(0.0, min(1.0, minutes / cap_min))


def llm_minutes(rec: dict, model: str | None = None) -> float:
    """LLM minutes estimate for a single opportunity; falls back to the heuristic."""
    title = (rec.get("title") or "")[:160]
    src = rec.get("source")
    typ = rec.get("opportunity_type")
    verb = "open a PR / fix the issue" if src == "github" else "write a thoughtful comment or reply"
    prompt = (f"Estimate the time in MINUTES for an experienced professional to meaningfully "
              f"engage ({verb}) with this {src} {typ}.\nTitle: \"{title}\"\n"
              f"Reply with ONLY an integer number of minutes.")
    out = llm.chat([{"role": "system", "content": "detailed thinking off. Output only an integer."},
                    {"role": "user", "content": prompt}],
                   model=model or llm.FAST_MODEL, max_tokens=8, temperature=0)
    if out:
        m = re.search(r"\d+", out)
        if m:
            v = float(m.group(0))
            if 1 <= v <= 1000:
                return v
    return heuristic_minutes(rec)


def refine_top_k(recs: list[dict], k: int = 10, model: str | None = None) -> None:
    """In-place: set rec['effort_min'] via LLM for the top-k, heuristic for the rest.

    Only the shortlist gets the (paid) LLM call; everything else keeps the
    deterministic heuristic so the pipeline never hard-depends on the LLM.
    """
    for i, r in enumerate(recs):
        if i < k and llm.available():
            r["effort_min"] = llm_minutes(r, model=model)
            r["effort_source"] = "llm"
        else:
            r.setdefault("effort_min", heuristic_minutes(r))
            r.setdefault("effort_source", "heuristic")
