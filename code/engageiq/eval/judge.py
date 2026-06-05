"""LLM-as-judge relevance grading (Phase-3 eval ground truth).

The honest way to grade ranking quality without real engagement history. The
judge is a THIRD mechanism, independent of both rankers under test:
  - embeddings rank by dense cosine,
  - TF-IDF ranks by lexical overlap,
  - the LLM judges semantic fit to the persona's FULL goal/criteria (0-3).

So neither retrieval method has a built-in advantage with the judge (unlike a
keyword judge, which TF-IDF trivially wins). We grade by POOLING (TREC-style):
only the union of the rankers' top-K is judged; everything else is grade 0.

Judgments are CACHED to data/judgments/{persona}.json and committed, so a grader
reproduces every NDCG number deterministically with no API key. Without a key and
without cache, callers fall back to the lexical judge in ranking_eval.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .. import llm

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "judgments"

# what each persona actually values — given to the judge, derived from the spec
PERSONA_VALUES = {
    "sofia": ("MS student building an open-source portfolio. IDEAL = beginner-friendly "
              "'good first issue' GitHub repos in Python/ML/data (NOT C++/Rust/low-level), "
              "plus ML blog/forum discussions for visibility; each <~1h. Penalize advanced, "
              "non-ML, or C++/Rust/systems items."),
    "david": ("Senior DevOps engineer building thought leadership. IDEAL = Kubernetes / "
              "cloud-native / infra content: high-signal discussion threads where expert "
              "commentary adds value, and ACTIVE GitHub repos with FEW contributors (room to "
              "stand out). Penalize general web dev, frontend, or beginner content."),
    "lina": ("Data journalist hunting story leads. IDEAL = FAST-RISING, RECENT, trending or "
             "viral tech across many topics (new tools, emerging AI, repos gaining traction). "
             "Values recency and momentum over personal skill match. Penalize old, niche, or "
             "low-traction items."),
    "raj": ("Developer-tools startup founder doing community marketing. IDEAL = discussion "
            "threads/posts about developer tools, APIs, CLIs, developer productivity where a "
            "dev-tools product is genuinely relevant to mention. Penalize generic programming, "
            "unrelated topics, or pure link-only announcements."),
}

_RUBRIC = ("Grade how well the opportunity fits THIS person, 0-3:\n"
           "  3 = ideal match for their goal\n  2 = relevant and useful\n"
           "  1 = weakly/tangentially related\n  0 = irrelevant or violates their constraints\n"
           "Reply with ONLY the single digit.")


def _prompt(persona: dict, m: dict, text: str) -> list[dict]:
    vals = PERSONA_VALUES.get(persona["id"], persona.get("goal", ""))
    opp = f"[{m['source']} / {m['type']}] {text[:300]}"
    if m["source"] == "github" and m.get("language"):
        opp += f"  (language: {m['language']})"
    return [
        {"role": "system", "content": "detailed thinking off. You are a strict relevance judge. Output only one digit 0-3."},
        {"role": "user", "content": f"PERSON: {persona['name']}.\nWHAT THEY WANT: {vals}\n\n"
                                    f"OPPORTUNITY: {opp}\n\n{_RUBRIC}"},
    ]


def _grade_llm(persona: dict, m: dict, text: str, tries: int = 3) -> int | None:
    """LLM grade 0-3 with retry+backoff (the free tier rate-limits in bursts)."""
    for attempt in range(tries):
        out = llm.chat(_prompt(persona, m, text), model=llm.FAST_MODEL, max_tokens=4, temperature=0)
        if out:
            mt = re.search(r"[0-3]", out)
            if mt:
                return int(mt.group(0))
        time.sleep(0.8 * (attempt + 1))           # back off on failure / unparseable
    return None


def judge_pool(persona: dict, pool_ids: list[str], meta: dict, text_by_id: dict,
               fallback=None, pace: float = 0.3, refresh: bool = False) -> dict[str, int]:
    """Return {id: grade} for the pool, using (and extending) the on-disk cache.

    Only successful LLM grades are persisted — a call that fails after retries
    uses `fallback` (lexical) transiently and is retried on the next run, so an
    API hiccup is never frozen into the cache as a real grade-0.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{persona['id']}.json"
    cache: dict[str, int] = {}
    if cache_path.exists() and not refresh:
        cache = json.loads(cache_path.read_text())

    out: dict[str, int] = {}
    newly = 0
    for i in pool_ids:
        if i in cache:
            out[i] = cache[i]
            continue
        g = _grade_llm(persona, meta[i], text_by_id.get(i, "")) if llm.available() else None
        if g is not None:
            cache[i] = out[i] = g
            newly += 1
            time.sleep(pace)
        elif fallback is not None:
            out[i] = fallback(persona, meta[i], text_by_id.get(i, "").lower())
        else:
            out[i] = 0
    if newly:
        cache_path.write_text(json.dumps(cache, indent=0))
    return out


def have_cache(persona_id: str) -> bool:
    return (CACHE_DIR / f"{persona_id}.json").exists()
