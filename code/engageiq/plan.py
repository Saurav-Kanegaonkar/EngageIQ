"""The plan-driven Hub: a weekly ACTIVITY SPLIT the user can tune.

After a persona/profile is chosen we propose a weekly split of the user's
attention across four buckets, then let them adjust it (including down to 0%,
which drops that bucket from the feed entirely). The confirmed split drives the
Hub two ways: it decides which sources the feed pulls from, and it interleaves
the "For you" feed so the visible mix matches the chosen emphasis
(40% Contribute -> roughly 4 of every 10 cards).

Buckets group our 5 sources by what the user actually DOES with each:

  Contribute       -> github            open issues / PRs, build a portfolio
  Learn            -> devto             read articles and write-ups
  Discuss & watch  -> reddit, bluesky   join conversations, watch communities
  News             -> hackernews        stay current on what is rising

The recommendation is mostly the user's stated goal PLUS a recommended taste of
the others, tuned by experience level. It is a transparent heuristic; the prose
"read" is optionally polished by the LLM (cached to disk, offline-safe).
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from . import llm, personas, score

# ── buckets ──────────────────────────────────────────────────────────────────
BUCKETS = [
    {"key": "contribute", "label": "Contribute", "sources": ["github"],
     "blurb": "Open issues and PRs that build a visible portfolio."},
    {"key": "learn", "label": "Learn", "sources": ["devto"],
     "blurb": "Articles and write-ups that grow your skills."},
    {"key": "discuss", "label": "Discuss & watch", "sources": ["reddit", "bluesky"],
     "blurb": "Threads and posts where your take adds value."},
    {"key": "news", "label": "News", "sources": ["hackernews"],
     "blurb": "What is rising in your space right now."},
]
BUCKET_BY_KEY = {b["key"]: b for b in BUCKETS}
_SOURCE_BUCKET = {s: b["key"] for b in BUCKETS for s in b["sources"]}


def bucket_for(source: str) -> str:
    """Which activity bucket a source belongs to (defaults to discuss)."""
    return _SOURCE_BUCKET.get(source, "discuss")


# ── recommended split (transparent heuristic) ────────────────────────────────
# Base lean per engagement mode: the user's goal gets the majority, with a
# recommended taste of the rest so we widen them a little (never just one bucket).
_BASE = {
    "contribute": {"contribute": 50, "learn": 25, "discuss": 15, "news": 10},
    "discuss":    {"discuss": 45, "contribute": 22, "news": 20, "learn": 13},
    "monitor":    {"news": 40, "discuss": 30, "learn": 18, "contribute": 12},
    "promote":    {"discuss": 42, "news": 30, "learn": 16, "contribute": 12},
}

_NEWBIE_KW = ("student", "beginner", "newbie", "first ", "first-", "graduat",
              "early-career", "junior", "comfortable with python", "new to",
              "portfolio", "good first issue", "good-first")
_EXPERT_KW = ("senior", "expert", "yrs", "5 yr", "mid-career", "years of",
              "lead ", "principal", "founder", "experienced", "thought leader",
              "deep kubernetes", "deep ")


def experience(persona: dict) -> str:
    """Crude experience read from the persona's own words: newbie / mid / expert.

    A newbie leans a little more on learning; an expert leans into hands-on work.
    """
    blob = " ".join([
        persona.get("goal", ""), persona.get("skills", ""),
        persona.get("name", ""), " ".join(persona.get("interests", [])),
    ]).lower()
    if any(k in blob for k in _NEWBIE_KW):
        return "newbie"
    if any(k in blob for k in _EXPERT_KW):
        return "expert"
    return "mid"


def _round_to_100(weights: dict) -> dict:
    """Scale to sum 100 and round to ints that still sum to exactly 100
    (largest-remainder method), preserving bucket order."""
    raw = {k: max(0.0, float(v)) for k, v in weights.items()}
    tot = sum(raw.values()) or 1.0
    scaled = {k: v * 100.0 / tot for k, v in raw.items()}
    floor = {k: int(v) for k, v in scaled.items()}
    rem = 100 - sum(floor.values())
    order = sorted(scaled, key=lambda k: (scaled[k] - floor[k]), reverse=True)
    for k in order[: max(0, rem)]:
        floor[k] += 1
    return floor


_REASON = {
    "contribute": "Hands-on work on GitHub: open issues and PRs that build a visible portfolio.",
    "learn": "Reading and learning on Dev.to (the developer blogging community): articles and write-ups that grow your skills.",
    "discuss": "Conversations on Reddit and Bluesky: threads and posts where your take adds value.",
    "news": "Staying current on Hacker News (the startup and tech news board): what is rising in your space right now.",
}


def _reasons(split: dict) -> dict:
    top = max(split, key=lambda k: split[k]) if split else None
    out = {}
    for k, base in _REASON.items():
        if k == top and split.get(k, 0) > 0:
            out[k] = "Most of your week, " + base[0].lower() + base[1:]
        else:
            out[k] = base
    return out


def recommend_split(persona: dict) -> dict:
    """The starting weekly split we propose for this persona, ordered like BUCKETS."""
    mode = score.mode_for_persona(persona)
    base = dict(_BASE.get(mode, _BASE["contribute"]))
    exp = experience(persona)
    if exp == "newbie":                       # lean toward learning / lower-pressure wins
        shift = min(12, base.get("contribute", 0) - 8)
        if shift > 0:
            base["contribute"] -= shift
            base["learn"] = base.get("learn", 0) + shift
    elif exp == "expert":                     # lean into hands-on contribution
        shift = min(8, base.get("learn", 0) - 6)
        if shift > 0:
            base["learn"] -= shift
            base["contribute"] = base.get("contribute", 0) + shift
    split = _round_to_100({b["key"]: base.get(b["key"], 0) for b in BUCKETS})
    return {"split": split, "reasons": _reasons(split), "mode": mode, "experience": exp}


def bucket_weights_from_split(split: dict) -> dict:
    """A gentle per-bucket prior the ranker multiplies into impact, so a higher
    share nudges that bucket's items up (and surfaces more of them in the top-k).
    Mild on purpose: relevance still leads. 0% buckets are dropped upstream."""
    pos = {k: float(v) for k, v in split.items() if float(v) > 0}
    if not pos:
        return {}
    mx = max(pos.values()) or 1.0
    return {k: round(0.80 + 0.45 * (v / mx), 4) for k, v in pos.items()}


# ── the prose "read" of what the user wants ──────────────────────────────────
_DATA = Path(__file__).resolve().parent.parent.parent / "data"
_READ_CACHE = _DATA / "plan_reads.json"
_LOCK = threading.Lock()

_MODE_VERB = {
    "contribute": "build things and contribute code",
    "discuss": "join high-signal conversations where your take matters",
    "monitor": "track what is rising before it goes mainstream",
    "promote": "find the right places to mention your work",
}
_EXP_CLAUSE = {
    "newbie": "You are earlier in the journey, so we lean a little more on learning and lower-pressure wins. ",
    "expert": "You have real depth, so we lean into hands-on work and high-signal discussion. ",
    "mid": "",
}


def _domains_phrase(persona: dict, dom_label) -> str:
    ds = [dom_label(d) for d in (persona.get("domains") or [])][:3]
    if not ds:
        return "a broad range of technology"
    if len(ds) == 1:
        return ds[0]
    return ", ".join(ds[:-1]) + " and " + ds[-1]


def deterministic_read(persona: dict, dom_label) -> str:
    exp = experience(persona)
    mode = score.mode_for_persona(persona)
    goal = (persona.get("goal") or "").strip().rstrip(".")
    verb = _MODE_VERB.get(mode, "engage where it counts")
    return (f"We read your goal as: {goal}. {_EXP_CLAUSE.get(exp, '')}"
            f"You focus on {_domains_phrase(persona, dom_label)}, and you like to {verb}. "
            f"Here is a weekly mix we would suggest. Move any slider, or set one to zero to drop it.")


def _cache_key(persona: dict) -> str:
    pid = persona.get("id")
    if pid and pid != "custom":
        return pid
    blob = ((persona.get("goal") or "") + "|" + ",".join(persona.get("interests", []))).encode()
    return "custom:" + hashlib.sha1(blob).hexdigest()[:10]


def _load_reads() -> dict:
    if _READ_CACHE.exists():
        try:
            return json.loads(_READ_CACHE.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _llm_read(persona: dict, dom_label) -> str | None:
    if not llm.available():
        return None
    mode = score.mode_for_persona(persona)
    prompt = (
        "A developer is setting up a weekly plan for where to spend their engagement time. "
        "In 2 to 3 warm, plain sentences, reflect back what they want and what we will emphasise "
        "(contributing, learning, discussion, or staying current). Address them as 'you'. "
        "No markdown, no preamble, no dashes.\n\n"
        f"Goal: {persona.get('goal','')}\n"
        f"Experience read: {experience(persona)}\n"
        f"Focus areas: {_domains_phrase(persona, dom_label)}\n"
        f"Primary engagement style: {mode}\n"
        f"About them: {persona.get('skills','') or persona.get('goal','')}")
    out = llm.chat(
        [{"role": "system", "content": "You write short, encouraging plans for a busy developer."},
         {"role": "user", "content": prompt}],
        model=llm.FAST_MODEL, max_tokens=150, temperature=0.4)
    if out:
        out = " ".join(out.split()).replace(" — ", ", ").replace("—", ", ").replace("–", "-")
    return out or None


def read(persona: dict, dom_label, use_llm: bool = True, save: bool = True) -> str:
    """Cached LLM read if present; otherwise generate (when a key exists) or fall
    back to the deterministic read. Always returns clean prose."""
    key = _cache_key(persona)
    cache = _load_reads()
    if key in cache:
        return cache[key]
    if use_llm:
        out = _llm_read(persona, dom_label)
        if out:
            if save:
                with _LOCK:
                    cache = _load_reads()
                    cache[key] = out
                    try:
                        _READ_CACHE.write_text(json.dumps(cache, indent=2))
                    except Exception:  # noqa: BLE001
                        pass
            return out
    return deterministic_read(persona, dom_label)


# ── this week's active areas (real, from the data) ───────────────────────────
def hot_topics(persona: dict, dom_label, limit: int = 5) -> list[str]:
    """The most active domains in the user's space, by record volume across all
    sources (narrowed to their chosen domains). Labelled for display. Opens its
    own short-lived SQLite connection (SQLite handles are thread-bound, so we must
    not reuse the engine's). Falls back to the persona's stated focus areas."""
    import sqlite3

    from .embed import DB_PATH
    doms = personas.domain_whitelist(persona)
    out: list[str] = []
    try:
        conn = sqlite3.connect(str(DB_PATH))
        try:
            rows = conn.execute(
                "SELECT domain, COUNT(*) c FROM opportunities "
                "WHERE domain IS NOT NULL AND domain != '' GROUP BY domain ORDER BY c DESC"
            ).fetchall()
        finally:
            conn.close()
        for d, _c in rows:
            if doms and d not in doms:
                continue
            out.append(dom_label(d))
            if len(out) >= limit:
                break
    except Exception:  # noqa: BLE001
        out = []
    if not out:                                   # fallback: the persona's own focus areas
        out = [dom_label(d) for d in (persona.get("domains") or [])][:limit]
    return out


def build_plan(persona: dict, dom_label) -> dict:
    """Everything the plan screen needs: the read, the recommended buckets
    (key/label/sources/pct/reason), and this week's active areas."""
    rec = recommend_split(persona)
    buckets = []
    for b in BUCKETS:
        k = b["key"]
        buckets.append({
            "key": k, "label": b["label"], "sources": b["sources"],
            "pct": rec["split"].get(k, 0), "reason": rec["reasons"].get(k, b["blurb"]),
        })
    return {
        "read": read(persona, dom_label),
        "buckets": buckets,
        "topics": hot_topics(persona, dom_label),
        "mode": rec["mode"], "experience": rec["experience"],
    }
