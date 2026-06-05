"""EngageIQ — FastAPI backend.

Serves the real ranking engine to the prototype frontend. The prototype
(mockups/prototype.html) is the product UI; this wraps `rank.Recommender` so
onboarding + the Hub render REAL recommendations instead of hardcoded cards.

Run:  PYTHONPATH=code .venv/bin/uvicorn api:app --port 8000 --app-dir code
Then open:  http://localhost:8000/prototype.html
"""
from __future__ import annotations

import os
import re
import json
import html as _html
import sqlite3
import threading
import datetime as dt

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import brand_logos
import trends_render
from engageiq import feedback, llm, personas, plan, rank, score, store, summarize
from engageiq.embed import DB_PATH
from sketches import HyperLogLog

ROOT = Path(__file__).resolve().parent.parent
MOCKUPS = ROOT / "mockups"

DOMAIN_LABELS = {
    "machine_learning": "Machine Learning", "ai_research": "AI Research",
    "python_data_eng": "Python / Data Eng", "developer_tools": "Developer Tools",
    "devops_k8s": "DevOps / K8s", "cloud_apis": "Cloud / APIs",
    "cybersecurity": "Cybersecurity", "frontend_web": "Frontend / Web",
    "mobile_dev": "Mobile Dev", "gamedev_cpp": "Gamedev / C++",
    "embedded_iot": "Embedded / IoT", "blockchain_web3": "Blockchain / Web3",
    "b2b_saas": "B2B SaaS", "open_source_trending": "Open Source Trending",
    "beginner_coding": "Beginner Coding",
}
SRC_LABELS = {"github": "GitHub", "hackernews": "Hacker News", "bluesky": "Bluesky",
              "devto": "Dev.to", "reddit": "Reddit"}
MODE_COLOR = {"contribute": "#4f46e5", "discuss": "#2f9e8f",
              "monitor": "#bd7f4e", "promote": "#b56a93"}

# the prototype's onboarding uses a few keys that differ from the engine's
DOMAIN_ALIAS = {"blockchain": "blockchain_web3", "embedded_systems": "embedded_iot"}
PLATFORM_ALIAS = {"blogs": "devto", "social": "bluesky"}


def dom_label(k: str) -> str:
    return DOMAIN_LABELS.get(k, (k or "").replace("_", " ").title())


def clean(s: str) -> str:
    """House style: no em/en-dashes in our own copy."""
    return (s or "").replace(" — ", ", ").replace("—", ", ").replace("–", "-")


def split_name(full: str) -> tuple[str, str]:
    for sep in ("—", " - ", "–"):
        if sep in full:
            a, b = full.split(sep, 1)
            return a.strip(), b.strip()
    return full.strip(), ""


# ── engine (loaded once at startup) ──────────────────────────────────────────
app = FastAPI(title="EngageIQ API")
_REC: rank.Recommender | None = None
_STORE = store.EngageStore()            # per-user SQLite: personas, interactions, events, RL
_SUMM = summarize.Summarizer()          # LLM card summaries (read-only cache; prewarmed offline)
# server-side context per (user, persona): the engagement mode + each surfaced item's
# signal vector / bucket / source, so /api/feedback can train + log without trusting the client.
_CTX: dict[str, dict] = {}


def _ctx_key(user_id: str, persona_key: str) -> str:
    return f"{user_id}|{persona_key}"


def engine() -> rank.Recommender:
    global _REC
    if _REC is None:
        _REC = rank.Recommender()
    return _REC


@app.on_event("startup")
def _warm() -> None:
    """Eager-load the FULL ML stack at boot so the first real request is fast.

    Constructing the Recommender alone is not enough: the query sentence-encoder and
    the cross-encoder re-ranker load lazily on the first actual recommend(), which is
    the ~8s cold start users felt on the first feed load after a restart. A tiny real
    recommend here warms all of them, moving that cost to boot (once)."""
    try:
        eng = engine()
        warm_persona = personas.PERSONAS.get("sofia") or next(iter(personas.PERSONAS.values()))
        eng.recommend(warm_persona, k=3)
    except Exception:  # noqa: BLE001 — warm-up must never crash startup
        pass
    try:                                          # capability 5: base analytics images
        trends_render.prerender_base()
    except Exception:  # noqa: BLE001
        pass


# ── serializers ──────────────────────────────────────────────────────────────
def _strip_md(s: str) -> str:
    """Flatten Markdown to clean plain text for the card snippet (drop ##, **, ` etc)."""
    if not s:
        return ""
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)         # images
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)      # links -> text
    s = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", s)        # code spans
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", s)         # headers
    s = re.sub(r"(?m)^\s{0,3}>\s?", "", s)              # blockquotes
    s = re.sub(r"(?m)^\s{0,3}[-*+]\s+", "", s)          # bullet markers
    s = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", s)      # bold / italic
    s = s.replace("**", "").replace("##", "").replace("```", "").replace("`", "")
    s = re.sub(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF️‍]", "", s)  # emoji / pictographs
    return re.sub(r"\s+", " ", s).strip()


def _clean_lines(s: str) -> str:
    """Like _strip_md but PRESERVES line breaks, so the detail popup keeps the
    issue/post structure (sections, numbered steps) instead of one flat line."""
    if not s:
        return ""
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", s)
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", s)
    s = re.sub(r"(?m)^\s*[-*_=]{3,}\s*$", "", s)        # horizontal rules
    s = re.sub(r"(?m)^\s{0,3}>\s?", "", s)
    s = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", s)
    s = s.replace("**", "").replace("##", "").replace("```", "").replace("`", "")
    s = re.sub(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF️‍]", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _effort_basis(source: str, typ: str, em: int) -> str:
    """Plain-language explanation of how the time estimate was derived (no magic numbers)."""
    if source == "github":
        if typ == "pull_request":
            return f"~{em} min: reviewing and iterating on a pull request (about a 90-min base, plus the size of the change)."
        kind = "a beginner-friendly issue (a 45-min base)" if em < 120 else "a standard issue (a 150-min base)"
        return (f"~{em} min, from a transparent heuristic: {kind}, plus reading time scaled to the "
                f"issue's length. GitHub work is the heaviest category, so the estimate is deliberately generous.")
    if source in ("hackernews", "reddit"):
        return f"~{em} min: time to read the thread and write one substantive, non-obvious comment."
    if source == "bluesky":
        return f"~{em} min: read the post and write a thoughtful reply."
    if source == "devto":
        return f"~{em} min: the article's reading time plus ~8 min to leave a comment that adds value."
    return f"~{em} min to meaningfully engage."


def _gh_repo(url: str) -> str:
    """owner/repo from a GitHub URL, for the standout-repos lane (show the repo, not the issue)."""
    m = re.search(r"github\.com/([^/]+/[^/]+)", url or "")
    return m.group(1) if m else ""


# Per-source engagement floors for the "Rising" badge. `velocity` is a per-source
# PERCENTILE (relative momentum), so on low-traffic sources (Bluesky / Dev.to) it fires
# even for posts with ~0 real engagement. "Rising" should mean real momentum AND real
# people, so we also require engagement (comments + score) above a per-source floor, set
# from the actual per-source distributions (roughly the upper quartile, ~20x apart).
_RISING_FLOOR = {"github": 4, "devto": 2, "bluesky": 4, "reddit": 15, "hackernews": 8}


def _load_engagement() -> dict:
    """oid -> num_comments + score, loaded once, to gate the Rising badge on real volume."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        return {oid: (int(nc or 0) + int(sc or 0))
                for oid, nc, sc in conn.execute(
                    "SELECT opportunity_id, num_comments, score FROM opportunities")}
    finally:
        conn.close()


_ENGAGEMENT = _load_engagement()


def _item(r: dict) -> dict:
    sc, f = r["score"], r["features"]
    title = (r.get("title") or "").strip()
    body_raw = (r.get("body_raw") or "").strip()
    snippet = _strip_md(body_raw)[:180] if (title and body_raw) else ""
    body_text = _clean_lines(body_raw)[:1200] if body_raw else ""
    drivers = [s for s, c in r["score"].get("contributions", [])[:3] if c > 0.01]
    return {
        "id": r["id"],
        "source": r["source"],
        "source_label": SRC_LABELS.get(r["source"], r["source"]),
        "type": r.get("type") or "",
        "domain": (r.get("domains") or [r.get("domain")])[0] or "",
        "domain_label": dom_label((r.get("domains") or [r.get("domain")])[0]),
        "title": (r.get("disp") or "").strip(),
        "summary": _SUMM.cached(r["id"]),
        "sections": _SUMM.sections(r["id"]),       # labeled what/task/do/gain (or null)
        "snippet": snippet,
        "body_text": body_text,
        "effort_basis": _effort_basis(r["source"], r.get("type") or "", round(float(sc.get("effort_min", f.get("effort_min", 0))))),
        "url": r.get("url") or "",
        "why": clean(r.get("why", "")),
        "action": clean(r.get("action", "engage")),
        "drivers": drivers,
        "roi": round(float(sc.get("roi", 0)), 1),
        "impact": round(float(sc.get("impact", 0)), 3),
        "relevance": round(float(f.get("relevance", 0)), 2),
        "community_health": round(float(f.get("community_health", 0)), 2),
        "visibility": round(float(f.get("visibility", 0)), 2),
        "discussion": round(float(f.get("discussion", 0)), 2),
        "contribution_worth": round(float(f.get("contribution_worth", 0)), 2),
        "repo_reputation": round(float(f.get("repo_reputation", 0)), 2),
        "issue_quality": round(float(f.get("issue_quality", 0)), 2),
        "stars": (int(r.get("community_size") or 0) if r["source"] == "github" else None),
        "repo": (_gh_repo(r.get("url")) if r["source"] == "github" else ""),
        "effort_min": round(float(sc.get("effort_min", f.get("effort_min", 0)))),
        "velocity": round(float(f.get("velocity", 0)), 2),
        "standout": round(float(f.get("standout", 0)), 2),
        "language": r.get("language") or "",
        "bucket": plan.bucket_for(r["source"]),
        "engagement": _ENGAGEMENT.get(r["id"], 0),
        "rising": (float(f.get("velocity", 0)) >= 0.6
                   and _ENGAGEMENT.get(r["id"], 0) >= _RISING_FLOOR.get(r["source"], 5)),
    }


def _serialize(persona: dict, out: dict) -> dict:
    name, role = split_name(persona.get("name", ""))
    mode = out["mode"]
    rail = None
    if out.get("basket") is not None:
        used = sum(float(b["features"]["effort_min"]) for b in out["basket"])
        rail = {"type": "basket", "budget_min": out.get("budget_min", 0),
                "used_min": round(used), "items": [_item(b) for b in out["basket"]]}
    elif out.get("radar"):
        rail = {"type": "radar", "items": [_item(r) for r in out["ranked"]]}
    return {
        "persona": {
            "id": persona.get("id"), "name": name, "role": role,
            "goal": persona.get("goal", ""),
            "domains": [dom_label(d) for d in persona.get("domains", [])] or ["All topics"],
            "platforms": [SRC_LABELS.get(s, s) for s in sorted(personas.allowed_sources(persona))],
            "time_budget_hours": persona.get("time_budget_hours", 5),
        },
        "mode": mode, "mode_label": out["mode_label"],
        "mode_blurb": clean(out.get("mode_blurb", "")), "mode_color": MODE_COLOR.get(mode, "#475569"),
        "n_candidates": out["n_candidates"],
        "ranked": [_item(r) for r in out["ranked"]],
        "rail": rail,
        "repos": [_item(r) for r in out.get("repos", [])],
    }


# ── request models ───────────────────────────────────────────────────────────
class Profile(BaseModel):
    id: str | None = None              # stable per-browser id, so edits update in place
    name: str | None = None
    goal: str | None = None
    domains: list[str] = []
    platforms: list[str] = []          # engine platform keys the user observes/engages
    time_budget_hours: float = 5
    avoid: str | None = None


class RecRequest(BaseModel):
    persona_id: str | None = None
    profile: Profile | None = None
    session_id: str | None = None
    k: int = 10
    split: dict[str, float] | None = None     # confirmed weekly activity split (bucket -> %)


class FeedbackRequest(BaseModel):
    session_id: str                      # the persistent per-browser user id
    persona_key: str | None = None       # which persona instance this feedback is for
    opportunity_id: str
    action: str                          # engage | dismiss | save | unsave | undo
    rank: int | None = None              # 1-based position in the full ranked list
    page: int | None = None              # which page the user was on
    reason: str | None = None            # why the user dismissed it (feedback loop)


class EventRequest(BaseModel):
    session_id: str
    persona_key: str | None = None
    type: str                            # page_view, open_detail, etc.
    opportunity_id: str | None = None
    rank: int | None = None
    page: int | None = None
    extra: dict | None = None


class ActivityRequest(BaseModel):
    session_id: str
    persona_key: str | None = None


def _profile_to_persona(p: Profile) -> dict:
    domains = [DOMAIN_ALIAS.get(d, d) for d in p.domains]
    plats = [PLATFORM_ALIAS.get(x, x) for x in p.platforms]
    interests = [dom_label(d) for d in domains] or ["technology", "open source"]
    excl = [x.strip() for x in re.split(r"[,/;]", p.avoid or "") if x.strip()]
    return {
        "id": (p.id or "custom"), "name": (p.name or "You"),
        "interests": interests,
        "goal": (p.goal or "Find relevant opportunities to contribute to and discuss"),
        "platforms": plats or list(personas.ALL_SOURCES),
        "domains": domains, "exclude_languages": excl,
        "time_budget_hours": p.time_budget_hours, "skills": (p.goal or ""),
    }


def _resolve_persona(persona_id: str | None, profile: Profile | None) -> dict:
    if persona_id and persona_id in personas.PERSONAS:
        return personas.PERSONAS[persona_id]
    if profile is not None:
        return _profile_to_persona(profile)
    return personas.PERSONAS["sofia"]


def _persona_key(persona: dict) -> str:
    """Stable key for a persona instance, so learning + history are scoped to it.
    Named personas key by id; custom profiles by a hash of their goal + interests."""
    import hashlib
    pid = persona.get("id")
    if pid and pid != "custom":
        return pid
    blob = ((persona.get("goal") or "") + "|" + ",".join(persona.get("interests", []))).encode()
    return "custom:" + hashlib.sha1(blob).hexdigest()[:10]


def _split_to_engine(split: dict | None) -> tuple[set[str] | None, dict | None, dict | None]:
    """Turn a confirmed bucket split into (allowed sources, bucket prior, active split).
    0% buckets are dropped, so their sources never enter the feed."""
    if not split:
        return None, None, None
    active = {k: float(v) for k, v in split.items() if float(v or 0) > 0}
    if not active:
        return None, None, None
    srcs: set[str] = set()
    for k in active:
        b = plan.BUCKET_BY_KEY.get(k)
        if b:
            srcs.update(b["sources"])
    srcs &= personas.ALL_SOURCES
    return (srcs or None), plan.bucket_weights_from_split(active), active


def _persona_card(persona: dict) -> dict:
    name, role = split_name(persona.get("name", ""))
    return {
        "id": persona.get("id"), "name": name, "role": role,
        "goal": persona.get("goal", ""),
        "domains": [dom_label(d) for d in persona.get("domains", [])] or ["All topics"],
        "platforms": [SRC_LABELS.get(s, s) for s in sorted(personas.allowed_sources(persona))],
        "platform_keys": sorted(personas.allowed_sources(persona)),
        "time_budget_hours": persona.get("time_budget_hours", 5),
    }


# ── routes ───────────────────────────────────────────────────────────────────
@app.get("/api/personas")
def list_personas() -> dict:
    out = []
    for pid, p in personas.PERSONAS.items():
        name, role = split_name(p["name"])
        out.append({"id": pid, "name": name, "role": role, "goal": p["goal"],
                    "domains": [dom_label(d) for d in p.get("domains", [])] or ["All topics"],
                    "platforms": [SRC_LABELS.get(s, s) for s in sorted(personas.allowed_sources(p))],
                    "time_budget_hours": p.get("time_budget_hours", 5)})
    return {"personas": out}


@app.post("/api/plan")
def plan_endpoint(req: RecRequest) -> dict:
    """The weekly plan screen: an LLM/templated read of the user's goal, the
    recommended activity split (tunable, 0% drops a bucket), and active areas."""
    persona = _resolve_persona(req.persona_id, req.profile)
    out = plan.build_plan(persona, dom_label)
    out["persona"] = _persona_card(persona)
    out["mode_label"] = score.MODES[out["mode"]].label
    out["mode_color"] = MODE_COLOR.get(out["mode"], "#475569")
    return out


@app.post("/api/recommend")
def recommend(req: RecRequest) -> dict:
    persona = _resolve_persona(req.persona_id, req.profile)
    sources, bucket_weights, active_split = _split_to_engine(req.split)

    mode_key = score.mode_for_persona(persona)
    pkey = _persona_key(persona)
    learned = _STORE.weights(req.session_id, pkey, mode_key) if req.session_id else None
    out = engine().recommend(persona, k=req.k, weights=learned,
                             sources=sources, bucket_weights=bucket_weights)
    resp = _serialize(persona, out)
    resp["persona_key"] = pkey

    # echo the confirmed split + which buckets actually have content, so the Hub
    # can build its bucket chips (logo + share) and interleave the "For you" feed.
    if active_split:
        present = {it["bucket"] for it in resp["ranked"]}
        resp["split"] = {k: round(v) for k, v in active_split.items()}
        resp["buckets"] = [
            {"key": b["key"], "label": b["label"], "sources": b["sources"],
             "pct": round(active_split.get(b["key"], 0))}
            for b in plan.BUCKETS
            if active_split.get(b["key"], 0) > 0 and b["key"] in present
        ]

    if req.session_id:
        saved = _STORE.save_persona(req.session_id, pkey, persona, active_split, mode_key)
        if saved.get("mode_changed"):
            resp["notice"] = ("Your focus changed, so we restarted learning for the new "
                              "mode. Your saved cards and history are kept.")
        # cache each surfaced item's signal vector + bucket + source so /api/feedback
        # trains + logs authoritatively (the client only sends ids + rank/page).
        items = {r["id"]: {"signals": {s: float(r["features"].get(s, 0.0) or 0.0) for s in score.SIGNALS},
                           "bucket": plan.bucket_for(r["source"]), "source": r["source"]}
                 for r in out["ranked"]}
        _CTX[_ctx_key(req.session_id, pkey)] = {"mode": mode_key, "items": items}
        # restore each card's saved disposition (interested / dismissed / saved) on reload
        st = _STORE.states(req.session_id, pkey)
        if st:
            lists = [resp["ranked"], resp.get("repos") or []]
            if resp.get("rail") and resp["rail"].get("items"):
                lists.append(resp["rail"]["items"])
            for lst in lists:
                for it in lst:
                    if it["id"] in st:
                        it["state"] = st[it["id"]]
            # passed cards LEAVE the feed (they live in the Activity 'Passed' section),
            # so a reload visibly reflects the user's feedback.
            def _keep(lst):
                return [it for it in lst if (it.get("state") or {}).get("status") != "dismissed"]
            resp["ranked"] = _keep(resp["ranked"])
            if resp.get("repos"):
                resp["repos"] = _keep(resp["repos"])
            if resp.get("rail") and resp["rail"].get("items"):
                resp["rail"]["items"] = _keep(resp["rail"]["items"])
        lr = _STORE.learner(req.session_id, pkey, mode_key)
        if lr.n > 0:
            resp["learned"] = {"n": lr.n, "summary": feedback.learned_summary(lr), "weights": lr.w}
    return resp


@app.post("/api/feedback")
def feedback_endpoint(req: FeedbackRequest) -> dict:
    """Idempotent: a repeated engage/save is ONE interest (the RL trains + the
    disposition changes only on the first transition); every click still logs an
    event. Signals/bucket/source come from the server cache, not the client."""
    pkey = req.persona_key or "_"
    ctx = _CTX.get(_ctx_key(req.session_id, pkey))
    if not ctx:
        return {"ok": False, "reason": "no context (reload recommendations first)"}
    item = (ctx.get("items") or {}).get(req.opportunity_id, {})
    res = _STORE.record(req.session_id, pkey, req.opportunity_id, req.action,
                        signals=item.get("signals", {}), mode=ctx["mode"],
                        rank=req.rank, page=req.page, bucket=item.get("bucket"),
                        source=item.get("source"), reason=req.reason)
    lr = res["learner"]
    return {"ok": True, "status": res["status"], "saved": res["saved"],
            "applied": res["applied"], "n": lr.n,
            "summary": feedback.learned_summary(lr) if lr.n else None}


@app.post("/api/feedback/clear-all")
def feedback_clear_all() -> dict:
    """Wipe ALL user state (personas, interactions, events, learned weights). Kept
    easy to reset while we iterate; backs the Hub's 'Reset learning' button."""
    _STORE.clear_all()
    _CTX.clear()
    return {"ok": True}


@app.post("/api/event")
def event_endpoint(req: EventRequest) -> dict:
    """Append-only telemetry (page_view, open_detail, ...) for the 'why' analysis:
    which ranks/buckets a user pages to and opens."""
    pkey = req.persona_key or "_"
    ctx = _CTX.get(_ctx_key(req.session_id, pkey)) or {}
    item = (ctx.get("items") or {}).get(req.opportunity_id or "", {})
    _STORE.log_event(req.session_id, pkey, req.type, oid=req.opportunity_id,
                     rank=req.rank, page=req.page, bucket=item.get("bucket"),
                     source=item.get("source"), mode=ctx.get("mode"))
    return {"ok": True}


@app.post("/api/activity")
def activity(req: ActivityRequest) -> dict:
    """Everything this (user, persona) has reacted to, so the loop is visible:
    engaged / saved / passed, plus what the feedback learner has picked up."""
    pkey = req.persona_key or "_"
    rows = _STORE.interactions_for(req.session_id, pkey)
    meta = engine().meta

    def enrich(r: dict) -> dict:
        m = meta.get(r["opportunity_id"], {})
        src = r.get("source") or m.get("source") or ""
        doms = m.get("domains") or ([m.get("domain")] if m.get("domain") else [])
        return {
            "id": r["opportunity_id"],
            "title": (m.get("disp") or m.get("title") or r["opportunity_id"]),
            "source": src, "source_label": SRC_LABELS.get(src, src),
            "url": m.get("url") or "", "bucket": r.get("bucket"),
            "domain_label": dom_label(doms[0]) if doms else "",
            "reason": r.get("reason"),
        }

    interested = [enrich(r) for r in rows if r.get("status") == "interested"]
    saved = [enrich(r) for r in rows if r.get("saved")]
    dismissed = [enrich(r) for r in rows if r.get("status") == "dismissed"]
    lr = _STORE.learner_row(req.session_id, pkey)
    return {
        "counts": {"interested": len(interested), "saved": len(saved),
                   "dismissed": len(dismissed), "total": len(rows)},
        "interested": interested, "saved": saved, "dismissed": dismissed,
        "learned": ({"n": lr.n, "summary": feedback.learned_summary(lr)} if lr else None),
    }


@app.post("/api/persona/delete")
def persona_delete(req: ActivityRequest) -> dict:
    """Remove a profile and everything scoped to it (when deleted from the gallery)."""
    if req.persona_key:
        _STORE.delete_persona(req.session_id, req.persona_key)
        _CTX.pop(_ctx_key(req.session_id, req.persona_key), None)
    return {"ok": True}


# ── accounts + DB-backed profiles (name-based identity, no password) ───────────
class LoginRequest(BaseModel):
    name: str | None = None


class ProfileSaveRequest(BaseModel):
    session_id: str
    profile_key: str | None = None
    name: str | None = None
    goal: str | None = None
    domains: list[str] = []
    platforms: list[str] = []
    time_budget_hours: float = 5
    avoid: str | None = None
    avatar: str | None = None


class ProfileDeleteRequest(BaseModel):
    session_id: str
    profile_key: str


@app.post("/api/login")
def login(req: LoginRequest) -> dict:
    """Find-or-create the account for this name (identity only, no password) so the
    person's profiles + learning persist server-side and reload by name."""
    return {"ok": True, **_STORE.find_or_create_user(req.name or "")}


@app.post("/api/profiles/list")
def profiles_list(req: ActivityRequest) -> dict:
    """This account's custom profiles (the gallery; the 4 example personas are added
    client-side), plus the user's confirmed weekly splits per persona so the client can
    route feed-vs-plan from the DB instead of browser localStorage."""
    return {"ok": True, "profiles": _STORE.list_profiles(req.session_id),
            "persona_splits": _STORE.persona_splits(req.session_id)}


@app.post("/api/profiles")
def profiles_save(req: ProfileSaveRequest) -> dict:
    """Create or update one of the account's profiles in the DB."""
    key = req.profile_key or _new_profile_key(req.name)
    _STORE.save_profile(
        req.session_id, key, name=(req.name or "My profile"), goal=(req.goal or ""),
        domains=req.domains, platforms=req.platforms, hours=req.time_budget_hours,
        avoid=(req.avoid or ""), avatar=(req.avatar or ""))
    return {"ok": True, "profile_key": key}


@app.post("/api/profiles/delete")
def profiles_delete(req: ProfileDeleteRequest) -> dict:
    """Delete a profile and everything scoped to it."""
    _STORE.delete_profile(req.session_id, req.profile_key)
    _CTX.pop(_ctx_key(req.session_id, req.profile_key), None)
    return {"ok": True}


def _new_profile_key(name: str | None) -> str:
    import hashlib
    slug = (re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:20]) or "profile"
    return slug + "-" + hashlib.sha1(os.urandom(8)).hexdigest()[:6]


# ── on-demand "concepts" for the opportunity detail page ──────────────────────
# Action-aware: a Contribute card shows the skills you'd use, a Learn card shows what
# you'd learn, a Discuss card shows what's being debated. Generated by the LLM the first
# time a card's detail page is opened, then cached to disk (built only for cards opened).
_CONCEPTS_PATH = summarize.CACHE_PATH.parent / "concepts.json"
_CONCEPTS: dict = json.loads(_CONCEPTS_PATH.read_text()) if _CONCEPTS_PATH.exists() else {}
_CONCEPTS_LOCK = threading.Lock()
_CONCEPT_STOP = {"the", "for", "and", "with", "add", "fix", "update", "new", "data", "node",
                 "issue", "post", "how", "why", "your", "you", "this", "that", "from", "into"}
_CONCEPT_FRAMING = {
    "contribute": ("Concepts and skills you'd use",
                   "the specific technical concepts, skills, libraries, and tools a contributor would use to do this"),
    "learn": ("What you'll learn",
              "the key topics, concepts, and ideas a reader would take away from this"),
    "discuss": ("What's being discussed",
                "the main points, claims, or open questions being debated here"),
    "news": ("What it covers",
             "the key topics, technologies, or developments this covers"),
}


def _gen_concepts(title: str, body: str, ask: str) -> list[str]:
    """Extract 4 to 6 short concept phrases via the LLM; fall back to title keywords."""
    prompt = (f"For the item below, list {ask}. Reply with 4 to 6 short phrases "
              "(1 to 4 words each), comma-separated, on ONE line. No numbering, no markdown, "
              f"no dashes, no preamble.\n\nTitle: {title}\nContent: {body[:1200]}")
    txt = llm.chat([{"role": "user", "content": prompt}], max_tokens=120, temperature=0.2)
    if txt:
        parts = [re.sub(r"^[\s\-*\d.)]+", "", p).strip(" .\"'") for p in re.split(r"[,\n;]+", txt)]
        parts = [p for p in parts if 2 <= len(p) <= 40
                 and not p.lower().startswith(("here", "the item", "concept", "sure", "i "))]
        seen, out = set(), []
        for p in parts:
            k = p.lower()
            if k and k not in seen:
                seen.add(k)
                out.append(p)
        if out:
            return out[:6]
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9+#.]{2,}", title)
             if w.lower() not in _CONCEPT_STOP]
    return list(dict.fromkeys(words))[:5]


def _concepts_for(oid: str, title: str, body: str, bucket: str) -> dict:
    cached = _CONCEPTS.get(oid)
    if cached:
        return cached
    heading, ask = _CONCEPT_FRAMING.get(bucket, _CONCEPT_FRAMING["contribute"])
    obj = {"heading": heading, "items": _gen_concepts(title, body, ask)}
    with _CONCEPTS_LOCK:
        _CONCEPTS[oid] = obj
        try:
            _CONCEPTS_PATH.write_text(json.dumps(_CONCEPTS))
        except Exception:  # noqa: BLE001
            pass
    return obj


# ── capability 6: LLM "Suggested Action" (a concrete post / comment / PR to open) ──
# The spec asks for "LLM-generated ideas on how to engage (suggested posts, comments, or
# GitHub PRs to open)". For each opportunity we draft ONE concrete, ready-to-use action,
# framed by what the user would DO with that source (open a PR, reply to a thread, comment
# on an article), plus a one-line rationale. Generated on demand, cached to disk.
_ACTIONS_PATH = summarize.CACHE_PATH.parent / "actions.json"
_ACTIONS: dict = json.loads(_ACTIONS_PATH.read_text()) if _ACTIONS_PATH.exists() else {}
_ACTIONS_LOCK = threading.Lock()
_ACTION_FRAMING = {
    "contribute": ("Suggested contribution",
                   "Draft a concrete first move for a developer who wants to contribute here: either a short "
                   "comment to post on the issue offering to take it (naming the approach), or a one-line PR "
                   "plan. Be specific to THIS item."),
    "learn": ("Suggested comment",
              "Draft a substantive comment this reader could leave on the article that adds a point, asks a "
              "sharp question, or shares a related experience. Specific to THIS article."),
    "discuss": ("Suggested reply",
                "Draft a thoughtful reply this person could post that adds real value to the thread (a concrete "
                "point, counterpoint, or resource), not a generic 'great post'. Specific to THIS discussion."),
    "news": ("Suggested comment",
             "Draft a comment that adds an informed angle, context, or a sharp question to this story. "
             "Specific to THIS item."),
}


def _gen_action(title: str, body: str, ask: str) -> dict:
    """Draft a concrete engagement action + one-line rationale via the LLM."""
    prompt = (f"{ask}\n\nReturn EXACTLY two lines, no markdown, no dashes:\n"
              "DRAFT: <2 to 3 sentences, first person, ready to paste>\n"
              "WHY: <one short line on why this is a high-value way to engage>\n\n"
              f"Title: {title}\nContent: {body[:1100]}")
    txt = llm.chat([{"role": "user", "content": prompt}], max_tokens=200, temperature=0.4)
    draft, why = "", ""
    if txt:
        flat = " ".join(txt.split())                   # collapse to one line (model may wrap)
        flat = re.sub(r"^\s*DRAFT:\s*", "", flat, flags=re.I)
        parts = re.split(r"\s*WHY:\s*", flat, maxsplit=1, flags=re.I)   # split the rationale off
        draft = parts[0].strip()
        why = parts[1].strip() if len(parts) > 1 else ""
    draft = clean(draft).strip(" \"'")
    why = clean(why).strip(" \"'")
    return {"text": draft[:600], "rationale": why[:240]}


def _action_for(oid: str, title: str, body: str, bucket: str) -> dict | None:
    cached = _ACTIONS.get(oid)
    if cached:
        return cached
    if not llm.available():
        return None
    label, ask = _ACTION_FRAMING.get(bucket, _ACTION_FRAMING["discuss"])
    gen = _gen_action(title, body, ask)
    if not gen.get("text"):
        return None
    obj = {"label": label, **gen}
    with _ACTIONS_LOCK:
        _ACTIONS[oid] = obj
        try:
            _ACTIONS_PATH.write_text(json.dumps(_ACTIONS))
        except Exception:  # noqa: BLE001
            pass
    return obj


class OppRequest(BaseModel):
    opportunity_id: str


@app.post("/api/opportunity")
def opportunity(req: OppRequest) -> dict:
    """Full detail for ONE opportunity (the detail page), self-contained so a direct
    link / refresh works: the corpus row + summary caches + on-demand concepts."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        row = conn.execute(
            "SELECT source, title, body, url, opportunity_type, domain, domains "
            "FROM opportunities WHERE opportunity_id=?", (req.opportunity_id,)).fetchone()
    finally:
        conn.close()
    if not row:
        return {"ok": False, "error": "not found"}
    src, title, body, url, otype, domain, domains_json = row
    title = (title or "").strip()
    body = body or ""
    try:
        doms = json.loads(domains_json) if domains_json else ([domain] if domain else [])
    except Exception:  # noqa: BLE001
        doms = [domain] if domain else []
    bucket = plan.bucket_for(src)
    return {
        "ok": True,
        "id": req.opportunity_id,
        "source": src, "source_label": SRC_LABELS.get(src, src),
        "type": otype or "",
        "domain_label": dom_label(doms[0]) if doms else "",
        "title": title,
        "url": url or "",
        "bucket": bucket,
        "summary": _SUMM.cached(req.opportunity_id),
        "sections": _SUMM.sections(req.opportunity_id),
        "body_text": _clean_lines(body)[:1800] if body else "",
        "repo": _gh_repo(url or "") if src == "github" else "",
        "concepts": _concepts_for(req.opportunity_id, title, body, bucket),
        "suggested_action": _action_for(req.opportunity_id, title, body, bucket),
    }


# ── capability 5: batch analytics & trend detection (persona-scoped) ──────────
# analytics.py precomputed data/trends.json (aggregates) + trends_coords.npz (t-SNE
# coords + per-domain HyperLogLog registers). Here we ROLL those domain summaries up to
# whatever domains a persona cares about, on the fly, by merging the sketches: HLL
# registers merge by element-wise max (distinct reach), per-domain term counts by sum
# (trending in their space). No rescan of the corpus per request.
_TRENDS_PATH = ROOT / "data" / "trends.json"
_TRENDS = json.loads(_TRENDS_PATH.read_text()) if _TRENDS_PATH.exists() else None
try:
    _Z = np.load(ROOT / "data" / "trends_coords.npz", allow_pickle=True)
    _HLL_KEYS = [str(k) for k in _Z["domain_keys"]]
    _HLL_REGS = _Z["hll_regs"]
    _HLL_P = int(round(np.log2(_HLL_REGS.shape[1]))) if _HLL_REGS.size else 14
except Exception:  # noqa: BLE001
    _HLL_KEYS, _HLL_REGS, _HLL_P = [], None, 14

# the onboarding form's two alias keys -> the corpus domain keys trends.json uses
_TRENDS_DOMAIN_ALIAS = {"blockchain_web3": "blockchain", "embedded_iot": "embedded_systems"}


def _scope_domains(persona: dict) -> list[str]:
    """The persona's domains as corpus keys present in the analytics (empty => breadth)."""
    if not _TRENDS:
        return []
    present = _TRENDS["corpus"]["domains_present"]
    doms = [_TRENDS_DOMAIN_ALIAS.get(d, d) for d in (persona.get("domains") or [])]
    return [d for d in doms if d in present]


def _merged_distinct(domain_keys: list[str]) -> int:
    """Distinct authors across these domains, by MERGING their HyperLogLog registers
    (element-wise max = set union). Empty => the global distinct count."""
    if not _TRENDS:
        return 0
    if _HLL_REGS is None or not domain_keys:
        return int(_TRENDS["global"]["distinct_authors"])
    idx = [_HLL_KEYS.index(d) for d in domain_keys if d in _HLL_KEYS]
    if not idx:
        return 0
    merged = _HLL_REGS[idx].max(axis=0).astype(np.uint8)
    h = HyperLogLog(p=_HLL_P)
    h.registers = merged
    return h.count()


def _rising_opps(domain_keys: list[str], breadth: bool, limit: int = 4) -> list[dict]:
    """A few breakout opportunities in the persona's space: high momentum (velocity)
    AND real engagement, recent. Linked into the detail page."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        if breadth or not domain_keys:
            rows = conn.execute(
                "SELECT o.opportunity_id, o.title, o.source, o.domain, o.url, "
                "o.num_comments, o.score, f.velocity FROM opportunities o "
                "JOIN opportunity_features f ON o.opportunity_id=f.opportunity_id "
                "ORDER BY f.velocity DESC LIMIT 120").fetchall()
        else:
            ph = ",".join("?" * len(domain_keys))
            rows = conn.execute(
                f"SELECT o.opportunity_id, o.title, o.source, o.domain, o.url, "
                f"o.num_comments, o.score, f.velocity FROM opportunities o "
                f"JOIN opportunity_features f ON o.opportunity_id=f.opportunity_id "
                f"WHERE o.domain IN ({ph}) ORDER BY f.velocity DESC LIMIT 120",
                domain_keys).fetchall()
    finally:
        conn.close()
    out = []
    for oid, title, src, dom, url, nc, sc, vel in rows:
        eng = int(nc or 0) + int(sc or 0)
        if eng < _RISING_FLOOR.get(src, 5) or not (title or "").strip():
            continue
        out.append({"id": oid, "title": (title or "").strip()[:140],
                    "source": src, "source_label": SRC_LABELS.get(src, src),
                    "domain_label": dom_label(dom), "url": url or "",
                    "engagement": eng, "velocity": round(float(vel or 0), 2)})
        if len(out) >= limit:
            break
    return out


def _trends_narrative(cards: list[dict], breadth: bool) -> str:
    """One honest plain sentence about what is moving in the persona's space."""
    risers = [c for c in cards if c["direction"] == "up"]
    coolers = [c for c in cards if c["direction"] == "down"]
    parts = []
    if risers:
        top = risers[0]
        parts.append(f"{top['label']} is gaining momentum (+{top['delta_pp']:.1f} share points "
                     f"in the latest week)")
    if coolers:
        parts.append(f"{coolers[-1]['label']} is cooling")
    scope = "across all topics" if breadth else "in your space"
    if not parts:
        return f"Activity {scope} is steady week over week."
    return f"This week {scope}, " + ", while ".join(parts) + "."


def _fresh_this_week(domain_keys: list[str], breadth: bool, days: int = 7, limit: int = 6) -> dict:
    """The newest opportunities in the persona's space: a by-source split of recent inflow
    plus the freshest items to open. 'This week' = the most recent `days` of the FROZEN
    snapshot (relative to the latest collected item), not wall-clock now. created_at is
    stored in mixed Z / +00:00 forms, so we parse in Python rather than string-compare."""
    from datetime import datetime, timedelta
    conn = sqlite3.connect(str(DB_PATH))
    try:
        if breadth or not domain_keys:
            where, params = "", []
        else:
            ph = ",".join("?" * len(domain_keys))
            where, params = f"WHERE domain IN ({ph})", list(domain_keys)
        meta_rows = conn.execute(
            f"SELECT created_at, source FROM opportunities {where}", params).fetchall()
        sample_rows = conn.execute(
            f"SELECT opportunity_id, title, source, domain, url, created_at, num_comments, score "
            f"FROM opportunities {where} ORDER BY created_at DESC LIMIT 60", params).fetchall()
    finally:
        conn.close()

    def _parse(s):
        try:
            return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            return None

    dated = [(_parse(c), src) for c, src in meta_rows]
    dated = [(d, src) for d, src in dated if d is not None]
    if not dated:
        return {"window_days": days, "since": None, "total": 0, "by_source": [], "items": []}
    newest = max(d for d, _ in dated)
    cutoff = newest - timedelta(days=days)
    by_source: dict[str, int] = {}
    total = 0
    for d, src in dated:
        if d >= cutoff:
            total += 1
            by_source[src] = by_source.get(src, 0) + 1
    bys = [{"source": s, "source_label": SRC_LABELS.get(s, s), "count": c}
           for s, c in sorted(by_source.items(), key=lambda x: -x[1])]
    items = []
    for oid, title, src, dom, url, cat, nc, sc in sample_rows:
        if len(items) >= limit:
            break
        dt = _parse(cat)
        if dt is None or not (title or "").strip():
            continue
        hrs = max(0, int((newest - dt).total_seconds() // 3600))
        items.append({"id": oid, "title": (title or "").strip()[:130],
                      "source": src, "source_label": SRC_LABELS.get(src, src),
                      "domain_label": dom_label(dom), "url": url or "",
                      "ago_hours": hrs, "engagement": int(nc or 0) + int(sc or 0)})
    return {"window_days": days, "since": cutoff.date().isoformat(),
            "total": total, "by_source": bys, "items": items}


_DOMAIN_TOPICS = None
_TOPIC_STOP = frozenset({
    "discussion", "built", "made", "using", "use", "new", "data", "code", "help", "question", "show",
    "showdev", "project", "best", "good", "need", "work", "app", "tools", "tool", "tips", "guide",
    "tutorial", "blog", "article", "ai", "programming", "github", "documentation", "help wanted",
    "good first issue", "i will not promote", "i will promote it", "promotional", "promote", "beginner",
    "beginners", "news", "release", "update", "personal project showcase", "alternatives", "opensource",
    "open source", "productivity", "other", "enhancement", "career", "needs help", "general", "bug",
    "feature", "wip", "todo", "misc", "meta", "off topic", "advice", "resources", "learning",
})


def _domain_topics() -> dict:
    """Distinctive, human-readable subtopics per domain for the topic mind map: the tags that are
    both common in a domain AND concentrated there (a TF-IDF-style score = in-domain share times
    concentration), with generic words, PR/flair labels, and the domain's own name filtered out, so
    a subtopic is an informative thing people actually search for (pytorch, pricing, terraform,
    solana) rather than a filler word (python, data, built). Computed once, then cached."""
    global _DOMAIN_TOPICS
    if _DOMAIN_TOPICS is not None:
        return _DOMAIN_TOPICS
    import collections
    dom_tags = collections.defaultdict(collections.Counter)
    glob = collections.Counter()
    dom_total = collections.Counter()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        for dom, tags in conn.execute("SELECT domain, tags FROM opportunities"):
            dom_total[dom] += 1
            try:
                tl = json.loads(tags or "[]")
            except Exception:  # noqa: BLE001
                tl = []
            for t in tl:
                t = (t or "").strip().lower()
                if t:
                    dom_tags[dom][t] += 1
                    glob[t] += 1
    finally:
        conn.close()

    def ok(t):
        if len(t) < 3 or len(t) > 20 or t in _TOPIC_STOP:
            return False
        if any(ch in t for ch in "&!:/"):
            return False
        if t.startswith(("missing", "level", "size", "dco", "status")) or t.endswith(("-classification", "-review", "-program")):
            return False
        return "news -" not in t and not t.isdigit()

    out = {}
    for dom, c in dom_tags.items():
        tot = dom_total[dom] or 1
        lab = dom_label(dom).lower()
        excl = set(re.findall(r"[a-z0-9]+", lab)) | {lab.replace(" ", ""), dom.replace("_", "")}
        scored = []
        for t, n in c.items():
            if n < 4 or not ok(t):
                continue
            if t in excl or t.replace(" ", "").replace("-", "") in excl:
                continue
            scored.append((t, (n / tot) * (n / (glob[t] + 2))))
        scored.sort(key=lambda x: -x[1])
        out[dom] = [t for t, _ in scored[:6]]
    _DOMAIN_TOPICS = out
    return out


def _persona_kpis(persona: dict, scope_keys: list[str], breadth: bool, *,
                  reach: int, mover: dict | None, cards: list[dict],
                  communities: list[dict], fresh: dict) -> list[dict]:
    """Mode-aware KPI cards: the same aggregation idea (count / sum / distinct / momentum over
    the whole snapshot) but framed as the decision the persona actually cares about, given HOW
    they engage. Each card carries a 'how' string for the click-to-flip back."""
    mode = score.mode_for_persona(persona) if persona else "contribute"
    ndom = 0 if breadth else len(scope_keys)
    up = sum(1 for c in cards if c.get("direction") == "up")

    def C(value, label, hint, how, tone=""):
        return {"value": str(value), "label": label, "hint": hint, "how": how, "tone": tone}

    out = [C(f"{reach:,}", "People active in your space",
             (f"across your {ndom} domains" if ndom else "across all topics"),
             "Distinct authors and contributors active across your domains, estimated with HyperLogLog: "
             "each domain's registers are merged element-wise by max (a set union), giving ~0.4% error in "
             "about 16 KB instead of storing every contributor id.")]
    if mover:
        sgn = "+" if mover["delta_pp"] > 0 else ""
        out.append(C(mover["label"], "Heating up fastest" if mover["direction"] == "up" else "Biggest mover this week",
                     f"{sgn}{mover['delta_pp']:.1f} share points this week",
                     "The domain with the largest week-over-week change in its share of all weekly activity in "
                     "your space (complete weeks only, since the snapshot skews recent). We compare share, not "
                     "raw volume, so collection bias does not distort it.", tone=mover["direction"]))

    conn = sqlite3.connect(str(DB_PATH))
    try:
        dom_clause = "" if (breadth or not scope_keys) else f"o.domain IN ({','.join('?' * len(scope_keys))})"
        dp = [] if (breadth or not scope_keys) else list(scope_keys)

        def W(*conds):
            cs = [c for c in ([dom_clause] + list(conds)) if c]
            return ("WHERE " + " AND ".join(cs)) if cs else ""

        if mode == "contribute":
            gh = conn.execute("SELECT COUNT(*) FROM opportunities o " + W("o.source='github'"), dp).fetchone()[0]
            efforts = sorted(e for (e,) in conn.execute(
                "SELECT f.effort_min FROM opportunities o JOIN opportunity_features f "
                "ON o.opportunity_id=f.opportunity_id " + W(), dp).fetchall() if e is not None)
            quick_pct = round(100 * sum(1 for e in efforts if e < 60) / len(efforts)) if efforts else 0
            med = int(efforts[len(efforts) // 2]) if efforts else 0
            standout = conn.execute(
                "SELECT COUNT(*) FROM opportunities o JOIN opportunity_features f "
                "ON o.opportunity_id=f.opportunity_id " + W("o.source='github'", "f.standout>=0.55"), dp).fetchone()[0]
            out += [
                C(f"{gh:,}", "Open issues you could take", "the directly contributable work",
                  "A count of open GitHub issues across your domains, filtered from the full snapshot. Unlike a "
                  "raw opportunity count, this is work you can actually pick up and ship."),
                C(f"{quick_pct}%", "Are quick wins, under an hour", f"median effort about {med} min",
                  "The share of in-scope opportunities whose estimated effort (an LLM estimate plus a per-type "
                  f"heuristic) is under 60 minutes; the median across your space is about {med} minutes. It tells "
                  "you how much you can realistically fit in a week."),
                C(f"{standout:,}", "Repos with room to be seen", "high activity, few contributors",
                  "Repositories scoring high on the standout signal (activity relative to contributor count): "
                  "busy projects that still have few hands, where a newcomer's work gets noticed."),
            ]
        elif mode in ("discuss", "promote"):
            social = conn.execute(
                "SELECT COUNT(*) FROM opportunities o " + W(
                    "o.source IN ('reddit','bluesky','hackernews')",
                    "(COALESCE(o.score,0)+COALESCE(o.num_comments,0))>=10"), dp).fetchone()[0]
            total_eng = conn.execute(
                "SELECT SUM(COALESCE(o.score,0)+COALESCE(o.num_comments,0)) FROM opportunities o " + W(), dp).fetchone()[0] or 0
            top = communities[0] if communities else None
            out += [
                C(f"{social:,}", "Conversations you could join", "live threads with real traction",
                  "Reddit, Bluesky and Hacker News threads in your domains that already have real engagement "
                  "(comments + score above a noise floor): discussions where your take could land."),
                C(f"{int(total_eng):,}", "Total engagement in your space", "comments + score, all sources",
                  "The sum of comments and score across every opportunity in your domains, the size of the "
                  "conversation you would be stepping into."),
                (C(top["community"], "Most active community",
                   f"{top['engagement']:,} engagement, {top['authors']:,} voices",
                   "The community with the highest total engagement in your space; its distinct voices are "
                   "counted with HyperLogLog.") if top else
                 C("-", "Most active community", "no community signal", "No community signal in your current scope.")),
            ]
        else:  # monitor / default
            brk = conn.execute(
                "SELECT COUNT(*) FROM opportunities o JOIN opportunity_features f ON o.opportunity_id=f.opportunity_id "
                + W("f.velocity>=0.6", "(COALESCE(o.score,0)+COALESCE(o.num_comments,0))>=8"), dp).fetchone()[0]
            out += [
                C(f"{up}", "Topics gaining steam", f"of your {ndom or len(cards)} domains",
                  "How many of your domains rose in their share of weekly activity versus the previous complete "
                  "week, where momentum is actually building."),
                C(f"{brk:,}", "Breakout items right now", "accelerating fastest",
                  "Opportunities in the top velocity band that also carry real engagement, the items picking up "
                  "speed fastest in your space right now."),
                C(f"{(fresh or {}).get('total', 0):,}", "New this week",
                  f"in the last {(fresh or {}).get('window_days', 7)} days",
                  "Opportunities created in the most recent week of the snapshot, the fresh inflow into your space."),
            ]
    finally:
        conn.close()
    return out[:5]


def _persona_trends(persona: dict) -> dict:
    if not _TRENDS:
        return {"ok": False, "error": "analytics not built (run code/analytics.py)"}
    doms = _scope_domains(persona)
    present = _TRENDS["corpus"]["domains_present"]
    scope_keys = doms or present
    breadth = not doms
    D = _TRENDS["domains"]
    chart_weeks = _TRENDS["corpus"]["weeks"]

    cards = []
    for k in scope_keys:
        d = D.get(k)
        if not d:
            continue
        w = d["wow"]
        cards.append({
            "key": k, "label": d["label"], "color": trends_render.color_for(k),
            "count": d["count"], "distinct_authors": d["distinct_authors"],
            "share_last": w["share_last"], "delta_pp": w["delta_pp"], "momentum": w["momentum"],
            "rising": d["rising_score"],
            "direction": "up" if w["delta_pp"] > 0.1 else ("down" if w["delta_pp"] < -0.1 else "flat"),
            "top_terms": [t for t, _ in d["top_terms"][:6]],
            "topics": _domain_topics().get(k, []),
            "links": [{"key": r["key"], "sim": round(float(r["sim"]), 3)}
                      for r in d.get("related", []) if r["key"] in scope_keys][:4],
        })
    cards.sort(key=lambda c: -c["delta_pp"])

    term_acc: dict[str, int] = {}
    for k in scope_keys:
        for t, c in D.get(k, {}).get("top_terms", []):
            term_acc[t] = term_acc.get(t, 0) + int(c)
    trending_terms = [{"term": t, "count": c}
                      for t, c in sorted(term_acc.items(), key=lambda x: -x[1])[:14]]

    comm_all = []
    for k in scope_keys:
        for c in D.get(k, {}).get("top_communities", []):
            comm_all.append({**c, "domain": D[k]["label"]})
    comm_all.sort(key=lambda c: -c["engagement"])
    seen, communities = set(), []
    for c in comm_all:
        if c["community"] in seen:
            continue
        seen.add(c["community"])
        communities.append(c)
        if len(communities) >= 6:
            break

    totals_by_week = [0] * len(chart_weeks)
    for k in scope_keys:
        for i, v in enumerate(D[k]["weekly_counts"]):
            totals_by_week[i] += v
    top_for_lines = sorted(scope_keys, key=lambda k: -D[k]["count"])[:5]
    series = []
    for k in top_for_lines:
        cc = D[k]["weekly_counts"]
        share = [round(cc[i] / totals_by_week[i] * 100, 1) if totals_by_week[i] else 0.0
                 for i in range(len(chart_weeks))]
        series.append({"key": k, "label": D[k]["label"], "color": trends_render.color_for(k),
                       "share": share, "counts": cc})

    related = []
    if not breadth:
        smap: dict[str, float] = {}
        for k in scope_keys:
            for r in D.get(k, {}).get("related", []):
                if r["key"] in scope_keys:
                    continue
                smap[r["key"]] = max(smap.get(r["key"], 0.0), r["sim"])
        related = [{"key": kk, "label": D[kk]["label"], "color": trends_render.color_for(kk),
                    "sim": round(s, 3), "topics": _domain_topics().get(kk, [])}
                   for kk, s in sorted(smap.items(), key=lambda x: -x[1])[:4]]

    mover = cards[0] if cards else None
    kpis = {
        "opportunities": sum(D[k]["count"] for k in scope_keys),
        "distinct_reach": _merged_distinct([] if breadth else scope_keys),
        "trending_terms": len(term_acc),
        "domains_active": len(scope_keys),
        "top_mover": ({"label": mover["label"], "delta_pp": mover["delta_pp"],
                       "direction": mover["direction"]} if mover else None),
    }
    fresh_data = _fresh_this_week(scope_keys, breadth)
    kpi_cards = _persona_kpis(persona, scope_keys, breadth, reach=kpis["distinct_reach"],
                              mover=mover, cards=cards, communities=communities, fresh=fresh_data)
    img = trends_render.persona_map_path(set() if breadth else set(scope_keys))
    return {
        "ok": True,
        "persona": _persona_card(persona),
        "scope": {"breadth": breadth, "n": len(scope_keys),
                  "domains": [D[k]["label"] for k in scope_keys],
                  "label": "all topics (breadth)" if breadth else f"your {len(scope_keys)} domains"},
        "kpis": kpis,
        "map_img": f"/trends_img/{img}",
        "domains": cards,
        "trending_terms": trending_terms,
        "communities": communities,
        "weekly": {"weeks": chart_weeks, "totals": totals_by_week, "series": series},
        "related": related,
        "rising_opps": _rising_opps(scope_keys, breadth),
        "fresh": fresh_data,
        "kpi_cards": kpi_cards,
        "narrative": _trends_narrative(cards, breadth),
        "snapshot_note": _TRENDS["corpus"].get("snapshot_note", ""),
        "benchmarks": _TRENDS["benchmarks"],
        "generated_at": _TRENDS["generated_at"],
    }


class TrendsRequest(BaseModel):
    persona_id: str | None = None
    profile: Profile | None = None
    session_id: str | None = None
    persona_key: str | None = None


@app.post("/api/trends")
def trends_endpoint(req: TrendsRequest) -> dict:
    """Persona-scoped batch-analytics view: their domains' distribution, trending terms,
    active communities, week-over-week share momentum, distinct reach (merged HLL),
    related domains, and breakout opportunities. Backs the Trends tab."""
    persona = _resolve_persona(req.persona_id, req.profile)
    return _persona_trends(persona)


# ── capability 6: the downloadable engagement brief ───────────────────────────
# A per-persona document pulling the whole product together: their weekly plan, their
# top ranked opportunities with "Why this?" + an LLM "Suggested action", what is trending
# in their space, and what the feedback learner has picked up. Rendered server-side as a
# self-contained, print-ready HTML file (the same markup powers the in-app view via an
# iframe and the one-click download), so it works offline and prints cleanly to PDF.
class BriefRequest(BaseModel):
    persona_id: str | None = None
    profile: Profile | None = None
    session_id: str | None = None
    persona_key: str | None = None
    split: dict[str, float] | None = None


def _brief_data(req: BriefRequest) -> dict:
    persona = _resolve_persona(req.persona_id, req.profile)
    pkey = req.persona_key or _persona_key(persona)
    mode_key = score.mode_for_persona(persona)
    sources, bucket_weights, active_split = _split_to_engine(req.split)
    learned = _STORE.weights(req.session_id, pkey, mode_key) if req.session_id else None

    # The plan split drives the per-bucket distribution of the top moves. If the request carried no
    # confirmed split, fall back to the recommended one AND drive the recommend with it too, so the
    # candidate pool is balanced across buckets (Contribute / Learn / Discuss / News).
    split_disp = active_split or plan.recommend_split(persona)["split"]
    if not active_split:
        sources, bucket_weights, _ = _split_to_engine(split_disp)
    out = engine().recommend(persona, k=30, weights=learned,
                             sources=sources, bucket_weights=bucket_weights)
    ranked = out.get("ranked", [])

    buckets = [{"key": b["key"], "label": b["label"], "pct": round(split_disp.get(b["key"], 0))}
               for b in plan.BUCKETS if split_disp.get(b["key"], 0) > 0]

    # Group the ranked pool by activity bucket, then take about pct/10 of every-ten from each, so
    # the brief's top moves mirror the user's plan split (Contribute / Learn / Discuss / News),
    # section by section, rather than one flat list.
    by_bucket = {}
    for r in ranked:
        it = _item(r)
        by_bucket.setdefault(it["bucket"], []).append((r, it))
    top = []
    for b in buckets:
        target = max(1, round(b["pct"] / 10))          # "about N of every 10 items"
        for r, it in by_bucket.get(b["key"], [])[:target]:
            rank = len(top) + 1
            action = _action_for(it["id"], it["title"], r.get("body_raw") or "", it["bucket"]) if rank <= 3 else None
            top.append({
                "rank": rank, "id": it["id"], "title": it["title"],
                "source": it["source"], "source_label": it["source_label"], "domain_label": it["domain_label"],
                "url": it["url"], "why": it["why"],
                "effort_min": it["effort_min"], "bucket": it["bucket"],
                "sections": it.get("sections"), "summary": it.get("summary"),
                "suggested_action": action,
            })

    tr = _persona_trends(persona)
    trends = None
    if tr.get("ok"):
        trends = {
            "narrative": tr.get("narrative"),
            "movers": [{"label": c["label"], "delta_pp": c["delta_pp"], "direction": c["direction"]}
                       for c in tr.get("domains", [])[:5]],
            "terms": [t["term"] for t in tr.get("trending_terms", [])[:10]],
            "kpis": tr.get("kpis"), "snapshot_note": tr.get("snapshot_note"),
        }

    lr = _STORE.learner_row(req.session_id, pkey) if req.session_id else None
    learned_obj = {"n": lr.n, "summary": feedback.learned_summary(lr)} if lr else None

    return {
        "ok": True, "persona": _persona_card(persona), "mode_label": score.MODES[mode_key].label,
        "plan": {"buckets": buckets}, "top": top, "trends": trends, "learned": learned_obj,
        "generated_at": dt.datetime.now().strftime("%B %-d, %Y"),
    }


def _brief_html(d: dict) -> str:
    """Render the brief as an ACADEMIC report document (centered title block, serif type,
    bold section headings, justified paragraphs, page numbers at the bottom centre), matching
    a course homework/report PDF rather than a slide deck. Used for the in-app preview iframe
    AND as the source WeasyPrint converts to a PDF. Small platform logos stay inline."""
    esc = _html.escape
    p = d["persona"]
    name = esc(p.get("name", "You"))
    role = esc(p.get("role", "") or d.get("mode_label", ""))
    mode_label = esc(d.get("mode_label", "") or role)
    doms = ", ".join(esc(x) for x in p.get("domains", [])) or "all topics"
    plat_keys = p.get("platform_keys") or []
    plats = ", ".join(esc(SRC_LABELS.get(s, s)) for s in plat_keys) or esc(", ".join(p.get("platforms", [])))
    hrs = esc(str(p.get("time_budget_hours", "")))
    goal = esc(p.get("goal", ""))

    _BK_SRC = {"contribute": ["github"], "learn": ["devto"],
               "discuss": ["reddit", "bluesky"], "news": ["hackernews"]}
    plan_rows = "".join(
        f'<tr><td class="plk">{"".join(brand_logos.logo(s, 11) for s in _BK_SRC.get(b["key"], []))} {esc(b["label"])}</td>'
        f'<td class="plv">{b["pct"]}%</td>'
        f'<td class="plnote">about {round(b["pct"] / 10)} of every 10 items</td></tr>'
        for b in d["plan"]["buckets"])

    def opp_item(o, full):
        slog = brand_logos.logo(o.get("source", ""), 11)
        meta = (f'<p class="ometa">Source: {slog} {esc(o["source_label"])}. '
                f'Domain: {esc(o["domain_label"])}. Estimated effort: about {o["effort_min"]} minutes.</p>')
        why = f'<p class="owhy"><b>Why this:</b> {esc(o["why"])}</p>' if o.get("why") else ""
        act = ""
        if full and o.get("suggested_action") and o["suggested_action"].get("text"):
            a = o["suggested_action"]
            act = ('<p class="oact"><b>Suggested action:</b> ' + esc(a["text"])
                   + (f' <span class="orat">({esc(a["rationale"])})</span>' if a.get("rationale") else "")
                   + '</p>')
        _u = o.get("url") or ""
        _t = esc(o["title"])
        _tl = f'<a href="{esc(_u)}">{_t}</a>' if _u else _t          # the opportunity title is the engagement link
        link = f'<p class="olnk"><a href="{esc(_u)}">{esc(_u)}</a></p>' if _u else ""
        return f'<li class="opp"><p class="ot">{_tl}</p>{meta}{why}{act}{link}</li>'

    # group the top moves into bucket sections (Contribute / Learn / Discuss & watch / News)
    _bk_label = {b["key"]: b["label"] for b in d["plan"]["buckets"]}
    _plan_order = [b["key"] for b in d["plan"]["buckets"]]
    _grouped = {}
    for o in d["top"]:
        _grouped.setdefault(o["bucket"], []).append(o)
    _order = _plan_order + [k for k in _grouped if k not in _plan_order]
    top_sections = ""
    for bk in _order:
        items = _grouped.get(bk)
        if not items:
            continue
        logos = "".join(brand_logos.logo(s, 12) for s in _BK_SRC.get(bk, []))
        top_sections += (f'<h3 class="bsec">{logos} {esc(_bk_label.get(bk, bk.title()))}'
                         f' <span class="bcount">{len(items)} pick{"" if len(items) == 1 else "s"}</span></h3>'
                         + '<ol class="tops">'
                         + "".join(opp_item(o, (o.get("rank", 9) <= 3)) for o in items)
                         + "</ol>")

    trends_html = ""
    if d.get("trends"):
        t = d["trends"]
        k = t.get("kpis") or {}
        movers = "; ".join(
            f'{esc(m["label"])} ({"+" if m["delta_pp"] > 0 else ""}{m["delta_pp"]:.1f} pp)'
            for m in t.get("movers", []))
        terms = ", ".join(esc(x) for x in t.get("terms", [])[:10])
        trends_html = (
            '<h2>Trends in Your Space</h2>'
            f'<p>{esc(t.get("narrative") or "")} In total there are about {k.get("opportunities", 0):,} '
            f'opportunities and {k.get("distinct_reach", 0):,} distinct contributors '
            f'(estimated with HyperLogLog) across {k.get("domains_active", 0)} active domains.</p>'
            + (f'<p><b>Momentum by domain:</b> {movers}.</p>' if movers else '')
            + (f'<p><b>Trending terms:</b> {terms}.</p>' if terms else ''))

    learned_html = ""
    if d.get("learned"):
        lr = d["learned"]
        learned_html = ('<h2>What EngageIQ Learned From You</h2>'
                        f'<p>Across {lr["n"]} feedback signal(s): {esc(lr["summary"])} This is now shaping '
                        'how your feed is ranked.</p>')

    note = esc((d.get("trends") or {}).get("snapshot_note", "")) if d.get("trends") else ""
    plan_sentence = ", ".join(f'{esc(b["label"])} {b["pct"]}%' for b in d["plan"]["buckets"]) or "no split set"
    _topo = d["top"][0] if d.get("top") else None
    top_title = esc(_topo["title"]) if _topo else ""
    _topu = (_topo.get("url") if _topo else "") or ""
    top_title_html = f'<a href="{esc(_topu)}">{top_title}</a>' if (top_title and _topu) else top_title
    tnarr = esc((d.get("trends") or {}).get("narrative", "")) if d.get("trends") else ""
    exec_summary = (
        f'This brief organises the engagement opportunities for {name} this week across {doms}, '
        f'for a {mode_label.lower() or "balanced"} style of engagement within a weekly budget of {hrs} hours. '
        + (f'The stated goal is "{goal}" ' if goal else '')
        + (f'{tnarr} ' if tnarr else '')
        + f'The recommended split of effort is {plan_sentence}'
        + (f', and the single highest-leverage move this week is "{top_title_html}".' if top_title else '.'))
    sub = name + ((" &middot; " + role) if role else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Weekly Engagement Brief - {name}</title>
<style>
@page {{
  size: A4; margin: 24mm 22mm 22mm;
  @bottom-center {{ content: counter(page); color: #333; font-size: 9pt;
    font-family: "Latin Modern Roman", "Times New Roman", Georgia, serif; }}
}}
* {{ box-sizing: border-box; }}
html {{ font-family: "Latin Modern Roman", "CMU Serif", "Times New Roman", Georgia, serif;
        color: #111; font-size: 11.2pt; line-height: 1.5; }}
body {{ margin: 0; }}
svg {{ vertical-align: -1.5px; }}
.title {{ text-align: center; }}
.title h1 {{ font-size: 21pt; font-weight: 700; margin: 2pt 0; }}
.title .sub {{ font-size: 12.5pt; margin: 8pt 0 3pt; }}
.title .date {{ font-size: 11pt; color: #333; margin: 3pt 0; }}
.title .meta {{ font-size: 9.5pt; color: #444; margin-top: 5pt; }}
hr.rule {{ border: none; border-top: 1px solid #999; margin: 15pt 0 16pt; }}
h2 {{ font-size: 13.5pt; font-weight: 700; margin: 17pt 0 6pt; }}
p {{ margin: 0 0 8pt; text-align: justify; }}
ol.tops {{ margin: 0; padding: 0; list-style: none; counter-reset: opp; }}
ol.tops > li {{ counter-increment: opp; margin-bottom: 11pt; padding-left: 22pt; position: relative; break-inside: avoid; }}
ol.tops > li::before {{ content: counter(opp) "."; position: absolute; left: 2pt; top: 0; font-weight: 700; }}
h3.bsec {{ font-size: 11.5pt; font-weight: 700; margin: 15pt 0 5pt; color: #1a1a1a; }}
h3.bsec svg {{ vertical-align: -1.5px; }}
.bcount {{ font-weight: 400; color: #777; font-size: 9.5pt; }}
.ot {{ font-weight: 700; margin: 0 0 2pt; text-align: left; }}
.ometa {{ font-size: 10pt; color: #333; margin: 0 0 2pt; text-align: left; }}
.owhy {{ margin: 0 0 2pt; }}
.oact {{ margin: 0 0 2pt; }}
.orat {{ color: #555; }}
.olnk {{ font-size: 9pt; color: #555; margin: 0; word-break: break-all; text-align: left; }}
table.plan {{ border-collapse: collapse; margin: 2pt 0 8pt; }}
table.plan td {{ padding: 3pt 16pt 3pt 0; font-size: 10.5pt; vertical-align: baseline; }}
table.plan td svg {{ margin-right: 2px; }}
table.plan .plv {{ font-weight: 700; }}
table.plan .plnote {{ color: #555; font-size: 10pt; }}
.foot {{ margin-top: 14pt; border-top: 1px solid #ccc; padding-top: 8pt; font-size: 9pt; color: #555; line-height: 1.45; }}
a {{ color: #4338ca; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.ot a {{ color: #4338ca; }}
.olnk a {{ color: #4338ca; }}
table.bhead {{ width: 100%; border-collapse: collapse; border-bottom: 2px solid #4f46e5; margin: 0 0 16pt; }}
table.bhead td {{ border: none; padding: 0 0 7pt; vertical-align: middle; }}
.bhead .bw {{ font-family: Arial, Helvetica, sans-serif; font-weight: 700; font-size: 15pt; color: #4f46e5; letter-spacing: -0.3pt; margin-left: 7pt; vertical-align: middle; }}
.bhead svg {{ vertical-align: middle; }}
.bhead .br {{ text-align: right; font-family: Arial, Helvetica, sans-serif; font-size: 9.5pt; color: #777; vertical-align: middle; }}
@media screen {{ body {{ padding: 40px 52px; }} }}
</style></head><body>
<table class="bhead"><tr><td><svg width="24" height="24" viewBox="0 0 64 64"><defs><linearGradient id="bhg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#4f46e5"/><stop offset="1" stop-color="#6e63ec"/></linearGradient></defs><rect width="64" height="64" rx="15" fill="url(#bhg)"/><text x="32" y="34" font-family="Arial,Helvetica,sans-serif" font-size="29" font-weight="700" fill="#ffffff" text-anchor="middle" dominant-baseline="central" letter-spacing="-1">IQ</text></svg><span class="bw">EngageIQ</span></td><td class="br">Where to invest your attention</td></tr></table>
<div class="title">
  <h1>Weekly Engagement Brief</h1>
  <div class="sub">{sub}</div>
  <div class="date">{esc(d["generated_at"])}</div>
  <div class="meta">Prepared by EngageIQ. Engagement mode: {mode_label}. Weekly budget: {hrs} hours. Sources: {plats}.</div>
</div>
<hr class="rule">
<h2>Executive Summary</h2>
<p>{exec_summary}</p>
<h2>Your Weekly Plan</h2>
<p>EngageIQ recommends dividing your week across activity types as follows, where each ten percent corresponds to roughly one of every ten ranked items in your feed.</p>
<table class="plan">{plan_rows or '<tr><td>No split set.</td></tr>'}</table>
<h2>Your Top Moves This Week</h2>
{top_sections}
{trends_html}
{learned_html}
<h2>Methodology</h2>
<p class="foot">EngageIQ scores each opportunity on relevance, community health, visibility, and effort, using Sentence-BERT retrieval, a cross-encoder re-rank, and MMR for diversity. It learns from your feedback with an online contextual bandit, and computes the trends above with a Count-Min Sketch and HyperLogLog over the full corpus. {note}</p>
</body></html>"""


@app.post("/api/brief")
def brief_endpoint(req: BriefRequest) -> dict:
    """Structured engagement-brief data (plan + top opportunities with why + suggested
    actions + trends + what was learned)."""
    return _brief_data(req)


@app.post("/api/brief/html", response_class=HTMLResponse)
def brief_html_endpoint(req: BriefRequest) -> HTMLResponse:
    """The same brief as a self-contained, print-ready HTML document (in-app iframe preview)."""
    return HTMLResponse(_brief_html(_brief_data(req)))


def _brief_pdf(d: dict) -> bytes:
    """Convert the print-designed brief HTML into a real PDF with WeasyPrint (a proper
    HTML-to-PDF render, not a browser screenshot). Imported lazily so the API still starts
    if WeasyPrint's native libraries are ever unavailable."""
    from weasyprint import HTML as _WeasyHTML
    return _WeasyHTML(string=_brief_html(d), base_url=str(MOCKUPS)).write_pdf()


@app.post("/api/brief/pdf")
def brief_pdf_endpoint(req: BriefRequest):
    """The engagement brief rendered to an authentic, downloadable PDF via WeasyPrint."""
    d = _brief_data(req)
    name = re.sub(r"[^A-Za-z0-9]+", "-", (d.get("persona") or {}).get("name", "brief")).strip("-") or "brief"
    try:
        pdf = _brief_pdf(d)
    except Exception as e:  # noqa: BLE001 -- fall back to the HTML doc if the PDF engine fails
        return HTMLResponse(f"<p>PDF generation unavailable: {_html.escape(str(e))}</p>", status_code=500)
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="EngageIQ-Brief-{name}.pdf"'})


APP_HTML = str(MOCKUPS / "index.html")


def _app_shell() -> FileResponse:
    # no-cache: the browser may keep a copy but MUST revalidate (via ETag) before using
    # it, so it never serves a stale SPA. This app's HTML+JS changes often, and a cached
    # old shell was showing stale UI (e.g. the old inline "create a profile" form).
    # Set on the response object explicitly (FileResponse ignores a headers= kwarg here).
    resp = FileResponse(APP_HTML)
    resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


@app.get("/")
def root() -> FileResponse:
    """The single-page app, served at the clean root URL."""
    return _app_shell()


@app.get("/plan")
@app.get("/plan/{rest:path}")
def plan_route(rest: str = "") -> FileResponse:
    """Client-side route: a refresh or direct link to /plan/<persona> serves the app."""
    return _app_shell()


@app.get("/hub")
@app.get("/hub/{rest:path}")
def hub_route(rest: str = "") -> FileResponse:
    """Client-side route: a refresh or direct link to /hub/<persona> serves the app."""
    return _app_shell()


@app.get("/activity")
@app.get("/activity/{rest:path}")
def activity_route(rest: str = "") -> FileResponse:
    """Client-side route: /activity/<persona> serves the app (the reactions view)."""
    return _app_shell()


@app.get("/build")
@app.get("/build/{rest:path}")
def build_route(rest: str = "") -> FileResponse:
    """Client-side route: /build (new) or /build/<persona> (edit) serves the app."""
    return _app_shell()


@app.get("/opportunity")
@app.get("/opportunity/{rest:path}")
def opportunity_route(rest: str = "") -> FileResponse:
    """Client-side route: /opportunity/<id> serves the app (the card detail page)."""
    return _app_shell()


@app.get("/trends")
@app.get("/trends/{rest:path}")
def trends_route(rest: str = "") -> FileResponse:
    """Client-side route: /trends/<persona> serves the app (the analytics overview)."""
    return _app_shell()


@app.get("/brief")
@app.get("/brief/{rest:path}")
def brief_route(rest: str = "") -> FileResponse:
    """Client-side route: /brief/<persona> serves the app (the engagement brief)."""
    return _app_shell()


@app.get("/prototype.html")
def legacy_prototype() -> RedirectResponse:
    """Old links keep working."""
    return RedirectResponse("/")


# Persona-highlighted topic maps + the similarity heatmap are rendered to data/trends_img
# (cached by domain set). Mount it BEFORE the catch-all static mount so it takes priority.
TRENDS_IMG = ROOT / "data" / "trends_img"
TRENDS_IMG.mkdir(parents=True, exist_ok=True)
app.mount("/trends_img", StaticFiles(directory=str(TRENDS_IMG)), name="trends_img")
app.mount("/", StaticFiles(directory=str(MOCKUPS), html=True), name="static")
