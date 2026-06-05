"""EngageIQ — Streamlit Hub (the real UI, wired to the live ranking engine).

This is NOT a mockup: every card on screen is produced by `rank.Recommender`
running over the real 20,995-record corpus. Pick a persona and you see exactly
what the algorithm chose, with its real "why", ROI, effort, and weekly plan.

Run:  PYTHONPATH=code streamlit run code/app.py
"""
from __future__ import annotations

import html
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import streamlit as st

from engageiq import personas, rank

# ─────────────────────────────────────────────────────────────────────────────
# Display maps (friendly labels — no raw underscore keys reach the screen)
# ─────────────────────────────────────────────────────────────────────────────
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
SRC_COLORS = {"github": "#24292f", "hackernews": "#ff6600", "bluesky": "#0085ff",
              "devto": "#1a1a1a", "reddit": "#ff4500"}
MODE_STYLE = {
    "contribute": ("#16a34a", "Build work that grows a visible portfolio, inside your weekly hours."),
    "discuss":    ("#2563eb", "Join live, high-signal conversations where your expertise adds value."),
    "monitor":    ("#d97706", "Spot what's rising before it goes mainstream. Watch, don't spend hours."),
    "promote":    ("#7c3aed", "Find threads where your product is genuinely relevant."),
}


def dom_label(k: str) -> str:
    return DOMAIN_LABELS.get(k, (k or "").replace("_", " ").title())


def split_name(full: str) -> tuple[str, str]:
    for sep in ("—", " - ", "–"):
        if sep in full:
            a, b = full.split(sep, 1)
            return a.strip(), b.strip()
    return full.strip(), ""


def clean(s: str) -> str:
    """Strip em/en-dashes from any engine-produced copy (house style: no em-dashes)."""
    return (s or "").replace(" — ", ", ").replace("—", ", ").replace("–", "-")


# ─────────────────────────────────────────────────────────────────────────────
# Engine (loaded once, cached across reruns)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading the ranking engine (embeddings + cross-encoder)…")
def get_recommender() -> rank.Recommender:
    return rank.Recommender()


@st.cache_data(show_spinner="Scoring opportunities for this persona…")
def get_recs(pid: str, k: int) -> dict:
    rec = get_recommender()
    return rec.recommend(personas.PERSONAS[pid], k=k)


# ─────────────────────────────────────────────────────────────────────────────
# Page + styling
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EngageIQ", page_icon="🎯", layout="wide")
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1320px; }
      .eq-badge { display:inline-block; padding:2px 9px; border-radius:999px;
                  color:#fff; font-size:.72rem; font-weight:600; letter-spacing:.2px; }
      .eq-meta  { color:#64748b; font-size:.82rem; }
      .eq-title { font-size:1.04rem; font-weight:650; margin:.35rem 0 .25rem; line-height:1.35; }
      .eq-why   { color:#475569; font-size:.85rem; }
      .eq-mode  { display:inline-block; padding:5px 14px; border-radius:999px;
                  color:#fff; font-weight:700; font-size:.9rem; }
      .eq-roi   { color:#0f766e; font-weight:650; }
      .eq-rail-item { font-size:.86rem; padding:6px 0; border-bottom:1px solid #eef2f7; }
      a.eq-link { text-decoration:none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — persona picker + profile
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 EngageIQ")
    st.caption("Smart Engagement Opportunity Scorer")
    st.divider()
    pids = list(personas.PERSONAS.keys())
    labels = {pid: split_name(personas.PERSONAS[pid]["name"])[0] for pid in pids}
    pid = st.radio("Who are you?", pids, format_func=lambda p: labels[p])
    k = st.slider("How many to show", 5, 20, 10)
    st.divider()
    p = personas.PERSONAS[pid]
    name, role = split_name(p["name"])
    st.markdown(f"**{name}**  \n{role}")
    st.caption(f"🎯 {p['goal']}")
    if p.get("domains"):
        st.caption("**Topics:** " + ", ".join(dom_label(d) for d in p["domains"]))
    else:
        st.caption("**Topics:** all (breadth)")
    st.caption("**Platforms:** " + ", ".join(SRC_LABELS.get(s, s) for s in sorted(personas.allowed_sources(p))))
    st.caption(f"**Time budget:** {p['time_budget_hours']}h / week")

# ─────────────────────────────────────────────────────────────────────────────
# Main — header, mode banner, feed + right rail
# ─────────────────────────────────────────────────────────────────────────────
recs = get_recs(pid, k)
mode = recs["mode"]
color, blurb = MODE_STYLE.get(mode, ("#475569", clean(recs.get("mode_blurb", ""))))

name, role = split_name(recs.get("name", p["name"]))
st.markdown(f"### {name}'s engagement hub")
n_src = len({r["source"] for r in recs["ranked"]})
st.markdown(
    f"<span class='eq-mode' style='background:{color}'>{recs['mode_label']} mode</span>"
    f"&nbsp;&nbsp;<span class='eq-meta'>{html.escape(blurb)}</span>",
    unsafe_allow_html=True,
)
st.caption(f"{len(recs['ranked'])} opportunities · {recs['n_candidates']} candidates scored · {n_src} sources")
st.write("")

feed_col, rail_col = st.columns([2, 1], gap="large")

# ── For You feed ─────────────────────────────────────────────────────────────
with feed_col:
    st.markdown("#### For you")
    for i, r in enumerate(recs["ranked"]):
        src = r["source"]
        sc, feats = r["score"], r["features"]
        badge_c = SRC_COLORS.get(src, "#475569")
        dom = dom_label((r.get("domains") or [r.get("domain")])[0])
        disp = html.escape((r.get("disp") or "")[:160])
        why = html.escape(clean(r.get("why", "")))
        action = html.escape(clean(r.get("action", "engage")))
        roi = float(sc.get("roi", 0))
        eff = float(sc.get("effort_min", feats.get("effort_min", 0)))
        rel = float(feats.get("relevance", 0))
        url = r.get("url") or "#"
        with st.container(border=True):
            st.markdown(
                f"<span class='eq-badge' style='background:{badge_c}'>{SRC_LABELS.get(src, src)}</span> "
                f"<span class='eq-meta'>· {dom} · ~{eff:.0f} min · "
                f"<span class='eq-roi'>ROI {roi:.1f}</span> · relevance {rel:.2f}</span>"
                f"<div class='eq-title'><a class='eq-link' href='{html.escape(url)}' target='_blank'>{disp}</a></div>"
                f"<div class='eq-why'>💡 {why}</div>"
                f"<div class='eq-meta' style='margin-top:3px'>→ Suggested: {action}</div>",
                unsafe_allow_html=True,
            )
            b1, b2, b3, _ = st.columns([1, 1, 1, 4])
            if b1.button("✅ Engage", key=f"eng_{i}"):
                st.session_state.setdefault("fb", {})[r["id"]] = "engage"
            if b2.button("⏭ Skip", key=f"skip_{i}"):
                st.session_state.setdefault("fb", {})[r["id"]] = "skip"
            if b3.button("🔖 Save", key=f"save_{i}"):
                st.session_state.setdefault("fb", {})[r["id"]] = "save"

# ── Right rail — This Week's Plan / Velocity Radar + standout repos ───────────
with rail_col:
    if recs.get("basket") is not None:
        used = sum(float(b["features"]["effort_min"]) for b in recs["basket"])
        budget = recs.get("budget_min", 0)
        st.markdown("#### 🗓️ This week's plan")
        st.caption(f"Time-budget knapsack · {used:.0f} of {budget} min")
        st.progress(min(1.0, used / budget) if budget else 0.0)
        for b in recs["basket"]:
            st.markdown(
                f"<div class='eq-rail-item'><b>{SRC_LABELS.get(b['source'], b['source'])}</b> "
                f"· {float(b['features']['effort_min']):.0f}m<br>{html.escape((b.get('disp') or '')[:70])}</div>",
                unsafe_allow_html=True,
            )
    elif recs.get("radar"):
        st.markdown("#### 📡 Velocity radar")
        st.caption("Monitor mode: rising fast, no time spent")
        for r in recs["ranked"]:
            v = float(r["features"].get("velocity", 0))
            st.markdown(
                f"<div class='eq-rail-item'><b>{SRC_LABELS.get(r['source'], r['source'])}</b> "
                f"· velocity {v:.2f}<br>{html.escape((r.get('disp') or '')[:70])}</div>",
                unsafe_allow_html=True,
            )

    if recs.get("repos"):
        st.markdown("#### ⭐ Standout repos")
        st.caption("High activity, few contributors — room to stand out")
        for b in recs["repos"]:
            so = float(b["features"].get("standout", 0))
            lang = b.get("language") or "—"
            st.markdown(
                f"<div class='eq-rail-item'>standout {so:.2f} · {html.escape(str(lang))}<br>"
                f"{html.escape((b.get('disp') or '')[:70])}</div>",
                unsafe_allow_html=True,
            )

fb = st.session_state.get("fb", {})
if fb:
    st.sidebar.divider()
    st.sidebar.caption(f"📝 You've reacted to {len(fb)} items (seeds Phase 4 learning).")
