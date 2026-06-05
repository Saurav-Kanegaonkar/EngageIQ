"""The 4 BAX-423 test personas as structured profiles.

Each persona drives the retrieval -> filter -> score pipeline three ways
(the Phase-2<->3 bridge):
  EMBED  interests/goal  -> query vector -> relevance               (Phase 2)
  FILTER platforms, domains, exclude_languages                      (hard cuts)
  SCORE  goal -> engagement mode -> signal weights                  (score.py)

`domains` is a whitelist of the 15 domain keys (empty = no filter, i.e. breadth —
Lina). `exclude_languages` hard-drops GitHub repos in those languages (Sofia must
get zero C++/Rust). Reddit has no viable free API in 2026, so persona "reddit" is
served by our discussion/blog substitutes (Hacker News, Dev.to, Bluesky).
"""
from __future__ import annotations

PERSONAS = {
    "sofia": {
        "id": "sofia",
        "name": "Sofia — ML Student / Portfolio Builder",
        "interests": ["machine learning", "NLP", "data pipelines", "Python", "pandas"],
        "goal": "Find beginner-friendly open-source repos to contribute to and ML discussions to engage with for visibility",
        "platforms": ["github", "reddit"],
        "domains": ["machine_learning", "ai_research", "python_data_eng"],
        "exclude_languages": ["C++", "Rust", "C"],
        "time_budget_hours": 5,
        "skills": "Comfortable with Python and pandas; wants good-first-issues; avoid C++/Rust-heavy work",
    },
    "david": {
        "id": "david",
        "name": "David — DevOps Engineer / Thought Leader",
        "interests": ["Kubernetes", "Terraform", "CI/CD", "observability", "cloud-native infrastructure"],
        "goal": "Find high-signal Kubernetes/infra projects and discussion threads where expert commentary adds value",
        "platforms": ["github", "reddit", "hackernews"],
        "domains": ["devops_k8s", "cloud_apis", "cybersecurity"],
        "exclude_languages": [],
        "time_budget_hours": 3,
        "skills": "Mid-career DevOps; deep Kubernetes/Terraform; wants high-activity, few-contributor repos to stand out",
    },
    "lina": {
        "id": "lina",
        "name": "Lina — Data Journalist / Trend Spotter",
        "interests": ["trending repos", "emerging tools", "viral tech discussions", "open-source", "AI"],
        "goal": "Surface fast-growing repos and trending discussions before they go mainstream; breadth over depth",
        "platforms": ["github", "hackernews", "bluesky", "reddit"],
        "domains": [],                       # breadth over depth — no domain filter
        "exclude_languages": [],
        "time_budget_hours": 10,
        "skills": "Monitors many communities; values recency and velocity over personal skill match",
    },
    "raj": {
        "id": "raj",
        "name": "Raj — Startup Founder / Marketing",
        "interests": ["developer tools", "APIs", "CLI tools", "developer productivity", "open-source business"],
        "goal": "Find discussion threads and posts where a developer-tools product is relevant, and repos where integration makes sense",
        "platforms": ["reddit", "hackernews", "github", "devto"],
        "domains": ["developer_tools", "b2b_saas", "open_source_trending"],
        "exclude_languages": [],
        "time_budget_hours": 4,
        "skills": "Technical co-founder of a developer-tools startup; wants developer-tools-relevant discussions, not general programming",
    },
}

# persona platform name -> our available sources (Reddit substituted in 2026)
SOURCE_MAP = {
    "github": ["github"], "hackernews": ["hackernews"], "bluesky": ["bluesky"],
    "devto": ["devto"], "blogs": ["devto"],
    "reddit": ["reddit"],
}
ALL_SOURCES = {"github", "hackernews", "bluesky", "devto", "reddit"}


def profile_text(p: dict) -> str:
    """Single enriched query vector (baseline retrieval)."""
    return f"{p['goal']}. Interests: {', '.join(p['interests'])}. {p.get('skills', '')}".strip()


def facets(p: dict) -> list[str]:
    """Multi-facet queries: the goal + each interest, retrieved separately then pooled.

    Helps breadth personas (Lina, Raj) whose interests a single averaged vector blurs.
    """
    return [f for f in ([p["goal"]] + list(p.get("interests", []))) if f]


def allowed_sources(p: dict) -> set[str]:
    out: set[str] = set()
    for plat in p.get("platforms", []):
        out.update(SOURCE_MAP.get(plat.lower(), []))
    return out or set(ALL_SOURCES)


def domain_whitelist(p: dict) -> set[str]:
    return set(p.get("domains") or [])      # empty => no domain filter (breadth)


def excluded_languages(p: dict) -> set[str]:
    return {x.lower() for x in p.get("exclude_languages", [])}
