"""Goal-conditioned engagement scoring + engagement modes (Phase 3).

The signature model. Engagement value is not one number for everyone — it's
conditioned on WHY a person engages. We capture that with an ENGAGEMENT MODE
(Contribute / Discuss / Monitor / Promote), derived from the persona's goal x
primary platform. The mode is a switch ABOVE the score:

  impact = Σ  weight_mode[signal] · signal_value         (signals all in [0,1])

The opportunity list is ranked by IMPACT (engagement value). Effort enters the
score only mildly (an `effort_fit` term) so the ranking is NOT dominated by
trivial 8-minute actions — instead effort does its real work in two places:
  - roi = impact / effort_hours   → the efficient-frontier view + a reported metric
  - the time-budget KNAPSACK       → "This Week's Plan" (doers, in rank.py)
A `platform_fit` term lets each mode prefer its natural platform (Contribute →
GitHub work; Discuss/Promote → discussion venues) without hard-excluding others.

Weights here are interpretable defaults; Phase-5's bandit LEARNS per-user
deviations, and "Why this?" reads straight off the contributions dict.
"""
from __future__ import annotations

SIGNALS = ("relevance", "community_health", "visibility", "velocity", "standout",
           "recency", "discussion", "platform_fit", "effort_fit", "contribution_worth")


class Mode:
    def __init__(self, key, label, weights, platform_pref, effort_aware, knapsack, action, blurb):
        self.key = key
        self.label = label
        self.weights = weights
        self.platform_pref = platform_pref      # source -> [0,1] preference
        self.effort_aware = effort_aware         # ROI divides by effort? knapsack uses it?
        self.knapsack = knapsack                 # time-budget basket vs. velocity radar
        self.action = action
        self.blurb = blurb

    def platform_fit(self, source: str) -> float:
        return self.platform_pref.get(source, 0.5)


MODES: dict[str, Mode] = {
    "contribute": Mode(
        "contribute", "Contribute",
        {"relevance": 0.30, "contribution_worth": 0.20, "platform_fit": 0.15,
         "standout": 0.12, "community_health": 0.10, "visibility": 0.08, "effort_fit": 0.05},
        {"github": 1.0, "devto": 0.45, "hackernews": 0.45, "reddit": 0.45, "bluesky": 0.30},
        effort_aware=True, knapsack=True,
        action="Open a PR / fix this issue",
        blurb="Do work that builds a visible portfolio you can point to."),
    "discuss": Mode(
        "discuss", "Discuss",
        {"relevance": 0.33, "community_health": 0.18, "discussion": 0.15,
         "platform_fit": 0.13, "standout": 0.13, "visibility": 0.03, "effort_fit": 0.05},
        {"reddit": 1.0, "hackernews": 1.0, "bluesky": 0.95, "devto": 0.85, "github": 0.82},
        effort_aware=True, knapsack=True,
        action="Add expert commentary",
        blurb="Join live, high-signal conversations where your expertise adds value."),
    "monitor": Mode(
        "monitor", "Monitor",
        {"velocity": 0.40, "recency": 0.25, "relevance": 0.20, "visibility": 0.15},
        {"github": 1.0, "hackernews": 1.0, "bluesky": 1.0, "devto": 1.0, "reddit": 1.0},
        effort_aware=False, knapsack=False,
        action="Track this",
        blurb="Spot what's rising before it goes mainstream — watch, don't spend hours."),
    "promote": Mode(
        "promote", "Promote",
        {"relevance": 0.33, "visibility": 0.22, "discussion": 0.18,
         "platform_fit": 0.12, "recency": 0.08, "effort_fit": 0.07},
        {"reddit": 1.0, "hackernews": 1.0, "devto": 0.95, "bluesky": 0.90, "github": 0.60},
        effort_aware=True, knapsack=True,
        action="Mention your product where relevant",
        blurb="Find discussions where your product is genuinely relevant to raise awareness."),
}

PERSONA_MODE = {"sofia": "contribute", "david": "discuss", "lina": "monitor", "raj": "promote"}

_MONITOR_KW = ("monitor", "trend", "surface", "spot", "before they go mainstream",
               "breadth", "rising", "fast-growing", "emerging")
_PROMOTE_KW = ("promote", "awareness", "marketing", "product is relevant", "integration",
               "partnership", "grow awareness", "relevant to the discussion")
_CONTRIB_KW = ("contribute", "portfolio", "good first issue", "repos to contribute",
               "fix", "build a visible")
_DISCUSS_KW = ("thought leadership", "expert commentary", "discussion", "commentary",
               "high-signal", "engage with")


def mode_for_persona(persona: dict) -> str:
    """Derive an engagement mode from a persona's goal + platforms.

    Named personas are pinned; unknown/hidden personas use goal-keyword + platform
    heuristics so the product generalizes beyond the 4 examples.
    """
    pid = (persona.get("id") or persona.get("name", "")).lower()
    for k in PERSONA_MODE:
        if k in pid:
            return PERSONA_MODE[k]
    goal = (persona.get("goal", "") + " " + " ".join(persona.get("interests", []))).lower()
    platforms = [p.lower() for p in persona.get("platforms", [])]
    if any(k in goal for k in _MONITOR_KW):
        return "monitor"
    if any(k in goal for k in _PROMOTE_KW):
        return "promote"
    if any(k in goal for k in _CONTRIB_KW) and platforms[:1] == ["github"]:
        return "contribute"
    if any(k in goal for k in _DISCUSS_KW):
        return "discuss"
    return "contribute" if platforms[:1] == ["github"] else "discuss"


def impact(feats: dict, mode: Mode, weights: dict | None = None) -> float:
    """Goal-weighted sum of normalized signals. Missing signal -> 0 contribution.

    `weights` is the Phase-4 per-user LEARNED weight vector; when given it
    overrides the mode's interpretable defaults so the ranking personalizes."""
    w = weights or mode.weights
    return sum(wt * float(feats.get(s, 0.0) or 0.0) for s, wt in w.items())


def contributions(feats: dict, mode: Mode, weights: dict | None = None) -> list[tuple[str, float]]:
    """(signal, weighted_contribution) sorted desc — the basis for 'Why this?'."""
    w = weights or mode.weights
    items = [(s, wt * float(feats.get(s, 0.0) or 0.0)) for s, wt in w.items()]
    return sorted(items, key=lambda x: -x[1])


def score_one(feats: dict, mode: Mode, weights: dict | None = None) -> dict:
    """Score a single opportunity. feats must include relevance, platform_fit,
    effort_fit, effort_min (rank.py injects the persona-specific ones).
    `weights` overrides the mode defaults with learned per-user weights."""
    imp = impact(feats, mode, weights)
    if mode.key == "contribute":
        # worthiness GATE: multiplicatively down-rank random/abandoned repos, so a famous,
        # well-run, well-specified repo outranks a 2-star personal repo even when both match
        # the user's interests equally. Research: responsiveness + issue quality > popularity.
        imp *= 0.45 + 0.55 * float(feats.get("contribution_worth", 0.5) or 0.5)
    eff_min = float(feats.get("effort_min", 30.0) or 30.0)
    roi = (imp / max(eff_min / 60.0, 0.1)) if mode.effort_aware else imp
    contribs = contributions(feats, mode, weights)
    return {"impact": round(imp, 4), "roi": round(roi, 4), "effort_min": round(eff_min, 1),
            "contributions": contribs, "top_drivers": [s for s, _ in contribs[:3]],
            "mode": mode.key}


_LABEL = {"relevance": "matches your interests", "community_health": "active, healthy community",
          "visibility": "high visibility potential", "velocity": "rising fast",
          "standout": "room to stand out (few contributors)", "recency": "fresh",
          "discussion": "live discussion to join", "platform_fit": "on your primary platform",
          "effort_fit": "quick to engage", "contribution_worth": "a worthwhile, well-run repo to contribute to"}


def why_this(scored: dict, feats: dict, mode: Mode) -> str:
    """Plain-language explanation from the top contributions (deterministic)."""
    parts = [_LABEL.get(s, s) for s, c in scored["contributions"][:3] if c > 0.01]
    eff = scored["effort_min"]
    tail = f" · ~{eff/60:.1f}h to engage" if mode.effort_aware else " · monitor only"
    return ("; ".join(parts) if parts else "weak match") + tail
