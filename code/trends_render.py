"""Render the persona-scoped topic map (and the global similarity heatmap) from the
precomputed t-SNE coords in data/trends_coords.npz.

The slow step (t-SNE) ran once in analytics.py and saved 2-D coordinates. Here we only
draw a scatter, so a persona-specific map (their domains in colour, the rest greyed) is
cheap and cached to disk by the set of highlighted domains. Colours are stable per
domain (shared with the UI chips) so the map and the chips line up.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hashlib
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engageiq.domains import DOMAIN_LABELS, DOMAINS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMG_DIR = DATA / "trends_img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Curated PREMIUM domain palette: muted jewel + earth tones at a consistent saturation,
# replacing matplotlib's bright clashing "tab20" categorical set (which read childish).
# Stable per domain in the fixed DOMAINS order, shared with the UI chips + the topic mind
# map so colours line up. Hues are spread but desaturated so they feel like one family.
_PREMIUM_PALETTE = [
    "#5b6ee1",  # 0  machine_learning      indigo
    "#6f78cc",  # 1  devops_k8s            periwinkle
    "#4f8fc9",  # 2  open_source_trending  azure
    "#4f9fb3",  # 3  developer_tools       muted cyan
    "#2f9e8f",  # 4  cybersecurity         teal
    "#4f9d7e",  # 5  frontend_web          emerald
    "#6a9b62",  # 6  b2b_saas              sage
    "#8c8c4e",  # 7  blockchain            muted olive
    "#c2923f",  # 8  python_data_eng       antique gold
    "#bd7f4e",  # 9  gamedev_cpp           bronze
    "#bb6f5c",  # 10 ai_research           clay
    "#c2718c",  # 11 embedded_systems      dusty rose
    "#b56a93",  # 12 cloud_apis            mauve
    "#9d6aa8",  # 13 mobile_dev            plum
    "#8c6cb5",  # 14 beginner_coding       amethyst
]
DOMAIN_COLORS = {d: _PREMIUM_PALETTE[i % len(_PREMIUM_PALETTE)] for i, d in enumerate(DOMAINS)}

_COORDS = None


def _coords():
    global _COORDS
    if _COORDS is None:
        z = np.load(DATA / "trends_coords.npz", allow_pickle=True)
        keys = [str(k) for k in z["domain_keys"]]
        _COORDS = {"xy": z["xy"], "dom_idx": z["dom_idx"], "keys": keys}
    return _COORDS


def color_for(domain_key: str) -> str:
    return DOMAIN_COLORS.get(domain_key, "#94a3b8")


def _key_hash(highlight: set[str]) -> str:
    return hashlib.sha1((",".join(sorted(highlight)) or "all").encode()).hexdigest()[:10]


def persona_map_path(highlight: set[str]) -> str:
    """Path to the cached topic map for this set of highlighted domains (render once)."""
    c = _coords()
    keys = c["keys"]
    hl = set(highlight) & set(keys)
    if not hl:                                  # breadth (e.g. Lina): colour everything
        hl = set(keys)
    out = IMG_DIR / f"map_{_key_hash(hl)}.png"
    if out.exists():
        return out.name
    _render(c, hl, out)
    return out.name


def _render(c, highlight: set[str], out: Path) -> None:
    xy, dom_idx, keys = c["xy"], c["dom_idx"], c["keys"]
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    fig.patch.set_facecolor("white")
    # greyed background: the domains NOT in this persona's space
    bg = np.array([keys[i] not in highlight for i in dom_idx])
    if bg.any():
        ax.scatter(xy[bg, 0], xy[bg, 1], s=8, c="#e2e5ea", alpha=0.55, linewidths=0, zorder=1)
    # highlighted domains, each in its stable colour
    for ki, k in enumerate(keys):
        if k not in highlight:
            continue
        m = dom_idx == ki
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=14, color=color_for(k),
                       label=DOMAIN_LABELS.get(k, k), alpha=0.85, linewidths=0, zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ncol = 1 if len(highlight) <= 8 else 2
    ax.legend(markerscale=1.7, fontsize=8, loc="center left",
              bbox_to_anchor=(1.0, 0.5), frameon=False, ncol=ncol)
    ax.set_title("Topic map: opportunities embedded (MiniLM 384-d) to t-SNE 2-D\n"
                 "Nearby points are semantically related; clusters are sub-topics",
                 fontsize=10.5, color="#1a1d24")
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor="white")
    plt.close(fig)


def render_similarity(matrix, labels, out: Path | None = None) -> str:
    """Global domain-similarity heatmap (cosine between embedding centroids)."""
    out = out or (IMG_DIR / "domsim.png")
    M = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    off = ~np.eye(len(labels), dtype=bool)
    im = ax.imshow(M, cmap="viridis", vmin=float(M[off].min()) if off.any() else 0.0)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(im, label="cosine similarity", fraction=0.046)
    ax.set_title("Which domains are semantically interrelated\n"
                 "(cosine similarity between domain embedding centroids)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor="white")
    plt.close(fig)
    return out.name


def prerender_base() -> None:
    """Generate the global map (all domains) + the similarity heatmap, used by the
    brief and the design mockups. Idempotent; safe to call at startup."""
    import json
    persona_map_path(set())                      # the all-domains map
    t = json.loads((DATA / "trends.json").read_text())
    ds = t["domain_similarity"]
    render_similarity(ds["matrix"], ds["labels"])


if __name__ == "__main__":
    prerender_base()
    print("prerendered:", sorted(p.name for p in IMG_DIR.glob("*.png")))
