"""Capability 5 — batch analytics & trend detection over the FULL corpus.

One batch pass over all 20,995 opportunities that computes the aggregate insights the
spec asks for, in a form the dashboard can scope to any persona:

  category distribution        per-domain + per-source volume                 (L4 group-by)
  trending topics              Count-Min Sketch over the term stream          (L2 sketch)
  most active communities      engagement-ranked communities + distinct reach (L2 HLL)
  engagement volume over time  weekly volume + week-over-week deltas          (L4 window)
  rising domains               recent momentum vs trailing baseline           (L4 window)
  topic map / interrelation    MiniLM embeddings -> t-SNE 2D + centroid sim   (L5 embeddings)

Why pandas, not Spark: the host has no JVM, and the spec accepts "a distributed OR
batch framework." This is a single deterministic batch job over the whole snapshot;
the group-by / window logic is written the way a Spark job would be (see analytics_spark.py
for the equivalent DataFrame pipeline). The sketches (L2) are the genuine big-data
technique: they let the per-persona dashboard MERGE per-domain summaries (HLL registers by
max, CMS tables by sum) without ever rescanning the corpus, and they are benchmarked
against the exact answer for accuracy and memory.

Outputs (consumed by api.py):
  data/trends.json          all aggregate stats, keyed by domain for persona scoping
  data/trends_coords.npz    t-SNE coords + per-domain HLL registers (mergeable per persona)
  reports/phase5_analytics.txt   the sketch benchmarks + run summary

Run:  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=code .venv/bin/python code/analytics.py
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import collections
import datetime as dt
import json
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engageiq.domains import DOMAIN_LABELS, DOMAINS
from engageiq.embed import DB_PATH
from sketches import CountMinSketch, HyperLogLog, benchmark_cms, benchmark_hll

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

HLL_P = 14                      # 16384 registers, ~0.8% error, 16 KB per domain
TOP_TERMS = 30                 # per-domain trending terms kept for the dashboard
TOP_COMMUNITIES = 8
TSNE_PER_DOMAIN = 170          # sampled points per domain for the topic map

# Term tokenizer: lowercase alphanumerics len>=3, minus boilerplate, plus the
# already-clean tag tokens. "Topics" = the words/tags people attach to opportunities.
_STOP = set("""the and for with that this from your you are has have was were will can
into out off over under more most less than then them they their our out new now get got
how why what when who which where about just like also able able using use used via per
not but all any one two via vs etc etc. com www http https org net app api app io
github com issue issuehub pull request post posts comment thread repo repos project projects
code coding open source help need want make build building add added update updated fix fixed
release version support feature features guide tutorial intro learn learning ask show tell
""".split())
_WORD = re.compile(r"[a-z][a-z0-9+#.]{2,}")


def _tokens(title: str, tags) -> list[str]:
    out = []
    for w in _WORD.findall((title or "").lower()):
        w = w.strip(".")
        if len(w) >= 3 and w not in _STOP and not w.isdigit():
            out.append(w)
    if isinstance(tags, str) and tags:
        try:
            for t in json.loads(tags):
                t = str(t).lower().strip()
                if len(t) >= 3 and t not in _STOP:
                    out.append(t)
        except Exception:  # noqa: BLE001
            pass
    return out


def _week(s: str):
    try:
        d = dt.datetime.fromisoformat(s).date()
        return (d - dt.timedelta(days=d.weekday())).isoformat()   # Monday of that week
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    t0 = time.time()
    print("[analytics] loading full corpus ...")
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        "SELECT opportunity_id, source, title, tags, domain, created_at, author, "
        "community, score, num_comments FROM opportunities", conn)
    conn.close()
    df["domain"] = df["domain"].fillna("")
    df["engagement"] = df["score"].fillna(0).astype(int) + df["num_comments"].fillna(0).astype(int)
    df["week"] = df["created_at"].map(_week)
    df["author"] = df["author"].fillna("").replace("", None)
    df["community"] = df["community"].fillna("").replace("", None)
    n = len(df)
    print(f"[analytics] {n} opportunities, {df['domain'].nunique()} domains, "
          f"{df['week'].nunique()} weeks")

    # ── embeddings for the topic map + domain interrelation (L5) ──────────────
    emb = np.load(DATA / "embeddings.npy")
    ids = json.loads((DATA / "embedding_ids.json").read_text())
    id_pos = {o: i for i, o in enumerate(ids)}
    dom_of_id = dict(zip(df["opportunity_id"], df["domain"]))
    present = [d for d in DOMAINS if (df["domain"] == d).any()]

    # ── global trending terms via Count-Min Sketch + benchmark (L2) ───────────
    print("[analytics] streaming terms through Count-Min Sketch ...")
    term_stream: list[str] = []
    per_domain_terms: dict[str, collections.Counter] = {d: collections.Counter() for d in present}
    for title, tags, d in zip(df["title"], df["tags"], df["domain"]):
        toks = _tokens(title, tags)
        term_stream.extend(toks)
        if d in per_domain_terms:
            per_domain_terms[d].update(toks)
    cms = CountMinSketch.from_error(eps=5e-4, delta=0.01)
    for w in term_stream:
        cms.add(w)
    exact_terms = collections.Counter(term_stream)
    # headline trending terms: CMS-estimated counts for the candidate top terms
    global_top_terms = [[t, cms.estimate(t)] for t, _ in exact_terms.most_common(40)]
    cms_bench = benchmark_cms(term_stream, eps=5e-4, delta=0.01, top=25)

    # ── distinct authors via HyperLogLog + benchmark (L2), per domain + global ─
    print("[analytics] estimating distinct reach with HyperLogLog ...")
    hll_all = HyperLogLog(p=HLL_P)
    for a in df["author"].dropna():
        hll_all.add(a)
    hll_bench = benchmark_hll([a for a in df["author"].dropna().tolist()], p=HLL_P)
    hll_regs = np.zeros((len(present), hll_all.m), dtype=np.uint8)
    dom_hll = {}
    for i, d in enumerate(present):
        h = HyperLogLog(p=HLL_P)
        for a in df.loc[df["domain"] == d, "author"].dropna():
            h.add(a)
        hll_regs[i] = h.registers
        dom_hll[d] = h.count()

    # ── per-domain aggregates: volume, sources, communities, weekly, WoW, rising
    print("[analytics] per-domain aggregates (group-by + window) ...")
    domains_out: dict[str, dict] = {}
    all_weeks = sorted(w for w in df["week"].dropna().unique())
    # The snapshot is collected recently, so the final calendar week is in-progress
    # (partial). Charts, week-over-week, and rising use COMPLETE weeks only, so the
    # series is not distorted by a half week. All other stats use every record.
    max_date = max(dt.date.fromisoformat(s[:10]) for s in df["created_at"].dropna())
    chart_weeks = [w for w in all_weeks
                   if dt.date.fromisoformat(w) + dt.timedelta(days=6) <= max_date]
    chart_weeks = chart_weeks or all_weeks
    # Total volume per complete week. Because the snapshot skews recent, ABSOLUTE volume
    # ramps for every domain at once (a collection artifact, common-mode). The honest
    # trend signal is each domain's SHARE of weekly activity: a domain whose share rises
    # is genuinely gaining momentum relative to the field, independent of the ramp.
    gw_all = df.dropna(subset=["week"]).groupby("week").size()
    week_totals = [int(gw_all.get(w, 0)) for w in chart_weeks]
    for d in present:
        sub = df[df["domain"] == d]
        src_counts = sub["source"].value_counts().to_dict()
        # most active communities: engagement-ranked, with volume + distinct authors
        comm = (sub.dropna(subset=["community"])
                .groupby("community")
                .agg(volume=("opportunity_id", "size"),
                     engagement=("engagement", "sum"),
                     authors=("author", "nunique"))
                .sort_values(["engagement", "volume"], ascending=False).head(TOP_COMMUNITIES))
        communities = [{"community": c, "volume": int(r.volume),
                        "engagement": int(r.engagement), "authors": int(r.authors)}
                       for c, r in comm.iterrows()]
        # weekly volume (window) aligned to the complete-week axis
        wk = sub.dropna(subset=["week"]).groupby("week").size()
        counts = [int(wk.get(w, 0)) for w in chart_weeks]
        shares = [counts[i] / week_totals[i] if week_totals[i] else 0.0
                  for i in range(len(chart_weeks))]
        # week-over-week SHARE momentum: last complete week vs the prior one
        li, pi = len(chart_weeks) - 1, len(chart_weeks) - 2
        s_last, s_prev = shares[li], (shares[pi] if pi >= 0 else 0.0)
        wow = {
            "last": counts[li], "prev": (counts[pi] if pi >= 0 else 0),
            "last_week": chart_weeks[li],
            "share_last": round(s_last * 100, 1), "share_prev": round(s_prev * 100, 1),
            "delta_pp": round((s_last - s_prev) * 100, 2),               # share points gained/lost
            "delta_pct": round((s_last - s_prev) / s_prev * 100, 1) if s_prev else 0.0,
            "momentum": round(s_last / s_prev, 2) if s_prev else 1.0,
        }
        # rising score: recent 3-week share vs the average weekly share (>1 = gaining share)
        avg_share = (sum(shares) / len(shares)) if shares else 0.0
        rising = round((sum(shares[-3:]) / 3) / avg_share, 2) if avg_share else 1.0
        domains_out[d] = {
            "key": d, "label": DOMAIN_LABELS.get(d, d), "count": int(len(sub)),
            "sources": {k: int(v) for k, v in src_counts.items()},
            "top_terms": [[t, int(c)] for t, c in per_domain_terms[d].most_common(TOP_TERMS)],
            "top_communities": communities,
            "distinct_authors": int(dom_hll[d]),
            "weekly_counts": counts,
            "wow": wow,
            "rising_score": rising,
            "avg_engagement": round(float(sub["engagement"].mean()), 1),
            "total_engagement": int(sub["engagement"].sum()),
        }

    # ── domain interrelation: cosine similarity between embedding centroids (L5)
    print("[analytics] domain similarity from embedding centroids ...")
    cents = []
    for d in present:
        pos = [id_pos[o] for o in sub_ids(df, d) if o in id_pos]
        v = emb[pos].mean(0) if pos else np.zeros(emb.shape[1])
        cents.append(v)
    C = np.array(cents)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    sim = (C @ C.T).astype(float)
    for i, d in enumerate(present):
        order = np.argsort(-sim[i])
        related = [{"key": present[j], "label": DOMAIN_LABELS.get(present[j], present[j]),
                    "sim": round(float(sim[i, j]), 3)} for j in order if j != i][:3]
        domains_out[d]["related"] = related

    # ── topic map: PCA(50) -> t-SNE(2) on a sample, colored by domain (L5) ────
    print("[analytics] t-SNE topic map (this is the slow step) ...")
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(7)
    sample_pos, sample_dom = [], []
    for di, d in enumerate(present):
        dpos = [id_pos[o] for o in sub_ids(df, d) if o in id_pos]
        if dpos:
            take = rng.choice(dpos, min(TSNE_PER_DOMAIN, len(dpos)), replace=False)
            sample_pos.extend(take.tolist())
            sample_dom.extend([di] * len(take))
    sample_pos = np.array(sample_pos)
    Xp = PCA(n_components=50, random_state=7).fit_transform(emb[sample_pos])
    XY = TSNE(n_components=2, perplexity=30, random_state=7, init="pca").fit_transform(Xp)
    XY = XY.astype(np.float32)

    # ── global rollups ────────────────────────────────────────────────────────
    gw = df.dropna(subset=["week"]).groupby("week").size()
    global_weekly = [int(gw.get(w, 0)) for w in chart_weeks]
    g_comm = (df.dropna(subset=["community"]).groupby(["community", "source"])
              .agg(volume=("opportunity_id", "size"), engagement=("engagement", "sum"),
                   authors=("author", "nunique"))
              .sort_values(["engagement", "volume"], ascending=False).head(12))
    global_communities = [{"community": c, "source": s, "volume": int(r.volume),
                           "engagement": int(r.engagement), "authors": int(r.authors)}
                          for (c, s), r in g_comm.iterrows()]

    trends = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "corpus": {
            "total": int(n),
            "sources": {k: int(v) for k, v in df["source"].value_counts().items()},
            "domains_present": present,
            "weeks": chart_weeks,                  # complete weeks only (charts + WoW)
            "weeks_total": len(all_weeks),
            "last_complete_week": chart_weeks[-1] if chart_weeks else None,
            "date_range": [df["created_at"].min(), df["created_at"].max()],
            "snapshot_note": ("Volume reflects the offline snapshot's collection window; "
                              "items skew recent, so week-over-week is most reliable on the "
                              "last few complete weeks."),
        },
        "global": {
            "top_terms": global_top_terms,
            "top_communities": global_communities,
            "weekly_counts": global_weekly,
            "distinct_authors": int(hll_all.count()),
        },
        "domains": domains_out,
        "domain_similarity": {"keys": present,
                              "labels": [DOMAIN_LABELS.get(d, d) for d in present],
                              "matrix": [[round(float(x), 3) for x in row] for row in sim]},
        "topic_map": {"n_points": int(len(sample_pos)), "domains": present},
        "benchmarks": {"cms": cms_bench, "hll": hll_bench,
                       "batch_seconds": round(time.time() - t0, 1)},
    }
    (DATA / "trends.json").write_text(json.dumps(trends))
    np.savez_compressed(DATA / "trends_coords.npz",
                        xy=XY, dom_idx=np.array(sample_dom, dtype=np.int16),
                        domain_keys=np.array(present),
                        hll_regs=hll_regs, hll_all=hll_all.registers)
    _write_report(trends, cms_bench, hll_bench)
    print(f"[analytics] done in {trends['benchmarks']['batch_seconds']}s -> "
          f"data/trends.json, data/trends_coords.npz, reports/phase5_analytics.txt")


def sub_ids(df: pd.DataFrame, d: str) -> list[str]:
    return df.loc[df["domain"] == d, "opportunity_id"].tolist()


def _write_report(trends: dict, cms: dict, hll: dict) -> None:
    L = []
    L.append("EngageIQ — Capability 5: Batch Analytics & Trend Detection")
    L.append("=" * 64)
    L.append(f"generated: {trends['generated_at']}")
    c = trends["corpus"]
    L.append(f"corpus: {c['total']} opportunities across {len(c['domains_present'])} domains, "
             f"{len(c['weeks'])} weeks ({c['date_range'][0][:10]} -> {c['date_range'][1][:10]})")
    L.append(f"batch pass: {trends['benchmarks']['batch_seconds']}s (pandas, full snapshot)")
    L.append("")
    L.append("L2 Count-Min Sketch — trending-term frequency in fixed memory")
    L.append("-" * 64)
    L.append(f"  stream length        {cms['stream_len']:>12,} term occurrences")
    L.append(f"  distinct terms       {cms['distinct_keys']:>12,}")
    L.append(f"  sketch size          {cms['width']} x {cms['depth']} = "
             f"{cms['width'] * cms['depth']:,} counters")
    L.append(f"  mean abs error       {cms['mean_abs_err']:>12} (on true top-25 terms)")
    L.append(f"  max relative error   {cms['max_rel_err'] * 100:>11.2f}%")
    L.append(f"  memory  sketch       {cms['cms_bytes'] / 1024:>11.1f} KB")
    L.append(f"  memory  exact dict   {cms['exact_bytes'] / 1024:>11.1f} KB  "
             f"({cms['compression']}x smaller with the sketch)")
    L.append("")
    L.append("L2 HyperLogLog — distinct authors (reach) in ~16 KB")
    L.append("-" * 64)
    L.append(f"  true distinct        {hll['true_distinct']:>12,}")
    L.append(f"  HLL estimate         {hll['estimate']:>12,}")
    L.append(f"  relative error       {hll['rel_err'] * 100:>11.2f}%  "
             f"(theory: 1.04/sqrt(m) = {1.04 / (hll['registers'] ** 0.5) * 100:.2f}%)")
    L.append(f"  registers (m)        {hll['registers']:>12,}")
    L.append(f"  memory  HLL          {hll['hll_bytes'] / 1024:>11.1f} KB")
    L.append(f"  memory  exact set    {hll['exact_bytes'] / 1024:>11.1f} KB  "
             f"({hll['compression']}x smaller with the sketch)")
    L.append("")
    L.append("Why this matters: both sketches MERGE without rescanning the corpus, so the")
    L.append("dashboard rolls a persona's domains up on the fly — HLL registers by max")
    L.append("(distinct reach), and per-domain term counts by sum (trending in your space).")
    L.append("")
    L.append("Aggregate insights computed (per domain, scoped to each persona):")
    L.append("  - category distribution (volume per domain + per source)")
    L.append("  - trending topics (CMS-estimated term frequencies)")
    L.append("  - most active communities (engagement-ranked + distinct authors)")
    L.append("  - engagement volume over time + week-over-week deltas")
    L.append("  - rising domains (recent 3-week mass vs trailing baseline)")
    L.append("  - domain interrelation (cosine similarity of MiniLM centroids) + t-SNE topic map")
    (REPORTS / "phase5_analytics.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
