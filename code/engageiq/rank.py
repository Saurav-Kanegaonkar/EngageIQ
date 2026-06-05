"""Multi-stage ranking funnel (Phase 3, Lecture 7).

    profile -> [candidate generation: FAISS top-N]
            -> [hard filters: platform, domain, language-exclusion]
            -> [score: goal-conditioned engagement-ROI by mode]
            -> [re-rank: MMR diversity]
            -> [pack: time-budget knapsack basket  |  velocity radar]

The mode (from score.py) decides whether ROI is effort-aware and whether the
output is a "This Week's Plan" knapsack basket (doers) or a velocity radar
(monitors). This is the single entry point the dashboard will call.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sqlite3

import numpy as np

from . import effort, features, personas, plan, rerank, score
from .embed import DB_PATH, encode
from .retrieve import Retriever


class Recommender:
    def __init__(self) -> None:
        self.ret = Retriever()
        self.id2row = {oid: i for i, oid in enumerate(self.ret.ids)}
        self.conn = sqlite3.connect(str(DB_PATH))
        self.feats = features.load_features(self.conn)
        self.meta = self._load_meta()
        self._src_rows = self._build_src_rows()

    def _build_src_rows(self) -> dict[str, np.ndarray]:
        by: dict[str, list[int]] = {}
        for oid, i in self.id2row.items():
            m = self.meta.get(oid)
            if m:
                by.setdefault(m["source"], []).append(i)
        return {s: np.array(rows, dtype=np.int64) for s, rows in by.items()}

    def _load_meta(self) -> dict:
        cols = "opportunity_id, source, opportunity_type, domain, domains, language, title, url, created_at, body, community_size"
        out = {}
        for r in self.conn.execute(f"SELECT {cols} FROM opportunities"):
            try:
                doms = json.loads(r[4]) if r[4] else [r[3]]
            except Exception:  # noqa: BLE001
                doms = [r[3]]
            title = (r[6] or "").strip()
            body = r[9] or ""
            disp = title or " ".join(body.split())[:90]           # Bluesky posts have no title
            out[r[0]] = {"id": r[0], "source": r[1], "type": r[2], "domain": r[3],
                         "domains": doms, "language": r[5], "title": title, "disp": disp,
                         "url": r[7], "created_at": r[8], "body_raw": body[:1600],
                         "community_size": r[10],
                         "text": f"{title} {body}".strip()[:400]}   # for cross-encoder re-rank
        return out

    # ── stage 1: STRATIFIED candidate generation ─────────────────────────
    def _candidates(self, persona: dict, per_source: int, multifacet: bool,
                    allow: set[str] | None = None) -> list[tuple[str, float]]:
        """Retrieve top-K *within each allowed source* and union.

        A single global top-N lets high-volume sources (Bluesky/Dev.to) crowd
        out GitHub (whose terse issue text embeds far from persona queries), so a
        Contribute persona would see ~zero repos. Per-source quotas guarantee
        recall across platforms; scoring then decides the final order.
        Similarity = best match across the persona's facets (multi-facet query).
        `allow` lets a confirmed activity split widen/narrow which sources we pull.
        """
        allow = allow or personas.allowed_sources(persona)
        texts = personas.facets(persona) if multifacet else [personas.profile_text(persona)]
        qvs = encode(texts).astype(np.float32)                 # (F, d), L2-normalized
        sims_all = (self.ret.embs @ qvs.T).max(axis=1)         # best facet sim per record
        best: dict[str, float] = {}
        for src in allow:
            rows = self._src_rows.get(src)
            if rows is None or len(rows) == 0:
                continue
            sub = sims_all[rows]
            kk = min(per_source, len(sub))
            top = np.argpartition(-sub, kk - 1)[:kk]
            for ri in top:
                best[self.ret.ids[int(rows[ri])]] = float(sub[ri])
        return sorted(best.items(), key=lambda x: -x[1])

    # ── stage 2: hard filters ────────────────────────────────────────────
    def _filter(self, cands: list[tuple[str, float]], persona: dict,
                allow: set[str] | None = None) -> list[tuple[str, float]]:
        allow = allow or personas.allowed_sources(persona)
        doms = personas.domain_whitelist(persona)
        excl = personas.excluded_languages(persona)
        out = []
        for oid, sim in cands:
            m = self.meta.get(oid)
            if not m or m["source"] not in allow:
                continue
            if doms and not (set(m["domains"]) & doms):
                continue
            if excl and m["source"] == "github" and (m["language"] or "").lower() in excl:
                continue
            out.append((oid, sim))
        return out

    # ── stage 2.5: cross-encoder relevance (benchmarked best Stage-2 re-ranker) ──
    def _relevance(self, persona: dict, cands: list[tuple[str, float]],
                   use_ce: bool, ce_pool: int | None = None):
        """Return (candidate_subset, {id: raw_relevance}). With the cross-encoder we
        keep the top `ce_pool` by dense recall and re-score relevance by reading
        query+doc jointly; otherwise relevance falls back to the dense cosine.

        `ce_pool` defaults to env EIQ_CE_POOL (120). Re-ranking only the top ~120 by
        dense recall leaves the final top-10 essentially unchanged (items ranked past
        ~120 almost never survive into it) but is ~2.5x faster on CPU than the old 300.
        Set EIQ_CE_POOL=300 to reproduce the original offline eval."""
        if not use_ce or not cands:
            return cands, {i: s for i, s in cands}
        if ce_pool is None:
            ce_pool = int(os.environ.get("EIQ_CE_POOL", "120"))
        head = cands[:ce_pool]
        try:
            q = personas.profile_text(persona)
            scores = rerank.cross_encoder_scores(q, [self.meta[i]["text"] for i, _ in head])
            return head, {i: float(sc) for (i, _), sc in zip(head, scores)}
        except Exception as e:  # noqa: BLE001
            print(f"  [rank] cross-encoder unavailable ({e}); using dense relevance")
            return cands, {i: s for i, s in cands}

    # ── stage 3: scoring (relevance normalized within the candidate set) ──
    def _score_all(self, cands: list[tuple[str, float]], mode: score.Mode, rel: dict,
                   weights: dict | None = None, bucket_weights: dict | None = None) -> list[dict]:
        if not cands:
            return []
        vals = [rel[i] for i, _ in cands]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        out = []
        for oid, sim in cands:
            m = self.meta[oid]
            f = dict(self.feats.get(oid, {}))           # copy: don't mutate cache
            f.setdefault("effort_min", 30.0)
            f["relevance"] = max(0.0, min(1.0, (rel[oid] - lo) / rng))   # normalized in candidate set
            f["platform_fit"] = mode.platform_fit(m["source"])
            f["effort_fit"] = 1.0 - effort.normalized_effort(f["effort_min"])
            sc = score.score_one(f, mode, weights)
            if bucket_weights:                          # confirmed-split prior (mild)
                bf = bucket_weights.get(plan.bucket_for(m["source"]))
                if bf is not None:
                    sc["impact"] = round(sc["impact"] * bf, 4)
                    sc["roi"] = round(sc["roi"] * bf, 4)
            out.append({**m, "sim": sim, "features": f, "score": sc})
        return out

    # ── stage 4: MMR diversity re-rank ───────────────────────────────────
    def _emb(self, oid: str) -> np.ndarray:
        return self.ret.embs[self.id2row[oid]]

    def _mmr(self, items: list[dict], k: int, lam: float, key) -> list[dict]:
        if not items:
            return []
        vals = [key(it) for it in items]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        util = {it["id"]: (key(it) - lo) / rng for it in items}
        selected: list[dict] = []
        remaining = items[:]
        while remaining and len(selected) < k:
            best, best_v = None, -1e9
            for it in remaining:
                div = max((float(np.dot(self._emb(it["id"]), self._emb(s["id"])))
                           for s in selected), default=0.0)
                v = lam * util[it["id"]] - (1 - lam) * div
                if v > best_v:
                    best_v, best = v, it
            selected.append(best)
            remaining.remove(best)
        return selected

    # ── stage 4': time-budget knapsack ("This Week's Plan") ──────────────
    def _knapsack(self, scored: list[dict], budget_min: int, pool: int = 40) -> list[dict]:
        items = sorted(scored, key=lambda x: -x["score"]["impact"])[:pool]
        cap = max(0, int(budget_min))
        W = [max(1, int(round(it["features"]["effort_min"]))) for it in items]
        V = [int(round(it["score"]["impact"] * 1000)) for it in items]
        n = len(items)
        if n == 0 or cap == 0:
            return []
        dp = [[0] * (cap + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            wi, vi = W[i - 1], V[i - 1]
            row, prev = dp[i], dp[i - 1]
            for c in range(cap + 1):
                row[c] = prev[c]
                if wi <= c and prev[c - wi] + vi > row[c]:
                    row[c] = prev[c - wi] + vi
        chosen, c = [], cap
        for i in range(n, 0, -1):
            if dp[i][c] != dp[i - 1][c]:
                chosen.append(items[i - 1])
                c -= W[i - 1]
        chosen.reverse()
        return chosen

    # ── orchestration ────────────────────────────────────────────────────
    def recommend(self, persona: dict, k: int = 10, per_source: int = 150,
                  multifacet: bool = True, mmr_lambda: float = 0.75,
                  use_llm_effort: bool = False, use_cross_encoder: bool = True,
                  weights: dict | None = None, sources: set[str] | None = None,
                  bucket_weights: dict | None = None) -> dict:
        mode = score.MODES[score.mode_for_persona(persona)]
        allow = set(sources) if sources else personas.allowed_sources(persona)
        cands = self._filter(self._candidates(persona, per_source, multifacet, allow), persona, allow)
        cands, rel = self._relevance(persona, cands, use_cross_encoder)
        scored = self._score_all(cands, mode, rel, weights, bucket_weights)

        # rank by engagement VALUE (impact), not impact-per-hour — effort does its
        # work via the knapsack + the roi metric, not by inflating trivial actions.
        key = lambda x: x["score"]["impact"]   # noqa: E731
        scored.sort(key=key, reverse=True)

        if use_llm_effort and scored:
            self._refine_effort(scored[:max(k, 12)], mode, k)
            scored.sort(key=key, reverse=True)

        ranked = self._mmr(scored[:max(k * 4, 40)], k, mmr_lambda, key)
        for r in ranked:
            r["why"] = score.why_this(r["score"], r["features"], mode)
            r["action"] = mode.action

        result = {"persona": persona.get("id"), "name": persona.get("name"),
                  "mode": mode.key, "mode_label": mode.label, "mode_blurb": mode.blurb,
                  "n_candidates": len(scored), "ranked": ranked}
        if mode.knapsack:
            budget = int(persona.get("time_budget_hours", 5) * 60)
            result["basket"] = self._knapsack(scored, budget)
            result["budget_min"] = budget
        else:
            result["radar"] = True

        # secondary lane: standout repos worth a PR (high activity, FEW contributors).
        # Kept out of the relevance-led main feed so it doesn't crowd discussion, but
        # surfaced for doers/discussers (e.g. David's "repos to stand out in").
        if mode.key in ("contribute", "discuss"):
            repos = [s for s in scored if s["source"] == "github"]
            repos.sort(key=lambda x: -(0.6 * x["features"].get("standout", 0)
                                       + 0.4 * x["features"].get("relevance", 0)))
            result["repos"] = repos[:3]
        return result

    def _refine_effort(self, shortlist: list[dict], mode: score.Mode, k: int) -> None:
        probes = [{"source": s["source"], "opportunity_type": s["type"],
                   "title": s["title"], "effort_min": s["features"]["effort_min"]}
                  for s in shortlist]
        effort.refine_top_k(probes, k=k)
        for s, p in zip(shortlist, probes):
            s["features"]["effort_min"] = p["effort_min"]
            s["score"] = score.score_one(s["features"], mode)


def _fmt(r: dict) -> str:
    sc = r["score"]
    return (f"  [{r['source']:10}] {(r['disp'] or '')[:54]:54}  "
            f"imp={sc['impact']:.2f} roi={sc['roi']:.2f} rel={r['features']['relevance']:.2f} "
            f"eff={sc['effort_min']:.0f}m  why={r['why']}")


if __name__ == "__main__":
    rec = Recommender()
    for pid, p in personas.PERSONAS.items():
        out = rec.recommend(p, k=8)
        print(f"\n{'='*100}\n{out['name']}  |  MODE={out['mode_label']}  "
              f"|  candidates={out['n_candidates']}\n  ({out['mode_blurb']})\n{'='*100}")
        for r in out["ranked"]:
            print(_fmt(r))
        if out.get("basket") is not None:
            tot = sum(b["features"]["effort_min"] for b in out["basket"])
            print(f"  -- This Week's Plan ({len(out['basket'])} items, "
                  f"{tot:.0f}m / {out['budget_min']}m budget) --")
            for b in out["basket"]:
                print(f"     · [{b['source']:10}] {(b['disp'] or '')[:58]:58} ({b['features']['effort_min']:.0f}m)")
        elif out.get("radar"):
            print("  -- velocity radar (monitor mode: watch, no time budget) --")
        if out.get("repos"):
            print("  -- standout repos worth a PR (high activity, few contributors) --")
            for b in out["repos"]:
                so = b["features"].get("standout", 0)
                print(f"     · [{b['source']:10}] {(b['disp'] or '')[:52]:52} standout={so:.2f} [{b.get('language')}]")
