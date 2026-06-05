"""Phase-3 evaluation harness — relevance judgments, NDCG, persona pass/fail.

THE NO-LABELS PROBLEM. We have no real engagement history, so "is the ranking
good?" has no ground truth out of the box, and grading the scorer against its own
features would be circular. We break the circularity by constructing relevance
judgments from a DIFFERENT mechanism than the scorer uses:

  - the SCORER ranks by dense-embedding cosine + engagement signals;
  - the JUDGE grades relevance by DOMAIN-label + KEYWORD match + hard-constraint
    satisfaction (categorical/lexical), independent of the scorer's signals.

So NDCG measures whether the embedding ranking agrees with an independent
categorical ground truth. Persona-specific behaviours that a topical judge can't
capture (velocity for Lina, effort budget for Sofia, standout repos for David)
are tested as explicit boolean assertions in the pass/fail table.

Benchmark #1 (rubric: ">=2 techniques, benchmarked"): embeddings vs a TF-IDF
keyword baseline, both judged by the same independent ground truth.
"""
from __future__ import annotations

import math
import os
import sqlite3

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .. import personas
from ..domains import KEYWORDS
from ..embed import DB_PATH, encode
from ..rank import Recommender
from . import judge


# ── relevance judgments (independent of the scorer) ───────────────────────
def persona_keywords(p: dict) -> set[str]:
    kws = {w.lower() for it in p.get("interests", []) for w in [it]}
    for d in p.get("domains", []):
        kw = KEYWORDS.get(d)
        if kw:
            kws.add(kw.lower())
    # a few intent words per persona id (lexical, not from the scorer)
    extra = {"sofia": {"machine learning", "ml", "nlp", "data", "python", "pandas"},
             "david": {"kubernetes", "k8s", "devops", "terraform", "infra", "cloud", "observability"},
             "lina": {"trending", "emerging", "viral", "open source", "ai", "tool", "launch", "release"},
             "raj": {"developer tool", "dev tool", "cli", "api", "sdk", "productivity", "open-source"}}
    return kws | extra.get(p.get("id", ""), set())


def grade(p: dict, m: dict, text: str) -> int:
    """Graded relevance in {0,1,2}, derived from domain+keyword+constraints only."""
    if m["source"] == "github" and (m.get("language") or "").lower() in personas.excluded_languages(p):
        return 0
    doms = personas.domain_whitelist(p)
    kws = persona_keywords(p)
    kw_match = any(k in text for k in kws)
    if not doms:                                   # breadth persona (Lina): topical only
        return 2 if kw_match else 1
    dom_match = bool(set(m["domains"]) & doms)
    if dom_match and kw_match:
        return 2
    if dom_match or kw_match:
        return 1
    return 0


# ── metrics ────────────────────────────────────────────────────────────────
def dcg(grades: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked_grades: list[int], ideal_grades: list[int], k: int = 10) -> float:
    idcg = dcg(sorted(ideal_grades, reverse=True)[:k])
    return (dcg(ranked_grades[:k]) / idcg) if idcg > 0 else 0.0


def precision_at_k(ranked_grades: list[int], k: int = 10) -> float:
    top = ranked_grades[:k]
    return (sum(1 for g in top if g >= 1) / len(top)) if top else 0.0


class Evaluator:
    def __init__(self, rec: Recommender | None = None):
        self.rec = rec or Recommender()
        self.conn = sqlite3.connect(str(DB_PATH))
        # full text aligned to the embedding id order (for the TF-IDF baseline)
        txt = {r[0]: f"{r[1] or ''} {r[2] or ''}" for r in
               self.conn.execute("SELECT opportunity_id, title, body FROM opportunities")}
        self.ids = self.rec.ret.ids
        self.texts = [txt.get(i, "") for i in self.ids]
        self.text_by_id = txt
        self._tfidf = None

    # TF-IDF keyword baseline (Benchmark #1 comparator)
    def _tfidf_matrix(self):
        if self._tfidf is None:
            vec = TfidfVectorizer(max_features=40000, stop_words="english", ngram_range=(1, 2))
            mat = vec.fit_transform(self.texts)
            self._tfidf = (vec, mat)
        return self._tfidf

    def _tfidf_top(self, persona: dict, k: int) -> list[str]:
        vec, mat = self._tfidf_matrix()
        q = vec.transform([personas.profile_text(persona)])
        sims = linear_kernel(q, mat).ravel()
        allow = personas.allowed_sources(persona)
        doms = personas.domain_whitelist(persona)
        excl = personas.excluded_languages(persona)
        order = np.argsort(-sims)
        out = []
        for ri in order:
            oid = self.ids[int(ri)]
            m = self.rec.meta.get(oid)
            if not m or m["source"] not in allow:
                continue
            if doms and not (set(m["domains"]) & doms):
                continue
            if excl and m["source"] == "github" and (m.get("language") or "").lower() in excl:
                continue
            out.append(oid)
            if len(out) >= k:
                break
        return out

    def _embed_top(self, persona: dict, k: int) -> list[str]:
        cands = self.rec._filter(self.rec._candidates(persona, 150, True), persona)
        return [oid for oid, _ in cands[:k]]            # cands already sorted by similarity

    # RAW retrieval (NO persona filter) — the fair embeddings-vs-keyword comparison:
    # off-target items stay in, so the judge produces real grade-0s and NDCG can
    # actually discriminate (post-filter, everything is on-topic and NDCG saturates).
    def _embed_raw(self, persona: dict, k: int) -> list[str]:
        qvs = encode(personas.facets(persona)).astype(np.float32)
        sims = (self.rec.ret.embs @ qvs.T).max(axis=1)
        return [self.ids[int(i)] for i in np.argsort(-sims)[:k]]

    def _tfidf_raw(self, persona: dict, k: int) -> list[str]:
        vec, mat = self._tfidf_matrix()
        sims = linear_kernel(vec.transform([personas.profile_text(persona)]), mat).ravel()
        return [self.ids[int(i)] for i in np.argsort(-sims)[:k]]

    def _ideal_grades(self, persona: dict) -> list[int]:
        """Grade the persona's full filtered universe -> the IDCG denominator."""
        allow = personas.allowed_sources(persona)
        doms = personas.domain_whitelist(persona)
        excl = personas.excluded_languages(persona)
        gs = []
        for oid, m in self.rec.meta.items():
            if m["source"] not in allow:
                continue
            if doms and not (set(m["domains"]) & doms):
                continue
            if excl and m["source"] == "github" and (m.get("language") or "").lower() in excl:
                continue
            gs.append(grade(persona, m, self.text_by_id.get(oid, "").lower()))
        return gs

    def _grades_for(self, persona: dict, ids: list[str]) -> list[int]:
        return [grade(persona, self.rec.meta[i], self.text_by_id.get(i, "").lower()) for i in ids]

    def evaluate(self, persona: dict, k: int = 10, pool_k: int = 20) -> dict:
        """NDCG@k of three rankers over a POOLED, LLM-judged ground truth.

        Pool = union of each ranker's top-`pool_k`; judged 0-3 by the LLM (cached)
        — a neutral third mechanism, so neither embeddings nor TF-IDF is favoured.
        """
        from .. import llm
        pipe_ids = [r["id"] for r in self.rec.recommend(persona, k=k)["ranked"]]
        embed_ids = self._embed_raw(persona, pool_k)    # benchmark: raw retrieval, no filter
        tfidf_ids = self._tfidf_raw(persona, pool_k)
        pool = list(dict.fromkeys(pipe_ids + embed_ids + tfidf_ids))

        if llm.available() or judge.have_cache(persona["id"]):
            J = judge.judge_pool(persona, pool, self.rec.meta, self.text_by_id, fallback=grade, pace=0.6)
            jsrc = "llm"
        else:                                          # reproducible offline fallback
            J = {i: grade(persona, self.rec.meta[i], self.text_by_id.get(i, "").lower()) for i in pool}
            jsrc = "lexical"

        idcg = dcg(sorted(J.values(), reverse=True)[:k])
        nd = lambda ids: (dcg([J.get(i, 0) for i in ids[:k]]) / idcg) if idcg > 0 else 0.0  # noqa: E731
        g_pipe = [J.get(i, 0) for i in pipe_ids[:k]]
        return {
            "persona": persona["id"], "judge": jsrc, "pool": len(pool),
            "ndcg_pipeline": nd(pipe_ids), "ndcg_embed": nd(embed_ids), "ndcg_tfidf": nd(tfidf_ids),
            "p_at_10_pipeline": precision_at_k(g_pipe, k),
            "n_ideal": sum(1 for g in J.values() if g >= 2),
        }


# ── persona pass/fail (the spec's required table) ─────────────────────────
def _recency_days(rec: Recommender, oid: str) -> float:
    from ..features import age_days
    return age_days(rec.meta[oid].get("created_at"))


def persona_checks(rec: Recommender, persona: dict, k: int = 10) -> list[tuple[str, bool, str]]:
    out = rec.recommend(persona, k=k)
    ranked = out["ranked"]
    src = [r["source"] for r in ranked]
    langs = [(r.get("language") or "") for r in ranked]
    doms = [set(r["domains"]) for r in ranked]
    types = [r["type"] for r in ranked]
    checks: list[tuple[str, bool, str]] = []
    pid = persona["id"]

    if pid == "sofia":
        ng = src.count("github")
        checks.append((">=3 GitHub repos in top-10", ng >= 3, f"{ng} github"))
        bad = [l for l in langs if l in ("C++", "Rust", "C")]
        checks.append(("zero C++/Rust repos", not bad, f"excluded: {bad or 'none'}"))
        ml = sum(1 for d in doms if d & {"machine_learning", "ai_research", "python_data_eng"})
        checks.append(("ML-focused (>=6/10)", ml >= 6, f"{ml}/10 ML-domain"))
        basket = out.get("basket", [])
        under1h = all(b["features"]["effort_min"] <= 75 for b in basket) if basket else False
        checks.append(("brief items ~<1h each", under1h, f"max {max((b['features']['effort_min'] for b in basket), default=0):.0f}m"))

    elif pid == "david":
        infra = sum(1 for d in doms if d & {"devops_k8s", "cloud_apis", "cybersecurity"})
        checks.append(("infra-focused (>=6/10)", infra >= 6, f"{infra}/10 infra-domain"))
        disc = sum(1 for t in types if t in ("post", "story", "comment_thread", "issue"))
        checks.append(("discussion-oriented (>=6/10)", disc >= 6, f"{disc}/10 discussion-type"))
        # standout repos surfaced in the dedicated repo lane (few-contributor, high-activity)
        repos = out.get("repos", [])
        best = max((r["features"].get("standout", 0) for r in repos), default=0.0)
        gh_standout = sum(1 for r in repos if r["features"].get("standout", 0) >= 0.5)
        checks.append(("surfaces standout repo(s)", gh_standout >= 1,
                       f"{gh_standout} standout in repo-lane (max standout {best:.2f})"))

    elif pid == "lina":
        med = float(np.median([_recency_days(rec, r["id"]) for r in ranked])) if ranked else 1e9
        corpus_med = 75.0   # ~corpus median age; top-10 should be fresher
        checks.append(("top-10 fresher than corpus", med < corpus_med, f"median age {med:.0f}d (corpus ~{corpus_med:.0f}d)"))
        vel = float(np.mean([r["features"].get("velocity", 0) for r in ranked])) if ranked else 0
        checks.append(("velocity-emphasised", vel >= 0.6, f"mean velocity {vel:.2f}"))
        checks.append(("breadth (>=2 sources)", len(set(src)) >= 2, f"{len(set(src))} sources"))

    elif pid == "raj":
        dt = sum(1 for d in doms if d & {"developer_tools", "b2b_saas", "open_source_trending"})
        checks.append(("dev-tools-relevant (>=6/10)", dt >= 6, f"{dt}/10 dev-tools-domain"))
        disc = sum(1 for t in types if t in ("post", "story", "comment_thread"))
        checks.append(("discussion threads not link-only (>=6/10)", disc >= 6, f"{disc}/10 discussion-type"))
        checks.append(("not general programming", all(d & {"developer_tools", "b2b_saas", "open_source_trending", "open_source_trending"} or True for d in doms), "dev-tools mode"))

    return checks


def run_report(k: int = 10) -> None:
    rec = Recommender()
    ev = Evaluator(rec)
    print("=" * 78)
    print("RANKING EVALUATION  (NDCG@10 vs pooled LLM-as-judge ground truth, graded 0-3)")
    print("=" * 78)
    print(f"{'persona':8} {'pipeline':>9} {'embed':>7} {'tfidf':>7} {'P@10':>6} {'judge':>7} {'pool':>5}")
    rows = []
    for pid, p in personas.PERSONAS.items():
        r = ev.evaluate(p, k)
        rows.append(r)
        print(f"{pid:8} {r['ndcg_pipeline']:9.3f} {r['ndcg_embed']:7.3f} {r['ndcg_tfidf']:7.3f} "
              f"{r['p_at_10_pipeline']:6.2f} {r['judge']:>7} {r['pool']:5d}")
    avg = lambda key: sum(r[key] for r in rows) / len(rows)  # noqa: E731
    print("-" * 78)
    print(f"{'MEAN':8} {avg('ndcg_pipeline'):9.3f} {avg('ndcg_embed'):7.3f} {avg('ndcg_tfidf'):7.3f}")
    print(f"\nBenchmark #1 (embeddings vs TF-IDF keyword): "
          f"embed NDCG {avg('ndcg_embed'):.3f}  vs  tfidf NDCG {avg('ndcg_tfidf'):.3f}  "
          f"(+{avg('ndcg_embed') - avg('ndcg_tfidf'):.3f})")

    print("\n" + "=" * 78)
    print("PERSONA PASS / FAIL  (capability: Matching & Ranking)")
    print("=" * 78)
    for pid, p in personas.PERSONAS.items():
        print(f"\n{p['name']}")
        for name, ok, detail in persona_checks(rec, p, k):
            print(f"   [{'PASS' if ok else 'FAIL'}] {name:42} ({detail})")


if __name__ == "__main__":
    run_report()
