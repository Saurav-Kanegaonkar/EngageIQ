"""Phase 4 evaluation: does the online learner adapt over 50+ feedback rounds?

Setup: take a real candidate pool for a persona, then invent a HIDDEN user
preference that differs from the engagement mode's defaults (a Contribute user who
is secretly velocity / recency driven). Each round we show the user the current
learner's top-K, the user reacts according to their hidden preference (engage the
genuinely-good ones, skip the rest), and the learner updates. We track NDCG@5 of
the learner's ranking against the hidden-optimal ranking and watch it climb from
the mode-default baseline toward the oracle.

Run:  PYTHONPATH=code .venv/bin/python -m engageiq.eval.adaptive_eval
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import math
import random
from pathlib import Path

from engageiq import feedback, personas, rank, score

REPORTS = Path(__file__).resolve().parent.parent.parent.parent / "reports"
K_PRESENT = 6        # how many we show the user each round
K_NDCG = 5
ROUNDS = 60
SEED = 7


def _ndcg(order, util, k=K_NDCG):
    def dcg(idxs):
        return sum(util[i] / math.log2(pos + 2) for pos, i in enumerate(idxs[:k]))
    ideal = sorted(range(len(util)), key=lambda i: -util[i])
    return dcg(order) / (dcg(ideal) or 1e-9)


def _rank_by(weights, X, n):
    return sorted(range(n), key=lambda i: -sum(weights.get(s, 0.0) * X[i][s] for s in weights))


def main() -> None:
    random.seed(SEED)
    rec = rank.Recommender()
    persona = personas.PERSONAS["sofia"]
    mode_key = score.mode_for_persona(persona)
    mode = score.MODES[mode_key]

    pool = rec.recommend(persona, k=50)["ranked"]
    X = [{s: float(it["features"].get(s, 0.0) or 0.0) for s in score.SIGNALS} for it in pool]
    n = len(X)

    # hidden preference: trend-chaser inside Contribute mode (defaults are relevance-heavy)
    w_true = {"velocity": 0.30, "recency": 0.25, "standout": 0.20, "relevance": 0.15, "community_health": 0.10}
    tot = sum(w_true.values())
    w_true = {k: v / tot for k, v in w_true.items()}
    util = [sum(w_true[s] * X[i].get(s, 0.0) for s in w_true) for i in range(n)]
    median = sorted(util)[n // 2]

    base_ndcg = _ndcg(_rank_by(mode.weights, X, n), util)        # default weights, fixed
    learner = feedback.Learner(mode_key)
    curve = []
    for t in range(1, ROUNDS + 1):
        order = _rank_by(learner.w, X, n)
        shown = order[:K_PRESENT - 1] + [random.randrange(n)]    # exploit + 1 explore
        for i in shown:
            learner.update(X[i], "engage" if util[i] >= median else "skip")
        curve.append((t, _ndcg(_rank_by(learner.w, X, n), util)))

    final = curve[-1][1]

    lines = []
    lines.append("=" * 70)
    lines.append("ADAPTIVE LEARNING EVAL  (online weight learner, 50+ feedback rounds)")
    lines.append("=" * 70)
    lines.append(f"persona=sofia  mode={mode_key}  pool={n}  rounds={ROUNDS}  present/round={K_PRESENT}")
    lines.append("Hidden user preference differs from the mode defaults (velocity/recency driven).")
    lines.append("")
    lines.append(f"baseline NDCG@{K_NDCG} (mode-default weights, no learning): {base_ndcg:.3f}")
    for t, v in curve:
        if t == 1 or t % 10 == 0:
            bar = "#" * int(v * 40)
            lines.append(f"  round {t:>2}: NDCG@{K_NDCG} {v:.3f}  {bar}")
    lines.append(f"final NDCG@{K_NDCG} (learned weights):                       {final:.3f}"
                 f"   ({'+' if final >= base_ndcg else ''}{final - base_ndcg:.3f} vs baseline)")
    lines.append("")
    lines.append("weight convergence (did the learner discover the hidden preference?):")
    lines.append(f"  {'signal':18} {'default':>8} {'learned':>8} {'true':>8}")
    for s in sorted(score.SIGNALS, key=lambda s: -w_true.get(s, 0.0)):
        dflt, lrn, tru = mode.weights.get(s, 0.0), learner.w.get(s, 0.0), w_true.get(s, 0.0)
        if dflt > 0.01 or tru > 0.01 or lrn > 0.05:
            lines.append(f"  {s:18} {dflt:>8.2f} {lrn:>8.2f} {tru:>8.2f}")
    lines.append("")
    lines.append(feedback.learned_summary(learner))
    report = "\n".join(lines)
    print(report)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "phase4_adaptive_eval.txt").write_text(report + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts = [t for t, _ in curve]
        vs = [v for _, v in curve]
        plt.figure(figsize=(7, 4))
        plt.axhline(base_ndcg, ls="--", color="#999", label=f"mode default ({base_ndcg:.2f})")
        plt.plot(ts, vs, color="#4f46e5", lw=2, label="learned (online)")
        plt.xlabel("feedback round"); plt.ylabel(f"NDCG@{K_NDCG} vs hidden optimal")
        plt.title("EngageIQ adaptive learning: ranking improves with feedback")
        plt.legend(); plt.tight_layout()
        plt.savefig(REPORTS / "phase4_adaptive_curve.png", dpi=120)
        print(f"\nsaved curve -> reports/phase4_adaptive_curve.png")
    except Exception as e:  # noqa: BLE001
        print(f"\n(plot skipped: {e})")


if __name__ == "__main__":
    main()
