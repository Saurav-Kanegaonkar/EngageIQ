"""Phase 4: online adaptive learning from engage / skip / save feedback.

The signature personalization loop. Each session carries a weight vector over the
scoring SIGNALS, warm-started from its engagement mode's interpretable defaults
and nudged online by an LMS (Widrow-Hoff) update toward the signals the user
actually engages with. `recommend()` then scores with the learned weights, so the
ranking adapts within a session. This is the reward model of a contextual bandit;
exploration is supplied by the MMR diversity stage in rank.py.

Design choices that matter:
  - Weights stay non-negative and renormalized to sum 1, so impact stays in [0,1]
    and the vector remains INTERPRETABLE ("you respond most to velocity, recency").
  - engage = reward 1.0, save = 0.8 (positive but weaker), skip = 0.0 (negative).
  - State persists to data/feedback_state.json; every event is appended to
    data/feedback_log.jsonl for the eval + the demo.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from . import score

REWARD = {"engage": 1.0, "save": 0.8, "skip": 0.0, "dismiss": 0.0}
_DATA = Path(__file__).resolve().parent.parent.parent / "data"
STATE_PATH = _DATA / "feedback_state.json"
LOG_PATH = _DATA / "feedback_log.jsonl"
_LOCK = threading.Lock()


def _normalize(w: dict) -> dict:
    tot = sum(max(0.0, v) for v in w.values()) or 1.0
    return {k: max(0.0, v) / tot for k, v in w.items()}


class Learner:
    """Online linear reward model over the mode's signal weights."""

    def __init__(self, mode_key: str, weights: dict | None = None, n: int = 0):
        self.mode = mode_key
        base = score.MODES[mode_key].weights
        src = weights if weights else base
        # span ALL signals (missing ones start at 0) so a user can learn to value
        # signals the mode defaults ignore, e.g. a Contribute user who is actually
        # velocity-driven. Initial behaviour still equals the mode defaults.
        self.w = {s: float(src.get(s, base.get(s, 0.0))) for s in score.SIGNALS}
        self.n = n

    def predict(self, feats: dict) -> float:
        return sum(self.w[s] * float(feats.get(s, 0.0) or 0.0) for s in self.w)

    def update(self, feats: dict, action: str, lr: float = 0.30) -> "Learner":
        """One LMS step: w += lr * (reward - prediction) * signal_value, then
        clip to non-negative and renormalize to sum 1."""
        y = REWARD.get(action, 0.0)
        err = y - self.predict(feats)
        for s in self.w:
            x = float(feats.get(s, 0.0) or 0.0)
            self.w[s] = max(0.0, self.w[s] + lr * err * x)
        self.w = _normalize(self.w)
        self.n += 1
        return self

    def to_dict(self) -> dict:
        return {"mode": self.mode, "n": self.n, "w": self.w}


# Short, natural labels for the Priority / Non-priority line. score._LABEL phrases were
# written for the "Why this?" context and read awkwardly after "Priority:".
_PRIO_LABEL = {
    "relevance": "topics that match you", "community_health": "active communities",
    "visibility": "high-visibility posts", "velocity": "fast-rising topics",
    "standout": "room to stand out", "recency": "fresh posts",
    "discussion": "live discussions", "platform_fit": "your main platform",
    "effort_fit": "quick wins", "contribution_worth": "well-run repos to contribute to"}


def learned_summary(learner: Learner) -> str:
    """A simple Priority / Non-priority read of how this user differs from the mode
    defaults, computed live from the learned weights after each reaction."""
    base = score.MODES[learner.mode].weights
    deltas = sorted(((s, learner.w.get(s, 0.0) - base.get(s, 0.0)) for s in learner.w),
                    key=lambda x: x[1])
    lab = _PRIO_LABEL
    up = [lab.get(s, s) for s, d in reversed(deltas) if d > 0.02][:2]
    down = [lab.get(s, s) for s, d in deltas if d < -0.02][:2]
    if not up and not down:
        return "Still learning what you respond to."
    bits = []
    if up:
        bits.append("Priority: " + ", ".join(up))
    if down:
        bits.append("Non-priority: " + ", ".join(down))
    return ". ".join(bits) + "."


class Store:
    """File-backed per-session learner state + an append-only event log."""

    def __init__(self):
        _DATA.mkdir(exist_ok=True)
        self.state = self._load()

    def _load(self) -> dict:
        if STATE_PATH.exists():
            try:
                return json.loads(STATE_PATH.read_text())
            except Exception:  # noqa: BLE001
                return {}
        return {}

    def _save(self) -> None:
        STATE_PATH.write_text(json.dumps(self.state, indent=2))

    def learner(self, session: str, mode_key: str) -> Learner:
        s = self.state.get(session)
        if s and s.get("mode") == mode_key:
            return Learner(mode_key, s.get("w"), s.get("n", 0))
        return Learner(mode_key)

    def weights(self, session: str, mode_key: str) -> dict | None:
        """Learned weights for (session, mode), or None if this session hasn't
        given enough feedback yet (so the caller falls back to mode defaults)."""
        s = self.state.get(session)
        if s and s.get("mode") == mode_key and s.get("n", 0) > 0:
            return dict(s["w"])
        return None

    def record(self, session: str, mode_key: str, oid: str, action: str, feats: dict,
               rank: int | None = None, page: int | None = None, reason: str | None = None) -> Learner:
        with _LOCK:
            lr = self.learner(session, mode_key)
            lr.update(feats, action)
            self.state[session] = lr.to_dict()
            self._save()
            self._append({"kind": "feedback", "session": session, "mode": mode_key, "oid": oid,
                          "action": action, "rank": rank, "page": page, "reason": reason,
                          "ts": time.time(), "n": lr.n})
            return lr

    def log_event(self, session: str, etype: str, page: int | None = None, extra: dict | None = None) -> None:
        self._append({"kind": "event", "session": session, "type": etype, "page": page,
                      "extra": extra, "ts": time.time()})

    def _append(self, row: dict) -> None:
        try:
            with LOG_PATH.open("a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def reset(self, session: str) -> None:
        with _LOCK:
            self.state.pop(session, None)
            self._save()

    def clear_all(self) -> None:
        """Wipe ALL stored reactions: every session's learned weights + the event
        log. Used by the temporary "Reset learning" button while we iterate."""
        with _LOCK:
            self.state = {}
            self._save()
            try:
                LOG_PATH.unlink()
            except FileNotFoundError:
                pass
            except Exception:  # noqa: BLE001
                pass
