"""Per-user persistent state (Phase: persistence + idempotent feedback).

A small SQLite database (data/engage.sqlite) that is SEPARATE from the read-only
opportunity corpus (data/engageiq.sqlite), so the corpus stays a clean offline
snapshot while user state is mutable. It holds, scoped to (user, persona):

  users          who has used the app (one stable id per browser)
  personas       each profile they picked or built, plus their confirmed split
  interactions   ONE row per card  -> the current disposition (idempotent state)
  events         append-only stream of every click (for "why" analysis later)
  learner_state  the online RL weights, so learning survives a refresh

Identity is the browser's localStorage id (no login). Learning is keyed by
(user, persona) because the RL weights are engagement-mode specific.

Idempotency: clicking "Leverage" twice is ONE interest. The RL update + the
disposition only change on the FIRST transition; repeat clicks still append an
event (full history) but never double-count or re-train.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path

from . import feedback

_DATA = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = _DATA / "engage.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  user_id TEXT PRIMARY KEY, name TEXT, created_at REAL, last_seen_at REAL);

CREATE TABLE IF NOT EXISTS profiles(
  user_id TEXT, profile_key TEXT, name TEXT, goal TEXT, domains TEXT, platforms TEXT,
  time_budget_hours REAL, avoid TEXT, avatar TEXT, created_at REAL, updated_at REAL,
  PRIMARY KEY(user_id, profile_key));

CREATE TABLE IF NOT EXISTS personas(
  user_id TEXT, persona_key TEXT, name TEXT, goal TEXT, domains TEXT, platforms TEXT,
  time_budget_hours REAL, avoid TEXT, mode TEXT, split TEXT, created_at REAL, updated_at REAL,
  PRIMARY KEY(user_id, persona_key));

CREATE TABLE IF NOT EXISTS interactions(
  user_id TEXT, persona_key TEXT, opportunity_id TEXT,
  status TEXT DEFAULT 'none', saved INTEGER DEFAULT 0, reason TEXT,
  rank INTEGER, page INTEGER, bucket TEXT, source TEXT, mode TEXT, signals TEXT,
  first_at REAL, last_at REAL,
  PRIMARY KEY(user_id, persona_key, opportunity_id));

CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, persona_key TEXT, opportunity_id TEXT,
  type TEXT, reason TEXT, rank INTEGER, page INTEGER, bucket TEXT, source TEXT, mode TEXT,
  signals TEXT, ts REAL);
CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id, persona_key);

CREATE TABLE IF NOT EXISTS learner_state(
  user_id TEXT, persona_key TEXT, mode TEXT, n INTEGER DEFAULT 0, weights TEXT, updated_at REAL,
  PRIMARY KEY(user_id, persona_key));
"""

# action -> RL reward label (feedback.REWARD). save is a positive-but-weaker signal.
_RL = {"engage": "engage", "dismiss": "skip", "save": "save"}


class EngageStore:
    def __init__(self, path: Path | None = None):
        _DATA.mkdir(exist_ok=True)
        self.path = path or DB_PATH
        self._lock = threading.Lock()
        # FastAPI serves sync endpoints on a threadpool; SQLite handles are thread
        # bound, so allow cross-thread use and serialize every access with the lock.
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.executescript(SCHEMA)
            self._migrate_locked()
            self.conn.commit()

    def _migrate_locked(self) -> None:
        """Add columns that older databases predate (CREATE TABLE IF NOT EXISTS does not
        alter an existing table). Idempotent: only adds what is missing."""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(users)")}
        if "name" not in cols:
            self.conn.execute("ALTER TABLE users ADD COLUMN name TEXT")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_users_name ON users(name)")

    # ── accounts (simple name-based identity, no password) ────────────────
    def find_or_create_user(self, name: str) -> dict:
        """Find the account for this name (case-insensitive) or create one. Returns
        {user_id, name, is_new}. Identity only, not authentication: a name maps to an
        account so a person's profiles + learning persist server-side and reload by name.
        Matching is case-insensitive (Saurav == saurav == SAURAV) and the display name is
        normalized to title case so the same person always renders the same way."""
        name = " ".join((name or "").split())[:60] or "Guest"
        disp = " ".join(w[:1].upper() + w[1:].lower() for w in name.split()) or "Guest"
        now = time.time()
        with self._lock:
            row = self.conn.execute(
                "SELECT user_id, name FROM users WHERE name IS NOT NULL AND "
                "lower(name)=lower(?) ORDER BY created_at LIMIT 1", (disp,)).fetchone()
            if row:
                # normalize any legacy/mixed-case stored name to the title-case display
                self.conn.execute("UPDATE users SET last_seen_at=?, name=? WHERE user_id=?", (now, disp, row["user_id"]))
                self.conn.commit()
                return {"user_id": row["user_id"], "name": disp, "is_new": False}
            slug = (re.sub(r"[^a-z0-9]+", "-", disp.lower()).strip("-")[:20]) or "user"
            uid = "u_" + slug + "_" + hashlib.sha1(f"{disp.lower()}{now}".encode()).hexdigest()[:6]
            self.conn.execute(
                "INSERT INTO users(user_id, name, created_at, last_seen_at) VALUES(?,?,?,?)",
                (uid, disp, now, now))
            self.conn.commit()
            return {"user_id": uid, "name": disp, "is_new": True}

    # ── DB-backed profile gallery (per account) ───────────────────────────
    def list_profiles(self, user_id: str) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT profile_key, name, goal, domains, platforms, time_budget_hours, "
                "avoid, avatar FROM profiles WHERE user_id=? ORDER BY updated_at DESC",
                (user_id,)).fetchall()
        out = []
        for r in rows:
            out.append({"key": r["profile_key"], "name": r["name"], "goal": r["goal"],
                        "domains": json.loads(r["domains"] or "[]"),
                        "platforms": json.loads(r["platforms"] or "[]"),
                        "hours": r["time_budget_hours"], "avoid": r["avoid"] or "",
                        "ava": r["avatar"] or ""})
        return out

    def persona_splits(self, user_id: str) -> dict:
        """{persona_key: split_dict} for every persona this user has a confirmed weekly
        split on. Lets the client decide feed-vs-plan from the DB (the source of truth),
        not a browser localStorage copy that can be cleared or differ across devices."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT persona_key, split FROM personas "
                "WHERE user_id=? AND split IS NOT NULL AND split != ''",
                (user_id,)).fetchall()
        out = {}
        for r in rows:
            try:
                sp = json.loads(r["split"])
            except Exception:  # noqa: BLE001
                sp = None
            if isinstance(sp, dict) and sp:
                out[r["persona_key"]] = sp
        return out

    def save_profile(self, user_id: str, profile_key: str, *, name: str, goal: str,
                     domains: list, platforms: list, hours: float, avoid: str, avatar: str) -> str:
        now = time.time()
        with self._lock:
            exists = self.conn.execute(
                "SELECT 1 FROM profiles WHERE user_id=? AND profile_key=?",
                (user_id, profile_key)).fetchone()
            args = (name, goal, json.dumps(domains or []), json.dumps(platforms or []),
                    float(hours or 5), avoid, avatar)
            if exists:
                self.conn.execute(
                    "UPDATE profiles SET name=?, goal=?, domains=?, platforms=?, "
                    "time_budget_hours=?, avoid=?, avatar=?, updated_at=? "
                    "WHERE user_id=? AND profile_key=?", (*args, now, user_id, profile_key))
            else:
                self.conn.execute(
                    "INSERT INTO profiles(user_id, profile_key, name, goal, domains, platforms, "
                    "time_budget_hours, avoid, avatar, created_at, updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)", (user_id, profile_key, *args, now, now))
            self.conn.commit()
        return profile_key

    def delete_profile(self, user_id: str, profile_key: str) -> None:
        """Delete a profile and everything scoped to it (its reactions, events, learning,
        and runtime persona row)."""
        with self._lock:
            self.conn.execute(
                "DELETE FROM profiles WHERE user_id=? AND profile_key=?", (user_id, profile_key))
            for t in ("interactions", "events", "learner_state", "personas"):
                self.conn.execute(
                    f"DELETE FROM {t} WHERE user_id=? AND persona_key=?", (user_id, profile_key))
            self.conn.commit()

    # ── identity + persona ───────────────────────────────────────────────
    def save_persona(self, user_id: str, persona_key: str, persona: dict,
                     split: dict | None, mode: str) -> dict:
        """Upsert the persona (so an edit updates it IN PLACE, keeping its learning +
        history). If the edit changed the engagement MODE, reset the learner to the
        new mode's defaults (old weights were for a different signal mix) but keep the
        interactions/events. Returns {existed, mode_changed}."""
        now = time.time()
        with self._lock:
            prev = self.conn.execute(
                "SELECT mode FROM personas WHERE user_id=? AND persona_key=?",
                (user_id, persona_key)).fetchone()
            old_mode = prev["mode"] if prev else None
            mode_changed = bool(old_mode and old_mode != mode)
            self.conn.execute(
                "INSERT INTO users(user_id, created_at, last_seen_at) VALUES(?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET last_seen_at=excluded.last_seen_at",
                (user_id, now, now))
            self.conn.execute(
                "INSERT INTO personas(user_id, persona_key, name, goal, domains, platforms, "
                "time_budget_hours, avoid, mode, split, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(user_id, persona_key) DO UPDATE SET name=excluded.name, "
                "goal=excluded.goal, domains=excluded.domains, platforms=excluded.platforms, "
                "time_budget_hours=excluded.time_budget_hours, avoid=excluded.avoid, "
                "mode=excluded.mode, split=excluded.split, updated_at=excluded.updated_at",
                (user_id, persona_key, persona.get("name", ""), persona.get("goal", ""),
                 json.dumps(persona.get("domains", [])), json.dumps(persona.get("platforms", [])),
                 float(persona.get("time_budget_hours", 5) or 5),
                 ",".join(persona.get("exclude_languages", []) or []), mode,
                 json.dumps(split or {}), now, now))
            if mode_changed:                       # restart learning for the new mode (keep history)
                self.conn.execute(
                    "DELETE FROM learner_state WHERE user_id=? AND persona_key=?",
                    (user_id, persona_key))
            self.conn.commit()
        return {"existed": prev is not None, "mode_changed": mode_changed}

    # ── RL learner (persisted per user+persona) ──────────────────────────
    def _learner_locked(self, user_id: str, persona_key: str, mode: str) -> feedback.Learner:
        row = self.conn.execute(
            "SELECT mode, n, weights FROM learner_state WHERE user_id=? AND persona_key=?",
            (user_id, persona_key)).fetchone()
        if row and row["mode"] == mode and (row["n"] or 0) > 0:
            return feedback.Learner(mode, json.loads(row["weights"]), row["n"])
        return feedback.Learner(mode)

    def _save_learner_locked(self, user_id: str, persona_key: str, lr: feedback.Learner) -> None:
        self.conn.execute(
            "INSERT INTO learner_state(user_id, persona_key, mode, n, weights, updated_at) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(user_id, persona_key) DO UPDATE SET "
            "mode=excluded.mode, n=excluded.n, weights=excluded.weights, updated_at=excluded.updated_at",
            (user_id, persona_key, lr.mode, lr.n, json.dumps(lr.w), time.time()))

    def learner(self, user_id: str, persona_key: str, mode: str) -> feedback.Learner:
        with self._lock:
            return self._learner_locked(user_id, persona_key, mode)

    def weights(self, user_id: str, persona_key: str, mode: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT mode, n, weights FROM learner_state WHERE user_id=? AND persona_key=?",
                (user_id, persona_key)).fetchone()
        if row and row["mode"] == mode and (row["n"] or 0) > 0:
            return json.loads(row["weights"])
        return None

    # ── card states (to restore the feed on reload) ──────────────────────
    def states(self, user_id: str, persona_key: str) -> dict[str, dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT opportunity_id, status, saved, reason FROM interactions "
                "WHERE user_id=? AND persona_key=?", (user_id, persona_key)).fetchall()
        return {r["opportunity_id"]: {"status": r["status"], "saved": bool(r["saved"]),
                                      "reason": r["reason"]} for r in rows}

    def explored_count(self, user_id: str, persona_key: str) -> int:
        """Distinct opportunities the user has explored for this persona: leveraged or
        marked not-for-me (interactions) OR opened (open_detail events), deduped by id."""
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(DISTINCT oid) AS c FROM ("
                " SELECT opportunity_id AS oid FROM interactions"
                "  WHERE user_id=? AND persona_key=? AND status IN ('interested','dismissed')"
                " UNION"
                " SELECT opportunity_id AS oid FROM events"
                "  WHERE user_id=? AND persona_key=? AND type='open_detail')",
                (user_id, persona_key, user_id, persona_key)).fetchone()
        return int(row["c"]) if row and row["c"] is not None else 0

    # ── the core: idempotent action recording ────────────────────────────
    def record(self, user_id: str, persona_key: str, oid: str, action: str, *,
               signals: dict, mode: str, rank: int | None = None, page: int | None = None,
               bucket: str | None = None, source: str | None = None,
               reason: str | None = None) -> dict:
        now = time.time()
        sig_json = json.dumps(signals or {})
        with self._lock:
            cur = self.conn.execute(
                "SELECT status, saved FROM interactions WHERE user_id=? AND persona_key=? AND opportunity_id=?",
                (user_id, persona_key, oid)).fetchone()
            prev_status = cur["status"] if cur else "none"
            prev_saved = (cur["saved"] if cur else 0) or 0
            status, saved, rl_action = prev_status, prev_saved, None

            if action == "engage":
                status = "interested"
                if prev_status != "interested":
                    rl_action = "engage"
            elif action == "dismiss":
                status = "dismissed"
                if prev_status != "dismissed":
                    rl_action = "dismiss"
            elif action == "undo":
                status = "none"
            elif action == "save":
                saved = 1
                if not prev_saved:
                    rl_action = "save"
            elif action == "unsave":
                saved = 0

            if cur:
                self.conn.execute(
                    "UPDATE interactions SET status=?, saved=?, reason=COALESCE(?, reason), last_at=? "
                    "WHERE user_id=? AND persona_key=? AND opportunity_id=?",
                    (status, saved, reason, now, user_id, persona_key, oid))
            else:
                self.conn.execute(
                    "INSERT INTO interactions(user_id, persona_key, opportunity_id, status, saved, "
                    "reason, rank, page, bucket, source, mode, signals, first_at, last_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (user_id, persona_key, oid, status, saved, reason, rank, page, bucket,
                     source, mode, sig_json, now, now))

            self.conn.execute(
                "INSERT INTO events(user_id, persona_key, opportunity_id, type, reason, rank, page, "
                "bucket, source, mode, signals, ts) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (user_id, persona_key, oid, action, reason, rank, page, bucket, source, mode, sig_json, now))

            lr = self._learner_locked(user_id, persona_key, mode)
            if rl_action and signals:                       # train ONLY on a new disposition
                lr.update(signals, _RL[rl_action])
                self._save_learner_locked(user_id, persona_key, lr)
            self.conn.commit()

        return {"status": status, "saved": bool(saved), "applied": rl_action is not None,
                "n": lr.n, "w": lr.w, "learner": lr}

    def log_event(self, user_id: str, persona_key: str, etype: str, *, oid: str | None = None,
                  rank: int | None = None, page: int | None = None, bucket: str | None = None,
                  source: str | None = None, mode: str | None = None, signals: dict | None = None,
                  reason: str | None = None) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO events(user_id, persona_key, opportunity_id, type, reason, rank, page, "
                "bucket, source, mode, signals, ts) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (user_id, persona_key, oid, etype, reason, rank, page, bucket, source, mode,
                 json.dumps(signals or {}), time.time()))
            self.conn.commit()

    def interactions_for(self, user_id: str, persona_key: str) -> list[dict]:
        """Every card this (user, persona) has acted on, newest first, so the user
        can SEE their reactions (engaged / saved / passed)."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT opportunity_id, status, saved, reason, bucket, source, last_at "
                "FROM interactions WHERE user_id=? AND persona_key=? ORDER BY last_at DESC",
                (user_id, persona_key)).fetchall()
        return [dict(r) for r in rows]

    def learner_row(self, user_id: str, persona_key: str) -> feedback.Learner | None:
        """The persisted learner for this (user, persona), or None if no feedback yet."""
        with self._lock:
            row = self.conn.execute(
                "SELECT mode, n, weights FROM learner_state WHERE user_id=? AND persona_key=?",
                (user_id, persona_key)).fetchone()
        if row and (row["n"] or 0) > 0:
            return feedback.Learner(row["mode"], json.loads(row["weights"]), row["n"])
        return None

    def delete_persona(self, user_id: str, persona_key: str) -> None:
        """Delete one persona and everything scoped to it (interactions, events,
        learned weights). Used when the user removes a profile from their gallery."""
        with self._lock:
            for t in ("interactions", "events", "learner_state", "personas"):
                self.conn.execute(
                    f"DELETE FROM {t} WHERE user_id=? AND persona_key=?",
                    (user_id, persona_key))
            self.conn.commit()

    def clear_all(self) -> None:
        """Wipe ALL user state (accounts, profiles, reactions, learning) for a clean slate."""
        with self._lock:
            for t in ("events", "interactions", "learner_state", "personas", "profiles", "users"):
                self.conn.execute(f"DELETE FROM {t}")
            self.conn.commit()
