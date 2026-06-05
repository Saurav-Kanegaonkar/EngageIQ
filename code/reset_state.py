"""Wipe ALL user state (accounts, profiles, personas, reactions, events, learning)
from the READ-WRITE user DB only (``data/engage.sqlite``).

This NEVER touches the read-only corpus (``data/engageiq.sqlite``), embeddings,
summaries, or trends. It is the fast, safe "clean slate" the user asks for
repeatedly. Idempotent: running it twice just reports zeros.

    PYTHONPATH=code .venv/bin/python code/reset_state.py            # wipe
    PYTHONPATH=code .venv/bin/python code/reset_state.py --dry-run  # just show counts

After wiping the DB, also clear the browser's local copy so the UI shows the
clean slate (run this in the page, or let the helper below print it):

    ["eiq-account","eiq-personas","eiq-profiles","eiq-custom-profile","eiq-custom-pid"]
      .forEach(k=>localStorage.removeItem(k)); location.reload();
"""
from __future__ import annotations

import sys

from engageiq.store import EngageStore

# the six user-state tables clear_all() wipes; corpus tables live in a different file
TABLES = ("users", "profiles", "personas", "interactions", "events", "learner_state")


def _counts(store: EngageStore) -> dict:
    out = {}
    for t in TABLES:
        try:
            out[t] = store.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception:  # noqa: BLE001 -- a missing table just reports as n/a
            out[t] = "n/a"
    return out


def reset(dry_run: bool = False) -> dict:
    store = EngageStore()
    before = _counts(store)
    if not dry_run:
        store.clear_all()
    after = _counts(store)
    store.conn.close()
    label = "Would wipe" if dry_run else "Wiped"
    print(f"{label} user state in data/engage.sqlite (corpus untouched):")
    for t in TABLES:
        print(f"  {t:14} {str(before[t]):>6} -> {after[t]}")
    return {"before": before, "after": after, "dry_run": dry_run}


if __name__ == "__main__":
    reset(dry_run="--dry-run" in sys.argv)
