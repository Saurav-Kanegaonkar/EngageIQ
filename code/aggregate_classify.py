"""Aggregate the multi-label classification chunks into the DB + structural check.

Reads data/classify_full/chunk_*.json (each {rowid: [domain,...]} primary-first),
validates keys/format, writes `domain` (= primary) and `domains` (= JSON list),
and reports coverage so any missing/failed chunk can be re-run.
"""
from __future__ import annotations

import glob
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engageiq.domains import DOMAIN_LABELS, DOMAINS  # noqa: E402

VALID = set(DOMAINS) | {"other"}
ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "engageiq.sqlite"
OUT = ROOT / "data" / "classify_full"


def main() -> None:
    conn = sqlite3.connect(str(DB))
    cols = [r[1] for r in conn.execute("PRAGMA table_info(opportunities)")]
    if "domains" not in cols:
        conn.execute("ALTER TABLE opportunities ADD COLUMN domains TEXT")

    all_rows = {r[0] for r in conn.execute("SELECT rowid FROM opportunities")}
    covered: set[int] = set()
    bad_files, invalid = [], 0
    updates = []
    for f in sorted(glob.glob(str(OUT / "chunk_*.json"))):
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001
            bad_files.append(f)
            continue
        for rid, doms in d.items():
            rid = int(rid)
            if not isinstance(doms, list):
                invalid += 1
                continue
            doms = [x for x in doms if x in VALID][:3]
            if not doms:
                doms = ["other"]
                invalid += 1
            covered.add(rid)
            updates.append((doms[0], json.dumps(doms), rid))

    conn.executemany("UPDATE opportunities SET domain=?, domains=? WHERE rowid=?", updates)
    conn.commit()

    missing = all_rows - covered
    print(f"=== structural check ===")
    print(f"classified: {len(covered)}/{len(all_rows)}  |  missing: {len(missing)}  |  invalid entries: {invalid}  |  bad files: {len(bad_files)}")
    if missing:
        print("first missing rowids:", sorted(missing)[:25])
    if bad_files:
        print("bad files:", bad_files)

    multi = conn.execute("SELECT COUNT(*) FROM opportunities WHERE domains LIKE '%,%'").fetchone()[0]
    other = conn.execute("SELECT COUNT(*) FROM opportunities WHERE domain='other'").fetchone()[0]
    print(f"\nmulti-domain records: {multi}  |  'other'/noise: {other}")
    print("new PRIMARY-domain split:")
    for d, n in conn.execute("SELECT domain, COUNT(*) FROM opportunities GROUP BY domain ORDER BY COUNT(*) DESC"):
        print(f"  {DOMAIN_LABELS.get(d, d):24} {n}")


if __name__ == "__main__":
    main()
