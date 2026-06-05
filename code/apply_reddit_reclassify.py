"""Apply the sub-agent Reddit reclassification: update labels, drop noise.

Reads data/reddit_classify/result_*.json (written by the reclassification
workflow's chunk agents), validates the domain keys, then:
  - non-noise posts: set `domain` (= new primary) and `domains` (= multi-label JSON)
  - noise posts: archive their rows to data/reddit_noise.json, then DELETE them
Reports coverage so any failed/short chunk can be re-run before re-embedding.
"""
from __future__ import annotations

import glob
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "engageiq.sqlite"
RESDIR = ROOT / "data" / "reddit_classify"
NOISE_ARCHIVE = ROOT / "data" / "reddit_noise.json"

VALID = {"machine_learning", "devops_k8s", "open_source_trending", "developer_tools",
         "cybersecurity", "frontend_web", "b2b_saas", "blockchain", "python_data_eng",
         "gamedev_cpp", "ai_research", "embedded_systems", "cloud_apis", "mobile_dev",
         "beginner_coding"}


def main() -> None:
    conn = sqlite3.connect(str(DB))
    reddit_ids = {r[0] for r in conn.execute(
        "SELECT source_native_id FROM opportunities WHERE source='reddit'")}
    print(f"reddit rows in DB: {len(reddit_ids)}")

    verdicts: dict[str, dict] = {}
    bad_files = []
    for f in sorted(glob.glob(str(RESDIR / "result_*.json"))):
        try:
            arr = json.loads(Path(f).read_text())
        except Exception as e:  # noqa: BLE001
            bad_files.append((Path(f).name, str(e)[:60]))
            continue
        for o in arr:
            oid = o.get("id")
            if not oid:
                continue
            doms = [d for d in (o.get("domains") or []) if d in VALID][:3]
            noise = bool(o.get("noise")) or not doms
            verdicts[oid] = {"domains": doms, "noise": noise}

    covered = reddit_ids & set(verdicts)
    missing = reddit_ids - set(verdicts)
    print(f"classified: {len(covered)}/{len(reddit_ids)}  |  missing: {len(missing)}  |  bad files: {bad_files}")
    if missing:
        # which chunks are missing? (chunk_i ids)
        miss_chunks = []
        for f in sorted(glob.glob(str(RESDIR / "chunk_*.json"))):
            ids = {x["id"] for x in json.loads(Path(f).read_text())}
            if ids & missing and not (RESDIR / Path(f).name.replace("chunk_", "result_")).exists():
                miss_chunks.append(Path(f).stem)
        print(f"  -> missing/failed chunks to re-run: {miss_chunks or 'none (partial coverage within chunks)'}")
        print(f"  -> missing ids keep their current subreddit label for now.")

    # apply updates for non-noise covered posts
    updates, noise_ids = [], []
    for oid in covered:
        v = verdicts[oid]
        if v["noise"]:
            noise_ids.append(oid)
        else:
            updates.append((v["domains"][0], json.dumps(v["domains"]), oid))
    conn.executemany(
        "UPDATE opportunities SET domain=?, domains=? WHERE source='reddit' AND source_native_id=?",
        updates)
    conn.commit()
    print(f"\nupdated labels on {len(updates)} reddit posts (multi-label)")

    # archive + drop noise
    if noise_ids:
        ph = ",".join("?" * len(noise_ids))
        cols = [r[1] for r in conn.execute("PRAGMA table_info(opportunities)")]
        rows = conn.execute(
            f"SELECT * FROM opportunities WHERE source='reddit' AND source_native_id IN ({ph})",
            noise_ids).fetchall()
        NOISE_ARCHIVE.write_text(json.dumps([dict(zip(cols, r)) for r in rows], indent=0))
        conn.execute(
            f"DELETE FROM opportunities WHERE source='reddit' AND source_native_id IN ({ph})",
            noise_ids)
        conn.commit()
        print(f"archived + dropped {len(noise_ids)} noise posts -> {NOISE_ARCHIVE.name}")

    total = conn.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0]
    rtotal = conn.execute("SELECT COUNT(*) FROM opportunities WHERE source='reddit'").fetchone()[0]
    print(f"\nDB total now: {total}  |  reddit now: {rtotal}")
    print("\nnew Reddit PRIMARY-domain split:")
    for d, n in conn.execute(
        "SELECT domain, COUNT(*) FROM opportunities WHERE source='reddit' GROUP BY domain ORDER BY COUNT(*) DESC"):
        print(f"  {d:22} {n}")
    multi = conn.execute(
        "SELECT COUNT(*) FROM opportunities WHERE source='reddit' AND domains LIKE '%,%'").fetchone()[0]
    print(f"\nmulti-domain reddit posts: {multi}")


if __name__ == "__main__":
    main()
