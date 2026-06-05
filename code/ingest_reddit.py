"""Ingest the browser-captured Reddit snapshot into the opportunity store.

Reddit blocks server-side API access (bot detection on non-browser traffic), so
the data was captured via a browser-console fetch on reddit.com (the TA-sanctioned
public .json endpoints, using the logged-in session) into data/reddit_snapshot.json.
This script normalizes those posts into the canonical Opportunity schema and stores
them with the same Bloom + primary-key dedup as every other source.

Domain labels come from the subreddit -> domain mapping baked into the snapshot,
which is a stronger signal than keyword matching (r/kubernetes is unambiguously
devops_k8s), so Reddit rows get a clean single-domain label.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engageiq.schema import Opportunity  # noqa: E402
from engageiq.storage import Store  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = sorted((ROOT / "data").glob("reddit_snapshot*.json"))


def _iso(utc: float | int | None) -> str:
    if not utc:
        return ""
    return datetime.fromtimestamp(float(utc), tz=timezone.utc).isoformat()


def to_opportunity(post: dict, domain: str, subreddit: str) -> Opportunity:
    flair = post.get("link_flair_text")
    return Opportunity(
        source="reddit",
        source_native_id=str(post.get("id")),
        url="https://www.reddit.com" + (post.get("permalink") or ""),
        title=(post.get("title") or "")[:300],
        body=(post.get("selftext") or "")[:4000],
        opportunity_type="post",            # a Reddit thread to discuss/engage in
        domain=domain,
        created_at=_iso(post.get("created_utc")),
        tags=[flair] if flair else [],
        author=post.get("author"),
        community="r/" + subreddit,
        score=post.get("score"),
        num_comments=post.get("num_comments"),
        signals={
            "upvote_ratio": post.get("upvote_ratio"),
            "is_self": post.get("is_self"),
            "flair": flair,
            "over_18": post.get("over_18"),
            "external_url": None if post.get("is_self") else post.get("url"),
        },
        raw={},
    )


def main() -> None:
    if not SNAPSHOTS:
        print("No reddit_snapshot*.json found in data/. Run the browser-console capture first.")
        return
    print("snapshot files:", [f.name for f in SNAPSHOTS])
    groups = []
    for f in SNAPSHOTS:
        groups.extend(json.loads(f.read_text()))
    opps, skipped_sticky = [], 0
    for g in groups:
        domain, sub = g.get("domain"), g.get("subreddit")
        for post in g.get("posts", []):
            if post.get("stickied"):            # skip pinned mod/announcement posts
                skipped_sticky += 1
                continue
            if not post.get("id") or not post.get("title"):
                continue
            opps.append(to_opportunity(post, domain, sub))

    store = Store()
    before = store.count()
    inserted, dup = store.add_many(opps)

    # set the multi-label `domains` column (JSON) for the new rows, mirroring `domain`
    cols = [r[1] for r in store.conn.execute("PRAGMA table_info(opportunities)")]
    if "domains" in cols:
        store.conn.execute(
            "UPDATE opportunities SET domains = '[\"' || domain || '\"]' "
            "WHERE source='reddit' AND (domains IS NULL OR domains='')")
        store.conn.commit()

    print(f"parsed posts: {len(opps)} (skipped stickied: {skipped_sticky})")
    print(f"inserted: {inserted}  |  duplicates skipped: {dup}")
    print(f"DB total: {before} -> {store.count()}")
    print("\nsource counts:")
    for s, n in store.counts_by("source").items():
        print(f"  {s:12} {n}")
    print("\nReddit by domain:")
    for d, n in store.conn.execute(
        "SELECT domain, COUNT(*) FROM opportunities WHERE source='reddit' GROUP BY domain ORDER BY COUNT(*) DESC"):
        print(f"  {d:22} {n}")


if __name__ == "__main__":
    main()
