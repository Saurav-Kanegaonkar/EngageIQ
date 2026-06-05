"""Profile the captured Reddit data: what fields/signals do we actually have?

Reads the raw browser-capture snapshots (full Reddit post objects, much richer
than what we stored), dedupes by post id, and reports distributions for the
signals that matter to scoring: conversation size (num_comments), score,
upvote_ratio, post length, recency, self-vs-link, and community size
(subreddit_subscribers). Also flags 'broad' subreddits whose posts may not all
match the single subreddit->domain label (the thing to verify with the LLM).
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
NOW = datetime.now(timezone.utc)


def dist(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    s = sorted(vals)
    p = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return f"min={s[0]:.0f} p50={st.median(s):.0f} p90={p(0.9):.0f} max={s[-1]:.0f} mean={st.mean(s):.1f}"


def main() -> None:
    posts = {}
    for f in sorted(DATA.glob("reddit_snapshot*.json")):
        for g in json.loads(f.read_text()):
            dom, sub = g.get("domain"), g.get("subreddit")
            for p in g.get("posts", []):
                pid = p.get("id")
                if pid:
                    posts[pid] = (dom, sub, p)   # dedupe by id (keep last)
    print(f"unique Reddit posts across snapshots: {len(posts)}\n")

    # which fields are present?
    sample = next(iter(posts.values()))[2]
    interesting = ["num_comments", "score", "upvote_ratio", "selftext", "is_self",
                   "created_utc", "subreddit_subscribers", "total_awards_received",
                   "num_crossposts", "over_18", "link_flair_text", "author",
                   "view_count", "post_hint", "domain"]
    print("-- field availability (of", len(posts), "posts) --")
    for k in interesting:
        present = sum(1 for _, _, p in posts.values() if p.get(k) not in (None, ""))
        print(f"   {k:24} {100*present/len(posts):5.1f}%")

    ncom = [p.get("num_comments") for _, _, p in posts.values()]
    score = [p.get("score") for _, _, p in posts.values()]
    upv = [p.get("upvote_ratio") for _, _, p in posts.values()]
    slen = [len(p.get("selftext") or "") for _, _, p in posts.values()]
    subs = [p.get("subreddit_subscribers") for _, _, p in posts.values()]
    ages = []
    for _, _, p in posts.values():
        if p.get("created_utc"):
            ages.append((NOW - datetime.fromtimestamp(p["created_utc"], tz=timezone.utc)).days)

    print("\n-- conversation size  num_comments --", dist(ncom))
    print("   threads with >10 comments:", f"{100*sum(1 for x in ncom if x and x>10)/len(ncom):.0f}%",
          "| >50:", f"{100*sum(1 for x in ncom if x and x>50)/len(ncom):.0f}%")
    print("-- score (upvotes)        --", dist(score))
    print("-- upvote_ratio           --", dist(upv))
    print("-- post length (chars)    --", dist(slen))
    self_posts = sum(1 for _, _, p in posts.values() if p.get("is_self"))
    print(f"-- self (text) posts: {100*self_posts/len(posts):.0f}%  |  link posts: {100*(1-self_posts/len(posts)):.0f}%")
    print("-- community size  subreddit_subscribers --", dist(subs))
    if ages:
        print(f"-- recency (days old)     -- min={min(ages)} p50={int(st.median(ages))} max={max(ages)}")
        yrs = defaultdict(int)
        for _, _, p in posts.values():
            if p.get("created_utc"):
                yrs[datetime.fromtimestamp(p["created_utc"], tz=timezone.utc).year] += 1
        print("   posts by year:", dict(sorted(yrs.items())))

    # per-subreddit: count + median engagement (broad subs flagged)
    BROAD = {"programming", "Python", "artificial", "startups", "opensource",
             "learnprogramming", "commandline", "webdev"}
    by_sub = defaultdict(list)
    for dom, sub, p in posts.values():
        by_sub[(dom, sub)].append(p)
    print("\n-- per subreddit (domain | sub | posts | median comments | subscribers | BROAD?) --")
    for (dom, sub), ps in sorted(by_sub.items()):
        med_c = st.median([x.get("num_comments") or 0 for x in ps])
        subc = next((x.get("subreddit_subscribers") for x in ps if x.get("subreddit_subscribers")), None)
        flag = "  <-- broad, verify" if sub in BROAD else ""
        print(f"   {dom:20} r/{sub:20} {len(ps):4}  comts~{med_c:<4.0f} subs={subc}{flag}")


if __name__ == "__main__":
    main()
