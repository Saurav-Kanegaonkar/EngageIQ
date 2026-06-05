"""Reddit adapter — read-only PRAW.

Needs only client_id + client_secret + user_agent from a registered "script"
app (NO username/password). Skips gracefully until those land in `.env`, so the
rest of the pipeline never blocks on Reddit access.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

from .. import config
from ..domains import SUBREDDITS
from ..schema import Opportunity
from .base import SourceAdapter


class RedditAdapter(SourceAdapter):
    name = "reddit"
    requires_auth = True

    def __init__(self):
        self._reddit = None

    def available(self) -> bool:
        return bool(config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET)

    def _client(self):
        if self._reddit is None:
            import praw  # lazy: only imported when we actually have credentials
            self._reddit = praw.Reddit(
                client_id=config.REDDIT_CLIENT_ID,
                client_secret=config.REDDIT_CLIENT_SECRET,
                user_agent=config.REDDIT_USER_AGENT,
            )
            self._reddit.read_only = True
        return self._reddit

    def fetch(self, domain: str, limit: int = 100) -> Iterator[Opportunity]:
        subs = SUBREDDITS.get(domain, [])
        if not subs:
            return
        try:
            reddit = self._client()
        except Exception as e:  # noqa: BLE001
            print(f"  [reddit] init: {e}")
            return
        per = max(1, limit // len(subs))
        for sub in subs:
            try:
                for post in reddit.subreddit(sub).hot(limit=per):
                    created = datetime.fromtimestamp(
                        getattr(post, "created_utc", 0) or 0, tz=timezone.utc
                    ).isoformat()
                    yield Opportunity(
                        source="reddit", source_native_id=str(post.id),
                        url=f"https://www.reddit.com{post.permalink}",
                        title=post.title or "", body=(post.selftext or "")[:4000],
                        opportunity_type="post", domain=domain, created_at=created,
                        author=str(post.author) if post.author else None,
                        score=int(getattr(post, "score", 0) or 0),
                        num_comments=int(getattr(post, "num_comments", 0) or 0),
                        community=f"r/{sub}",
                        tags=[post.link_flair_text] if getattr(post, "link_flair_text", None) else [],
                        signals={"upvote_ratio": getattr(post, "upvote_ratio", None),
                                 "subreddit": sub}, raw={},
                    )
            except Exception as e:  # noqa: BLE001
                print(f"  [reddit] r/{sub}: {e}")
