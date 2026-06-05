"""Bluesky adapter — authenticated search via the AT Protocol.

The public searchPosts endpoint now 403s, so this logs in with an APP PASSWORD
(`com.atproto.server.createSession`) and calls `app.bsky.feed.searchPosts` with
the bearer token, paginated via cursor. Needs BLUESKY_HANDLE +
BLUESKY_APP_PASSWORD in `.env`; skips gracefully otherwise.
"""
from __future__ import annotations

import time
from collections.abc import Iterator

from .. import config
from ..domains import KEYWORDS
from ..schema import Opportunity
from .base import SourceAdapter
from ._util import get_json, post_json


class BlueskyAdapter(SourceAdapter):
    name = "bluesky"
    requires_auth = True
    HOST = "https://bsky.social"

    def __init__(self):
        self._jwt = None

    def available(self) -> bool:
        return bool(config.BLUESKY_HANDLE and config.BLUESKY_APP_PASSWORD)

    def _token(self) -> str:
        if self._jwt is None:
            data = post_json(
                f"{self.HOST}/xrpc/com.atproto.server.createSession",
                json={"identifier": config.BLUESKY_HANDLE,
                      "password": config.BLUESKY_APP_PASSWORD},
            )
            self._jwt = data["accessJwt"]
        return self._jwt

    def fetch(self, domain: str, limit: int = 100) -> Iterator[Opportunity]:
        q = KEYWORDS.get(domain, domain.replace("_", " "))
        try:
            headers = {"Authorization": f"Bearer {self._token()}"}
        except Exception as e:  # noqa: BLE001
            print(f"  [bluesky] auth: {e}")
            return
        fetched = 0
        cursor = None
        while fetched < limit:
            params = {"q": q, "limit": min(100, limit - fetched)}
            if cursor:
                params["cursor"] = cursor
            try:
                data = get_json(f"{self.HOST}/xrpc/app.bsky.feed.searchPosts",
                                params=params, headers=headers)
            except Exception as e:  # noqa: BLE001
                print(f"  [bluesky] {domain}: {e}")
                return
            posts = data.get("posts", [])
            if not posts:
                break
            for p in posts:
                uri = p.get("uri", "")
                rkey = uri.split("/")[-1]
                handle = (p.get("author") or {}).get("handle", "")
                rec = p.get("record") or {}
                yield Opportunity(
                    source="bluesky", source_native_id=uri,
                    url=f"https://bsky.app/profile/{handle}/post/{rkey}" if handle else uri,
                    title="", body=rec.get("text") or "",
                    opportunity_type="post", domain=domain,
                    created_at=rec.get("createdAt") or "",
                    author=handle, score=p.get("likeCount"),
                    num_comments=p.get("replyCount"), community="bluesky",
                    signals={"repostCount": p.get("repostCount"),
                             "quoteCount": p.get("quoteCount")}, raw={},
                )
                fetched += 1
                if fetched >= limit:
                    break
            cursor = data.get("cursor")
            if not cursor:
                break
            time.sleep(0.25)
