"""Hacker News adapter (no auth) via the Algolia HN search API — CURRENT only.

Restricts to 2026+ stories with a numericFilters cutoff on created_at_i. Without
it, the relevance-ranked /search endpoint returns all-time-popular classics
(2007-era), which are useless as live engagement opportunities.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import datetime, timezone

from ..domains import KEYWORDS
from ..schema import Opportunity
from .base import SourceAdapter
from ._util import get_json

# "current" boundary = start of 2026 (epoch seconds)
CURRENT_SINCE = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())


class HackerNewsAdapter(SourceAdapter):
    name = "hackernews"
    requires_auth = False
    API = "https://hn.algolia.com/api/v1/search"

    def fetch(self, domain: str, limit: int = 100) -> Iterator[Opportunity]:
        q = KEYWORDS.get(domain, domain.replace("_", " "))
        fetched = 0
        page = 0
        while fetched < limit:
            try:
                data = get_json(self.API, params={
                    "query": q, "tags": "story", "page": page, "hitsPerPage": 100,
                    "numericFilters": f"created_at_i>={CURRENT_SINCE}",
                })
            except Exception as e:  # noqa: BLE001
                print(f"  [hackernews] {domain} p{page}: {e}")
                return
            hits = data.get("hits", [])
            if not hits:
                break
            for h in hits:
                oid = str(h.get("objectID"))
                url = h.get("url") or f"https://news.ycombinator.com/item?id={oid}"
                yield Opportunity(
                    source="hackernews", source_native_id=oid, url=url,
                    title=h.get("title") or "", body=h.get("story_text") or "",
                    opportunity_type="story", domain=domain,
                    created_at=h.get("created_at") or "",
                    author=h.get("author"), score=h.get("points"),
                    num_comments=h.get("num_comments"), community="hackernews",
                    signals={"query": q}, raw={},
                )
                fetched += 1
                if fetched >= limit:
                    break
            if page >= data.get("nbPages", 1) - 1:
                break
            page += 1
            time.sleep(0.25)
