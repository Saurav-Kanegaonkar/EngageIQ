"""Dev.to adapter (no auth) — articles per domain tag, paginated."""
from __future__ import annotations

import time
from collections.abc import Iterator

from ..domains import DEVTO_TAGS
from ..schema import Opportunity
from .base import SourceAdapter
from ._util import get_json


class DevToAdapter(SourceAdapter):
    name = "devto"
    requires_auth = False
    API = "https://dev.to/api/articles"

    def fetch(self, domain: str, limit: int = 100) -> Iterator[Opportunity]:
        tags = DEVTO_TAGS.get(domain, [])
        tag = tags[0] if tags else ""
        fetched = 0
        page = 1
        while fetched < limit:
            try:
                data = get_json(self.API, params={"tag": tag, "per_page": 100, "page": page})
            except Exception as e:  # noqa: BLE001
                print(f"  [devto] {domain} p{page}: {e}")
                return
            if not data:
                break
            for a in data:
                yield Opportunity(
                    source="devto", source_native_id=str(a.get("id")),
                    url=a.get("url", ""), title=a.get("title") or "",
                    body=a.get("description") or "",
                    opportunity_type="article", domain=domain,
                    created_at=a.get("published_at") or "",
                    author=(a.get("user") or {}).get("username"),
                    score=a.get("public_reactions_count"),
                    num_comments=a.get("comments_count"), community="dev.to",
                    tags=a.get("tag_list") or [],
                    signals={"reading_time_min": a.get("reading_time_minutes"),
                             "tag": tag}, raw={},
                )
                fetched += 1
                if fetched >= limit:
                    break
            if len(data) < 100:
                break
            page += 1
            time.sleep(0.25)
