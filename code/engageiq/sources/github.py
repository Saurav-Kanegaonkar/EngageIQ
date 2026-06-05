"""GitHub adapter (auth) — finds open, beginner-friendly issues per domain.

Uses the Search Issues API filtered to `good first issue` so results map to the
"contribute to open source" use case (Sofia / beginner personas).
"""
from __future__ import annotations

import re
from collections.abc import Iterator

from .. import config
from ..domains import KEYWORDS
from ..schema import Opportunity
from .base import SourceAdapter
from ._util import get_json


class GitHubAdapter(SourceAdapter):
    name = "github"
    requires_auth = True
    API = "https://api.github.com/search/issues"

    def available(self) -> bool:
        return bool(config.GITHUB_TOKEN)

    def fetch(self, domain: str, limit: int = 100) -> Iterator[Opportunity]:
        kw = KEYWORDS.get(domain, domain.replace("_", " "))
        q = f'{kw} label:"good first issue" state:open'
        headers = {
            "Authorization": f"Bearer {config.GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        }
        try:
            data = get_json(self.API, params={"q": q, "sort": "created",
                                              "order": "desc",
                                              "per_page": min(limit, 100)},
                            headers=headers)
        except Exception as e:  # noqa: BLE001
            print(f"  [github] {domain}: {e}")
            return
        for it in data.get("items", []):
            repo = ""
            m = re.search(r"repos/([^/]+/[^/]+)$", it.get("repository_url", ""))
            if m:
                repo = m.group(1)
            yield Opportunity(
                source="github", source_native_id=str(it.get("id")),
                url=it.get("html_url", ""), title=it.get("title") or "",
                body=(it.get("body") or "")[:4000],
                opportunity_type="pull_request" if "pull_request" in it else "issue",
                domain=domain, created_at=it.get("created_at") or "",
                last_activity_at=it.get("updated_at"),
                author=(it.get("user") or {}).get("login"),
                score=(it.get("reactions") or {}).get("total_count"),
                num_comments=it.get("comments"), community=repo,
                tags=[l.get("name") for l in it.get("labels", []) if l.get("name")],
                signals={"state": it.get("state"), "query": q}, raw={},
            )
