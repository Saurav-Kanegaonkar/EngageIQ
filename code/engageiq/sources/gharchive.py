"""GH Archive adapter — most ACTIVELY-DEVELOPED notable repos (no auth for archive).

NOTE (2026): GitHub removed star events (WatchEvent) from the public firehose, so
"trending by stars" is no longer possible from GH Archive (a typical hour has
~137k PushEvents but ~150 WatchEvents). Instead we measure *development activity*
velocity — weighted push / PR / issue-comment events per repo over recent hours —
then enrich the top repos via the GitHub API and keep only those with real
traction (stars >= floor) to filter out bots/mirrors.

This is GH Archive's surviving strength: rate-of-change/activity in bulk, which the
REST API can't give. Current-by-construction (recent hours) -> fits the 2026 rule.
"""
from __future__ import annotations

import gzip
import json
import time
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

from .. import config
from ..domains import DEVTO_TAGS, DOMAINS, KEYWORDS
from ..schema import Opportunity
from .base import SourceAdapter
from ._util import _session, get_json

# weight human-collaboration events above raw pushes (which are bot-heavy)
ACTIVITY_WEIGHTS = {
    "PushEvent": 1, "PullRequestEvent": 3, "IssuesEvent": 3,
    "IssueCommentEvent": 2, "PullRequestReviewCommentEvent": 2,
}
_LANG_HINTS = {
    "gamedev_cpp": ["c++", "cmake"], "embedded_systems": ["rust", "zig"],
    "frontend_web": ["javascript", "typescript", "vue", "svelte"],
    "python_data_eng": ["python", "jupyter notebook"],
    "mobile_dev": ["swift", "kotlin", "dart"], "blockchain": ["solidity"],
}


def _match_terms() -> dict[str, list[str]]:
    terms: dict[str, list[str]] = {}
    for d in DOMAINS:
        ts = {d.replace("_", " ")}
        if KEYWORDS.get(d):
            ts.add(KEYWORDS[d].lower())
        for t in DEVTO_TAGS.get(d, []):
            ts.add(t.lower())
        for t in _LANG_HINTS.get(d, []):
            ts.add(t)
        terms[d] = [t for t in ts if len(t) > 2]  # drop ultra-short tokens (false positives)
    return terms


class GHArchiveAdapter(SourceAdapter):
    name = "gharchive"
    requires_auth = False
    BASE = "https://data.gharchive.org"

    def __init__(self, hours: int = 6, top_n: int = 400, star_floor: int = 50):
        self.hours = hours
        self.top_n = top_n
        self.star_floor = star_floor
        self._buckets: dict[str, list[Opportunity]] | None = None
        self._terms = _match_terms()

    def available(self) -> bool:
        return bool(config.GITHUB_TOKEN)  # enrichment needs the GitHub API

    def _hour_urls(self):
        start = datetime.now(timezone.utc) - timedelta(hours=2)  # ~2h publish lag
        for i in range(self.hours):
            t = start - timedelta(hours=i)
            yield f"{self.BASE}/{t.year}-{t.month:02d}-{t.day:02d}-{t.hour}.json.gz"

    def _domain_for(self, text: str) -> str | None:
        text = text.lower()
        for d in DOMAINS:
            for term in self._terms[d]:
                if term in text:
                    return d
        return None

    def _activity_counts(self) -> Counter:
        activity: Counter = Counter()
        for url in self._hour_urls():
            try:
                r = _session.get(url, timeout=120)
                if r.status_code != 200:
                    print(f"  [gharchive] {url} -> HTTP {r.status_code}")
                    continue
                try:
                    raw = gzip.decompress(r.content)
                except (OSError, EOFError):
                    raw = r.content
            except Exception as e:  # noqa: BLE001
                print(f"  [gharchive] {url}: {e}")
                continue
            for line in raw.splitlines():
                try:
                    ev = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                w = ACTIVITY_WEIGHTS.get(ev.get("type"))
                if not w:
                    continue
                repo = (ev.get("repo") or {}).get("name")
                if repo:
                    activity[repo] += w
            print(f"  [gharchive] {url.rsplit('/', 1)[-1]}: {sum(activity.values())} cumulative activity pts")
        return activity

    def _enrich(self, repo: str, activity: int) -> Opportunity | None:
        try:
            d = get_json(
                f"https://api.github.com/repos/{repo}",
                headers={"Authorization": f"Bearer {config.GITHUB_TOKEN}",
                         "Accept": "application/vnd.github+json"},
            )
        except Exception:  # noqa: BLE001 - deleted/renamed/private
            return None
        stars = d.get("stargazers_count") or 0
        if stars < self.star_floor:          # filter out bots / no-traction repos
            return None
        topics = [t.lower() for t in (d.get("topics") or [])]
        lang = d.get("language") or ""
        desc = d.get("description") or ""
        domain = self._domain_for(f"{repo} {lang} {' '.join(topics)} {desc}") or "open_source_trending"
        return Opportunity(
            source="gharchive", source_native_id=f"repo:{repo}",
            url=d.get("html_url", f"https://github.com/{repo}"),
            title=repo, body=desc, opportunity_type="repo", domain=domain,
            created_at=d.get("pushed_at") or d.get("updated_at") or "",
            author=(d.get("owner") or {}).get("login"),
            community=repo, community_size=stars, language=d.get("language"),
            score=activity,                   # activity velocity = headline signal
            num_comments=d.get("open_issues_count"), tags=topics,
            signals={"activity_in_window": activity, "window_hours": self.hours,
                     "total_stars": stars, "forks": d.get("forks_count")}, raw={},
        )

    def _ensure(self) -> None:
        if self._buckets is not None:
            return
        self._buckets = {d: [] for d in DOMAINS}
        activity = self._activity_counts()
        top = activity.most_common(self.top_n)
        print(f"  [gharchive] {len(activity)} active repos; enriching top {len(top)} (star floor {self.star_floor})...")
        kept = 0
        for repo, act in top:
            opp = self._enrich(repo, act)
            if opp:
                self._buckets[opp.domain].append(opp)
                kept += 1
            time.sleep(0.05)
        print(f"  [gharchive] kept {kept} notable active repos")

    def fetch(self, domain: str, limit: int = 100) -> Iterator[Opportunity]:
        self._ensure()
        assert self._buckets is not None
        for opp in self._buckets.get(domain, [])[:limit]:
            yield opp
