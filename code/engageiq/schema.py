"""The canonical data contract for EngageIQ.

Every source adapter normalizes its raw API response into an `Opportunity`.
After this point, NOTHING downstream (dedup, embedding, scoring, ranking, UI)
ever touches a raw API payload — it only reads these fields. This is the
"library catalog card" every record is filled into, regardless of source.

🟢 PERMANENT — the core fields are the contract the whole pipeline depends on.
🟡 PROVISIONAL — `signals` is an open dict: add source-specific extras freely
   (forks, upvote_ratio, repostCount, reactions, ...) without breaking anything.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Allowed values (documented, not enforced — kept permissive on purpose).
SOURCES = ("github", "gharchive", "hackernews", "reddit", "bluesky", "devto")
TYPES = (
    "issue", "pull_request", "repo", "discussion",
    "post", "story", "comment_thread", "article",
)


@dataclass
class Opportunity:
    # ── identity ──────────────────────────────────────────────
    source: str                 # one of SOURCES
    source_native_id: str       # the id within that source
    url: str
    title: str
    body: str                   # may be "" (e.g. a Bluesky post has no title)
    opportunity_type: str       # one of TYPES
    domain: str                 # primary technical domain (one of the 15)
    created_at: str             # ISO-8601 UTC

    # ── enrichment (adapters fill what they have) ─────────────
    tags: list[str] = field(default_factory=list)       # labels / flair / hashtags
    language: str | None = None                         # programming language if any
    author: str | None = None
    author_reputation: float | None = None              # karma / followers / ...
    community: str | None = None                        # repo full-name, subreddit, feed
    community_size: int | None = None                   # stars / subscribers / ...

    # ── engagement signals (the two ~universal ones) ──────────
    score: int | None = None        # primary popularity count: stars|points|upvotes|likes|reactions
    num_comments: int | None = None

    # ── temporal / provenance ─────────────────────────────────
    last_activity_at: str | None = None
    signals: dict = field(default_factory=dict)         # 🟡 source-specific extras, verbatim
    raw: dict = field(default_factory=dict)             # original payload, for re-processing
    fetched_at: str = field(default_factory=_now_iso)

    @property
    def opportunity_id(self) -> str:
        """Stable, source-scoped id — the dedup key (Bloom filter / DB primary key)."""
        return hashlib.sha1(f"{self.source}:{self.source_native_id}".encode()).hexdigest()

    @property
    def text(self) -> str:
        """Title + body, the text we embed downstream (Phase 2)."""
        return (self.title + "\n\n" + self.body).strip()

    def to_row(self) -> dict:
        """Flatten to a DB-insertable row (list/dict fields JSON-encoded)."""
        d = asdict(self)
        d["opportunity_id"] = self.opportunity_id
        d["tags"] = json.dumps(d["tags"])
        d["signals"] = json.dumps(d["signals"])
        d["raw"] = json.dumps(d["raw"])
        return d
