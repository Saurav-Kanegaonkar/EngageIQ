"""SQLite storage + Bloom-filter deduplication (Capability 1).

🟢 PERMANENT — storage interface + the dedup contract.
Dedup is two-layer:
  1. A Bloom filter (L2, from scratch) = fast in-memory "definitely-new vs maybe-seen".
  2. The DB PRIMARY KEY on opportunity_id = the certain backstop.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
from pathlib import Path

from .schema import Opportunity

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "engageiq.sqlite"

_INT_COLS = {"score", "num_comments", "community_size"}
_COLS = [
    "opportunity_id", "source", "source_native_id", "url", "title", "body",
    "opportunity_type", "domain", "created_at", "tags", "language", "author",
    "author_reputation", "community", "community_size", "score", "num_comments",
    "last_activity_at", "signals", "raw", "fetched_at",
]


class BloomFilter:
    """Probabilistic set membership (BAX-423 Lecture 2).

    No false negatives. ~`error` false-positive rate at `capacity` items.
    Sized via m = -n·ln(p)/(ln2)^2 and k = (m/n)·ln2.
    """

    def __init__(self, capacity: int = 500_000, error: float = 0.01):
        self.m = max(8, int(-(capacity * math.log(error)) / (math.log(2) ** 2)))
        self.k = max(1, round((self.m / capacity) * math.log(2)))
        self.bits = bytearray((self.m + 7) // 8)

    def _indexes(self, item: str):
        # double hashing: g_i(x) = (h1 + i*h2) mod m  (Kirsch–Mitzenmacher)
        digest = hashlib.sha256(item.encode()).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:16], "big") | 1  # force odd -> coprime-ish
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, item: str) -> None:
        for b in self._indexes(item):
            self.bits[b >> 3] |= 1 << (b & 7)

    def __contains__(self, item: str) -> bool:
        return all(self.bits[b >> 3] & (1 << (b & 7)) for b in self._indexes(item))


class Store:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self._create()
        self.bloom = BloomFilter()
        for (oid,) in self.conn.execute("SELECT opportunity_id FROM opportunities"):
            self.bloom.add(oid)

    def _create(self) -> None:
        cols = ", ".join(
            f"{c} INTEGER" if c in _INT_COLS else f"{c} TEXT" for c in _COLS
        )
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS opportunities ({cols}, PRIMARY KEY(opportunity_id))"
        )
        self.conn.commit()

    def add(self, opp: Opportunity) -> bool:
        """Insert if new. Returns True if inserted, False if a duplicate."""
        oid = opp.opportunity_id
        if oid in self.bloom:  # maybe-seen -> confirm against the DB
            if self.conn.execute(
                "SELECT 1 FROM opportunities WHERE opportunity_id=?", (oid,)
            ).fetchone():
                return False
        row = opp.to_row()
        self.conn.execute(
            f"INSERT OR IGNORE INTO opportunities ({','.join(_COLS)}) "
            f"VALUES ({','.join('?' for _ in _COLS)})",
            [row[c] for c in _COLS],
        )
        self.bloom.add(oid)
        return True

    def add_many(self, opps) -> tuple[int, int]:
        inserted = skipped = 0
        for o in opps:
            if self.add(o):
                inserted += 1
            else:
                skipped += 1
        self.conn.commit()
        return inserted, skipped

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0]

    def counts_by(self, field: str) -> dict:
        return dict(
            self.conn.execute(
                f"SELECT {field}, COUNT(*) FROM opportunities GROUP BY {field} "
                f"ORDER BY COUNT(*) DESC"
            )
        )
