"""LLM card summaries: a concrete 3-4 sentence "what this is and what you'd work on"
per opportunity, so the feed DESCRIBES the task before the user opens it (instead of a
raw body truncation). Cached to data/summaries.json (keyed by opportunity id) so it is
instant on reload and works offline for graders; the clean snippet is the fallback when
the LLM is unavailable or an item has not been pre-warmed.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from . import llm

_DATA = Path(__file__).resolve().parent.parent.parent / "data"
CACHE_PATH = _DATA / "summaries.json"
SECTIONS_PATH = _DATA / "summary_sections.json"     # {oid: {what, task, do, gain}}
_LOCK = threading.Lock()

_SRC_NOUN = {"github": "GitHub issue", "hackernews": "Hacker News post", "reddit": "Reddit thread",
             "devto": "Dev.to article", "bluesky": "Bluesky post"}
_SRC_VERB = {"github": "what a contributor would build or fix",
             "hackernews": "what the news/discussion is about and why it matters",
             "reddit": "what the thread is discussing and what you could add",
             "devto": "what the article teaches and what you'd take away",
             "bluesky": "what the post is saying and what you could reply"}


class Summarizer:
    """File-backed cache of LLM summaries. The API only READS the cache (fast); a
    separate prewarm pass writes it."""

    def __init__(self) -> None:
        _DATA.mkdir(exist_ok=True)
        self.cache: dict[str, str] = self._load(CACHE_PATH)
        self.sections_cache: dict[str, dict] = self._load(SECTIONS_PATH)   # labeled what/task/do/gain

    def _load(self, path: Path = CACHE_PATH) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:  # noqa: BLE001
                return {}
        return {}

    def sections(self, oid: str) -> dict | None:
        """The labeled {what, task, do, gain} for a card, or None (then fall back
        to the paragraph summary)."""
        s = self.sections_cache.get(oid)
        return s if isinstance(s, dict) else None

    def _save(self) -> None:
        try:
            CACHE_PATH.write_text(json.dumps(self.cache))
        except Exception:  # noqa: BLE001
            pass

    def cached(self, oid: str) -> str | None:
        return self.cache.get(oid)

    def generate(self, oid: str, title: str, body: str, source: str, save: bool = True) -> str | None:
        if oid in self.cache:
            return self.cache[oid]
        if not llm.available():
            return None
        noun = _SRC_NOUN.get(source, "opportunity")
        focus = _SRC_VERB.get(source, "what it is about and what you'd do")
        prompt = (
            f"Write a clear 3-4 sentence summary of this {noun} a developer can understand at a glance. "
            f"Cover, in order: what the project/thread is about; the specific task or question raised; "
            f"what the person would concretely do ({focus}); and the scope/level and what they'd gain. "
            f"Be specific and concrete; never just restate the title. Plain text, no markdown, no preamble.\n\n"
            f"Title: {title}\n\nContent:\n{(body or '')[:1600]}")
        out = llm.chat(
            [{"role": "system", "content": "You write tight, concrete summaries for a busy developer."},
             {"role": "user", "content": prompt}],
            model=llm.FAST_MODEL, max_tokens=170, temperature=0.3)
        if out:
            out = " ".join(out.split())[:650]
            with _LOCK:
                self.cache[oid] = out
                if save:
                    self._save()
        return out

    def prewarm(self, items: list[tuple], log_every: int = 20, pace: float = 0.8) -> int:
        """items: list of (oid, title, body, source). Generate + cache the missing ones,
        paced + with one backoff-retry to ride out the free tier's rate limit (429)."""
        n = 0
        for oid, title, body, source in items:
            if oid in self.cache:
                continue
            s = self.generate(oid, title, body, source, save=False)
            if not s:
                time.sleep(6)                                   # rate-limit backoff
                s = self.generate(oid, title, body, source, save=False)
            if s:
                n += 1
                if n % log_every == 0:
                    with _LOCK:
                        self._save()
                    print(f"  summarized {n} new (cache {len(self.cache)})")
            time.sleep(pace)
        with _LOCK:
            self._save()
        return n
