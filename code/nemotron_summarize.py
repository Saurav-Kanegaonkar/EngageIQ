"""Summarize the remaining corpus with NVIDIA Nemotron (free API, ZERO Claude tokens).

Resumable + incremental: skips items already in the cache, flushes every FLUSH_EVERY
items with an ATOMIC write, so a stop / crash / rate-limit never loses or corrupts
progress. Self-throttles via backoff when the free tier rate-limits. Runs in background.

Run: PYTHONPATH=code .venv/bin/python -m nemotron_summarize
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from engageiq import llm, summarize
from engageiq.embed import DB_PATH

WORKERS = 5            # concurrent NVIDIA calls (backoff handles rate limits)
FLUSH_EVERY = 40       # write caches to disk every N completed items
MODEL = llm.SMART_MODEL  # nvidia/llama-3.3-nemotron-super-49b-v1
_KEYS = ("what", "task", "do", "gain")
_lock = threading.Lock()


def _clean(s: str, cap: int) -> str:
    s = re.sub(r"[*#`_]+", "", s or "")
    s = s.replace(" — ", ", ").replace("—", ", ").replace(" – ", ", ").replace("–", "-")
    return " ".join(s.split())[:cap]


def _atomic_write(path, obj) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(obj))
    os.replace(tmp, path)


def _gen(title: str, body: str):
    prompt = ("Write a card summary for a developer engagement product. Return STRICT JSON with keys "
              "summary, what, task, do, gain (each a short plain sentence; summary 2 to 3 sentences). "
              "No markdown, no em-dashes, grounded only in the text.\n\n"
              f"Title: {title}\nContent: {body[:1200]}\n\nReturn ONLY the JSON.")
    for attempt in range(4):
        txt = llm.chat([{"role": "user", "content": prompt}], model=MODEL,
                       max_tokens=400, temperature=0.2, timeout=60)
        if txt:
            m = re.search(r"\{.*\}", txt, re.S)
            if m:
                try:
                    d = json.loads(m.group(0))
                    summ = _clean(d.get("summary", ""), 650)
                    sec = {k: _clean(d.get(k, ""), 240) for k in _KEYS}
                    if summ or any(sec.values()):
                        return summ, sec
                except Exception:  # noqa: BLE001
                    pass
        time.sleep(3 * (attempt + 1))   # backoff: also throttles when rate-limited
    return None, None


def main() -> None:
    summ_cache = json.loads(summarize.CACHE_PATH.read_text()) if summarize.CACHE_PATH.exists() else {}
    sec_cache = json.loads(summarize.SECTIONS_PATH.read_text()) if summarize.SECTIONS_PATH.exists() else {}
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("SELECT opportunity_id, title, body FROM opportunities").fetchall()
    conn.close()
    todo = [(oid, (t or "").strip(), b or "") for (oid, t, b) in rows if oid not in sec_cache]
    total = len(todo)
    print(f"[nemo] start: {total} to summarize | {len(sec_cache)} already cached | model={MODEL}", flush=True)
    if not total:
        print("[nemo] nothing to do.", flush=True)
        return

    done = fail = 0

    def work(item):
        oid, title, body = item
        s, sec = _gen(title, body)
        return oid, s, sec

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for oid, s, sec in (f.result() for f in as_completed([ex.submit(work, it) for it in todo])):
            with _lock:
                if sec is not None:
                    if s:
                        summ_cache[oid] = s
                    sec_cache[oid] = sec
                    done += 1
                else:
                    fail += 1
                n = done + fail
                if n % FLUSH_EVERY == 0:
                    _atomic_write(summarize.CACHE_PATH, summ_cache)
                    _atomic_write(summarize.SECTIONS_PATH, sec_cache)
                    rate = n / max(1e-9, (time.time() - t0)) * 60
                    print(f"[nemo] {n}/{total}  done={done} fail={fail}  cached={len(sec_cache)}  "
                          f"~{rate:.0f}/min", flush=True)

    with _lock:
        _atomic_write(summarize.CACHE_PATH, summ_cache)
        _atomic_write(summarize.SECTIONS_PATH, sec_cache)
    print(f"[nemo] FINISHED: done={done} fail={fail} total_cached={len(sec_cache)}", flush=True)


if __name__ == "__main__":
    main()
