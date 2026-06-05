"""Enrich GitHub records with repo-level signals the scoring engine needs.

WHY: the Phase-1 scrape stored issues but not their repo's stars, language, or
contributor count. Phase-3 scoring + the persona pass-criteria depend on these:
  - Sofia  -> `language` (must exclude C++/Rust repos)
  - David  -> contributors (few contributors + high activity = room to stand out)
  - Lina   -> stars + star_velocity (fast-growing repos)
  - all    -> stars = community_size = a real visibility/health signal

Re-fetches each unique repo once via GitHub REST `/repos/{owner}/{name}`, plus a
1-call contributor-count trick (per_page=1 -> Link rel=last page == count).
Results are cached to data/github_repo_enrich.json so this is idempotent and a
grader never has to hit the API. Writes into `community_size`, `language`, and
merges extras into the per-row `signals` JSON.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engageiq import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "engageiq.sqlite"
CACHE = ROOT / "data" / "github_repo_enrich.json"

API = "https://api.github.com"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "EngageIQ/0.1 (BAX-423 UC Davis project)",
    "X-GitHub-Api-Version": "2022-11-28",
}
if config.GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"

_session = requests.Session()
_session.headers.update(HEADERS)


def _get(url: str, params: dict | None = None):
    """GET with rate-limit awareness. Returns (response | None)."""
    for attempt in range(4):
        try:
            r = _session.get(url, params=params, timeout=15)
        except Exception as e:  # noqa: BLE001
            print(f"  net error {url}: {e}; retry")
            time.sleep(2)
            continue
        rem = r.headers.get("X-RateLimit-Remaining")
        if r.status_code == 403 and rem == "0":
            reset = int(r.headers.get("X-RateLimit-Reset", "0"))
            wait = max(5, reset - int(time.time()) + 2)
            print(f"  rate-limited; sleeping {wait}s")
            time.sleep(wait)
            continue
        return r
    return None


def _age_days(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400)
    except Exception:  # noqa: BLE001
        return None


def fetch_repo(full_name: str) -> dict:
    """Return the enrichment dict for one repo (nulls on failure/404)."""
    out: dict = {"stars": None, "language": None, "forks": None, "watchers": None,
                 "open_issues": None, "contributors": None, "repo_pushed_at": None,
                 "repo_created_at": None, "repo_age_days": None, "star_velocity": None,
                 "archived": None, "status": None}
    r = _get(f"{API}/repos/{full_name}")
    if r is None or r.status_code != 200:
        out["status"] = r.status_code if r is not None else "neterr"
        return out
    d = r.json()
    out.update(
        stars=d.get("stargazers_count"), language=d.get("language"),
        forks=d.get("forks_count"), watchers=d.get("subscribers_count"),
        open_issues=d.get("open_issues_count"), repo_pushed_at=d.get("pushed_at"),
        repo_created_at=d.get("created_at"), archived=d.get("archived"), status=200,
    )
    age = _age_days(d.get("created_at"))
    out["repo_age_days"] = round(age, 1) if age else None
    if out["stars"] is not None and age and age > 0:
        out["star_velocity"] = round(out["stars"] / age, 3)  # stars/day since creation

    # contributor count via Link-header pagination trick (1 row/page -> last page == count)
    rc = _get(f"{API}/repos/{full_name}/contributors", params={"per_page": 1, "anon": "true"})
    if rc is not None and rc.status_code == 200:
        link = rc.headers.get("Link", "")
        m = re.search(r'[?&]page=(\d+)>;\s*rel="last"', link)
        if m:
            out["contributors"] = int(m.group(1))
        else:
            try:
                out["contributors"] = len(rc.json())
            except Exception:  # noqa: BLE001
                pass
    return out


def main() -> None:
    if not config.GITHUB_TOKEN:
        print("No GITHUB_TOKEN in .env — cannot enrich. Aborting.")
        return

    conn = sqlite3.connect(str(DB))
    repos = [r[0] for r in conn.execute(
        "SELECT DISTINCT community FROM opportunities WHERE source='github' AND community<>''")]
    print(f"unique GitHub repos to enrich: {len(repos)}")

    cache: dict = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    print(f"already cached: {len(cache)}")

    todo = [r for r in repos if r not in cache]
    for i, full in enumerate(todo, 1):
        cache[full] = fetch_repo(full)
        if i % 25 == 0 or i == len(todo):
            CACHE.write_text(json.dumps(cache, indent=0))
            ok = sum(1 for v in cache.values() if v.get("status") == 200)
            print(f"  {i}/{len(todo)}  (ok={ok})")
        time.sleep(0.03)
    CACHE.write_text(json.dumps(cache, indent=0))

    # write into DB
    n_stars = n_lang = 0
    for full, e in cache.items():
        rows = conn.execute(
            "SELECT rowid, signals FROM opportunities WHERE source='github' AND community=?",
            (full,)).fetchall()
        for rowid, sig_json in rows:
            try:
                sig = json.loads(sig_json) if sig_json else {}
            except Exception:  # noqa: BLE001
                sig = {}
            sig.update({k: e.get(k) for k in (
                "forks", "watchers", "open_issues", "contributors",
                "repo_pushed_at", "repo_created_at", "repo_age_days",
                "star_velocity", "archived")})
            stars = e.get("stars")
            lang = e.get("language")
            conn.execute(
                "UPDATE opportunities SET community_size=?, language=COALESCE(?,language), signals=? WHERE rowid=?",
                (stars, lang, json.dumps(sig), rowid))
            if stars is not None:
                n_stars += 1
            if lang:
                n_lang += 1
    conn.commit()

    ok = sum(1 for v in cache.values() if v.get("status") == 200)
    failed = [k for k, v in cache.items() if v.get("status") != 200]
    print(f"\nenriched repos: ok={ok}/{len(cache)}  failed={len(failed)}")
    if failed:
        print("  failed (deleted/renamed/private):", failed[:10], "..." if len(failed) > 10 else "")
    print(f"rows updated: community_size set={n_stars}, language set={n_lang}")

    # quick distributions
    print("\n-- post-enrichment GitHub signal coverage --")
    langs = conn.execute(
        "SELECT language, COUNT(*) FROM opportunities WHERE source='github' AND language IS NOT NULL "
        "GROUP BY language ORDER BY COUNT(*) DESC LIMIT 12").fetchall()
    print("top languages:", dict(langs))
    star_rows = [r[0] for r in conn.execute(
        "SELECT community_size FROM opportunities WHERE source='github' AND community_size IS NOT NULL")]
    if star_rows:
        star_rows.sort()
        print(f"stars: min={star_rows[0]} p50={star_rows[len(star_rows)//2]} "
              f"p90={star_rows[int(0.9*len(star_rows))]} max={star_rows[-1]}")


if __name__ == "__main__":
    main()
