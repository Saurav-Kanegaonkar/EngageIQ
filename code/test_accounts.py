"""Tests for the name-based accounts + DB-backed profiles. Run directly:

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=code .venv/bin/python code/test_accounts.py

Store-level tests use a throwaway DB; endpoint tests run if the API is up on :8000.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from engageiq.store import EngageStore

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def test_store() -> None:
    print("Store: accounts + profiles (throwaway DB)")
    p = Path("/tmp/eiq_accounts_test.sqlite")
    if p.exists():
        p.unlink()
    s = EngageStore(path=p)

    a = s.find_or_create_user("Saurav")
    check("new account is_new", a["is_new"] is True)
    b = s.find_or_create_user("  saurav ")          # case + whitespace insensitive
    check("re-login same account", b["user_id"] == a["user_id"] and b["is_new"] is False)
    c = s.find_or_create_user("Alex")
    check("different name => different account", c["user_id"] != a["user_id"])

    check("empty gallery for new account", s.list_profiles(a["user_id"]) == [])
    s.save_profile(a["user_id"], "priya-1", name="Priya", goal="Find ML issues",
                   domains=["machine_learning", "python_data_eng"], platforms=["github", "reddit"],
                   hours=6, avoid="C++", avatar="")
    pl = s.list_profiles(a["user_id"])
    check("profile saved + listed", len(pl) == 1 and pl[0]["name"] == "Priya")
    check("profile round-trips domains/platforms",
          pl[0]["domains"] == ["machine_learning", "python_data_eng"] and "reddit" in pl[0]["platforms"])
    check("other account is isolated", s.list_profiles(c["user_id"]) == [])

    s.save_profile(a["user_id"], "priya-1", name="Priya 2", goal="Updated",
                   domains=["ai_research"], platforms=["github"], hours=4, avoid="", avatar="")
    pl = s.list_profiles(a["user_id"])
    check("upsert updates in place (no dup)", len(pl) == 1 and pl[0]["name"] == "Priya 2")

    s.delete_profile(a["user_id"], "priya-1")
    check("delete removes the profile", s.list_profiles(a["user_id"]) == [])

    s.save_profile(a["user_id"], "k2", name="X", goal="g", domains=[], platforms=[], hours=5, avoid="", avatar="")
    s.clear_all()
    check("clear_all wipes profiles too", s.list_profiles(a["user_id"]) == [])
    check("clear_all wipes accounts", s.find_or_create_user("Saurav")["is_new"] is True)
    s.conn.close()
    p.unlink()


def _post(path: str, body: dict):
    req = urllib.request.Request("http://127.0.0.1:8000" + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def test_endpoints() -> None:
    print("Endpoints: /api/login + /api/profiles (needs API on :8000)")
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/api/personas", timeout=3)
    except Exception:  # noqa: BLE001
        print("  SKIP  API not reachable")
        return
    a = _post("/api/login", {"name": "EndpointTester_zzz"})
    check("login returns user_id", bool(a.get("user_id")))
    uid = a["user_id"]
    key = _post("/api/profiles", {"session_id": uid, "profile_key": "ep-1", "name": "EP",
                                  "goal": "g", "domains": ["machine_learning"],
                                  "platforms": ["github", "reddit"], "time_budget_hours": 5})["profile_key"]
    check("profile create returns key", key == "ep-1")
    pl = _post("/api/profiles/list", {"session_id": uid})["profiles"]
    check("profile list has it", any(p["key"] == "ep-1" for p in pl))
    _post("/api/profiles/delete", {"session_id": uid, "profile_key": "ep-1"})
    pl = _post("/api/profiles/list", {"session_id": uid})["profiles"]
    check("profile deleted", not any(p["key"] == "ep-1" for p in pl))


if __name__ == "__main__":
    test_store()
    test_endpoints()
    print(f"\n{'=' * 44}\n  {_passed} passed, {_failed} failed\n{'=' * 44}")
    raise SystemExit(1 if _failed else 0)
