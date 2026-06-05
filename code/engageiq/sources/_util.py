"""Shared HTTP helper for source adapters."""
from __future__ import annotations

import requests

_session = requests.Session()
_session.headers.update(
    {"User-Agent": "EngageIQ/0.1 (BAX-423 UC Davis project; contact via GitHub)"}
)


def get_json(url: str, params: dict | None = None, headers: dict | None = None,
             timeout: int = 12):
    r = _session.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def post_json(url: str, json: dict | None = None, headers: dict | None = None,
              timeout: int = 12):
    r = _session.post(url, json=json, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()
