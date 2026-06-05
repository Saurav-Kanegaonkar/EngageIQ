"""Loads credentials/config from the project-root `.env` (never committed).

Tiny zero-dependency .env reader so the package doesn't need python-dotenv.
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv(_ENV_PATH)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "EngageIQ/0.1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
BLUESKY_HANDLE = os.environ.get("BLUESKY_HANDLE", "")
BLUESKY_APP_PASSWORD = os.environ.get("BLUESKY_APP_PASSWORD", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")  # free hosted LLMs (build.nvidia.com)


def have(*names: str) -> bool:
    """True only if every named credential is present and non-empty."""
    return all(os.environ.get(n) for n in names)
