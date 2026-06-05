"""LLM client for generation tasks — Stage-2 relevance re-rank, "Suggested
Actions", and "Why this?" explanations.

Uses NVIDIA's FREE hosted API (build.nvidia.com / integrate.api.nvidia.com),
which is OpenAI-compatible. Falls back gracefully (returns None) when no key is
present, so the pipeline never hard-depends on the LLM — deterministic fallbacks
keep everything working offline for graders.
"""
from __future__ import annotations

import requests

from . import config

BASE = "https://integrate.api.nvidia.com/v1"

# fast + cheap for high-volume calls (e.g. re-ranking many candidates)
FAST_MODEL = "meta/llama-3.1-8b-instruct"
# stronger, reasoning-tuned for quality tasks (Suggested Actions, Why-this)
SMART_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"


def available() -> bool:
    return bool(config.NVIDIA_API_KEY)


def chat(messages: list[dict], model: str = FAST_MODEL, max_tokens: int = 512,
         temperature: float = 0.3, timeout: int = 120) -> str | None:
    """Return the assistant message text, or None on any failure / no key."""
    if not config.NVIDIA_API_KEY:
        return None
    try:
        r = requests.post(
            f"{BASE}/chat/completions",
            headers={"Authorization": f"Bearer {config.NVIDIA_API_KEY}",
                     "Accept": "application/json"},
            json={"model": model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": temperature},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:  # noqa: BLE001
        print(f"  [llm] {model}: {e}")
        return None
