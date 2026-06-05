"""List NVIDIA-hosted models available to our key, and test a chat model.

The 253B Nemotron Ultra is self-host-only (NIM container). This finds which
models ARE callable via the hosted OpenAI-compatible API and picks a good one
for our LLM tasks (relevance re-rank, Suggested Actions).
"""
from __future__ import annotations

from pathlib import Path

import requests

ENV = Path(__file__).resolve().parents[1] / ".env"
BASE = "https://integrate.api.nvidia.com/v1"


def find_key():
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if v and any(t in k.upper() for t in ("NVIDIA", "NGC", "NIM")):
            return v
    return None


key = find_key()
auth = {"Authorization": f"Bearer {key}", "Accept": "application/json"}

r = requests.get(f"{BASE}/models", headers=auth, timeout=60)
ids = sorted(m["id"] for m in r.json().get("data", []))
print(f"total hosted models available: {len(ids)}")
rel = [i for i in ids if any(t in i.lower() for t in ("nemotron", "llama-3", "instruct"))]
print("relevant chat models:")
for i in rel:
    print("  ", i)

prio = [
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-8b-instruct",
]
cand = next((c for c in prio if c in ids), rel[0] if rel else None)
print(f"\n--- testing: {cand} ---")
r2 = requests.post(
    f"{BASE}/chat/completions", headers=auth,
    json={"model": cand,
          "messages": [{"role": "system", "content": "detailed thinking off"},
                       {"role": "user", "content": "Reply with exactly: EngageIQ NVIDIA test OK"}],
          "max_tokens": 40, "temperature": 0.2},
    timeout=120)
print("HTTP", r2.status_code)
print(r2.json()["choices"][0]["message"]["content"].strip() if r2.ok else r2.text[:400])
