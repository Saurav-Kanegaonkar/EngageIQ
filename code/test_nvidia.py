"""Quick test of NVIDIA's HOSTED LLM API (build.nvidia.com / integrate.api.nvidia.com).

No GPU / Docker needed — it's a hosted, OpenAI-compatible REST endpoint. Auto-detects
the API key from .env (never prints the value).
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

ENV = Path(__file__).resolve().parents[1] / ".env"


def find_key():
    lines = ENV.read_text().splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if v and any(t in k.upper() for t in ("NVIDIA", "NGC", "NIM")):
            return v, k
    for line in lines:  # fallback: key pasted into OPENAI_API_KEY
        k, _, v = line.partition("=")
        if k.strip() == "OPENAI_API_KEY" and v.strip().startswith("nvapi"):
            return v.strip(), "OPENAI_API_KEY"
    return None, None


key, keyname = find_key()
if not key:
    print("NO NVIDIA KEY FOUND in .env. Add a line:  NVIDIA_API_KEY=nvapi-...")
    sys.exit(1)
print(f"found NVIDIA key in .env var: {keyname}")

MODEL = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
try:
    r = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "detailed thinking off"},
                {"role": "user", "content": "Reply with exactly: EngageIQ NVIDIA test OK"},
            ],
            "max_tokens": 60,
            "temperature": 0.2,
        },
        timeout=120,
    )
    print("HTTP", r.status_code)
    if r.ok:
        d = r.json()
        print("model:", d.get("model"))
        print("reply:", d["choices"][0]["message"]["content"].strip())
        print("usage:", d.get("usage"))
    else:
        print("error:", r.text[:700])
except Exception as e:  # noqa: BLE001
    print("request failed:", type(e).__name__, e)
