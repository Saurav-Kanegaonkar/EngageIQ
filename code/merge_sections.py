"""Merge the sub-agent section result files (data/sec_results/chunk_*.json, each
{oid: {what, task, do, gain}}) into data/summary_sections.json, which the API reads.
Run after the summary-sections Workflow finishes."""
from __future__ import annotations

import json
import re
from pathlib import Path

from engageiq.summarize import SECTIONS_PATH

RESULTS = Path("data/sec_results")
_KEYS = ("what", "task", "do", "gain")


def clean(s: str) -> str:
    s = re.sub(r"[*#`]+", "", s or "")
    s = s.replace(" — ", ", ").replace("—", ", ").replace("–", "-")
    return " ".join(s.split())[:240]


def main() -> None:
    out = json.loads(SECTIONS_PATH.read_text()) if SECTIONS_PATH.exists() else {}
    added = 0
    for f in sorted(RESULTS.glob("chunk_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f.name}: {e}")
            continue
        for oid, sec in (d.items() if isinstance(d, dict) else []):
            if isinstance(sec, dict):
                obj = {k: clean(sec.get(k, "")) for k in _KEYS}
                if any(obj.values()):
                    out[oid] = obj
                    added += 1
    SECTIONS_PATH.write_text(json.dumps(out))
    print(f"merged: {added} section sets -> {len(out)} total in summary_sections.json")


if __name__ == "__main__":
    main()
