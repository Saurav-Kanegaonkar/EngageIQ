"""Semantic domain re-classification + near-dedup (Phase 2).

Fixes the keyword-matching noise found in the data-quality audit (~62% label
precision). Instead of trusting the scrape-query bucket, we:
  1. embed a rich description of each of the 15 domains,
  2. re-assign every record to its most semantically-similar domain (zero-shot),
  3. drop records that match NO domain well (true noise — the bottom slice by
     best-domain similarity),
  4. remove near-duplicates (cross-posts the exact-ID Bloom dedup missed).

Keeps embeddings.npy aligned with the surviving DB rows. The original
keyword label is preserved in a new `domain_scraped` column for comparison.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sqlite3

import numpy as np

from .domains import DOMAIN_LABELS, DOMAINS
from .embed import DB_PATH, EMB_PATH, IDS_PATH, encode, load_cached
from .eval.embedding_eval import find_near_duplicates

DEDUP_THRESHOLD = 0.97       # cosine >= this == near-duplicate
NOISE_PERCENTILE = 5         # drop the bottom N% by best-domain similarity

DOMAIN_DESCRIPTIONS = {
    "machine_learning": "Machine learning and deep learning: training neural networks, model architectures, ML frameworks like PyTorch and scikit-learn, computer vision, and applied machine learning.",
    "ai_research": "Artificial intelligence research: large language models (LLMs), generative AI, AI agents, transformers, model alignment, prompt engineering, and frontier AI research papers.",
    "python_data_eng": "Python data engineering: data pipelines, ETL, pandas dataframes, data warehousing, analytics engineering, and processing datasets in Python.",
    "devops_k8s": "DevOps and infrastructure: Kubernetes, Docker, containers, CI/CD, Terraform, Helm, observability, and cloud-native platform engineering.",
    "cloud_apis": "Cloud platforms and APIs: AWS, Google Cloud, Azure, serverless functions, Lambda, REST APIs, and cloud infrastructure services.",
    "cybersecurity": "Cybersecurity: vulnerabilities, CVEs, penetration testing, exploits, malware, application and network security, and infosec.",
    "frontend_web": "Frontend web development: React, Vue, Svelte, JavaScript, TypeScript, CSS, HTML, web UI, and responsive browser applications.",
    "developer_tools": "Developer tools and productivity: command-line tools, IDEs and editors, build systems, debuggers, SDKs, and engineering workflow tooling.",
    "b2b_saas": "B2B SaaS and startups: software-as-a-service products, business software, pricing, go-to-market, founders, and subscription product growth.",
    "blockchain": "Blockchain and web3: cryptocurrency, smart contracts, Solidity, Ethereum, DeFi, NFTs, and decentralized applications.",
    "gamedev_cpp": "Game development: game engines like Unreal, Unity and Godot, C++ graphics and rendering, game physics, shaders, and building video games.",
    "mobile_dev": "Mobile app development: iOS, Android, Swift, SwiftUI, Kotlin, Flutter, and React Native mobile applications.",
    "embedded_systems": "Embedded systems and hardware: microcontrollers, firmware, RTOS, Arduino, STM32, C programming for devices, and IoT hardware.",
    "beginner_coding": "Learning to code for beginners: programming tutorials for newcomers, first projects, coding fundamentals, and career-starter guides.",
    "open_source_trending": "Trending open-source software: popular and notable open-source projects, GitHub repositories, releases, and community contributions.",
}


def run() -> None:
    ids, embs = load_cached()
    n = len(ids)
    dom_emb = encode([DOMAIN_DESCRIPTIONS[d] for d in DOMAINS])   # (15, 384), normalized
    sims = embs @ dom_emb.T                                       # (N, 15)
    best = sims.argmax(1)
    best_sim = sims.max(1)
    new_domain = [DOMAINS[b] for b in best]

    pct = {p: float(np.percentile(best_sim, p)) for p in (1, 5, 10, 25, 50, 90)}
    print("best-domain-similarity percentiles:", {k: round(v, 3) for k, v in pct.items()})
    noise_floor = float(np.percentile(best_sim, NOISE_PERCENTILE))
    print(f"noise floor (p{NOISE_PERCENTILE}) = {noise_floor:.3f}")

    # near-dedup: greedily keep the first of each near-duplicate pair
    pairs = find_near_duplicates(embs, ids, threshold=DEDUP_THRESHOLD)
    removed_dup: set[str] = set()
    for a, b, _ in pairs:
        if a in removed_dup or b in removed_dup:
            continue
        removed_dup.add(b)

    noise = {ids[i] for i in range(n) if best_sim[i] < noise_floor}
    drop = removed_dup | noise
    survivors = [i for i in ids if i not in drop]

    print(f"\nrecords: {n}")
    print(f"  near-duplicates removed: {len(removed_dup)}")
    print(f"  noise removed:           {len(noise - removed_dup)}")
    print(f"  -> survivors:            {len(survivors)}")

    # ---- apply to DB ----
    conn = sqlite3.connect(str(DB_PATH))
    cols = [r[1] for r in conn.execute("PRAGMA table_info(opportunities)")]
    if "domain_scraped" not in cols:
        conn.execute("ALTER TABLE opportunities ADD COLUMN domain_scraped TEXT")
        conn.execute("UPDATE opportunities SET domain_scraped = domain")
    # count how many survivors actually change domain
    pos = {i: p for p, i in enumerate(ids)}
    old = dict(conn.execute("SELECT opportunity_id, domain FROM opportunities"))
    changed = sum(1 for i in survivors if old.get(i) != new_domain[pos[i]])
    conn.executemany("UPDATE opportunities SET domain=? WHERE opportunity_id=?",
                     [(new_domain[i], ids[i]) for i in range(n)])
    conn.executemany("DELETE FROM opportunities WHERE opportunity_id=?", [(i,) for i in drop])
    conn.commit()

    # ---- re-align cached embeddings to survivors ----
    keep = [pos[i] for i in survivors]
    np.save(EMB_PATH, embs[keep])
    IDS_PATH.write_text(json.dumps(survivors))

    total = conn.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0]
    print(f"\n{changed} survivors re-assigned to a different domain ({100*changed/max(1,len(survivors)):.0f}%)")
    print(f"final total: {total}")
    print("final per-domain counts:")
    for d, c in conn.execute("SELECT domain, COUNT(*) FROM opportunities GROUP BY domain ORDER BY COUNT(*) DESC"):
        print(f"  {DOMAIN_LABELS.get(d, d):24} {c}")


if __name__ == "__main__":
    run()
