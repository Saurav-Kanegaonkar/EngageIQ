"""
engageiq.eval.embedding_eval
============================
Model-agnostic embedding evaluation module.

All public functions accept precomputed embeddings as a numpy ndarray (N x D),
assumed to be L2-normalised (unit vectors).  They never call any model directly;
callers must supply an `encode_fn(list[str]) -> np.ndarray` where one is needed.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional

import faiss
import numpy as np

# ---------------------------------------------------------------------------
# Built-in default queries: one natural-language query per domain.
# Keys are the canonical domain strings used in the `opportunities` table.
# ---------------------------------------------------------------------------

DEFAULT_QUERIES: Dict[str, str] = {
    "machine_learning":
        "gradient boosting hyperparameter tuning scikit-learn XGBoost",
    "ai_research":
        "large language model pre-training transformer architecture research paper",
    "python_data_eng":
        "pandas dataframe ETL pipeline data cleaning Python",
    "devops_k8s":
        "kubernetes helm chart deployment container orchestration",
    "cloud_apis":
        "REST API cloud infrastructure AWS Lambda serverless functions",
    "cybersecurity":
        "vulnerability CVE penetration testing security patch exploit",
    "frontend_web":
        "React component CSS animation responsive web design",
    "developer_tools":
        "VS Code extension developer productivity CLI tooling",
    "b2b_saas":
        "SaaS startup product launch customer acquisition B2B pricing",
    "blockchain":
        "smart contract Solidity DeFi Ethereum Web3",
    "gamedev_cpp":
        "Unreal Engine C++ game physics rendering game development",
    "mobile_dev":
        "Flutter iOS Android app SwiftUI mobile development",
    "embedded_systems":
        "Arduino microcontroller RTOS firmware embedded C",
    "beginner_coding":
        "learn programming beginner Python tutorial first project",
    "open_source_trending":
        "GitHub trending open source project community contribution",
}


# ---------------------------------------------------------------------------
# 1. Domain separation
# ---------------------------------------------------------------------------

def domain_separation(
    embeddings: np.ndarray,
    labels: List[str],
) -> dict:
    """
    Compute cluster-quality metrics for the embedding space.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D), L2-normalised
    labels     : list of N domain label strings

    Returns
    -------
    dict with keys:
        silhouette_score      – sklearn silhouette score using cosine distance
        mean_intra_cosine     – average pairwise cosine similarity within domains
        mean_inter_cosine     – average pairwise cosine similarity across domains
        separation_gap        – mean_intra_cosine - mean_inter_cosine
    """
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder

    embeddings = np.asarray(embeddings, dtype=np.float32)
    label_arr = np.array(labels)
    le = LabelEncoder()
    encoded = le.fit_transform(label_arr)

    # Silhouette uses cosine distance (1 - cosine_sim).
    # For L2-normalised vectors, dot product == cosine similarity.
    # sklearn silhouette_score accepts metric='cosine'.
    sil = float(silhouette_score(embeddings, encoded, metric="cosine"))

    # Intra / inter cosine means -------------------------------------------
    unique_labels = le.classes_
    intra_sims: List[float] = []
    inter_sims: List[float] = []

    # Sample cap so this stays tractable on large N
    MAX_PAIRS_PER_DOMAIN = 50_000
    rng = random.Random(42)

    for lbl in unique_labels:
        mask = label_arr == lbl
        idx_in = np.where(mask)[0].tolist()
        idx_out = np.where(~mask)[0].tolist()

        # Intra-domain pairs (sample if large)
        if len(idx_in) >= 2:
            pairs_in = [(i, j) for i in idx_in for j in idx_in if j > i]
            if len(pairs_in) > MAX_PAIRS_PER_DOMAIN:
                pairs_in = rng.sample(pairs_in, MAX_PAIRS_PER_DOMAIN)
            for i, j in pairs_in:
                sim = float(np.dot(embeddings[i], embeddings[j]))
                intra_sims.append(sim)

        # Inter-domain pairs (sample)
        if idx_in and idx_out:
            inter_sample = min(MAX_PAIRS_PER_DOMAIN, len(idx_in) * len(idx_out))
            pairs_out = rng.sample(
                [(i, j) for i in idx_in for j in rng.sample(idx_out, min(20, len(idx_out)))],
                min(inter_sample, len(idx_in) * min(20, len(idx_out))),
            )
            for i, j in pairs_out:
                sim = float(np.dot(embeddings[i], embeddings[j]))
                inter_sims.append(sim)

    mean_intra = float(np.mean(intra_sims)) if intra_sims else float("nan")
    mean_inter = float(np.mean(inter_sims)) if inter_sims else float("nan")

    return {
        "silhouette_score": sil,
        "mean_intra_cosine": mean_intra,
        "mean_inter_cosine": mean_inter,
        "separation_gap": mean_intra - mean_inter,
    }


# ---------------------------------------------------------------------------
# 2. Retrieval precision@k
# ---------------------------------------------------------------------------

def retrieval_precision_at_k(
    embeddings: np.ndarray,
    labels: List[str],
    encode_fn: Callable[[List[str]], np.ndarray],
    queries: Optional[Dict[str, str]] = None,
    k: int = 10,
) -> dict:
    """
    For each query, encode it, retrieve top-k items from `embeddings` by cosine
    similarity, and compute precision@k = fraction of top-k with matching domain.

    Parameters
    ----------
    embeddings : np.ndarray (N, D), L2-normalised
    labels     : list of N domain label strings
    encode_fn  : callable that takes list[str] and returns np.ndarray (M, D).
                 The returned vectors should be L2-normalised (or this function
                 will normalise them before scoring).
    queries    : dict mapping expected_domain -> natural-language query string.
                 Defaults to DEFAULT_QUERIES if None.
    k          : number of neighbours to retrieve

    Returns
    -------
    dict with keys:
        per_query : dict mapping domain -> {"query": str, "precision_at_k": float, "k": int}
        mean_precision_at_k : float
    """
    if queries is None:
        queries = DEFAULT_QUERIES

    embeddings = np.asarray(embeddings, dtype=np.float32)
    label_arr = np.array(labels)

    per_query: Dict[str, dict] = {}

    query_texts = list(queries.values())
    query_domains = list(queries.keys())

    # Encode all queries in one batch
    raw_vecs = encode_fn(query_texts)
    raw_vecs = np.asarray(raw_vecs, dtype=np.float32)

    # Normalise query vectors (robustness if encode_fn doesn't normalise)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    query_vecs = raw_vecs / norms

    for domain, qvec, qtext in zip(query_domains, query_vecs, query_texts):
        # Cosine similarity via dot product (embeddings assumed normalised)
        sims = embeddings @ qvec  # shape (N,)
        top_k_idx = np.argsort(sims)[::-1][:k]
        top_k_labels = label_arr[top_k_idx]
        hits = int(np.sum(top_k_labels == domain))
        precision = hits / k
        per_query[domain] = {
            "query": qtext,
            "precision_at_k": precision,
            "k": k,
            "hits": hits,
        }

    all_precisions = [v["precision_at_k"] for v in per_query.values()]
    mean_p = float(np.mean(all_precisions)) if all_precisions else float("nan")

    return {
        "per_query": per_query,
        "mean_precision_at_k": mean_p,
    }


# ---------------------------------------------------------------------------
# 3. Nearest-neighbour coherence
# ---------------------------------------------------------------------------

def nearest_neighbor_coherence(
    embeddings: np.ndarray,
    labels: List[str],
    sample: int = 500,
) -> float:
    """
    Fraction of (sampled) items whose nearest neighbour (excluding self) shares
    the same domain label.

    Parameters
    ----------
    embeddings : np.ndarray (N, D), L2-normalised
    labels     : list of N domain label strings
    sample     : number of items to sample; if N <= sample, all items are used

    Returns
    -------
    float in [0, 1]
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    label_arr = np.array(labels)
    N = len(embeddings)

    if N <= sample:
        idx = np.arange(N)
    else:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, size=sample, replace=False)

    sampled_emb = embeddings[idx]       # (S, D)
    sampled_labels = label_arr[idx]

    # Build FAISS index over ALL embeddings for neighbour lookup
    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product == cosine for L2-norm
    index.add(embeddings)

    # Query with k=2: first result is self (sim=1), second is NN
    sims, nn_indices = index.search(sampled_emb, 2)
    # nn_indices[:, 0] might not always be self if there are near-duplicates,
    # but we take [:, 1] which is always "not the query itself" at min.
    nn_idx = nn_indices[:, 1]
    nn_labels = label_arr[nn_idx]

    coherent = int(np.sum(sampled_labels == nn_labels))
    return coherent / len(idx)


# ---------------------------------------------------------------------------
# 4. Near-duplicate detection
# ---------------------------------------------------------------------------

def find_near_duplicates(
    embeddings: np.ndarray,
    ids: List,
    threshold: float = 0.95,
    max_pairs: int = 100_000,
) -> List[tuple]:
    """
    Find pairs of items whose cosine similarity >= threshold.

    Uses a FAISS IndexFlatIP (inner product = cosine for L2-normalised vecs).
    For each item we retrieve enough neighbours to find all above threshold,
    capped at max_pairs total.

    Parameters
    ----------
    embeddings : np.ndarray (N, D), L2-normalised
    ids        : list of N item identifiers (any type; returned in pairs)
    threshold  : minimum cosine similarity to flag as near-duplicate (default 0.95)
    max_pairs  : hard cap on returned pairs to avoid memory blow-up

    Returns
    -------
    list of (id_a, id_b, similarity) tuples, sorted by similarity descending
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    N = embeddings.shape[1]
    ids_arr = list(ids)
    n = len(embeddings)

    # We search for top-K neighbours per item.
    # For a 22k×384 corpus, searching k=50 is cheap and likely catches all dups.
    k = min(50, n)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    sims_all, idx_all = index.search(embeddings, k)

    pairs = []
    seen = set()

    for i in range(n):
        for j_pos in range(k):
            j = int(idx_all[i, j_pos])
            sim = float(sims_all[i, j_pos])
            if j <= i:
                # Skip self and already-seen reverse pairs
                continue
            if sim < threshold:
                break  # FAISS returns sorted desc; no point continuing
            key = (i, j)
            if key not in seen:
                seen.add(key)
                pairs.append((ids_arr[i], ids_arr[j], sim))
                if len(pairs) >= max_pairs:
                    break
        if len(pairs) >= max_pairs:
            break

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# 5. run_report — convenience wrapper
# ---------------------------------------------------------------------------

def run_report(
    embeddings: np.ndarray,
    labels: List[str],
    ids: List,
    encode_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
    queries: Optional[Dict[str, str]] = None,
) -> None:
    """
    Run all available metrics and print a human-readable report.

    Parameters
    ----------
    embeddings : np.ndarray (N, D), L2-normalised
    labels     : list of N domain label strings
    ids        : list of N item identifiers
    encode_fn  : optional; required for retrieval_precision_at_k
    queries    : optional custom query dict; defaults to DEFAULT_QUERIES
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    N = len(embeddings)

    print("=" * 60)
    print("EngageIQ Embedding Evaluation Report")
    print(f"Corpus size : {N:,} items  |  Dims : {embeddings.shape[1]}")
    print(f"Domains     : {len(set(labels))}")
    print("=" * 60)

    # --- 1. Domain separation ---
    print("\n[1] Domain Separation")
    sep = domain_separation(embeddings, labels)
    print(f"  Silhouette score        : {sep['silhouette_score']:+.4f}")
    print(f"  Mean intra-domain cos   : {sep['mean_intra_cosine']:.4f}")
    print(f"  Mean inter-domain cos   : {sep['mean_inter_cosine']:.4f}")
    print(f"  Separation gap          : {sep['separation_gap']:+.4f}")

    # --- 2. Retrieval precision@k ---
    if encode_fn is not None:
        print("\n[2] Retrieval Precision@10")
        ret = retrieval_precision_at_k(embeddings, labels, encode_fn, queries=queries)
        for domain, info in ret["per_query"].items():
            bar = "#" * int(info["precision_at_k"] * 20)
            print(
                f"  {domain:<25s}  P@{info['k']}={info['precision_at_k']:.2f}  "
                f"({info['hits']}/{info['k']})  {bar}"
            )
        print(f"\n  Mean P@10 : {ret['mean_precision_at_k']:.4f}")
    else:
        print("\n[2] Retrieval Precision@10  — SKIPPED (no encode_fn provided)")

    # --- 3. NN coherence ---
    print("\n[3] Nearest-Neighbour Coherence")
    sample_n = min(500, N)
    coherence = nearest_neighbor_coherence(embeddings, labels, sample=sample_n)
    print(f"  NN coherence ({sample_n} sampled) : {coherence:.4f}")

    # --- 4. Near duplicates ---
    print("\n[4] Near Duplicates (threshold=0.95)")
    dups = find_near_duplicates(embeddings, ids, threshold=0.95)
    print(f"  Pairs found : {len(dups)}")
    if dups:
        for a, b, s in dups[:5]:
            print(f"    {a!r} <-> {b!r}  sim={s:.4f}")
        if len(dups) > 5:
            print(f"    ... and {len(dups)-5} more")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Self-contained toy-data validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running toy-data validation for embedding_eval.py ...")
    rng = np.random.default_rng(0)

    # Build 3 tight clusters in 64-D space, 30 items each = 90 total
    N_PER_CLUSTER = 30
    D = 64

    def _make_cluster(center: np.ndarray, n: int, noise: float) -> np.ndarray:
        vecs = center + rng.normal(0, noise, size=(n, D))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    centers = [
        np.array([1.0] + [0.0] * (D - 1)),
        np.array([0.0, 1.0] + [0.0] * (D - 2)),
        np.array([0.0, 0.0, 1.0] + [0.0] * (D - 3)),
    ]
    cluster_names = ["alpha", "beta", "gamma"]

    emb_parts = [_make_cluster(c, N_PER_CLUSTER, 0.05) for c in centers]
    embeddings = np.vstack(emb_parts).astype(np.float32)
    labels = [name for name in cluster_names for _ in range(N_PER_CLUSTER)]
    ids = [f"{name}_{i}" for name in cluster_names for i in range(N_PER_CLUSTER)]

    print(f"\nToy corpus: {len(embeddings)} items, {D} dims, 3 domains")

    # --- Test 1: domain_separation ---
    sep = domain_separation(embeddings, labels)
    print(f"\ndomain_separation()")
    print(f"  silhouette_score   = {sep['silhouette_score']:+.4f}  (expect > 0)")
    print(f"  mean_intra_cosine  = {sep['mean_intra_cosine']:.4f}")
    print(f"  mean_inter_cosine  = {sep['mean_inter_cosine']:.4f}")
    print(f"  separation_gap     = {sep['separation_gap']:+.4f}  (expect > 0)")

    assert sep["silhouette_score"] > 0, "silhouette should be positive for clear clusters"
    assert sep["mean_intra_cosine"] > sep["mean_inter_cosine"], \
        "intra-domain cosine should exceed inter-domain cosine"
    assert sep["separation_gap"] > 0, "separation gap should be positive"
    print("  ASSERTIONS PASSED")

    # --- Test 2: retrieval_precision_at_k ---
    # encode_fn: for the toy test we just map query strings to the cluster center
    center_map = {
        "alpha": centers[0].astype(np.float32),
        "beta":  centers[1].astype(np.float32),
        "gamma": centers[2].astype(np.float32),
    }
    toy_queries = {
        "alpha": "query for alpha cluster",
        "beta":  "query for beta cluster",
        "gamma": "query for gamma cluster",
    }

    def toy_encode_fn(texts: List[str]) -> np.ndarray:
        """Map query text -> the corresponding cluster center."""
        out = []
        for t in texts:
            for domain, vec in center_map.items():
                if domain in t:
                    out.append(vec)
                    break
            else:
                out.append(rng.standard_normal(D).astype(np.float32))
        vecs = np.vstack(out)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    ret = retrieval_precision_at_k(embeddings, labels, toy_encode_fn, queries=toy_queries, k=10)
    print(f"\nretrieval_precision_at_k()")
    for domain, info in ret["per_query"].items():
        print(f"  {domain}: P@10 = {info['precision_at_k']:.2f}")
    print(f"  mean P@10 = {ret['mean_precision_at_k']:.4f}  (expect ~1.0)")

    assert ret["mean_precision_at_k"] >= 0.9, \
        f"Expected high precision on clear clusters, got {ret['mean_precision_at_k']}"
    print("  ASSERTIONS PASSED")

    # --- Test 3: nearest_neighbor_coherence ---
    coh = nearest_neighbor_coherence(embeddings, labels, sample=90)
    print(f"\nnearest_neighbor_coherence()")
    print(f"  coherence = {coh:.4f}  (expect ~1.0)")

    assert coh >= 0.9, f"Expected high NN coherence on clear clusters, got {coh}"
    print("  ASSERTIONS PASSED")

    # --- Test 4: find_near_duplicates ---
    # Add two near-identical vectors to the corpus to test this function
    dup_a = embeddings[0].copy()
    dup_b = embeddings[0] + rng.normal(0, 0.001, D).astype(np.float32)
    dup_b /= np.linalg.norm(dup_b)
    emb_with_dups = np.vstack([embeddings, dup_a[None, :], dup_b[None, :]]).astype(np.float32)
    ids_with_dups = ids + ["dup_A", "dup_B"]
    labels_with_dups = labels + ["alpha", "alpha"]

    dups = find_near_duplicates(emb_with_dups, ids_with_dups, threshold=0.95)
    print(f"\nfind_near_duplicates(threshold=0.95)")
    print(f"  Pairs found: {len(dups)}")
    if dups:
        for a, b, s in dups[:3]:
            print(f"    {a!r} <-> {b!r}  sim={s:.4f}")

    dup_ids_found = {frozenset([a, b]) for a, b, _ in dups}
    assert frozenset(["dup_A", "dup_B"]) in dup_ids_found, \
        "Expected dup_A <-> dup_B to be flagged as near-duplicate"
    print("  ASSERTIONS PASSED")

    # --- Test 5: run_report (smoke test) ---
    print("\nrun_report() — smoke test on toy data:")
    run_report(embeddings, labels, ids, encode_fn=toy_encode_fn, queries=toy_queries)

    print("\nAll toy-data validations PASSED.")
