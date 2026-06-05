"""
Embedding model comparison: all-MiniLM-L6-v2 vs BAAI/bge-small-en-v1.5
Metrics: domain silhouette, intra/inter cosine, query retrieval relevant@10
"""

import sqlite3
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

DB_PATH = "/Users/sauravkanegaonkar/Desktop/Pet Projects/EngageIQ/data/engageiq.sqlite"

MODELS = [
    {
        "name": "all-MiniLM-L6-v2",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": None,
    },
    {
        "name": "bge-small-en-v1.5",
        "model_id": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
]

QUERIES = [
    ("kubernetes deployment and cluster infrastructure", "devops_k8s"),
    ("training a deep learning neural network", "machine_learning"),
    ("react frontend component state bug", "frontend_web"),
    ("smart contract solidity security", "blockchain"),
    ("ios swiftui mobile app", "mobile_dev"),
    ("learning to code as a beginner", "beginner_coding"),
]

SAMPLE_PER_DOMAIN = 20


def load_sample(conn):
    domains = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT domain FROM opportunities ORDER BY domain"
        ).fetchall()
    ]
    texts, labels = [], []
    for domain in domains:
        rows = conn.execute(
            "SELECT title, body FROM opportunities WHERE domain=? ORDER BY RANDOM() LIMIT ?",
            (domain, SAMPLE_PER_DOMAIN),
        ).fetchall()
        for title, body in rows:
            text = (title or "").strip() + "\n" + (body or "").strip()
            text = text.strip()
            texts.append(text)
            labels.append(domain)
    return texts, labels


def l2_normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return vecs / norms


def domain_metrics(embeddings, labels):
    label_list = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    label_ids = np.array([label_to_idx[l] for l in labels])

    # Silhouette score (cosine distance = 1 - cosine_sim)
    sil = silhouette_score(embeddings, label_ids, metric="cosine")

    # Intra-domain cosine similarity
    intra_sims = []
    for lbl in label_list:
        mask = label_ids == label_to_idx[lbl]
        vecs = embeddings[mask]
        if len(vecs) < 2:
            continue
        sim_mat = cosine_similarity(vecs)
        # upper triangle only (exclude diagonal)
        n = len(vecs)
        upper = sim_mat[np.triu_indices(n, k=1)]
        intra_sims.extend(upper.tolist())
    mean_intra = np.mean(intra_sims)

    # Inter-domain cosine similarity (sample pairs across domains to keep fast)
    rng = np.random.default_rng(42)
    n = len(embeddings)
    pairs_i = rng.integers(0, n, size=5000)
    pairs_j = rng.integers(0, n, size=5000)
    # keep only cross-domain pairs
    cross = pairs_i != pairs_j
    cross &= label_ids[pairs_i] != label_ids[pairs_j]
    pi, pj = pairs_i[cross], pairs_j[cross]
    inter_sims = np.sum(embeddings[pi] * embeddings[pj], axis=1)
    mean_inter = float(np.mean(inter_sims))

    return sil, mean_intra, mean_inter


def retrieval_metrics(model, cfg, sample_texts, sample_labels, query_prefix):
    results = []
    for query_text, expected_domain in QUERIES:
        if query_prefix:
            query_enc = query_text  # encode separately with prefix
        else:
            query_enc = query_text

        if query_prefix:
            q_vec = model.encode([query_prefix + query_enc], normalize_embeddings=True)[0]
        else:
            q_vec = model.encode([query_enc], normalize_embeddings=True)[0]

        q_vec = q_vec.reshape(1, -1)
        doc_vecs = np.array(model.encode(sample_texts, normalize_embeddings=True))

        sims = cosine_similarity(q_vec, doc_vecs)[0]
        top10_idx = np.argsort(sims)[::-1][:10]
        top10_labels = [sample_labels[i] for i in top10_idx]

        # Relevant = matches expected domain
        relevant = sum(1 for lbl in top10_labels if lbl == expected_domain)
        results.append((query_text, expected_domain, relevant, top10_labels))

    return results


def main():
    conn = sqlite3.connect(DB_PATH)
    print("Loading stratified sample...")
    texts, labels = load_sample(conn)
    conn.close()
    print(f"Loaded {len(texts)} records across {len(set(labels))} domains\n")

    all_results = {}

    for cfg in MODELS:
        print(f"{'='*60}")
        print(f"Model: {cfg['name']} ({cfg['model_id']})")
        print(f"{'='*60}")

        t0 = time.time()
        model = SentenceTransformer(cfg["model_id"])
        load_time = time.time() - t0
        print(f"  Load time: {load_time:.1f}s")

        t0 = time.time()
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                                  normalize_embeddings=True)
        encode_time = time.time() - t0
        print(f"  Encode time ({len(texts)} texts): {encode_time:.1f}s")

        embeddings = np.array(embeddings, dtype=np.float32)

        # Domain metrics
        sil, mean_intra, mean_inter = domain_metrics(embeddings, labels)
        print(f"\n  METRIC 1 - Domain Separation:")
        print(f"    Silhouette score (cosine): {sil:.4f}")
        print(f"    Mean intra-domain cosine:  {mean_intra:.4f}")
        print(f"    Mean inter-domain cosine:  {mean_inter:.4f}")
        print(f"    Intra/Inter ratio:         {mean_intra/mean_inter:.3f}x")

        # Retrieval metrics
        print(f"\n  METRIC 2 - Query Retrieval (relevant@10):")
        retrieval_results = retrieval_metrics(
            model, cfg, texts, labels, cfg["query_prefix"]
        )
        total_rel = 0
        for q_text, exp_domain, relevant, top_labels in retrieval_results:
            print(f"    [{relevant:2d}/10] {q_text[:55]:<55} (expected: {exp_domain})")
            total_rel += relevant

        mean_rel_at_10 = total_rel / len(QUERIES)
        print(f"    Mean relevant@10: {mean_rel_at_10:.1f}/10  ({total_rel}/{len(QUERIES)*10} total)")

        all_results[cfg["name"]] = {
            "silhouette": sil,
            "mean_intra": mean_intra,
            "mean_inter": mean_inter,
            "ratio": mean_intra / mean_inter,
            "mean_rel_at_10": mean_rel_at_10,
            "encode_time": encode_time,
            "load_time": load_time,
            "retrieval_detail": retrieval_results,
        }
        print()

    # Final comparison table
    print("\n" + "="*75)
    print("COMPARISON SUMMARY")
    print("="*75)
    print(f"{'Metric':<35} {'all-MiniLM-L6-v2':>18} {'bge-small-en-v1.5':>18}")
    print(f"{'-'*35} {'-'*18} {'-'*18}")

    m1 = all_results["all-MiniLM-L6-v2"]
    m2 = all_results["bge-small-en-v1.5"]

    rows = [
        ("Silhouette (cosine, higher=better)", m1["silhouette"], m2["silhouette"]),
        ("Mean intra-domain cosine", m1["mean_intra"], m2["mean_intra"]),
        ("Mean inter-domain cosine", m1["mean_inter"], m2["mean_inter"]),
        ("Intra/inter ratio (higher=better)", m1["ratio"], m2["ratio"]),
        ("Mean relevant@10 (higher=better)", m1["mean_rel_at_10"], m2["mean_rel_at_10"]),
        ("Encode time (s, lower=better)", m1["encode_time"], m2["encode_time"]),
    ]

    for label, v1, v2 in rows:
        print(f"  {label:<33} {v1:>18.4f} {v2:>18.4f}")

    print("="*75)
    print("\nPer-query relevant@10 detail:")
    print(f"  {'Query':<55} {'MiniLM':>8} {'bge':>8}")
    print(f"  {'-'*55} {'-'*8} {'-'*8}")
    for i, (q_text, exp_domain, _, _) in enumerate(m1["retrieval_detail"]):
        v1 = m1["retrieval_detail"][i][2]
        v2 = m2["retrieval_detail"][i][2]
        print(f"  {q_text[:55]:<55} {v1:>8} {v2:>8}")

    print("\n" + "="*75)

    # Winner determination
    minilm_wins = sum([
        m1["silhouette"] > m2["silhouette"],
        m1["ratio"] > m2["ratio"],
        m1["mean_rel_at_10"] > m2["mean_rel_at_10"],
    ])
    bge_wins = sum([
        m2["silhouette"] > m1["silhouette"],
        m2["ratio"] > m1["ratio"],
        m2["mean_rel_at_10"] > m1["mean_rel_at_10"],
    ])

    print(f"\nMetric wins: all-MiniLM-L6-v2 = {minilm_wins}/3, bge-small-en-v1.5 = {bge_wins}/3")

    if bge_wins > minilm_wins:
        rec = "RECOMMENDATION: Use BAAI/bge-small-en-v1.5"
    elif minilm_wins > bge_wins:
        rec = "RECOMMENDATION: Use sentence-transformers/all-MiniLM-L6-v2"
    else:
        # Tiebreak on retrieval
        if m2["mean_rel_at_10"] >= m1["mean_rel_at_10"]:
            rec = "RECOMMENDATION: Use BAAI/bge-small-en-v1.5 (tiebreak: retrieval quality)"
        else:
            rec = "RECOMMENDATION: Use sentence-transformers/all-MiniLM-L6-v2 (tiebreak: retrieval quality)"

    print(f"\n{rec}")
    print("="*75)


if __name__ == "__main__":
    main()
