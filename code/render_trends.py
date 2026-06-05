"""Render the capability-5 analytics graphs from the real corpus (design exploration).
L5 embeddings -> topic map + domain-similarity; L4 window -> topic-over-time."""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, sqlite3, collections, datetime as dt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from engageiq.embed import DB_PATH

OUT = "reports/trends"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "font.size": 10})


def lab(d):
    return (d or "?").replace("_", " ").title()


emb = np.load("data/embeddings.npy")
ids = json.loads(open("data/embedding_ids.json").read())
conn = sqlite3.connect(str(DB_PATH))
meta = {o: (dm, ca, (nc or 0) + (sc or 0)) for o, dm, ca, nc, sc in
        conn.execute("SELECT opportunity_id, domain, created_at, num_comments, score FROM opportunities")}
conn.close()
dom_of = np.array([meta.get(o, (None,))[0] for o in ids])
domains = [d for d in sorted(set(d for d in dom_of if d))]
rng = np.random.default_rng(7)

# ---- 1. topic map: embeddings -> t-SNE 2D, colored by domain (L5) ----
idx = []
for d in domains:
    di = np.where(dom_of == d)[0]
    if len(di):
        idx.extend(rng.choice(di, min(180, len(di)), replace=False))
idx = np.array(idx)
Xp = PCA(n_components=50, random_state=7).fit_transform(emb[idx])
Y = TSNE(n_components=2, perplexity=30, random_state=7, init="pca").fit_transform(Xp)
plt.figure(figsize=(11, 8))
cmap = plt.get_cmap("tab20")
for i, d in enumerate(domains):
    m = dom_of[idx] == d
    plt.scatter(Y[m, 0], Y[m, 1], s=11, color=cmap(i % 20), label=lab(d), alpha=.75, linewidths=0)
plt.legend(markerscale=2, fontsize=8, loc="center left", bbox_to_anchor=(1.0, .5), frameon=False)
plt.title("Topic map: opportunities embedded (MiniLM 384-d) -> t-SNE 2D, colored by domain\nClusters = interrelated topics", fontsize=12)
plt.xticks([]); plt.yticks([])
plt.tight_layout(); plt.savefig(f"{OUT}/topicmap.png", dpi=115); plt.close()

# ---- 2. domain similarity heatmap (L5) ----
cent = np.array([emb[dom_of == d].mean(0) for d in domains])
cent = cent / np.linalg.norm(cent, axis=1, keepdims=True)
sim = cent @ cent.T
plt.figure(figsize=(9.5, 8))
im = plt.imshow(sim, cmap="viridis", vmin=sim[~np.eye(len(domains), dtype=bool)].min())
plt.xticks(range(len(domains)), [lab(d) for d in domains], rotation=55, ha="right", fontsize=8)
plt.yticks(range(len(domains)), [lab(d) for d in domains], fontsize=8)
plt.colorbar(im, label="cosine similarity", fraction=.046)
plt.title("Which domains are semantically interrelated\n(cosine similarity between domain embedding centroids)", fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/domsim.png", dpi=115); plt.close()

# ---- 3. topic progression over time (L4 window) ----
vol = collections.Counter(d for d in dom_of if d)
top = [d for d, _ in vol.most_common(6)]
recs = []
for o in ids:
    dm, ca, _ = meta.get(o, (None, None, 0))
    if dm in top and ca:
        try:
            recs.append((dm, dt.datetime.fromisoformat(ca).date()))
        except Exception:
            pass
df = pd.DataFrame(recs, columns=["domain", "date"])
df["week"] = pd.to_datetime(df["date"]).dt.to_period("W").dt.start_time
piv = df[df["week"] >= "2026-01-01"].groupby(["week", "domain"]).size().unstack(fill_value=0)
plt.figure(figsize=(11, 6))
for d in top:
    if d in piv:
        plt.plot(piv.index, piv[d].rolling(2, min_periods=1).mean(), marker="o", ms=3, label=lab(d))
plt.legend(fontsize=9, frameon=False)
plt.title("Topic progression over time (weekly opportunity volume by domain)", fontsize=12)
plt.ylabel("opportunities / week")
plt.tight_layout(); plt.savefig(f"{OUT}/overtime.png", dpi=115); plt.close()

print("rendered:", sorted(os.listdir(OUT)))
