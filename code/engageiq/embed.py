"""Sentence-BERT embeddings for all opportunities (Phase 2, Lecture 5).

Model: all-MiniLM-L6-v2 (384-d) — chosen by empirical comparison vs bge-small
(better domain separation: 1.82x vs 1.08x intra/inter cosine ratio, and no
query-prefix complexity). Vectors are L2-normalized so dot product == cosine.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid faiss/torch OpenMP segfault (macOS)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "engageiq.sqlite"
EMB_PATH = ROOT / "data" / "embeddings.npy"
IDS_PATH = ROOT / "data" / "embedding_ids.json"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode(texts, batch_size: int = 64) -> np.ndarray:
    """Encode a list of texts to L2-normalized float32 vectors."""
    v = get_model().encode(list(texts), batch_size=batch_size,
                           show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)


def _text(title: str | None, body: str | None) -> str:
    return ((title or "") + "\n" + (body or "")).strip()


def build_all(db_path: Path = DB_PATH) -> tuple[list[str], np.ndarray]:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT opportunity_id, title, body FROM opportunities ORDER BY rowid"
    ).fetchall()
    ids = [r[0] for r in rows]
    texts = [_text(r[1], r[2]) for r in rows]
    embs = encode(texts)
    np.save(EMB_PATH, embs)
    IDS_PATH.write_text(json.dumps(ids))
    print(f"encoded {len(ids)} records -> {embs.shape}; saved {EMB_PATH.name} + {IDS_PATH.name}")
    return ids, embs


def load_cached() -> tuple[list[str], np.ndarray]:
    return json.loads(IDS_PATH.read_text()), np.load(EMB_PATH)


if __name__ == "__main__":
    build_all()
