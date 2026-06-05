# engageiq.eval — model-agnostic embedding evaluation utilities
from .embedding_eval import (
    domain_separation,
    retrieval_precision_at_k,
    nearest_neighbor_coherence,
    find_near_duplicates,
    run_report,
    DEFAULT_QUERIES,
)

__all__ = [
    "domain_separation",
    "retrieval_precision_at_k",
    "nearest_neighbor_coherence",
    "find_near_duplicates",
    "run_report",
    "DEFAULT_QUERIES",
]
