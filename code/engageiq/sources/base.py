"""SourceAdapter — the pluggable interface every data source implements.

A source adapter has ONE job: hit its API and yield canonical `Opportunity`
objects (it owns the messy source-specific -> canonical translation). Adding a
new platform later = implement this interface; nothing else in the pipeline
changes.

🟢 PERMANENT — this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..schema import Opportunity


class SourceAdapter(ABC):
    #: short id, must match a value in schema.SOURCES
    name: str = "base"
    #: whether the adapter needs credentials to run
    requires_auth: bool = False

    @abstractmethod
    def fetch(self, domain: str, limit: int = 100) -> Iterator[Opportunity]:
        """Yield up to `limit` Opportunity items for a single domain."""
        raise NotImplementedError

    def available(self) -> bool:
        """Whether this adapter can run right now (e.g. has its credentials)."""
        return True
