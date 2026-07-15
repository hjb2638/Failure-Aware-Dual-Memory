"""
Memory module for Failure-Aware Dual-Memory.

Provides FAISS-based vector storage for material composition memory,
using CBFV (Composition-Based Feature Vector) for encoding.
"""

from .data_models import MemoryEntry
from .composition_encoder import CompositionEncoder
from .memory_store import MemoryStore

__all__ = [
    "MemoryEntry",
    "CompositionEncoder",
    "MemoryStore",
]
