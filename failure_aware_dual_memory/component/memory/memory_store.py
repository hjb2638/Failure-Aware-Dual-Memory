"""
Memory store using FAISS for vector storage.

Unified storage architecture:
- All cases (success/failure) stored in single FAISS index
- is_success flag distinguishes status
- Only is_success=True cases proceed to subsequent stages
"""

import faiss
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from .data_models import MemoryEntry, RetrievalResult
from .composition_encoder import CompositionEncoder


class MemoryStore:
    """
    FAISS-based memory storage system.
    
    Uses unified storage architecture:
    - Single FAISS index for all cases
    - is_success flag to distinguish status
    - Success cases can proceed to subsequent stages
    - Failure cases are for reference only
    """
    
    def __init__(
        self,
        storage_dir: str,
        use_onehot: bool = True,
        use_magpie: bool = True,
        nlist: int = 100,
    ):
        """
        Initialize memory store.
        
        Args:
            storage_dir: Directory for storing FAISS index and metadata
            use_onehot: Whether to use 714-dim onehot features
            use_magpie: Whether to use 132-dim magpie features
            nlist: FAISS IVF parameter for clustering
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encoder
        self.encoder = CompositionEncoder(use_onehot=use_onehot, use_magpie=use_magpie)
        self.dim = self.encoder.expected_dim
        
        # File paths
        self.index_path = self.storage_dir / "memory.index"
        self.meta_path = self.storage_dir / "entries.json"
        self.idmap_path = self.storage_dir / "idmap.json"
        
        # Initialize FAISS index
        self.index = self._create_index(nlist)
        
        # Metadata storage
        self.entries: List[MemoryEntry] = []
        self.id_to_index: Dict[str, int] = {}  # entry_id -> faiss_index
        
        # Load existing data
        self.load()
    
    def _create_index(self, nlist: int) -> faiss.Index:
        """
        Create FAISS index.
        
        For small datasets, use IndexFlatIP directly.
        For larger datasets, IndexIVFFlat would be more efficient.
        """
        # Use flat index for simplicity and small dataset efficiency
        index = faiss.IndexFlatIP(self.dim)
        return index
    
    def add(self, entry: MemoryEntry) -> int:
        """
        Add entry to memory store.
        
        Args:
            entry: MemoryEntry to add
            
        Returns:
            Index in FAISS (faiss_id)
        """
        # Encode composition if vector not provided
        if not entry.composition_vector:
            vector = self.encoder.encode(entry.composition)
            entry.composition_vector = vector.tolist()
        
        # Convert to numpy array
        vector = np.array([entry.composition_vector], dtype='float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(vector)
        
        # Train index if first entry
        if self.index.ntotal == 0:
            self.index.train(vector)
        
        # Add to FAISS
        faiss_id = self.index.ntotal
        self.index.add(vector)
        
        # Store metadata
        self.entries.append(entry)
        self.id_to_index[entry.entry_id] = faiss_id
        
        return faiss_id
    
    def add_batch(self, entries: List[MemoryEntry]) -> List[int]:
        """
        Add multiple entries in batch.
        
        Args:
            entries: List of MemoryEntry to add
            
        Returns:
            List of FAISS indices
        """
        if not entries:
            return []
        
        # Encode compositions
        vectors = []
        for entry in entries:
            if not entry.composition_vector:
                vec = self.encoder.encode(entry.composition)
                entry.composition_vector = vec.tolist()
            vectors.append(entry.composition_vector)
        
        # Convert to numpy array
        vectors = np.array(vectors, dtype='float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)
        
        # Train index if first entry
        if self.index.ntotal == 0:
            self.index.train(vectors)
        
        # Add to FAISS
        start_id = self.index.ntotal
        self.index.add(vectors)
        
        # Store metadata
        faiss_ids = []
        for i, entry in enumerate(entries):
            faiss_id = start_id + i
            self.entries.append(entry)
            self.id_to_index[entry.entry_id] = faiss_id
            faiss_ids.append(faiss_id)
        
        return faiss_ids
    
    def search(
        self,
        composition: str,
        k: int = 6,
        target_value: Optional[float] = None,
        target_tolerance: float = 0.5,
    ) -> RetrievalResult:
        """
        Search for similar compositions.
        
        Args:
            composition: Query composition
            k: Number of results to return
            target_value: Optional target value for filtering
            target_tolerance: Tolerance for target value filtering
            
        Returns:
            RetrievalResult with success and failure cases separated
        """
        if self.index.ntotal == 0:
            return RetrievalResult()
        
        # Handle empty or None composition
        if not composition:
            print(f"[Warning] Empty composition provided for search, returning empty result")
            return RetrievalResult()
        
        # Encode query
        try:
            query_vector = self.encoder.encode(composition)
        except Exception as e:
            print(f"[Warning] Failed to encode composition '{composition}': {e}")
            return RetrievalResult()
        query_vector = np.array([query_vector], dtype='float32')
        faiss.normalize_L2(query_vector)
        
        # Search
        k_search = min(k * 2, self.index.ntotal)  # Search more for filtering
        distances, indices = self.index.search(query_vector, k_search)
        
        # Build result
        result = RetrievalResult()
        
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.entries):
                continue
            
            entry = self.entries[idx]
            
            # Filter by target value if specified
            if target_value is not None:
                if abs(entry.target_value - target_value) > target_tolerance:
                    continue
            
            # Separate by success status
            if entry.is_success:
                if len(result.success_cases) < k // 2:
                    result.success_cases.append(entry)
            else:
                if len(result.failure_cases) < k // 2:
                    result.failure_cases.append(entry)
            
            # Add to similar compositions
            result.similar_compositions.append(entry)
            
            # Check if we have enough
            if len(result.success_cases) >= k // 2 and len(result.failure_cases) >= k // 2:
                break
        
        return result
    
    def search_by_target(
        self,
        target_value: float,
        tolerance: float = 0.5,
        k: int = 10,
    ) -> List[MemoryEntry]:
        """
        Search for entries with similar target values.
        
        Args:
            target_value: Target value to search for
            tolerance: Tolerance for matching
            k: Maximum number of results
            
        Returns:
            List of matching MemoryEntry
        """
        results = []
        for entry in self.entries:
            if abs(entry.target_value - target_value) <= tolerance:
                results.append(entry)
            if len(results) >= k:
                break
        return results
    
    def get_success_cases(self, k: int = 10) -> List[MemoryEntry]:
        """Get recent successful cases."""
        success_entries = [e for e in self.entries if e.is_success]
        return success_entries[-k:]
    
    def get_failure_cases(self, k: int = 10) -> List[MemoryEntry]:
        """Get recent failed cases."""
        failure_entries = [e for e in self.entries if not e.is_success]
        return failure_entries[-k:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total = len(self.entries)
        success = sum(1 for e in self.entries if e.is_success)
        failure = total - success
        
        return {
            "total_entries": total,
            "success_count": success,
            "failure_count": failure,
            "success_rate": success / total if total > 0 else 0.0,
            "faiss_index_size": self.index.ntotal,
        }
    
    def save(self):
        """Save FAISS index and metadata."""
        # Save FAISS index
        if self.index.ntotal > 0:
            faiss.write_index(self.index, str(self.index_path))
        
        # Save metadata
        with open(self.meta_path, 'w') as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2)
        
        # Save ID mapping
        with open(self.idmap_path, 'w') as f:
            json.dump(self.id_to_index, f, indent=2)
    
    def load(self):
        """Load FAISS index and metadata."""
        # Load FAISS index
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        
        # Load metadata
        if self.meta_path.exists():
            with open(self.meta_path, 'r') as f:
                data = json.load(f)
                self.entries = [MemoryEntry.from_dict(d) for d in data]
        
        # Load ID mapping
        if self.idmap_path.exists():
            with open(self.idmap_path, 'r') as f:
                self.id_to_index = {k: int(v) for k, v in json.load(f).items()}
    
    def clear(self):
        """Clear all data."""
        self.entries = []
        self.id_to_index = {}
        self.index = self._create_index(100)
        
        # Remove files
        for path in [self.index_path, self.meta_path, self.idmap_path]:
            if path.exists():
                path.unlink()


# For testing
def test_memory_store():
    """Test memory store functionality."""
    import tempfile
    import shutil
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create store
        store = MemoryStore(temp_dir)
        
        # Create test entries
        from datetime import datetime
        
        entry1 = MemoryEntry(
            composition="Li2O",
            target_value=-3.0,
            predicted_value=-2.95,
            is_success=True,
            distance_to_target=0.05,
        )
        
        entry2 = MemoryEntry(
            composition="NaCl",
            target_value=-3.0,
            predicted_value=-2.0,
            is_success=False,
            distance_to_target=1.0,
            failure_reason="Distance too large",
        )
        
        entry3 = MemoryEntry(
            composition="LiFeO2",
            target_value=-3.0,
            predicted_value=-3.02,
            is_success=True,
            distance_to_target=0.02,
        )
        
        # Add entries
        store.add(entry1)
        store.add(entry2)
        store.add(entry3)
        
        print(f"Added {len(store.entries)} entries")
        print(f"Statistics: {store.get_statistics()}")
        
        # Search
        result = store.search("Li2O", k=4)
        print(f"\nSearch results for 'Li2O':")
        print(f"Success cases: {len(result.success_cases)}")
        print(f"Failure cases: {len(result.failure_cases)}")
        
        # Save and reload
        store.save()
        
        store2 = MemoryStore(temp_dir)
        print(f"\nAfter reload: {len(store2.entries)} entries")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_memory_store()
