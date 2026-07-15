"""
Initialize memory system from MP20 test dataset.

This script loads the MP20 test set and initializes the memory store
with the existing data for use in the Failure-Aware Dual-Memory.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from failure_aware_dual_memory.component.memory import MemoryStore, MemoryEntry
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialize memory from MP20-like dataset"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/mp_20/train.csv",
        help="Path to MP20-like CSV file",
    )
    
    parser.add_argument(
        "--memory_storage_dir",
        type=str,
        default="./memory_storage_mp20_init",
        help="Directory for memory storage",
    )
    
    parser.add_argument(
        "--max_entries",
        type=int,
        default=None,
        help="Maximum number of entries to load (None for all)",
    )
    
    parser.add_argument(
        "--target_tolerance",
        type=float,
        default=0.1,
        help="Tolerance for considering predictions successful",
    )
    
    parser.add_argument(
        "--use_onehot",
        action="store_true",
        default=True,
        help="Use onehot features (714-dim)",
    )
    
    parser.add_argument(
        "--use_magpie",
        action="store_true",
        default=False,
        help="Use magpie features (132-dim)",
    )
    
    return parser.parse_args()


def load_mp20_test_data(csv_path: str, max_entries: int = None) -> pd.DataFrame:
    """Load data from CSV."""
    print(f"Loading  data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if max_entries is not None:
        df = df.head(max_entries)
    
    print(f"Loaded {len(df)} entries")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFormation energy stats:")
    print(f"  Min: {df['formation_energy_per_atom'].min():.3f} eV/atom")
    print(f"  Max: {df['formation_energy_per_atom'].max():.3f} eV/atom")
    print(f"  Mean: {df['formation_energy_per_atom'].mean():.3f} eV/atom")
    print(f"  Std: {df['formation_energy_per_atom'].std():.3f} eV/atom")
    
    return df


def create_memory_entries(
    df: pd.DataFrame,
    target_tolerance: float = 0.1
) -> list:
    """
    Create MemoryEntry objects from data.
    
    Since data contains ground truth formation energies,
    we treat these as "successful" reference cases.
    """
    entries = []
    
    for idx, row in df.iterrows():
        composition = row['pretty_formula']
        formation_energy = row['formation_energy_per_atom']
        material_id = row['material_id']
        
        # Create memory entry
        # We treat all MP20 entries as "successful" reference data
        entry = MemoryEntry(
            entry_id=f"mp_material_id_{material_id}",
            timestamp=datetime.now(),
            iteration=0,
            init_id=0,
            composition=composition,
            composition_vector=[],  # Will be computed by MemoryStore
            target_value=formation_energy,  # Use actual formation energy as target
            target_tolerance=target_tolerance,
            predicted_value=formation_energy,  # Ground truth
            is_success=True,  # All MP20 data is considered successful
            best_structure_cif=row.get('cif', None),
            all_predictions=None,
            failure_reason=None,
            distance_to_target=0.0,  # Perfect match
            previous_composition=None,
            feedback_given=f"Ground truth from MP: {formation_energy:.3f} eV/atom",
            metadata={
                'source': 'mp20_train',
                'material_id': material_id,
                'band_gap': row.get('band_gap', None),
                'e_above_hull': row.get('e_above_hull', None),
                'spacegroup': row.get('spacegroup.number', None),
            }
        )
        entries.append(entry)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Created {idx + 1} entries...")
    
    return entries


def initialize_memory_store(
    entries: list,
    storage_dir: str,
    use_onehot: bool = True,
    use_magpie: bool = False,
) -> MemoryStore:
    """Initialize memory store with entries."""
    print(f"\nInitializing memory store at {storage_dir}...")
    
    store = MemoryStore(
        storage_dir=storage_dir,
        use_onehot=use_onehot,
        use_magpie=use_magpie,
    )
    
    print(f"Memory store initialized with dim={store.dim}")
    
    # Add entries in batch
    print(f"Adding {len(entries)} entries to memory store...")
    
    batch_size = 100
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i+batch_size]
        store.add_batch(batch)
        if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(entries):
            print(f"  Added {min(i + batch_size, len(entries))} entries...")
    
    # Save store
    store.save()
    print(f"Memory store saved to {storage_dir}")
    
    return store


def print_statistics(store: MemoryStore):
    """Print memory store statistics."""
    stats = store.get_statistics()
    print("\n" + "="*50)
    print("Memory Store Statistics")
    print("="*50)
    print(f"Total entries: {stats['total_entries']}")
    print(f"Success count: {stats['success_count']}")
    print(f"Failure count: {stats['failure_count']}")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"FAISS index size: {stats['faiss_index_size']}")
    print("="*50)


def test_memory_retrieval(store: MemoryStore):
    """Test memory retrieval with sample queries."""
    print("\n" + "="*50)
    print("Testing Memory Retrieval")
    print("="*50)
    
    test_compositions = ["Li2O", "NaCl", "SiO2", "Fe2O3", "CaTiO3"]
    
    for comp in test_compositions:
        result = store.search(comp, k=5)
        print(f"\nQuery: {comp}")
        print(f"  Success cases found: {len(result.success_cases)}")
        print(f"  Failure cases found: {len(result.failure_cases)}")
        if result.success_cases:
            print(f"  Top match: {result.success_cases[0].composition}")
            print(f"    Target: {result.success_cases[0].target_value:.3f} eV/atom")


def main():
    args = parse_args()
    
    print("="*60)
    print("Data Memory Initialization")
    print("="*60)
    
    # Load MP20 test data
    df = load_mp20_test_data(args.data_path, args.max_entries)
    
    # Create memory entries
    print("\nCreating memory entries...")
    entries = create_memory_entries(df, args.target_tolerance)
    
    # Initialize memory store
    store = initialize_memory_store(
        entries,
        args.memory_storage_dir,
        args.use_onehot,
        args.use_magpie,
    )
    
    # Print statistics
    print_statistics(store)
    
    # Test retrieval
    test_memory_retrieval(store)
    
    print("\n" + "="*60)
    print("Memory initialization complete!")
    print(f"Storage location: {args.memory_storage_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
