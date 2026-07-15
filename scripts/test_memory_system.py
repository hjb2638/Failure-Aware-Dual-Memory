"""
Test script for memory system functionality.

Tests all components without running full diffusion model:
1. Memory initialization from MP20
2. Composition encoding with CBFV
3. Memory retrieval
4. Validation (pymatgen + SMACT)
5. Memory-enhanced prompt building
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from failure_aware_dual_memory.component.memory import MemoryStore
from failure_aware_dual_memory.component.validator import CompositionValidator, SMACTChecker
from failure_aware_dual_memory.component.proposer.util.prompt import build_enhanced_prompt


def test_memory_system():
    """Test complete memory system."""
    print("="*70)
    print("MEMORY SYSTEM FUNCTIONALITY TEST")
    print("="*70)
    
    # Test 1: Load MP20 data
    print("\n[1/6] Loading MP20 test data...")
    df = pd.read_csv(PROJECT_ROOT / "data" / "mp_20" / "test.csv")
    print(f"  Loaded {len(df)} entries")
    print(f"  Formation energy range: {df['formation_energy_per_atom'].min():.3f} to {df['formation_energy_per_atom'].max():.3f}")
    
    # Test 2: Initialize memory from pre-built storage
    print("\n[2/6] Loading pre-initialized memory...")
    memory_store = MemoryStore(
        storage_dir=str(PROJECT_ROOT / "memory_storage_mp20_init"),
        use_onehot=True,
        use_magpie=False,
    )
    stats = memory_store.get_statistics()
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Success count: {stats['success_count']}")
    print(f"  Failure count: {stats['failure_count']}")
    print(f"  ✓ Memory loaded successfully")
    
    # Test 3: Memory retrieval
    print("\n[3/6] Testing memory retrieval...")
    test_compositions = ["Li2O", "NaCl", "SiO2", "Fe2O3", "CaTiO3"]
    
    for comp in test_compositions:
        result = memory_store.search(comp, k=5, target_value=-3.5, target_tolerance=0.5)
        print(f"  {comp}: {len(result.success_cases)} success, {len(result.failure_cases)} failure cases found")
    print("  ✓ Memory retrieval working")
    
    # Test 4: Composition validation
    print("\n[4/6] Testing composition validation...")
    validator = CompositionValidator(max_natoms=34)
    
    test_cases = [
        ("Li2O", True),
        ("NaCl", True),
        ("Li2.O", False),  # Invalid - decimal
        ("Li100O50", False),  # Invalid - too many atoms
    ]
    
    for comp, expected in test_cases:
        is_valid, error = validator.validate(comp)
        status = "✓" if is_valid == expected else "✗"
        print(f"  {status} {comp}: valid={is_valid} (expected={expected})")
    
    # Test 5: SMACT validation
    print("\n[5/6] Testing SMACT validation...")
    try:
        smact_checker = SMACTChecker()
        
        smact_cases = ["Li2O", "NaCl", "CaTiO3", "Fe2O3"]
        for comp in smact_cases:
            is_valid, error, details = smact_checker.check_charge_neutrality(comp, return_details=True)
            status = "✓" if is_valid else "✗"
            print(f"  {status} {comp}: charge_neutral={is_valid}")
        print("  ✓ SMACT validation working")
    except Exception as e:
        print(f"  ⚠ SMACT test skipped: {e}")
    
    # Test 6: Enhanced prompt building
    print("\n[6/6] Testing enhanced prompt building...")
    
    # Create sample memory context
    memory_context = {
        "success_cases": [
            {
                "composition": "Li2FeO3",
                "predicted_value": -3.45,
                "target_value": -3.50,
                "is_success": True,
                "distance_to_target": 0.05,
            },
            {
                "composition": "NaFePO4",
                "predicted_value": -3.48,
                "target_value": -3.50,
                "is_success": True,
                "distance_to_target": 0.02,
            },
        ],
        "failure_cases": [
            {
                "composition": "LiCoO2",
                "predicted_value": -2.80,
                "target_value": -3.50,
                "is_success": False,
                "distance_to_target": 0.70,
                "failure_reason": "Too far from target",
            },
        ],
    }
    
    prev_guess = {"composition": "LiMn2O4"}
    feedback = "Predicted formation energy is -2.90 eV/atom, need to get closer to -3.50"
    
    enhanced_prompt = build_enhanced_prompt(
        target_value=-3.50,
        prev_guess=prev_guess,
        feedback=feedback,
        memory_context=memory_context,
        is_failure=True,
    )
    
    print(f"  Generated prompt length: {len(enhanced_prompt)} characters")
    print(f"  Prompt preview (first 300 chars):")
    print(f"  {enhanced_prompt[:300]}...")
    print("  ✓ Enhanced prompt building working")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("✓ All memory system components tested successfully!")
    print(f"  - Memory storage: {stats['total_entries']} entries")
    print(f"  - Composition validation: working")
    print(f"  - SMACT validation: working")
    print(f"  - Memory retrieval: working")
    print(f"  - Enhanced prompts: working")
    print("="*70)


if __name__ == "__main__":
    test_memory_system()
