"""
SMACT checker for chemical reasonableness validation.

Uses SMACT (Semiconducting Materials by Analogy and Chemical Theory) to check
chemical reasonableness, particularly charge neutrality.

Reference: https://github.com/WMD-group/SMACT
"""

from typing import Tuple, Optional, List, Dict, Any
try:
    from pymatgen.core.composition import Composition
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

try:
    import smact
    from smact import screening
    SMACT_AVAILABLE = True
except ImportError:
    SMACT_AVAILABLE = False


class SMACTChecker:
    """
    Chemical reasonableness checker using SMACT.
    
    Performs charge neutrality checks.
    """
    
    def __init__(self):
        """Initialize SMACT checker."""
        if not PYMATGEN_AVAILABLE:
            raise ImportError(
                "pymatgen is required. Install with: pip install pymatgen"
            )
        if not SMACT_AVAILABLE:
            raise ImportError(
                "SMACT is required. Install with: pip install smact"
            )
    
    def check_charge_neutrality(
        self,
        composition: str,
        return_details: bool = False
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Check if composition can be charge neutral.
        
        Uses SMACT's Element data to check if there's a valid combination
        of oxidation states that results in charge neutrality.
        
        Args:
            composition: Chemical formula string
            return_details: Whether to return detailed information
            
        Returns:
            Tuple of (is_valid, error_message, details)
            - is_valid: True if charge-neutral combination exists
            - error_message: Description if invalid
            - details: Dictionary with details if return_details=True
        """
        try:
            comp = Composition(composition)
            elements = [str(e) for e in comp.elements]
            counts = [int(comp[e]) for e in comp.elements]
            
            # Get allowed oxidation states for each element
            oxidation_states = []
            for elem in elements:
                elem_data = smact.Element(elem)
                oxidation_states.append(list(elem_data.oxidation_states))
            
            # Check all combinations of oxidation states for charge neutrality
            from itertools import product
            allowed_combinations = []
            
            for ox_combo in product(*oxidation_states):
                # Calculate total charge
                total_charge = sum(ox * count for ox, count in zip(ox_combo, counts))
                if total_charge == 0:
                    allowed_combinations.append(ox_combo)
            
            if not allowed_combinations:
                error_msg = (
                    f"SMACT validation failed for {composition}: "
                    "No charge-neutral combination found"
                )
                details = {
                    "elements": elements,
                    "counts": counts,
                    "oxidation_states": oxidation_states,
                    "allowed_combinations": []
                } if return_details else None
                return False, error_msg, details
            
            # Valid combination found
            details = {
                "elements": elements,
                "counts": counts,
                "oxidation_states": oxidation_states,
                "allowed_combinations": allowed_combinations[:5]  # First 5
            } if return_details else None
            
            return True, None, details
            
        except Exception as e:
            error_msg = f"SMACT validation error: {str(e)}"
            return False, error_msg, None
    
    def validate(
        self,
        composition: str,
        strict: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate composition using SMACT.
        
        Args:
            composition: Chemical formula string
            strict: If True, require charge neutrality; if False, be more lenient
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        is_valid, error_msg, _ = self.check_charge_neutrality(composition)
        
        if not is_valid and not strict:
            # In non-strict mode, some failures are warnings
            if "No charge-neutral combination" in (error_msg or ""):
                return True, f"Warning: {error_msg}"
        
        return is_valid, error_msg
    
    def get_possible_oxidation_states(
        self,
        composition: str
    ) -> Dict[str, List[int]]:
        """
        Get possible oxidation states for each element in composition.
        
        Args:
            composition: Chemical formula string
            
        Returns:
            Dictionary mapping element to list of possible oxidation states
        """
        try:
            comp = Composition(composition)
            result = {}
            
            for elem in comp.elements:
                elem_str = str(elem)
                elem_data = smact.Element(elem_str)
                result[elem_str] = list(elem_data.oxidation_states)
            
            return result
            
        except Exception:
            return {}
    
    def is_likely_ionic(self, composition: str) -> bool:
        """
        Check if composition is likely to be ionic.
        
        Args:
            composition: Chemical formula string
            
        Returns:
            True if likely ionic (contains metal + non-metal)
        """
        try:
            comp = Composition(composition)
            
            # Common metals
            metals = {
                'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',  # Alkali
                'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',  # Alkaline earth
                'Al', 'Ga', 'In', 'Tl',  # Post-transition
                'Sn', 'Pb', 'Bi', 'Po',
                'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',  # Transition
                'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            }
            
            # Common non-metals (that form anions)
            non_metals = {'O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I', 'N', 'P'}
            
            elements = {str(e) for e in comp.elements}
            
            has_metal = bool(elements & metals)
            has_non_metal = bool(elements & non_metals)
            
            return has_metal and has_non_metal
            
        except Exception:
            return False
    
    def check_composition_list(
        self,
        compositions: List[str],
        strict: bool = False
    ) -> List[Tuple[bool, Optional[str]]]:
        """
        Check multiple compositions.
        
        Args:
            compositions: List of chemical formula strings
            strict: Whether to use strict validation
            
        Returns:
            List of (is_valid, error_message) tuples
        """
        return [self.validate(comp, strict=strict) for comp in compositions]


# For testing
def test_smact_checker():
    """Test the SMACT checker."""
    checker = SMACTChecker()
    
    test_cases = [
        "Li2O",      # Should pass - common oxide
        "NaCl",      # Should pass - simple salt
        "CaTiO3",    # Should pass - perovskite
        "SiO2",      # May pass or fail depending on oxidation states
        "Fe2O3",     # Should pass - common oxide
        "LiFePO4",   # Should pass - battery material
    ]
    
    print("Testing SMACTChecker:")
    for comp in test_cases:
        is_valid, error, details = checker.check_charge_neutrality(
            comp, return_details=True
        )
        status = "PASS" if is_valid else "FAIL"
        print(f"\n[{status}] {comp}")
        
        if details:
            print(f"  Elements: {details['elements']}")
            print(f"  Counts: {details['counts']}")
            print(f"  Allowed combinations: {len(details['allowed_combinations'])}")
        
        if error:
            print(f"  Error: {error}")
    
    # Test oxidation states
    print("\nOxidation states for LiFePO4:")
    ox_states = checker.get_possible_oxidation_states("LiFePO4")
    for elem, states in ox_states.items():
        print(f"  {elem}: {states}")


if __name__ == "__main__":
    test_smact_checker()
