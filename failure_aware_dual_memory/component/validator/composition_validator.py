"""
Composition validator using pymatgen.

Validates chemical formula format, element validity, and basic constraints.
"""

from typing import Tuple, Optional, List
import re

try:
    from pymatgen.core.composition import Composition
    from pymatgen.core.periodic_table import DummySpecie, DummySpecies
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False


class CompositionValidator:
    """
    Chemical formula validator using pymatgen.
    
    Performs the following checks:
    1. Basic format validation (no decimals, valid characters)
    2. pymatgen parsing validation
    3. Atom count constraints
    4. Element validity (no dummy/virtual elements)
    """
    
    def __init__(self, max_natoms: int = 34):
        """
        Initialize validator.
        
        Args:
            max_natoms: Maximum number of atoms allowed in formula
        """
        if not PYMATGEN_AVAILABLE:
            raise ImportError(
                "pymatgen is required. Install with: pip install pymatgen"
            )
        
        self.max_natoms = max_natoms
        
        # Valid element symbols pattern
        self.element_pattern = re.compile(r'^[A-Z][a-z]*$')
    
    def validate(self, composition: str) -> Tuple[bool, Optional[str]]:
        """
        Validate chemical formula.
        
        Args:
            composition: Chemical formula string, e.g., "Li2O"
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if valid, False otherwise
            - error_message: Description of error if invalid, None if valid
        """
        # 1. Basic format checks
        if composition is None:
            return False, "Composition is None."
        
        if not isinstance(composition, str):
            return False, f"Composition must be string, got {type(composition)}."
        
        composition = composition.strip()
        
        if not composition:
            return False, "Composition is empty."
        
        # Check for decimal numbers
        if "." in composition:
            return False, f"Decimal numbers not allowed: {composition}"
        
        # Check for invalid characters
        if not re.match(r'^[A-Za-z0-9]+$', composition):
            return False, f"Invalid characters in composition: {composition}"
        
        # 2. pymatgen parsing check
        try:
            comp = Composition(composition)
        except Exception as e:
            return False, f"Invalid composition format: {composition} (Error: {str(e)})"
        
        # 3. Atom count check
        if comp.num_atoms > self.max_natoms:
            return False, (
                f"Too many atoms ({comp.num_atoms} > {self.max_natoms}): "
                f"{composition}"
            )
        
        # 4. Check for dummy/virtual elements
        for elem in comp.elements:
            elem_str = str(elem)
            if elem_str.startswith("X") or elem_str.startswith("D"):
                return False, f"Invalid element (dummy/species): {elem}"
        
        return True, None
    
    def validate_batch(
        self,
        compositions: List[str]
    ) -> List[Tuple[bool, Optional[str]]]:
        """
        Validate multiple compositions in batch.
        
        Args:
            compositions: List of chemical formula strings
            
        Returns:
            List of (is_valid, error_message) tuples
        """
        return [self.validate(comp) for comp in compositions]
    
    def get_composition_info(self, composition: str) -> Optional[dict]:
        """
        Get detailed composition information.
        
        Args:
            composition: Chemical formula string
            
        Returns:
            Dictionary with composition info, or None if invalid
        """
        is_valid, error = self.validate(composition)
        if not is_valid:
            return None
        
        try:
            comp = Composition(composition)
            
            return {
                "formula": composition,
                "formula_pretty": comp.formula,
                "reduced_formula": comp.reduced_formula,
                "num_atoms": comp.num_atoms,
                "num_elements": len(comp.elements),
                "elements": [str(e) for e in comp.elements],
                "element_fractions": {str(e): comp.get_atomic_fraction(e) 
                                      for e in comp.elements},
                "weight": comp.weight,
            }
        except Exception:
            return None
    
    def is_oxide(self, composition: str) -> bool:
        """
        Check if composition is an oxide.
        
        Args:
            composition: Chemical formula string
            
        Returns:
            True if composition contains oxygen
        """
        is_valid, _ = self.validate(composition)
        if not is_valid:
            return False
        
        try:
            comp = Composition(composition)
            return any(str(e) == "O" for e in comp.elements)
        except Exception:
            return False
    
    def contains_element(self, composition: str, element: str) -> bool:
        """
        Check if composition contains a specific element.
        
        Args:
            composition: Chemical formula string
            element: Element symbol to check
            
        Returns:
            True if composition contains the element
        """
        is_valid, _ = self.validate(composition)
        if not is_valid:
            return False
        
        try:
            comp = Composition(composition)
            return any(str(e) == element for e in comp.elements)
        except Exception:
            return False
    
    def get_element_count(self, composition: str) -> int:
        """
        Get number of unique elements in composition.
        
        Args:
            composition: Chemical formula string
            
        Returns:
            Number of unique elements
        """
        is_valid, _ = self.validate(composition)
        if not is_valid:
            return 0
        
        try:
            comp = Composition(composition)
            return len(comp.elements)
        except Exception:
            return 0


# For testing
def test_validator():
    """Test the composition validator."""
    validator = CompositionValidator(max_natoms=34)
    
    test_cases = [
        ("Li2O", True),
        ("NaCl", True),
        ("SiO2", True),
        ("LiFePO4", True),
        ("CaTiO3", True),
        ("", False),  # Empty
        ("Li2.O", False),  # Decimal
        ("Invalid123", False),  # Invalid format
        ("Li100O50", False),  # Too many atoms
    ]
    
    print("Testing CompositionValidator:")
    for comp, expected in test_cases:
        is_valid, error = validator.validate(comp)
        status = "PASS" if is_valid == expected else "FAIL"
        print(f"  [{status}] {comp}: valid={is_valid}, expected={expected}")
        if error:
            print(f"         Error: {error}")
    
    # Test info extraction
    print("\nComposition info for 'Li2O':")
    info = validator.get_composition_info("Li2O")
    if info:
        for key, value in info.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    test_validator()
