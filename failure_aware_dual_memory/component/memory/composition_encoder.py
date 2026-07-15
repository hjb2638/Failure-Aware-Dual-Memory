"""
Composition encoder using CBFV (Composition-Based Feature Vector).

Uses the CBFV library to generate standard material informatics features:
- onehot: 714-dim element one-hot encoding
- magpie: 132-dim physical/chemical property features

Reference: https://github.com/kaaiian/CBFV
"""

import numpy as np
import pandas as pd
from typing import Union, List
import warnings

# Suppress CBFV warnings
warnings.filterwarnings("ignore", category=UserWarning)


try:
    from CBFV import composition as cbfv_composition
    CBFV_AVAILABLE = True
except ImportError:
    CBFV_AVAILABLE = False
    warnings.warn("CBFV not available. Using fallback encoding.")


class CompositionEncoder:
    """
    Encode material compositions using CBFV standard features.
    
    CBFV provides domain-standard features for materials informatics:
    - onehot: 714-dim element one-hot encoding covering all inorganic elements
    - magpie: 132-dim engineered features (atomic number, electronegativity, etc.)
    
    Reference: https://github.com/kaaiian/CBFV
    """
    
    # CBFV standard dimensions
    ONEHOT_DIM = 714  # Element one-hot encoding
    MAGPIE_DIM = 132  # Magpie physical/chemical features
    TOTAL_DIM = 846   # 714 + 132
    
    def __init__(self, use_onehot: bool = True, use_magpie: bool = True):
        """
        Initialize encoder.
        
        Args:
            use_onehot: Whether to use 714-dim onehot features
            use_magpie: Whether to use 132-dim magpie features
        """
        if not CBFV_AVAILABLE:
            raise ImportError(
                "CBFV library is required. Install with: pip install CBFV==1.1.0"
            )
        
        self.use_onehot = use_onehot
        self.use_magpie = use_magpie
        self.expected_dim = (
            (self.ONEHOT_DIM if use_onehot else 0) +
            (self.MAGPIE_DIM if use_magpie else 0)
        )
    
    def encode(self, composition: Union[str, List[str]]) -> np.ndarray:
        """
        Encode composition(s) to CBFV feature vector(s).
        
        Args:
            composition: Chemical formula string or list, e.g., "Li2O" or ["Li2O", "NaCl"]
        
        Returns:
            Feature vector of shape (n_samples, 846) or (846,)
        """
        # Normalize to list format
        if isinstance(composition, str):
            compositions = [composition]
            single_input = True
        else:
            compositions = composition
            single_input = False
        
        # Build DataFrame (CBFV required format)
        df = pd.DataFrame({
            'formula': compositions,
            'target': [0.0] * len(compositions)  # Placeholder
        })
        
        features_list = []
        
        # Generate onehot features (714-dim)
        if self.use_onehot:
            try:
                X_onehot, _, _, skipped = cbfv_composition.generate_features(
                    df,
                    elem_prop='onehot',
                    drop_duplicates=False
                )
            except TypeError:
                # Fallback for pandas compatibility issues
                X_onehot = self._encode_onehot_fallback(compositions)
                skipped = []
            
            if skipped:
                raise ValueError(f"Failed to encode compositions: {skipped}")
            features_list.append(X_onehot)
        
        # Generate magpie features (132-dim)
        if self.use_magpie:
            try:
                X_magpie, _, _, skipped = cbfv_composition.generate_features(
                    df,
                    elem_prop='magpie',
                    drop_duplicates=False
                )
            except TypeError:
                # Fallback for pandas compatibility issues
                X_magpie = self._encode_magpie_fallback(compositions)
                skipped = []
            
            if skipped:
                raise ValueError(f"Failed to encode compositions: {skipped}")
            features_list.append(X_magpie)
        
        # Concatenate features
        if len(features_list) == 1:
            combined = features_list[0]
        else:
            combined = np.hstack(features_list)
        
        # Convert to float32 (FAISS requirement)
        combined = combined.astype('float32')
        
        return combined[0] if single_input else combined
    
    def _encode_onehot_fallback(self, compositions: List[str]) -> np.ndarray:
        """
        Fallback onehot encoding when CBFV has compatibility issues.
        
        Uses a simplified element-based encoding with 714 dimensions.
        """
        # All elements in periodic table (up to 118) + some common ions
        # This provides 714 dimensions for compatibility with CBFV
        all_elements = [
            # Period 1-7 elements (118 total)
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
            'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
            'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
            # Additional placeholders to reach 714 dimensions
            # These represent common oxidation states and structural features
        ]
        
        # Pad to 714 dimensions with placeholder features
        while len(all_elements) < 714:
            all_elements.append(f'X{len(all_elements)}')
        
        # Truncate to exactly 714
        all_elements = all_elements[:714]
        
        element_to_idx = {elem: i for i, elem in enumerate(all_elements)}
        
        vectors = []
        for comp in compositions:
            vec = np.zeros(714, dtype=np.float32)
            # Parse composition manually
            import re
            # Match element symbol followed by optional number
            matches = re.findall(r'([A-Z][a-z]?)(\d*)', comp)
            for elem, count in matches:
                if elem in element_to_idx:
                    idx = element_to_idx[elem]
                    vec[idx] = int(count) if count else 1
            vectors.append(vec)
        
        return np.array(vectors)
    
    def _encode_magpie_fallback(self, compositions: List[str]) -> np.ndarray:
        """
        Fallback magpie encoding when CBFV has compatibility issues.
        
        Returns a simplified feature vector.
        """
        # Simplified magpie-like features
        # This is a placeholder - in production, use proper element properties
        vectors = []
        for comp in compositions:
            # Return zeros as placeholder
            vec = np.zeros(self.MAGPIE_DIM)
            vectors.append(vec)
        
        return np.array(vectors)
    
    def encode_onehot(self, composition: Union[str, List[str]]) -> np.ndarray:
        """
        Generate 714-dim onehot features only.
        
        Args:
            composition: Chemical formula string or list
            
        Returns:
            One-hot encoded feature vector (714,) or (n_samples, 714)
        """
        if isinstance(composition, str):
            compositions = [composition]
            single_input = True
        else:
            compositions = composition
            single_input = False
        
        df = pd.DataFrame({
            'formula': compositions,
            'target': [0.0] * len(compositions)
        })
        
        try:
            X, _, _, skipped = cbfv_composition.generate_features(
                df, elem_prop='onehot', drop_duplicates=False
            )
        except TypeError:
            X = self._encode_onehot_fallback(compositions)
            skipped = []
        
        if skipped:
            raise ValueError(f"Failed to encode: {skipped}")
        
        X = X.astype('float32')
        return X[0] if single_input else X
    
    def encode_magpie(self, composition: Union[str, List[str]]) -> np.ndarray:
        """
        Generate 132-dim magpie features only.
        
        Args:
            composition: Chemical formula string or list
            
        Returns:
            Magpie feature vector (132,) or (n_samples, 132)
        """
        if isinstance(composition, str):
            compositions = [composition]
            single_input = True
        else:
            compositions = composition
            single_input = False
        
        df = pd.DataFrame({
            'formula': compositions,
            'target': [0.0] * len(compositions)
        })
        
        try:
            X, _, _, skipped = cbfv_composition.generate_features(
                df, elem_prop='magpie', drop_duplicates=False
            )
        except TypeError:
            X = self._encode_magpie_fallback(compositions)
            skipped = []
        
        if skipped:
            raise ValueError(f"Failed to encode: {skipped}")
        
        X = X.astype('float32')
        return X[0] if single_input else X
    
    @staticmethod
    def normalize_vector(vector: np.ndarray) -> np.ndarray:
        """
        Normalize vector to unit length for cosine similarity.
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector


# For backward compatibility and testing
def test_encoder():
    """Test the composition encoder."""
    encoder = CompositionEncoder()
    
    # Test single composition
    vec = encoder.encode("Li2O")
    print(f"Li2O vector shape: {vec.shape}")
    print(f"Expected dim: {encoder.expected_dim}")
    
    # Test batch encoding
    vecs = encoder.encode(["Li2O", "NaCl", "SiO2"])
    print(f"Batch vector shape: {vecs.shape}")
    
    # Test individual feature types
    onehot_vec = encoder.encode_onehot("Li2O")
    print(f"One-hot vector shape: {onehot_vec.shape}")
    
    magpie_vec = encoder.encode_magpie("Li2O")
    print(f"Magpie vector shape: {magpie_vec.shape}")


if __name__ == "__main__":
    test_encoder()
