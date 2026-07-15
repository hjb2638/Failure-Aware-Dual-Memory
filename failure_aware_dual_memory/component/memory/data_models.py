"""
Data models for memory storage.

Defines the MemoryEntry dataclass for storing composition,
prediction results, and feedback information.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid


@dataclass
class MemoryEntry:
    """
    Memory entry data structure for storing composition attempts.
    
    Stores both successful and failed attempts in a unified format,
    with is_success flag to distinguish status.
    """
    
    # Unique identifier
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Iteration information
    iteration: int = 0
    init_id: int = 0
    
    # Composition information
    composition: str = ""  # Chemical formula, e.g., "Li2O"
    composition_vector: List[float] = field(default_factory=list)
    
    # Target information
    target_value: float = 0.0
    target_tolerance: float = 0.1
    
    # Prediction results
    predicted_value: Optional[float] = None
    
    # Status flag
    is_success: bool = False
    
    # Detailed results (for successful cases)
    best_structure_cif: Optional[str] = None
    all_predictions: Optional[List[float]] = None
    
    # Failure information (for failed cases)
    failure_reason: Optional[str] = None
    reason_summary: Optional[str] = None
    distance_to_target: Optional[float] = None
    
    # Context information
    previous_composition: Optional[str] = None
    feedback_given: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        import numpy as np
        
        # Helper function to convert numpy types to Python native types
        def convert_value(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, np.bool_):
                return bool(v)
            elif isinstance(v, (np.float32, np.float64)):
                return float(v)
            elif isinstance(v, (np.int32, np.int64)):
                return int(v)
            elif isinstance(v, list):
                return [convert_value(item) for item in v]
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            return v
        
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "iteration": self.iteration,
            "init_id": self.init_id,
            "composition": self.composition,
            "composition_vector": convert_value(self.composition_vector),
            "target_value": convert_value(self.target_value),
            "target_tolerance": convert_value(self.target_tolerance),
            "predicted_value": convert_value(self.predicted_value),
            "is_success": convert_value(self.is_success),
            "best_structure_cif": self.best_structure_cif,
            "all_predictions": convert_value(self.all_predictions),
            "failure_reason": self.failure_reason,
            "reason_summary": self.reason_summary,
            "distance_to_target": convert_value(self.distance_to_target),
            "previous_composition": self.previous_composition,
            "feedback_given": self.feedback_given,
            "metadata": convert_value(self.metadata),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create MemoryEntry from dictionary."""
        # Parse timestamp
        timestamp = datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now()
        
        return cls(
            entry_id=data.get("entry_id", str(uuid.uuid4())),
            timestamp=timestamp,
            iteration=data.get("iteration", 0),
            init_id=data.get("init_id", 0),
            composition=data.get("composition", ""),
            composition_vector=data.get("composition_vector", []),
            target_value=data.get("target_value", 0.0),
            target_tolerance=data.get("target_tolerance", 0.1),
            predicted_value=data.get("predicted_value"),
            is_success=data.get("is_success", False),
            best_structure_cif=data.get("best_structure_cif"),
            all_predictions=data.get("all_predictions"),
            failure_reason=data.get("failure_reason"),
            reason_summary=data.get("reason_summary"),
            distance_to_target=data.get("distance_to_target"),
            previous_composition=data.get("previous_composition"),
            feedback_given=data.get("feedback_given"),
            metadata=data.get("metadata", {}),
        )
    
    def get_summary(self) -> str:
        """Get a summary string for display."""
        status = "SUCCESS" if self.is_success else "FAILED"
        return (
            f"[{status}] {self.composition} | "
            f"Predicted: {self.predicted_value:.3f} | "
            f"Target: {self.target_value:.3f} | "
            f"Distance: {self.distance_to_target:.3f}"
        )


@dataclass
class RetrievalResult:
    """
    Result structure for memory retrieval.
    
    Contains retrieved entries separated by success status.
    """
    
    success_cases: List[MemoryEntry] = field(default_factory=list)
    failure_cases: List[MemoryEntry] = field(default_factory=list)
    similar_compositions: List[MemoryEntry] = field(default_factory=list)
    
    def is_empty(self) -> bool:
        """Check if retrieval result is empty."""
        return len(self.success_cases) == 0 and len(self.failure_cases) == 0
    
    def get_all_cases(self) -> List[MemoryEntry]:
        """Get all retrieved cases."""
        return self.success_cases + self.failure_cases
    
    def to_prompt_context(self, max_success: int = 3, max_failure: int = 3) -> str:
        """
        Convert retrieval result to prompt context string.
        
        Args:
            max_success: Maximum number of success cases to include
            max_failure: Maximum number of failure cases to include
            
        Returns:
            Formatted context string for prompt
        """
        context_parts = []
        
        # Add successful cases
        if self.success_cases:
            context_parts.append("=== Successful Cases ===")
            for i, entry in enumerate(self.success_cases[:max_success], 1):
                context_parts.append(
                    f"{i}. Composition: {entry.composition} | "
                    f"Predicted: {entry.predicted_value:.3f} eV/atom | "
                    f"Target: {entry.target_value:.3f} eV/atom"
                )
        
        # Add failed cases
        if self.failure_cases:
            context_parts.append("\n=== Failed Cases (Learn from these) ===")
            for i, entry in enumerate(self.failure_cases[:max_failure], 1):
                context_parts.append(
                    f"{i}. Composition: {entry.composition} | "
                    f"Predicted: {entry.predicted_value:.3f} eV/atom | "
                    f"Distance: {entry.distance_to_target:.3f} eV/atom | "
                    f"Reason: {entry.failure_reason or 'N/A'}"
                )
        
        return "\n".join(context_parts) if context_parts else "No relevant cases found in memory."
