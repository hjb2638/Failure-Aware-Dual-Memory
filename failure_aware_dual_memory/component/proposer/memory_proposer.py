"""
Memory-enhanced proposer that integrates with FAISS memory store.

Wraps a base proposer and adds memory retrieval and enhanced prompting.
"""

from typing import Dict, Any, Optional
from .proposer import Proposer
from ..memory.memory_store import MemoryStore
from ..memory.data_models import MemoryEntry, RetrievalResult
from .reason_generator import MemoryReasonGenerator
from .util.prompt import build_enhanced_prompt


class MemoryEnhancedProposer(Proposer):
    """
    Proposer with integrated memory functionality.
    
    Wraps a base proposer and adds:
    - Memory retrieval before generation
    - Enhanced prompts with success/failure cases
    - Automatic memory storage after evaluation
    """
    
    def __init__(
        self,
        base_proposer: Proposer,
        memory_store: MemoryStore,
        success_threshold: float = 0.25,
        k_success: int = 3,
        k_failure: int = 3,
        target_tolerance: float = 0.5,
    ):
        """
        Initialize memory-enhanced proposer.
        
        Args:
            base_proposer: Base proposer to wrap
            memory_store: FAISS memory store
            success_threshold: Threshold for success (eV/atom)
            k_success: Number of success cases to retrieve
            k_failure: Number of failure cases to retrieve
            target_tolerance: Tolerance for target value matching
        """
        self.base_proposer = base_proposer
        self.memory_store = memory_store
        self.success_threshold = success_threshold
        self.k_success = k_success
        self.k_failure = k_failure
        self.target_tolerance = target_tolerance
        
        # Inherit base properties
        self.target_val = base_proposer.target_val
        self.target_prompt = base_proposer.target_prompt
        self.system_prompt_one = base_proposer.system_prompt_one
        self.reason_generator = MemoryReasonGenerator(base_proposer)
    
    def propose_one(
        self,
        prev_guess: Dict[str, Any],
        feedback: str,
        file=None,
        additional_prompt: str = "",
        iteration: int = 0,
        init_id: int = 0,
        predicted_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate new composition with memory enhancement.
        
        Args:
            prev_guess: Previous guess with 'composition' key
            feedback: Feedback from evaluator
            file: Log file
            additional_prompt: Additional prompt text
            iteration: Current iteration number
            init_id: Initialization ID
            predicted_value: Predicted value for failure detection
            
        Returns:
            Dictionary with 'composition'
        """
        # Determine if previous attempt failed
        is_failure = False
        if predicted_value is not None:
            distance = abs(predicted_value - self.target_val)
            is_failure = distance > self.success_threshold
        
        # Retrieve similar cases from memory
        composition = prev_guess.get("composition", "")
        if not composition:
            # If composition is None or empty, use a default or skip memory retrieval
            print("[Warning] No valid composition for memory search, using empty search")
            composition = ""
        
        retrieval_result = self.memory_store.search(
            composition=composition,
            k=self.k_success + self.k_failure,
            target_value=self.target_val,
            target_tolerance=self.target_tolerance,
        )
        
        # Build memory context
        memory_context = self._build_memory_context(
            retrieval_result,
            prev_guess,
            predicted_value,
            is_failure,
        )
        
        # Build enhanced prompt
        enhanced_prompt = build_enhanced_prompt(
            target_value=self.target_val,
            prev_guess=prev_guess,
            feedback=feedback,
            memory_context=memory_context,
            is_failure=is_failure,
            additional_prompt=additional_prompt,
        )
        
        # Generate using base proposer
        print(f"\n[LLM] Sending enhanced prompt to model...")
        response = self.base_proposer.generate(
            self.system_prompt_one,
            enhanced_prompt,
        )
        
        # Log raw response
        print(f"[LLM] Raw response: {response[:200]}..." if len(str(response)) > 200 else f"[LLM] Raw response: {response}")
        
        # Extract and parse output
        next_guess = self.base_proposer.extract_outputs(response)
        print(f"[LLM] Parsed output: {next_guess}")
        
        # Log if file provided
        if file is not None:
            file.write("# Enhanced Prompt:\n")
            file.write(enhanced_prompt + "\n\n")
            file.write("# Memory Context:\n")
            file.write(f"Success cases: {len(retrieval_result.success_cases)}\n")
            file.write(f"Failure cases: {len(retrieval_result.failure_cases)}\n")
            file.write(f"Similar compositions: {len(retrieval_result.similar_compositions)}\n")
            file.write("# LLM Raw Response:\n")
            file.write(str(response) + "\n\n")
            file.write("# LLM Parsed Output:\n")
            file.write(str(next_guess) + "\n\n")
        
        return next_guess
    
    def _build_memory_context(
        self,
        retrieval_result: RetrievalResult,
        prev_guess: Dict[str, Any],
        predicted_value: Optional[float],
        is_failure: bool,
    ) -> Dict[str, Any]:
        """
        Build memory context dictionary for prompt building.
        
        Args:
            retrieval_result: Retrieved cases from memory
            prev_guess: Previous guess
            predicted_value: Predicted value
            is_failure: Whether previous attempt failed
            
        Returns:
            Memory context dictionary
        """
        context = {}
        
        # Convert entries to dictionaries
        if retrieval_result.success_cases:
            context["success_cases"] = [
                entry.to_dict() for entry in retrieval_result.success_cases
            ]
        
        if retrieval_result.failure_cases:
            context["failure_cases"] = [
                entry.to_dict() for entry in retrieval_result.failure_cases
            ]
        
        if retrieval_result.similar_compositions:
            context["similar_compositions"] = [
                entry.to_dict() for entry in retrieval_result.similar_compositions
            ]
        
        # Add failure analysis if applicable
        if is_failure and predicted_value is not None:
            context["failure_analysis"] = {
                "composition": prev_guess.get("composition", ""),
                "predicted_value": predicted_value,
                "target_value": self.target_val,
                "distance": abs(predicted_value - self.target_val),
                "failure_reason": "Previous attempt missed the target and needs chemistry-aware refinement.",
            }
        
        return context
    
    def generate(self, system_prompt: str, prompt: str) -> str:
        """Delegate generation to base proposer."""
        return self.base_proposer.generate(system_prompt, prompt)
    
    def extract_outputs(self, response: str) -> Dict[str, Any]:
        """Delegate output extraction to base proposer."""
        return self.base_proposer.extract_outputs(response)
    
    def create_memory_entry(
        self,
        composition: str,
        predicted_value: float,
        iteration: int,
        init_id: int,
        feedback: str,
        all_predictions: Optional[list] = None,
        best_structure_cif: Optional[str] = None,
        previous_composition: Optional[str] = None,
    ) -> MemoryEntry:
        """
        Create a memory entry for storage.
        
        Args:
            composition: Chemical formula
            predicted_value: Predicted formation energy
            iteration: Iteration number
            init_id: Initialization ID
            feedback: Feedback string
            all_predictions: All structure predictions
            best_structure_cif: Best structure CIF
            previous_composition: Previous composition
            
        Returns:
            MemoryEntry object
        """
        distance = abs(predicted_value - self.target_val)
        is_success = distance <= self.success_threshold
        reason_summary = self.reason_generator.generate_reason(
            composition=composition,
            predicted_value=predicted_value,
            target_value=self.target_val,
            is_success=is_success,
            feedback=feedback,
            distance_to_target=distance,
            previous_composition=previous_composition,
            all_predictions=all_predictions,
        )
        
        entry = MemoryEntry(
            iteration=iteration,
            init_id=init_id,
            composition=composition,
            target_value=self.target_val,
            target_tolerance=self.success_threshold,
            predicted_value=predicted_value,
            is_success=is_success,
            all_predictions=all_predictions,
            best_structure_cif=best_structure_cif,
            distance_to_target=distance,
            failure_reason=reason_summary,
            reason_summary=reason_summary,
            feedback_given=feedback,
            previous_composition=previous_composition,
        )
        
        return entry
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self.memory_store.get_statistics()


def wrap_proposer_with_memory(
    base_proposer: Proposer,
    memory_storage_dir: str,
    success_threshold: float = 0.25,
    k_success: int = 3,
    k_failure: int = 3,
    target_tolerance: float = 0.5,
    use_onehot: bool = True,
    use_magpie: bool = False,
) -> MemoryEnhancedProposer:
    """
    Wrap a base proposer with memory functionality.
    
    Args:
        base_proposer: Base proposer to wrap
        memory_storage_dir: Directory for memory storage
        success_threshold: Success threshold (eV/atom)
        k_success: Number of success cases to retrieve
        k_failure: Number of failure cases to retrieve
        target_tolerance: Target value matching tolerance
        use_onehot: Whether to use onehot features
        use_magpie: Whether to use magpie features
        
    Returns:
        MemoryEnhancedProposer instance
    """
    memory_store = MemoryStore(
        storage_dir=memory_storage_dir,
        use_onehot=use_onehot,
        use_magpie=use_magpie,
    )
    
    return MemoryEnhancedProposer(
        base_proposer=base_proposer,
        memory_store=memory_store,
        success_threshold=success_threshold,
        k_success=k_success,
        k_failure=k_failure,
        target_tolerance=target_tolerance,
    )
