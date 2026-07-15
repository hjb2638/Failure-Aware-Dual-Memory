"""
LLM-based explanation generation for memory entries.

Produces short materials-science-oriented explanations describing why a
composition likely succeeded or missed the target, plus a suggested direction
for improvement when the attempt failed.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from .proposer import Proposer


REASON_SYSTEM_PROMPT = """
You are a materials scientist analyzing inorganic crystal compositions for
formation-energy-targeted design. Explain outcomes using domain-aware language:
chemical bonding, charge balance, coordination chemistry, electronegativity,
anion/cation chemistry, and composition trends. Be cautious and frame claims as
plausible hypotheses rather than certainties.
"""


class MemoryReasonGenerator:
    """Generate a short reason string for storing in memory entries."""

    def __init__(self, base_proposer: Proposer, max_words: int = 90):
        self.base_proposer = base_proposer
        self.max_words = max_words

    def generate_reason(
        self,
        composition: str,
        predicted_value: float,
        target_value: float,
        is_success: bool,
        feedback: str,
        distance_to_target: float,
        previous_composition: Optional[str] = None,
        all_predictions: Optional[Sequence[float]] = None,
    ) -> str:
        """Return a concise explanation suitable for memory storage."""
        prompt = self._build_prompt(
            composition=composition,
            predicted_value=predicted_value,
            target_value=target_value,
            is_success=is_success,
            feedback=feedback,
            distance_to_target=distance_to_target,
            previous_composition=previous_composition,
            all_predictions=all_predictions,
        )

        try:
            raw_reason = self.base_proposer.generate(REASON_SYSTEM_PROMPT, prompt)
            cleaned_reason = self._clean_reason(raw_reason)
            if cleaned_reason:
                return cleaned_reason
        except Exception as exc:
            print(f"[Warning] Failed to generate memory reason with LLM: {exc}")

        return self._fallback_reason(
            composition=composition,
            predicted_value=predicted_value,
            target_value=target_value,
            is_success=is_success,
            distance_to_target=distance_to_target,
        )

    def _build_prompt(
        self,
        composition: str,
        predicted_value: float,
        target_value: float,
        is_success: bool,
        feedback: str,
        distance_to_target: float,
        previous_composition: Optional[str],
        all_predictions: Optional[Sequence[float]],
    ) -> str:
        status = "success" if is_success else "failure"
        direction = (
            "The prediction is too high (insufficiently stable). Suggest how to lower the formation energy."
            if predicted_value > target_value
            else "The prediction is too low (over-stabilized). Suggest how to raise the formation energy slightly while staying chemically plausible."
        )
        prediction_summary = "N/A"
        if all_predictions:
            prediction_summary = ", ".join(f"{float(val):.3f}" for val in all_predictions[:8])

        instructions = (
            "Write 2-3 sentences explaining why this composition likely lands near the target. "
            "Mention likely chemistry patterns that help."
            if is_success
            else "Write 2-3 sentences explaining why this composition likely misses the target and propose one chemically plausible direction for improvement."
        )

        previous_line = (
            f"Previous composition before this step: {previous_composition}\n"
            if previous_composition
            else ""
        )

        return (
            f"Target formation energy: {target_value:.3f} eV/atom\n"
            f"Predicted formation energy: {predicted_value:.3f} eV/atom\n"
            f"Absolute distance to target: {distance_to_target:.3f} eV/atom\n"
            f"Outcome label: {status}\n"
            f"Composition: {composition}\n"
            f"{previous_line}"
            f"Evaluator feedback:\n{feedback}\n\n"
            f"Representative structure predictions: {prediction_summary}\n\n"
            f"{direction}\n"
            f"{instructions}\n"
            f"Keep the answer under {self.max_words} words. Do not use bullets, markdown, JSON, or code fences."
        )

    def _clean_reason(self, reason: str) -> str:
        if reason is None:
            return ""
        cleaned = reason.strip()
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.replace("\n", " ").strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned[:700]

    def _fallback_reason(
        self,
        composition: str,
        predicted_value: float,
        target_value: float,
        is_success: bool,
        distance_to_target: float,
    ) -> str:
        if is_success:
            return (
                f"{composition} lands close to the target formation energy, suggesting its element mix and stoichiometry provide a plausible balance of bonding and charge compensation for this objective."
            )
        direction = (
            "lower the formation energy by introducing more stabilizing chemistry or stronger anion-cation bonding"
            if predicted_value > target_value
            else "raise the formation energy slightly by reducing over-stabilizing chemistry while preserving chemical plausibility"
        )
        return (
            f"{composition} misses the target by {distance_to_target:.3f} eV/atom. A reasonable next step is to {direction}."
        )
