GENERAL_SYSTEM_PROMPT = """
You are an expert in materials science who suggests compositions of crystal structures."""

SYSTEM_PROMPT = """
You are an expert in materials science who suggests compositions of crystal structures.
"Composition" describes the types and proportions of elements in a material.

Your final answer must be given as a Python dictionary in the following format:
{"composition": $COMPOSITION}
"""

SYSTEM_PROMPT_ONE = """
You are an expert in materials science who suggests compositions of crystal structures.
"Composition" describes the types and proportions of elements in a material.

Your final answer must be given as a Python dictionary in the following format:
{"composition": $COMPOSITION}
"""


# ==================== Memory-Enhanced Prompts ====================

MEMORY_CONTEXT_PROMPT = """
Based on your previous attempts and similar cases from memory, here are relevant references:

{memory_context}

Use these examples to guide your next proposal. Learn from both successful and failed attempts.
Consider the patterns that led to success and avoid the mistakes that caused failures.
"""

SUCCESS_CASES_PROMPT = """
=== Reference Successful Cases ===
{success_cases}

These compositions achieved similar targets. Analyze their patterns:
- What elements are commonly used?
- What stoichiometric ratios work well?
- Apply similar strategies to your proposal.
"""

FAILURE_CASES_PROMPT = """
=== Learn from Failed Cases ===
{failure_cases}

These compositions failed to achieve the target. Understand why:
- What made these compositions unsuitable?
- How far were they from the target?
- Avoid similar mistakes in your proposal.
"""

SIMILAR_COMPOSITIONS_PROMPT = """
=== Similar Compositions from Previous Iterations ===
{similar_compositions}

These are chemically similar to your current attempt. Review their outcomes carefully.
"""

FAILURE_FEEDBACK_PROMPT = """
=== Failure Analysis ===
Your previous attempt did not achieve the target. Here is the detailed analysis:

Previous Composition: {composition}
Predicted Formation Energy: {predicted_value:.3f} eV/atom
Target Formation Energy: {target_value:.3f} eV/atom
Distance to Target: {distance:.3f} eV/atom

Failure Reason: {failure_reason}

Consider why this attempt failed and how to improve.
"""


def instruct_target(prev_guess, feedback):
    prompt = f"""
Your objective is to propose a new material composition that achieves the desired property by exploring the materials space across the periodic table.
In the previous step, you suggested the composition {prev_guess["composition"]}, and got the following feedback:
{feedback}
"""
    return prompt


def instruct_output_format_one():
    prompt = """
Based on the feedback, propose a new material composition to better achieve the desired property.
If previous suggestions are not successful enough, you may need to consider other material systems.

Your final answer must be given as a Python dictionary in the following format:
{"composition": $COMPOSITION}
Here are some requirements:
$COMPOSITION should be composed only of element symbols and digits, and should not include decimal numbers or symbols such as '-', '', '.', '{', '}', etc.
"""
    return prompt


def instruct_output_format(chars=300):
    prompt = """
Based on the above information, propose a new material composition to better achieve the desired property.
If previous suggestions are not successful enough, you may need to consider other material systems.

Your final answer must be given as a Python dictionary in the following format:
{"composition": $COMPOSITION}
Here are some requirements:
$COMPOSITION should be composed only of element symbols and digits, and should not include decimal numbers or symbols such as '-', '', '.', '{', '}', etc.
"""
    return prompt


def instruct_simple_output_format():
    prompt = """
Your answer must be given as a Python dictionary in the following format:
{"composition": $COMPOSITION}
Here is the requirements:"""

    prompt += """
$COMPOSITION should be composed only of element symbols and digits, and should not include decimal numbers or symbols such as '-', '', '.', '{', '}', etc.
"""
    return prompt


# ==================== Memory-Enhanced Prompt Builders ====================

def format_memory_entry(entry: dict, index: int = 1) -> str:
    """
    Format a memory entry for prompt display.
    
    Args:
        entry: Memory entry dictionary
        index: Entry number for display
        
    Returns:
        Formatted string
    """
    status = "SUCCESS" if entry.get("is_success", False) else "FAILED"
    composition = entry.get("composition", "Unknown")
    predicted = entry.get("predicted_value", "N/A")
    target = entry.get("target_value", "N/A")
    distance = entry.get("distance_to_target", "N/A")
    
    if status == "SUCCESS":
        reason = entry.get("reason_summary") or entry.get("failure_reason")
        return (
            f"{index}. [{status}] {composition} | "
            f"Predicted: {predicted:.3f} eV/atom | "
            f"Target: {target:.3f} eV/atom"
            + (f" | Why it helped: {reason}" if reason else "")
        )
    else:
        reason = (
            entry.get("reason_summary")
            or entry.get("failure_reason")
            or "Unknown"
        )
        return (
            f"{index}. [{status}] {composition} | "
            f"Predicted: {predicted:.3f} eV/atom | "
            f"Distance: {distance:.3f} eV/atom | "
            f"Reason: {reason}"
        )


def build_enhanced_prompt(
    target_value: float,
    prev_guess: dict,
    feedback: str,
    memory_context: dict = None,
    is_failure: bool = False,
    additional_prompt: str = ""
) -> str:
    """
    Build an enhanced prompt with memory context.
    
    Args:
        target_value: Target formation energy
        prev_guess: Previous guess dictionary with 'composition' key
        feedback: Feedback string from evaluator
        memory_context: Dictionary with 'success_cases', 'failure_cases', 'similar_compositions'
        is_failure: Whether the previous attempt failed
        additional_prompt: Additional prompt text
        
    Returns:
        Complete prompt string
    """
    # Base target information
    prompt_parts = [
        f"I am looking to design a material with a formation energy per atom of {target_value} eV/atom.",
        ""
    ]
    
    # Memory context
    if memory_context:
        # Add successful cases
        success_cases = memory_context.get("success_cases", [])
        if success_cases:
            prompt_parts.append("=== Reference Successful Cases ===")
            for i, entry in enumerate(success_cases[:3], 1):
                prompt_parts.append(format_memory_entry(entry, i))
            prompt_parts.append("")
            prompt_parts.append(
                "Analyze these successful patterns and apply similar strategies."
            )
            prompt_parts.append("")
        
        # Add failed cases
        failure_cases = memory_context.get("failure_cases", [])
        if failure_cases:
            prompt_parts.append("=== Learn from Failed Cases ===")
            for i, entry in enumerate(failure_cases[:3], 1):
                prompt_parts.append(format_memory_entry(entry, i))
            prompt_parts.append("")
            prompt_parts.append(
                "Understand why these failed and avoid similar mistakes."
            )
            prompt_parts.append("")
    
    # Previous attempt feedback
    prompt_parts.append("=== Previous Attempt ===")
    prompt_parts.append(f"Composition: {prev_guess.get('composition', 'Unknown')}")
    prompt_parts.append(f"Feedback: {feedback}")
    prompt_parts.append("")
    
    # Failure analysis
    if is_failure and memory_context and "failure_analysis" in memory_context:
        analysis = memory_context["failure_analysis"]
        prompt_parts.append("=== Failure Analysis ===")
        prompt_parts.append(
            f"Previous Composition: {analysis.get('composition', 'Unknown')}"
        )
        prompt_parts.append(
            f"Predicted Formation Energy: {analysis.get('predicted_value', 0):.3f} eV/atom"
        )
        prompt_parts.append(
            f"Target Formation Energy: {analysis.get('target_value', 0):.3f} eV/atom"
        )
        prompt_parts.append(
            f"Distance to Target: {analysis.get('distance', 0):.3f} eV/atom"
        )
        prompt_parts.append("")
    
    # Output requirements
    prompt_parts.append("=== Your Task ===")
    prompt_parts.append(
        "Based on all the above information, propose a new material composition "
        "to better achieve the target property."
    )
    prompt_parts.append("")
    prompt_parts.append(
        "Your final answer must be given as a Python dictionary in the following format:"
    )
    prompt_parts.append('{"composition": $COMPOSITION}')
    prompt_parts.append("")
    prompt_parts.append("Requirements:")
    prompt_parts.append("- $COMPOSITION should be composed only of element symbols and digits")
    prompt_parts.append("- Do not use decimal numbers or special symbols")
    prompt_parts.append("- Consider the patterns from successful cases")
    prompt_parts.append("- Avoid the mistakes seen in failed cases")
    
    if additional_prompt:
        prompt_parts.append("")
        prompt_parts.append(f"Additional constraints: {additional_prompt}")
    
    return "\n".join(prompt_parts)
