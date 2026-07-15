import ast
import re

from .util.prompt import (
    instruct_target,
    instruct_output_format_one,
)
from .util.prompt import SYSTEM_PROMPT_ONE
from .util.parse import remove_subscripts


class Proposer:
    def __init__(self, target_val, target_prompt):
        self.target_val = target_val
        self.target_prompt = target_prompt
        self.system_prompt_one = SYSTEM_PROMPT_ONE

    def propose_one(
        self, prev_guess, feedback, file=None, additional_prompt=""
    ):
        prompt = self.target_prompt + instruct_target(prev_guess, feedback)
        prompt += instruct_output_format_one()
        prompt += additional_prompt
        next_guess = self.generate(self.system_prompt_one, prompt)
        next_guess = remove_subscripts(next_guess)
        next_guess = self.extract_outputs(next_guess)

        if file is not None:
            file.write("# Prompt:\n")
            file.write(prompt + "\n\n")
            file.write("# Next guess:\n")
            file.write(str(next_guess) + "\n\n")

        return next_guess

    def extract_outputs(self, response):
        if response is None:
            return {"composition": None}
        
        try:
            # Method 1: Find {"composition": "..."} pattern directly (most reliable)
            # Find all occurrences and pick the one with a valid composition (not empty)
            all_comp_matches = re.findall(r'"composition"\s*:\s*"([^"]+)"', response)
            for comp in all_comp_matches:
                if comp and comp.strip() and len(comp.strip()) > 1:
                    return self._parse_dict({"composition": comp.strip()})
            
            # Method 2: Find any dict with composition key
            matches = re.findall(r'\{.*?\}', response, re.DOTALL)
            for match in matches:
                try:
                    parsed_dict = ast.literal_eval(match)
                    if isinstance(parsed_dict, dict) and "composition" in parsed_dict:
                        comp = parsed_dict.get("composition")
                        if comp and str(comp).strip() and len(str(comp).strip()) > 1:
                            return self._parse_dict(parsed_dict)
                except:
                    continue
            
            # Method 3: Try to extract JSON from markdown code blocks (least reliable)
            code_block_patterns = [
                r'```json\s*\n?(.*?)\n?```',
                r'```python\s*\n?(.*?)\n?```',
                r'```\s*\n?(.*?)\n?```',
            ]
            
            for pattern in code_block_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                if matches:
                    content = matches[-1].strip()
                    # Try to find {"composition": "..."} pattern
                    comp_matches = re.findall(r'"composition"\s*:\s*"([^"]+)"', content)
                    for comp in comp_matches:
                        if comp and comp.strip() and len(comp.strip()) > 1:
                            return self._parse_dict({"composition": comp.strip()})
                    # Try literal_eval
                    try:
                        parsed_dict = ast.literal_eval(content)
                        if isinstance(parsed_dict, dict) and "composition" in parsed_dict:
                            comp = parsed_dict.get("composition")
                            if comp and str(comp).strip() and len(str(comp).strip()) > 1:
                                return self._parse_dict(parsed_dict)
                    except:
                        pass
                    # Try to find dict inside content
                    dict_matches = re.findall(r'\{.*?\}', content, re.DOTALL)
                    for dm in dict_matches:
                        try:
                            parsed_dict = ast.literal_eval(dm)
                            if isinstance(parsed_dict, dict) and "composition" in parsed_dict:
                                comp = parsed_dict.get("composition")
                                if comp and str(comp).strip() and len(str(comp).strip()) > 1:
                                    return self._parse_dict(parsed_dict)
                        except:
                            continue
            
            # Method 4: Look for composition = "..." pattern (Python code)
            comp_assign = re.search(r'composition\s*=\s*["\']([^"\']+)["\']', response)
            if comp_assign:
                comp = comp_assign.group(1)
                if comp and comp.strip() and len(comp.strip()) > 1:
                    return self._parse_dict({"composition": comp.strip()})
            
            # No composition found
            print(f"[Warning] No composition found in response: {response[:200]}...")
            return {"composition": None}
            
        except Exception as e:
            print(f"[Warning] Failed to parse response: {e}")
            print(f"[Warning] Response: {response[:200]}...")
            return {"composition": None}
    
    def _parse_dict(self, parsed_dict):
        """Parse dictionary and extract composition only."""
        composition = parsed_dict.get("composition", None)
        
        # Clean composition string
        if composition is not None:
            composition = str(composition).strip()
            # Remove any extra whitespace or newlines
            composition = re.sub(r'\s+', '', composition)
            # If composition becomes empty after cleaning, set to None
            if not composition:
                composition = None
        
        return {"composition": composition}
