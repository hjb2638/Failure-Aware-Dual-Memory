import random
from pathlib import Path

import pandas as pd
from pymatgen.core.structure import Structure

from failure_aware_dual_memory.component.proposer.util.prompt import (
    instruct_simple_output_format,
    GENERAL_SYSTEM_PROMPT,
)


def get_initial_guess(args, proposer, prompt, additional_prompt="", device="cuda", log_file=None):
    if args.initial_guess == "random":
        out_dict = get_init_from_random(args)
        return out_dict
    elif args.initial_guess == "llm":
        out_dict = get_init_from_llm(args, proposer, prompt, additional_prompt, log_file)
        return out_dict
    elif args.initial_guess == "from_file":
        out_dict = get_init_from_file(args)
        return out_dict
    else:
        raise ValueError(f"Unknown initial guess: {args.initial_guess}")


def get_init_from_file(args):
    path = Path(args.initial_guess_file)
    if not path.exists():
        raise FileNotFoundError(f"Initial guess file not found: {args.initial_guess_file}")
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if len(lines) == 0:
        raise ValueError("Initial guess file is empty.")
    if len(lines) == 1 and args.n_init > 1:
        comps = lines * args.n_init
    elif len(lines) != args.n_init:
        raise ValueError(
            f"Number of initial guesses in file ({len(lines)}) does not match n_init ({args.n_init}). "
            "Provide exactly n_init non-empty lines, or provide a single line to reuse the same composition for every initialization."
        )
    else:
        comps = lines
    outputs = [{"composition": comp} for comp in comps]
    return outputs

def get_init_from_random(args):
    # generate initial composition from training data
    path = Path(args.data_path)
    df = pd.read_csv(path)
    idx = random.randint(0, len(df)-1)
    crystal = Structure.from_str(df.iloc[idx]["cif"], fmt="cif")
    comp = crystal.composition.reduced_formula
    out_dict = {"composition": comp}
    return out_dict


def get_init_from_llm(args, proposer, target_prompt, additional_prompt, log_file=None):
    system_prompt = GENERAL_SYSTEM_PROMPT
    prompt = (
        target_prompt
        + " Could you suggest one possible material composition?"
        + instruct_simple_output_format()
        + additional_prompt
    )
    
    # Log the prompt
    if log_file:
        log_file.write("\n=== LLM Prompt ===\n")
        log_file.write(f"System: {system_prompt}\n")
        log_file.write(f"User: {prompt}\n")
    
    print(f"\n[LLM] Sending prompt to model...")
    response = proposer.generate(system_prompt=system_prompt, prompt=prompt)
    
    # Log the raw response
    if log_file:
        log_file.write(f"\n=== LLM Raw Response ===\n")
        log_file.write(f"{response}\n")
    
    print(f"[LLM] Raw response: {response[:200]}..." if len(str(response)) > 200 else f"[LLM] Raw response: {response}")
    
    out_dict = proposer.extract_outputs(response)
    
    # Log the parsed output
    if log_file:
        log_file.write(f"\n=== LLM Parsed Output ===\n")
        log_file.write(f"{out_dict}\n")
    
    print(f"[LLM] Parsed output: {out_dict}")
    
    return out_dict
