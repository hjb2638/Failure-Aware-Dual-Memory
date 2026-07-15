"""
Full pipeline test with MP20 initialized memory.

This script runs the complete Failure-Aware Dual-Memory pipeline with:
1. Pre-initialized memory from MP20 test set
2. Chemical formula validation (pymatgen + SMACT)
3. Memory-enhanced proposer
4. Success/failure tracking and storage
"""

import os
# Fix CUDA deterministic behavior issue
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, Optional
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add parent directory to path
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from failure_aware_dual_memory.util import set_seed, get_prompt
from failure_aware_dual_memory.diffusion.pretrain import load_diffusion_model, run_diffusion_model
from failure_aware_dual_memory.component import load_evaluator, load_proposer
from failure_aware_dual_memory.component.proposer.util import get_initial_guess
from failure_aware_dual_memory.component.proposer.memory_proposer import wrap_proposer_with_memory
from failure_aware_dual_memory.component.validator.composition_validator import CompositionValidator
from failure_aware_dual_memory.component.validator.smact_checker import SMACTChecker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full pipeline test with MP20 memory"
    )
    
    # Data paths
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "mp_20" / "train.csv"),
        help="Path to training data",
    )
    
    parser.add_argument(
        "--initial_guess",
        type=str,
        choices=["random", "llm", "from_file"],
        default="llm",
        help="Initial guess method",
    )
    
    parser.add_argument(
        "--initial_guess_file",
        type=str,
        default=str(PROJECT_ROOT / "initial_guesses.txt"),
        help="File containing initial guesses",
    )
    
    parser.add_argument(
        "--memory_storage_dir",
        type=str,
        default=str(PROJECT_ROOT / "memory_storage_mp20_init"),
        help="Pre-initialized memory storage directory",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "test_outputs"),
        help="Output directory for results",
    )
    
    # Model settings
    parser.add_argument(
        "--llm_model",
        "--llm_model_path",
        dest="llm_model",
        type=str,
        default=os.environ.get("FADM_LLM_MODEL"),
        help="Local model path or Hugging Face model id for the proposer",
    )

    parser.add_argument(
        "--llm_device_map",
        type=str,
        default=os.environ.get("FADM_LLM_DEVICE_MAP", "auto"),
        help="Transformers device_map passed to the local model loader",
    )

    parser.add_argument(
        "--llm_torch_dtype",
        type=str,
        default=os.environ.get("FADM_LLM_TORCH_DTYPE", "auto"),
        help="Torch dtype for local model loading: auto, bfloat16, float16, float32",
    )

    parser.add_argument(
        "--llm_trust_remote_code",
        action="store_true",
        default=False,
        help="Allow transformers to execute custom remote model code",
    )
    
    parser.add_argument(
        "--target_value",
        type=float,
        default=-3.5,
        help="Target formation energy",
    )
    
    # Test settings
    parser.add_argument(
        "--n_init",
        type=int,
        default=2,
        help="Number of initial guesses",
    )
    
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=5,
        help="Max iterations per initial guess",
    )
    
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=0.25,
        help="Success threshold (eV/atom)",
    )
    
    # Feature flags
    parser.add_argument(
        "--enable_smact",
        action="store_true",
        default=True,
        help="Enable SMACT validation",
    )
    
    parser.add_argument(
        "--enable_memory",
        action="store_true",
        default=False,
        help="Enable memory-enhanced proposer",
    )

    parser.add_argument(
        "--freeze_memory",
        action="store_true",
        default=False,
        help="Use the memory database for retrieval only and skip writing new entries",
    )
    
    parser.add_argument(
        "--k_success",
        type=int,
        default=3,
        help="Number of success cases to retrieve",
    )
    
    parser.add_argument(
        "--k_failure",
        type=int,
        default=2,
        help="Number of failure cases to retrieve",
    )
    
    # Other settings
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max new tokens for proposer",
    )

    parser.add_argument(
        "--additional_prompt",
        type=str,
        default="",
        help="Extra instruction appended to the proposer prompt",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        default=False,
        help="Disable CUDA",
    )
    
    args = parser.parse_args()
    if not args.llm_model:
        parser.error("Missing local LLM configuration: pass --llm_model or set FADM_LLM_MODEL.")
    return args


def setup_output_dir(output_dir: str):
    """Create output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "logs").mkdir(exist_ok=True)
    (output_path / "memory").mkdir(exist_ok=True)
    (output_path / "results").mkdir(exist_ok=True)
    
    return output_path


def print_test_header():
    """Print test header."""
    print("\n" + "="*70)
    print("FULL PIPELINE TEST WITH MP20 MEMORY")
    print("="*70)


def print_section(title: str):
    """Print section header."""
    print("\n" + "-"*70)
    print(f"  {title}")
    print("-"*70)


def run_pipeline(
    args,
    proposer,
    evaluator,
    diffusion_model,
    composition_validator,
    prompt: str,
    device,
    output_dir: Optional[Path] = None,
    structured_output: bool = False,
    smact_checker=None,
    additional_prompt: str = "",
    initial_guesses=None,
    results_summary: Optional[Dict[str, int]] = None,
    print_run_header: Optional[Callable[[int, int], None]] = None,
    init_log_writer: Optional[Callable[[object, int], None]] = None,
    log_initial_guess_prompt: bool = False,
):
    if results_summary is None:
        results_summary = {
            "total_iterations": 0,
            "successful_iterations": 0,
            "failed_iterations": 0,
            "validation_failures": 0,
            "smact_warnings": 0,
        }

    base_output_dir = Path(output_dir) if output_dir is not None else Path(".")
    log_dir = base_output_dir / "logs" if structured_output else base_output_dir
    result_dir = base_output_dir / "results" if structured_output else base_output_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.initial_guess == "from_file" and initial_guesses is None:
        initial_guesses = get_initial_guess(
            args,
            proposer,
            prompt,
            additional_prompt=additional_prompt,
            device=device,
        )

    for init_id in range(args.n_init):
        if print_run_header is not None:
            print_run_header(init_id, args.n_init)

        prev_composition = None
        log_path = log_dir / f"response_log_{init_id:02d}.txt"
        with open(log_path, "w") as file:
            if init_log_writer is not None:
                init_log_writer(file, init_id)

            if args.initial_guess == "from_file":
                next_guess = initial_guesses[init_id]
            else:
                next_guess = get_initial_guess(
                    args,
                    proposer,
                    prompt,
                    additional_prompt=additional_prompt,
                    device=device,
                    log_file=file if log_initial_guess_prompt else None,
                )

            composition = next_guess.get("composition", "") if next_guess else None
            if not composition:
                print("✗ No valid initial composition generated")
                file.write("No valid initial composition generated\n")
                results_summary["validation_failures"] += 1
                continue

            file.write("##################################################\n")
            file.write(" Iteration 0 (Initialization)\n")
            file.write("##################################################\n")
            file.write(str(next_guess) + "\n")
            file.write(f"Initial guess: {composition}\n")
            print(f"Initial guess: {composition}")

            if not _validate_composition(
                composition,
                composition_validator,
                smact_checker,
                file,
                results_summary,
            ):
                continue

            success, feedback, predicted_val = _evaluate_and_record(
                args=args,
                proposer=proposer,
                evaluator=evaluator,
                diffusion_model=diffusion_model,
                next_guess=next_guess,
                composition=composition,
                iteration=0,
                init_id=init_id,
                file=file,
                result_dir=result_dir,
                results_summary=results_summary,
                previous_composition=None,
            )

            prev_composition = composition
            if success:
                file.write("Stop: success found at initial guess\n")
                continue

            for iter_num in range(1, args.n_iterations + 1):
                print(f"\n--- Iteration {iter_num}/{args.n_iterations} ---")
                file.write("##################################################\n")
                file.write(f" Iteration {iter_num}\n")
                file.write("##################################################\n")

                if args.enable_memory:
                    next_guess = proposer.propose_one(
                        prev_guess=next_guess,
                        feedback=feedback,
                        file=file,
                        additional_prompt=additional_prompt,
                        iteration=iter_num,
                        init_id=init_id,
                        predicted_value=predicted_val,
                    )
                else:
                    next_guess = proposer.propose_one(
                        prev_guess=next_guess,
                        feedback=feedback,
                        file=file,
                        additional_prompt=additional_prompt,
                    )

                prev_valid, validity_feedback = evaluator.check_validity(next_guess)
                if not prev_valid:
                    file.write("# Feedback:\n")
                    file.write(f"{validity_feedback}\n")
                    print("✗ No valid composition generated")
                    results_summary["validation_failures"] += 1
                    continue

                composition = next_guess.get("composition", "") if next_guess else None
                if not composition:
                    print("✗ No valid composition generated")
                    file.write("No valid composition generated\n")
                    results_summary["validation_failures"] += 1
                    continue

                print(f"Proposed: {composition}")
                file.write(f"Proposed: {composition}\n")

                if not _validate_composition(
                    composition,
                    composition_validator,
                    smact_checker,
                    file,
                    results_summary,
                ):
                    feedback = validity_feedback if not prev_valid else f"Invalid composition: {composition}"
                    continue

                success, feedback, predicted_val = _evaluate_and_record(
                    args=args,
                    proposer=proposer,
                    evaluator=evaluator,
                    diffusion_model=diffusion_model,
                    next_guess=next_guess,
                    composition=composition,
                    iteration=iter_num,
                    init_id=init_id,
                    file=file,
                    result_dir=result_dir,
                    results_summary=results_summary,
                    previous_composition=prev_composition,
                )

                prev_composition = composition
                if success:
                    file.write(f"Stop: success found at iteration {iter_num}\n")
                    break

                print(f"Iteration {iter_num} done.")

    return results_summary


def _validate_composition(
    composition: str,
    composition_validator,
    smact_checker,
    file,
    results_summary: Dict[str, int],
) -> bool:
    is_valid, error_msg = composition_validator.validate(composition)
    if not is_valid:
        print(f"✗ Validation failed: {error_msg}")
        file.write(f"Validation failed: {error_msg}\n")
        results_summary["validation_failures"] += 1
        return False

    print("✓ Composition validation passed")

    if smact_checker is not None:
        is_valid_smact, error_msg_smact = smact_checker.validate(composition, strict=False)
        if not is_valid_smact:
            print(f"⚠ SMACT warning: {error_msg_smact}")
            file.write(f"SMACT warning: {error_msg_smact}\n")
            results_summary["smact_warnings"] += 1
        else:
            print("✓ SMACT validation passed")

    return True


def _evaluate_and_record(
    args,
    proposer,
    evaluator,
    diffusion_model,
    next_guess,
    composition: str,
    iteration: int,
    init_id: int,
    file,
    result_dir: Path,
    results_summary: Dict[str, int],
    previous_composition: Optional[str],
):
    print("Running diffusion model...")
    gen_mat_df = run_diffusion_model(next_guess, diffusion_model)
    gen_mat_df = evaluator.evaluate(gen_mat_df)

    csv_path = result_dir / f"gen_mat_init{init_id:02d}_iter{iteration:02d}.csv"
    gen_mat_df.to_csv(csv_path)

    feedback, predicted_val = evaluator.feedback(gen_mat_df)
    print(f"Predicted: {predicted_val:.3f} eV/atom")
    file.write(f"Predicted: {predicted_val:.3f} eV/atom\n")
    file.write(f"Feedback: {feedback}\n")

    distance = abs(predicted_val - args.target_value)
    is_success = distance <= args.success_threshold

    if is_success:
        print(f"✓ SUCCESS! Distance: {distance:.3f} eV/atom")
        results_summary["successful_iterations"] += 1
    else:
        label = "Failed" if iteration == 0 else "Distance"
        print(f"✗ {label}: {distance:.3f} eV/atom")
        results_summary["failed_iterations"] += 1

    results_summary["total_iterations"] += 1

    if args.enable_memory and not args.freeze_memory:
        all_preds = None
        if "predicted" in gen_mat_df.columns:
            all_preds = [float(x) for x in gen_mat_df["predicted"].tolist()]

        best_cif = None
        if "cif" in gen_mat_df.columns and "predicted" in gen_mat_df.columns:
            best_cif = gen_mat_df.iloc[gen_mat_df["predicted"].idxmin()]["cif"]

        entry = proposer.create_memory_entry(
            composition=composition,
            predicted_value=float(predicted_val),
            iteration=iteration,
            init_id=init_id,
            feedback=feedback,
            all_predictions=all_preds,
            best_structure_cif=best_cif,
            previous_composition=previous_composition,
        )
        try:
            proposer.memory_store.add(entry)
            proposer.memory_store.save()
            print("✓ Stored in memory")
        except Exception as exc:
            warning_msg = (
                f"memory write skipped for composition {composition}:{exc}"
            )
            print(f"{warning_msg}")
            file.write(f"{warning_msg}\n")
    elif args.enable_memory and args.freeze_memory:
        print("✓ Memory frozen: retrieval only, skipped saving new entry")

    return is_success, feedback, predicted_val


def run_workflow(
    args,
    *,
    output_dir: Optional[Path] = None,
    structured_output: bool = True,
    show_header: bool = True,
    show_sections: bool = True,
    log_initial_guess_prompt: bool = True,
    section_printer: Callable[[str], None] = print_section,
):
    if show_header:
        print_test_header()

    output_path = setup_output_dir(output_dir or args.output_dir) if structured_output else Path(output_dir or ".")
    if not structured_output:
        output_path.mkdir(parents=True, exist_ok=True)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"\nDevice: {device}" if show_header else f"Device: {device}")

    if getattr(args, "set_seed", False):
        set_seed(args.seed)
    elif hasattr(args, "seed"):
        set_seed(args.seed)

    prompt = get_prompt(args.target_value)
    if hasattr(args, "target_value"):
        print(f"Target value: {args.target_value} eV/atom")

    if show_sections:
        section_printer("Initializing Validators")
    composition_validator = CompositionValidator(max_natoms=getattr(args, "max_natoms", 34))
    print("✓ Composition validator initialized")

    smact_checker = None
    if args.enable_smact:
        try:
            smact_checker = SMACTChecker()
            print("✓ SMACT checker initialized" if show_sections else "SMACT validation enabled.")
        except Exception as e:
            print(f"⚠ SMACT checker failed: {e}" if show_sections else f"Warning: Could not initialize SMACT checker: {e}")

    if show_sections:
        section_printer("Loading Models")
    print("Loading diffusion model...")
    diffusion_model = load_diffusion_model().to(device)
    diffusion_model.device = device
    print("✓ Diffusion model loaded")

    print("Loading evaluator...")
    evaluator = load_evaluator(args, device)
    print("✓ Evaluator loaded")

    print("Loading proposer...")
    max_tokens = getattr(args, "max_new_tokens", getattr(args, "max_new_tokens_for_tf_proposer", 2048))
    base_proposer = load_proposer(
        args=args,
        target_prompt=prompt,
        max_new_tokens=max_tokens,
    )
    print("✓ Base proposer loaded")

    if args.enable_memory:
        if show_sections:
            section_printer("Initializing Memory")
            print(f"Loading memory from: {args.memory_storage_dir}")
        else:
            print(f"Enabling memory-enhanced proposer (storage: {args.memory_storage_dir})")

        proposer = wrap_proposer_with_memory(
            base_proposer=base_proposer,
            memory_storage_dir=args.memory_storage_dir,
            success_threshold=args.success_threshold,
            k_success=args.k_success,
            k_failure=args.k_failure,
            target_tolerance=getattr(args, "target_tolerance", 0.5),
        )

        stats = proposer.get_memory_stats()
        if show_sections:
            print(f"\nMemory Statistics:")
            print(f"  Total entries: {stats['total_entries']}")
            print(f"  Success count: {stats['success_count']}")
            print(f"  Failure count: {stats['failure_count']}")
            print(f"  Success rate: {stats['success_rate']:.2%}")
            if args.freeze_memory:
                print("  Write mode: frozen (no new entries will be added)")
            print("✓ Memory-enhanced proposer ready")
        else:
            print(f"Memory stats: {stats}")
            if args.freeze_memory:
                print("Memory write mode: frozen (retrieval only, no new entries will be saved)")
    else:
        proposer = base_proposer

    if show_sections:
        section_printer("Running Inference")
    else:
        print("Start inference.")

    initial_guesses = None
    if args.initial_guess == "from_file":
        initial_guesses = get_initial_guess(
            args,
            proposer,
            prompt,
            additional_prompt=getattr(args, "additional_prompt", ""),
            device=device,
        )

    def print_run_header(init_id: int, total: int):
        if not show_sections:
            return
        print(f"\n{'='*70}")
        print(f"Initial Guess {init_id + 1}/{total}")
        print(f"{'='*70}")

    def init_log_writer(file, init_id: int):
        if show_sections:
            file.write(f"# Full Pipeline Test - Initial Guess {init_id}\n")
            file.write(f"# Target: {args.target_value} eV/atom\n")
            file.write(f"# Success threshold: {args.success_threshold} eV/atom\n")
            file.write("=" * 70 + "\n\n")

    results_summary = run_pipeline(
        args=args,
        proposer=proposer,
        evaluator=evaluator,
        diffusion_model=diffusion_model,
        composition_validator=composition_validator,
        prompt=prompt,
        device=device,
        output_dir=output_path,
        structured_output=structured_output,
        smact_checker=smact_checker,
        additional_prompt=getattr(args, "additional_prompt", ""),
        initial_guesses=initial_guesses,
        print_run_header=print_run_header if show_sections else None,
        init_log_writer=init_log_writer if show_sections else None,
        log_initial_guess_prompt=log_initial_guess_prompt,
    )

    return proposer, results_summary, output_path


def main():
    args = parse_args()
    proposer, results_summary, output_path = run_workflow(
        args,
        output_dir=args.output_dir,
        structured_output=True,
        show_header=True,
        show_sections=True,
        log_initial_guess_prompt=True,
    )
    
    # Print summary
    print_section("TEST SUMMARY")
    print(f"Total iterations: {results_summary['total_iterations']}")
    print(f"Successful: {results_summary['successful_iterations']}")
    print(f"Failed: {results_summary['failed_iterations']}")
    print(f"Validation failures: {results_summary['validation_failures']}")
    print(f"SMACT warnings: {results_summary['smact_warnings']}")
    
    if results_summary['total_iterations'] > 0:
        success_rate = results_summary['successful_iterations'] / results_summary['total_iterations']
        print(f"Success rate: {success_rate:.2%}")
    
    # Final memory stats
    if args.enable_memory:
        print_section("Final Memory Statistics")
        stats = proposer.get_memory_stats()
        print(f"Total entries: {stats['total_entries']}")
        print(f"Success count: {stats['success_count']}")
        print(f"Failure count: {stats['failure_count']}")
    
    print(f"\nOutput saved to: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    import pandas as pd  # Import here to avoid issues
    main()
