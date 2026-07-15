#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from pymatgen.core import Composition
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: pymatgen. Please run this script inside the Failure-Aware Dual-Memory "
        "environment where project dependencies are installed."
    ) from exc

try:
    from failure_aware_dual_memory.util.eval.composition import smact_validity
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing project dependencies for validity checking. Please run this script "
        "inside the Failure-Aware Dual-Memory environment after installing the package dependencies."
    ) from exc


LOG_PATTERN = re.compile(r"response_log_(\d+)\.txt$")
CSV_PATTERN = re.compile(r"gen_mat_init(\d+)_iter(\d+)\.csv$")
ITER_HEADER_PATTERN = re.compile(r"^\s*Iteration\s+(\d+)")
ITER_LOWER_PATTERN = re.compile(r"^\s*iteration\s+(\d+)")


@dataclass
class IterationResult:
    run_id: int
    iteration: int
    composition_raw: str
    composition_reduced: str
    best_predicted: float
    csv_path: Path


@dataclass
class SelectedResult:
    run_id: int
    iteration: int
    composition_raw: str
    composition_reduced: str
    best_predicted: float
    abs_error: float
    csv_path: Path
    valid: bool
    unique: bool = False
    novel: bool = False
    vun: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process Failure-Aware Dual-Memory inference artifacts and compute "
            "validity / uniqueness / novelty / VUN following the paper logic."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT,
        help=(
            "Run output directory. Supports either a flat directory containing "
            "response_log_XX.txt and gen_mat_initXX_iterYY.csv, or the newer "
            "nested layout with logs/ and results/ subdirectories."
        ),
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=REPO_ROOT / "data" / "mp_20" / "train.csv",
        help="Training CSV used as the novelty reference set",
    )
    parser.add_argument(
        "--target-value",
        type=float,
        default=-3.8,
        help="Target property value used to select the best composition per run",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save summary JSON",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save per-run selected results",
    )
    return parser.parse_args()


def canonical_formula(formula: str) -> str:
    comp = Composition(formula)
    return comp.reduced_formula


def load_train_formulas(train_csv: Path) -> set[str]:
    formulas = set()
    with open(train_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        pretty_idx = header.index("pretty_formula")
        for row in reader:
            formulas.add(canonical_formula(row[pretty_idx]))
    return formulas


def extract_run_id(log_path: Path) -> int:
    match = LOG_PATTERN.search(log_path.name)
    if not match:
        raise ValueError(f"Unexpected log filename: {log_path}")
    return int(match.group(1))


def resolve_output_layout(data_dir: Path) -> tuple[Path, Path]:
    """Resolve run outputs for both flat and nested directory layouts."""
    logs_dir = data_dir / "logs"
    results_dir = data_dir / "results"

    if logs_dir.is_dir() and results_dir.is_dir():
        return logs_dir, results_dir
    return data_dir, data_dir


def parse_log(log_path: Path) -> Dict[int, str]:
    iteration_to_composition: Dict[int, str] = {}
    current_iter: Optional[int] = None
    waiting_for_guess = False

    with open(log_path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            match_init = ITER_HEADER_PATTERN.match(line)
            if match_init:
                current_iter = int(match_init.group(1))
                waiting_for_guess = current_iter == 0
                continue

            match_iter = ITER_LOWER_PATTERN.match(line)
            if match_iter:
                current_iter = int(match_iter.group(1))
                waiting_for_guess = False
                continue

            if line == "# Next guess:":
                waiting_for_guess = True
                continue

            if line.startswith("Initial guess: "):
                current_iter = 0
                composition = line.split(": ", 1)[1].strip()
                if composition:
                    iteration_to_composition[current_iter] = composition
                waiting_for_guess = False
                continue

            if line.startswith("Proposed: "):
                if current_iter is None:
                    continue
                composition = line.split(": ", 1)[1].strip()
                if composition:
                    iteration_to_composition[current_iter] = composition
                waiting_for_guess = False
                continue

            if line == "# LLM Parsed Output:":
                waiting_for_guess = True
                continue

            if current_iter is None:
                continue

            if waiting_for_guess and line.startswith("{") and "composition" in line:
                try:
                    parsed = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    waiting_for_guess = False
                    continue
                composition = parsed.get("composition")
                if composition:
                    iteration_to_composition[current_iter] = composition
                waiting_for_guess = False

    return iteration_to_composition


def load_best_predicted(csv_path: Path) -> float:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        values = [float(row["predicted"]) for row in reader if row.get("predicted")]
    if not values:
        raise ValueError(f"No predicted values found in {csv_path}")
    return min(values)


def collect_csvs(results_dir: Path) -> Dict[int, Dict[int, Path]]:
    csv_map: Dict[int, Dict[int, Path]] = {}
    for path in sorted(results_dir.glob("gen_mat_init*_iter*.csv")):
        match = CSV_PATTERN.search(path.name)
        if not match:
            continue
        run_id = int(match.group(1))
        iteration = int(match.group(2))
        csv_map.setdefault(run_id, {})[iteration] = path
    return csv_map


def build_iteration_results(data_dir: Path) -> List[IterationResult]:
    logs_dir, results_dir = resolve_output_layout(data_dir)
    csv_map = collect_csvs(results_dir)
    results: List[IterationResult] = []

    for log_path in sorted(logs_dir.glob("response_log_*.txt")):
        run_id = extract_run_id(log_path)
        iteration_to_composition = parse_log(log_path)
        run_csvs = csv_map.get(run_id, {})

        for iteration, composition_raw in sorted(iteration_to_composition.items()):
            csv_path = run_csvs.get(iteration)
            if csv_path is None:
                continue
            try:
                composition_reduced = canonical_formula(composition_raw)
                best_predicted = load_best_predicted(csv_path)
            except Exception:
                continue

            results.append(
                IterationResult(
                    run_id=run_id,
                    iteration=iteration,
                    composition_raw=composition_raw,
                    composition_reduced=composition_reduced,
                    best_predicted=best_predicted,
                    csv_path=csv_path,
                )
            )

    return results


def composition_is_valid(formula: str) -> bool:
    comp = Composition(formula)
    reduced = comp.reduced_composition
    atomic_numbers = [el.Z for el in reduced.elements]
    counts = [int(round(reduced[el])) for el in reduced.elements]
    return smact_validity(atomic_numbers, counts)


def select_best_per_run(
    iteration_results: Iterable[IterationResult], target_value: float
) -> List[SelectedResult]:
    grouped: Dict[int, List[IterationResult]] = {}
    for item in iteration_results:
        grouped.setdefault(item.run_id, []).append(item)

    selected: List[SelectedResult] = []
    for run_id, items in sorted(grouped.items()):
        best = min(
            items,
            key=lambda x: (abs(x.best_predicted - target_value), x.iteration),
        )
        abs_error = abs(best.best_predicted - target_value)
        selected.append(
            SelectedResult(
                run_id=run_id,
                iteration=best.iteration,
                composition_raw=best.composition_raw,
                composition_reduced=best.composition_reduced,
                best_predicted=best.best_predicted,
                abs_error=abs_error,
                csv_path=best.csv_path,
                valid=composition_is_valid(best.composition_reduced),
            )
        )
    return selected


def annotate_metrics(selected: List[SelectedResult], train_formulas: set[str]) -> None:
    counts = Counter(item.composition_reduced for item in selected)
    for item in selected:
        item.unique = counts[item.composition_reduced] == 1
        item.novel = item.composition_reduced not in train_formulas
        item.vun = item.valid and item.unique and item.novel


def rate(values: Iterable[bool]) -> float:
    values = list(values)
    if not values:
        return math.nan
    return sum(bool(v) for v in values) / len(values)


def save_selected_csv(path: Path, selected: List[SelectedResult]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_id",
                "selected_iteration",
                "composition_raw",
                "composition_reduced",
                "best_predicted",
                "abs_error_to_target",
                "valid",
                "unique",
                "novel",
                "vun",
                "csv_path",
            ]
        )
        for item in selected:
            writer.writerow(
                [
                    item.run_id,
                    item.iteration,
                    item.composition_raw,
                    item.composition_reduced,
                    item.best_predicted,
                    item.abs_error,
                    item.valid,
                    item.unique,
                    item.novel,
                    item.vun,
                    str(item.csv_path),
                ]
            )


def build_summary(
    selected: List[SelectedResult],
    train_formulas: set[str],
    data_dir: Path,
    target_value: float,
) -> dict:
    return {
        "data_dir": str(data_dir),
        "target_value": target_value,
        "num_runs_with_valid_records": len(selected),
        "num_train_formulas": len(train_formulas),
        "validity": rate(item.valid for item in selected),
        "uniqueness": rate(item.unique for item in selected),
        "novelty": rate(item.novel for item in selected),
        "vun": rate(item.vun for item in selected),
        "selected_results": [
            {
                "run_id": item.run_id,
                "selected_iteration": item.iteration,
                "composition_raw": item.composition_raw,
                "composition_reduced": item.composition_reduced,
                "best_predicted": item.best_predicted,
                "abs_error_to_target": item.abs_error,
                "valid": item.valid,
                "unique": item.unique,
                "novel": item.novel,
                "vun": item.vun,
                "csv_path": str(item.csv_path),
            }
            for item in selected
        ],
    }


def main() -> None:
    args = parse_args()

    iteration_results = build_iteration_results(args.data_dir)
    if not iteration_results:
        raise SystemExit(
            f"No usable iteration records found under {args.data_dir}. "
            "Check filenames and log format."
        )

    train_formulas = load_train_formulas(args.train_csv)
    selected = select_best_per_run(iteration_results, args.target_value)
    annotate_metrics(selected, train_formulas)
    summary = build_summary(selected, train_formulas, args.data_dir, args.target_value)

    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        save_selected_csv(args.output_csv, selected)


if __name__ == "__main__":
    main()
