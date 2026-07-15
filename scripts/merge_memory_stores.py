#!/usr/bin/env python3
"""
Merge multiple memory-store directories into one rebuilt FAISS memory store.

This script treats each input directory's ``entries.json`` as the source of
truth, filters malformed or non-encodable entries, deduplicates repeated
records, and rebuilds a fresh ``memory.index`` / ``entries.json`` /
``idmap.json`` triple in the output directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from failure_aware_dual_memory.component.memory import MemoryEntry, MemoryStore
from failure_aware_dual_memory.component.validator.composition_validator import CompositionValidator



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple Failure-Aware Dual-Memory memory stores into a single rebuilt "
            "memory store with filtering and deduplication."
        )
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="One or more memory store directories to merge.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for the merged memory store.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing memory files in the output directory.",
    )
    parser.add_argument(
        "--use-onehot",
        action="store_true",
        default=True,
        help="Use onehot CBFV features when rebuilding the merged index.",
    )
    parser.add_argument(
        "--use-magpie",
        action="store_true",
        default=False,
        help="Use magpie CBFV features when rebuilding the merged index.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to save a JSON merge report.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _canonical_formula(formula: str) -> str:
    from pymatgen.core.composition import Composition

    return Composition(formula).reduced_formula


def _entry_score(entry: MemoryEntry) -> Tuple[int, int, int]:
    """Prefer richer entries, then earlier timestamps deterministically."""
    filled_fields = sum(
        1
        for value in [
            entry.best_structure_cif,
            entry.all_predictions,
            entry.failure_reason,
            entry.reason_summary,
            entry.previous_composition,
            entry.feedback_given,
            entry.metadata,
        ]
        if value
    )
    prediction_count = len(entry.all_predictions or [])
    has_vector = 1 if entry.composition_vector else 0
    return (filled_fields, prediction_count, has_vector)


def _semantic_key(entry: MemoryEntry) -> Tuple[Any, ...]:
    predicted_value = _safe_float(entry.predicted_value)
    target_value = _safe_float(entry.target_value)
    distance = _safe_float(entry.distance_to_target)

    return (
        _canonical_formula(entry.composition),
        round(target_value, 8) if target_value is not None else None,
        round(predicted_value, 8) if predicted_value is not None else None,
        bool(entry.is_success),
        round(distance, 8) if distance is not None else None,
        _safe_int(entry.iteration),
        _safe_int(entry.init_id),
        (entry.previous_composition or "").strip(),
    )


def _load_raw_entries(entries_path: Path) -> Tuple[List[Dict[str, Any]], str | None]:
    try:
        with open(entries_path, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        return [], f"failed to read {entries_path}: {exc}"

    if not isinstance(payload, list):
        return [], f"{entries_path} does not contain a JSON list"
    return payload, None


def _clean_entry(
    raw: Dict[str, Any],
    source_dir: Path,
    validator: CompositionValidator,
    probe_store: MemoryStore,
) -> Tuple[MemoryEntry | None, str | None]:
    try:
        entry = MemoryEntry.from_dict(raw)
    except Exception as exc:
        return None, f"invalid entry schema: {exc}"

    composition = (entry.composition or "").strip()
    if not composition:
        return None, "empty composition"

    is_valid, error_msg = validator.validate(composition)
    if not is_valid:
        return None, f"invalid composition: {error_msg}"

    predicted_value = _safe_float(entry.predicted_value)
    target_value = _safe_float(entry.target_value)
    if predicted_value is None or target_value is None:
        return None, "missing numeric target/predicted value"

    entry.composition = composition
    entry.predicted_value = predicted_value
    entry.target_value = target_value
    entry.target_tolerance = _safe_float(entry.target_tolerance) or 0.1
    entry.iteration = _safe_int(entry.iteration)
    entry.init_id = _safe_int(entry.init_id)
    entry.is_success = bool(entry.is_success)
    entry.distance_to_target = (
        _safe_float(entry.distance_to_target)
        if entry.distance_to_target is not None
        else abs(predicted_value - target_value)
    )

    if entry.distance_to_target is None:
        entry.distance_to_target = abs(predicted_value - target_value)

    if entry.previous_composition is not None:
        entry.previous_composition = str(entry.previous_composition).strip() or None
    if entry.feedback_given is not None:
        entry.feedback_given = str(entry.feedback_given).strip() or None
    if entry.failure_reason is not None:
        entry.failure_reason = str(entry.failure_reason).strip() or None
    if entry.reason_summary is not None:
        entry.reason_summary = str(entry.reason_summary).strip() or None

    if not isinstance(entry.metadata, dict):
        entry.metadata = {}
    entry.metadata = dict(entry.metadata)
    entry.metadata.setdefault("merged_source_dirs", [])
    if str(source_dir) not in entry.metadata["merged_source_dirs"]:
        entry.metadata["merged_source_dirs"].append(str(source_dir))

    # Force re-encoding later; this also validates CBFV compatibility now.
    entry.composition_vector = []
    try:
        probe_store.encoder.encode(entry.composition)
    except Exception as exc:
        return None, f"CBFV encoding failed: {exc}"

    return entry, None


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("memory.index", "entries.json", "idmap.json"):
        path = output_dir / filename
        if path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{path} already exists. Pass --overwrite to replace existing memory files."
                )
            path.unlink()


def _merge_entries(entries: Iterable[MemoryEntry]) -> Tuple[List[MemoryEntry], Dict[str, int]]:
    by_entry_id: Dict[str, MemoryEntry] = {}
    semantic_seen: Dict[Tuple[Any, ...], MemoryEntry] = {}
    stats = {
        "deduped_by_entry_id": 0,
        "deduped_by_semantic_key": 0,
    }

    for entry in entries:
        existing = by_entry_id.get(entry.entry_id)
        if existing is not None:
            stats["deduped_by_entry_id"] += 1
            winner = max([existing, entry], key=_entry_score)
            by_entry_id[entry.entry_id] = winner
            continue
        by_entry_id[entry.entry_id] = entry

    merged: List[MemoryEntry] = []
    for entry in by_entry_id.values():
        key = _semantic_key(entry)
        existing = semantic_seen.get(key)
        if existing is None:
            semantic_seen[key] = entry
            merged.append(entry)
            continue

        stats["deduped_by_semantic_key"] += 1
        winner = max([existing, entry], key=_entry_score)
        if winner is not existing:
            semantic_seen[key] = winner
            merged[merged.index(existing)] = winner

        # Keep provenance from both entries.
        sources = set(existing.metadata.get("merged_source_dirs", []))
        sources.update(entry.metadata.get("merged_source_dirs", []))
        winner.metadata["merged_source_dirs"] = sorted(sources)

    return merged, stats


def build_report(
    input_dirs: List[Path],
    output_dir: Path,
    accepted_entries: List[MemoryEntry],
    skipped_reasons: Dict[str, int],
    merge_stats: Dict[str, int],
) -> Dict[str, Any]:
    success_count = sum(1 for entry in accepted_entries if entry.is_success)
    failure_count = len(accepted_entries) - success_count
    return {
        "input_dirs": [str(path) for path in input_dirs],
        "output_dir": str(output_dir),
        "accepted_entries": len(accepted_entries),
        "success_entries": success_count,
        "failure_entries": failure_count,
        "skipped_reasons": skipped_reasons,
        "merge_stats": merge_stats,
    }


def main() -> None:
    args = parse_args()

    input_dirs = [Path(path).resolve() for path in args.input_dirs]
    output_dir = Path(args.output_dir).resolve()

    validator = CompositionValidator()

    # Probe store is only used for consistent CBFV compatibility checks.
    probe_dir = output_dir / ".merge_probe"
    probe_store = MemoryStore(
        storage_dir=str(probe_dir),
        use_onehot=args.use_onehot,
        use_magpie=args.use_magpie,
    )
    probe_store.clear()
    probe_dir.rmdir()

    accepted_entries: List[MemoryEntry] = []
    skipped_reasons: Dict[str, int] = {}

    for input_dir in input_dirs:
        entries_path = input_dir / "entries.json"
        if not entries_path.exists():
            reason = f"missing entries.json in {input_dir}"
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue

        raw_entries, error = _load_raw_entries(entries_path)
        if error is not None:
            skipped_reasons[error] = skipped_reasons.get(error, 0) + 1
            continue

        for raw in raw_entries:
            entry, reason = _clean_entry(raw, input_dir, validator, probe_store)
            if entry is None:
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                continue
            accepted_entries.append(entry)

    merged_entries, merge_stats = _merge_entries(accepted_entries)
    _prepare_output_dir(output_dir, args.overwrite)

    merged_store = MemoryStore(
        storage_dir=str(output_dir),
        use_onehot=args.use_onehot,
        use_magpie=args.use_magpie,
    )
    merged_store.clear()
    merged_store.add_batch(merged_entries)
    merged_store.save()

    report = build_report(
        input_dirs=input_dirs,
        output_dir=output_dir,
        accepted_entries=merged_entries,
        skipped_reasons=skipped_reasons,
        merge_stats=merge_stats,
    )

    print(json.dumps(report, indent=2, sort_keys=True))

    if args.report_json:
        report_path = Path(args.report_json).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
