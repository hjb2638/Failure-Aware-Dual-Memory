#!/usr/bin/env python3
"""
Build a filtered / stratified memory sub-store from a merged memory store.

This script reads a YAML config, filters and deduplicates entries from an input
memory store, applies hierarchical quota-based sampling, and rebuilds a fresh
FAISS memory store with the same on-disk format:

- entries.json
- memory.index
- idmap.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sub-memory store from an existing merged memory store."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: PyYAML. Install it in the active environment "
            "before running build_memory_subset.py."
        ) from exc

    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


@dataclass
class NodeSpec:
    name: str
    filters: Dict[str, Any] = field(default_factory=dict)
    pool_fraction: Optional[float] = None
    pool_count: Optional[int] = None
    output_fraction: Optional[float] = None
    output_count: Optional[int] = None
    remainder: bool = False
    children: List["NodeSpec"] = field(default_factory=list)


def build_node_spec(payload: Dict[str, Any], *, default_name: str) -> NodeSpec:
    if not isinstance(payload, dict):
        raise ValueError(f"Node spec must be a mapping, got: {type(payload)}")

    children_payload = payload.get("children", []) or []
    if not isinstance(children_payload, list):
        raise ValueError("Node field 'children' must be a list when provided.")

    return NodeSpec(
        name=str(payload.get("name", default_name)),
        filters=dict(payload.get("filters", {}) or {}),
        pool_fraction=payload.get("pool_fraction"),
        pool_count=payload.get("pool_count"),
        output_fraction=payload.get("output_fraction"),
        output_count=payload.get("output_count"),
        remainder=bool(payload.get("remainder", False)),
        children=[
            build_node_spec(child, default_name=f"{default_name}_child_{idx}")
            for idx, child in enumerate(children_payload)
        ],
    )


def _safe_float(value: Any) -> Optional[float]:
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


def _canonical_formula(formula: str) -> str:
    from pymatgen.core.composition import Composition

    return Composition(formula).reduced_formula


def _entry_score(entry: "MemoryEntry") -> Tuple[int, int, int]:
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


def _semantic_key(entry: "MemoryEntry") -> Tuple[Any, ...]:
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


def _reduced_formula_key(entry: "MemoryEntry") -> Tuple[str]:
    return (_canonical_formula(entry.composition),)


def _cap_from_pool(size: int, fraction: Optional[float], count: Optional[int]) -> int:
    cap = size
    if fraction is not None:
        if not (0 <= float(fraction) <= 1):
            raise ValueError(f"pool_fraction must be within [0, 1], got {fraction}")
        cap = min(cap, int(math.floor(size * float(fraction))))
    if count is not None:
        if int(count) < 0:
            raise ValueError(f"pool_count must be non-negative, got {count}")
        cap = min(cap, int(count))
    return max(cap, 0)


def _approx_equal(a: float, b: float, tol: float = 1e-8) -> bool:
    return abs(a - b) <= tol


def _match_target_filter(target_value: float, target_filter: Dict[str, Any]) -> bool:
    eq_val = target_filter.get("eq")
    if eq_val is not None and not _approx_equal(target_value, float(eq_val)):
        return False

    if "in" in target_filter:
        allowed = [float(v) for v in target_filter["in"]]
        if not any(_approx_equal(target_value, candidate) for candidate in allowed):
            return False

    if "not_in" in target_filter:
        blocked = [float(v) for v in target_filter["not_in"]]
        if any(_approx_equal(target_value, candidate) for candidate in blocked):
            return False

    min_val = target_filter.get("min")
    max_val = target_filter.get("max")
    inclusive_min = bool(target_filter.get("inclusive_min", True))
    inclusive_max = bool(target_filter.get("inclusive_max", True))

    if min_val is not None:
        min_val = float(min_val)
        if inclusive_min:
            if target_value < min_val:
                return False
        else:
            if target_value <= min_val:
                return False

    if max_val is not None:
        max_val = float(max_val)
        if inclusive_max:
            if target_value > max_val:
                return False
        else:
            if target_value >= max_val:
                return False

    ranges = target_filter.get("ranges", [])
    if ranges:
        matched_any_range = False
        for current in ranges:
            if not isinstance(current, dict):
                raise ValueError("Each target_value range must be a mapping.")
            if _match_target_filter(target_value, current):
                matched_any_range = True
                break
        if not matched_any_range:
            return False

    return True


def _match_filters(entry: "MemoryEntry", filters: Dict[str, Any]) -> bool:
    if not filters:
        return True

    if "is_success" in filters and bool(entry.is_success) != bool(filters["is_success"]):
        return False

    if "target_value" in filters:
        target_value = _safe_float(entry.target_value)
        if target_value is None:
            return False
        target_filter = filters["target_value"]
        if isinstance(target_filter, dict):
            if not _match_target_filter(target_value, target_filter):
                return False
        else:
            if not _approx_equal(target_value, float(target_filter)):
                return False

    return True


def _filter_ids(ids: Sequence[int], entries: Sequence["MemoryEntry"], filters: Dict[str, Any]) -> List[int]:
    if not filters:
        return list(ids)
    return [idx for idx in ids if _match_filters(entries[idx], filters)]


def _load_entries(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{path} does not contain a JSON list.")
    return payload


def _clean_entry(
    raw: Dict[str, Any],
    validator: "CompositionValidator",
    probe_store: "MemoryStore",
) -> Tuple[Optional["MemoryEntry"], Optional[str]]:
    from failure_aware_dual_memory.component.memory import MemoryEntry

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

    entry.composition_vector = []
    try:
        probe_store.encoder.encode(entry.composition)
    except Exception as exc:
        return None, f"CBFV encoding failed: {exc}"

    return entry, None


def _dedupe_entries(entries: List["MemoryEntry"], mode: str) -> Tuple[List["MemoryEntry"], Dict[str, int]]:
    if mode == "none":
        return entries, {"deduped": 0}

    stats = {"deduped": 0}
    by_key: Dict[Any, "MemoryEntry"] = {}
    order: List[Any] = []

    for entry in entries:
        if mode == "entry_id":
            key = entry.entry_id
        elif mode == "semantic":
            key = _semantic_key(entry)
        elif mode == "reduced_formula":
            key = _reduced_formula_key(entry)
        else:
            raise ValueError(f"Unsupported dedup mode: {mode}")

        existing = by_key.get(key)
        if existing is None:
            by_key[key] = entry
            order.append(key)
            continue

        stats["deduped"] += 1
        winner = max([existing, entry], key=_entry_score)
        by_key[key] = winner

    return [by_key[key] for key in order], stats


def _compute_max_output(
    node: NodeSpec,
    available_ids: Sequence[int],
    entries: Sequence["MemoryEntry"],
) -> int:
    candidate_ids = _filter_ids(available_ids, entries, node.filters)
    node_cap = _cap_from_pool(len(candidate_ids), node.pool_fraction, node.pool_count)

    if not node.children:
        return node_cap

    remainder_children = [child for child in node.children if child.remainder]
    if len(remainder_children) > 1:
        raise ValueError(f"Node {node.name} has more than one remainder child.")

    fixed_sum = 0
    fraction_sum = 0.0
    max_total = node_cap
    remainder_child: Optional[NodeSpec] = None

    for child in node.children:
        child_max = _compute_max_output(child, candidate_ids, entries)
        child._computed_max = child_max  # type: ignore[attr-defined]

        if child.remainder:
            remainder_child = child
            continue

        if child.output_count is not None:
            needed = int(child.output_count)
            if child_max < needed:
                raise ValueError(
                    f"Node {node.name}: child {child.name} requires {needed} entries "
                    f"but only {child_max} are available."
                )
            fixed_sum += needed
        elif child.output_fraction is not None:
            fraction = float(child.output_fraction)
            if not (0 <= fraction <= 1):
                raise ValueError(
                    f"Node {node.name}: child {child.name} has invalid output_fraction {fraction}."
                )
            if fraction == 0:
                continue
            fraction_sum += fraction
            max_total = min(max_total, int(math.floor(child_max / fraction)))
        else:
            raise ValueError(
                f"Node {node.name}: child {child.name} must define output_count, "
                "output_fraction, or remainder."
            )

    if fraction_sum > 1 + 1e-9:
        raise ValueError(f"Node {node.name}: child output fractions sum to more than 1.")

    leftover_fraction = 1.0 - fraction_sum
    if fixed_sum > 0 and leftover_fraction <= 0 and remainder_child is None:
        raise ValueError(
            f"Node {node.name}: fixed child counts are incompatible with fraction-only layout."
        )

    if remainder_child is None and leftover_fraction < 1 - 1e-9 and fixed_sum == 0 and fraction_sum < 1 - 1e-9:
        raise ValueError(
            f"Node {node.name}: child fractions do not cover the parent output. "
            "Add a remainder child or make child fractions sum to 1."
        )

    if remainder_child is not None and leftover_fraction <= 0:
        raise ValueError(
            f"Node {node.name}: remainder child is present but no output mass is left for it."
        )

    lower_bound = fixed_sum
    if fixed_sum > 0 and leftover_fraction > 0:
        lower_bound = max(lower_bound, int(math.ceil(fixed_sum / leftover_fraction)))

    if remainder_child is not None:
        remainder_max = remainder_child._computed_max  # type: ignore[attr-defined]
        max_total = min(
            max_total,
            int(math.floor((remainder_max + fixed_sum) / leftover_fraction)),
        )

    if max_total < lower_bound:
        raise ValueError(
            f"Node {node.name}: constraints are infeasible. "
            f"Lower bound {lower_bound}, upper bound {max_total}."
        )

    return max_total


def _allocate_child_counts(node: NodeSpec, target_count: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    remainder_child = next((child for child in node.children if child.remainder), None)

    fixed_total = 0
    fractional_children: List[Tuple[NodeSpec, float, int, float]] = []

    for child in node.children:
        child_max = child._computed_max  # type: ignore[attr-defined]
        if child.remainder:
            continue
        if child.output_count is not None:
            count = int(child.output_count)
            if count > child_max:
                raise ValueError(
                    f"Node {node.name}: child {child.name} requested {count} "
                    f"but only {child_max} are available."
                )
            counts[child.name] = count
            fixed_total += count
            continue

        fraction = float(child.output_fraction)
        raw = fraction * target_count
        base = int(math.floor(raw))
        base = min(base, child_max)
        counts[child.name] = base
        fractional_children.append((child, fraction, child_max, raw - math.floor(raw)))

    used = sum(counts.values())
    remaining = target_count - used

    if remainder_child is not None:
        remainder_max = remainder_child._computed_max  # type: ignore[attr-defined]
        if remaining > remainder_max:
            raise ValueError(
                f"Node {node.name}: remainder child {remainder_child.name} cannot absorb {remaining} entries."
            )
        counts[remainder_child.name] = remaining
        remaining = 0
    else:
        fractional_children.sort(key=lambda item: item[3], reverse=True)
        for child, _, child_max, _ in fractional_children:
            if remaining <= 0:
                break
            if counts[child.name] < child_max:
                counts[child.name] += 1
                remaining -= 1

    if remaining != 0:
        raise ValueError(
            f"Node {node.name}: could not allocate all {target_count} requested entries."
        )

    return counts


def _sample_ids(
    ids: Sequence[int],
    count: int,
    rng: random.Random,
) -> List[int]:
    if count < 0:
        raise ValueError(f"Sample count must be non-negative, got {count}")
    if count > len(ids):
        raise ValueError(f"Cannot sample {count} items from pool of size {len(ids)}")
    if count == len(ids):
        return list(ids)
    return rng.sample(list(ids), count)


def _select_node(
    node: NodeSpec,
    available_ids: Sequence[int],
    entries: Sequence["MemoryEntry"],
    rng: random.Random,
    target_count: Optional[int] = None,
    counts_report: Optional[Dict[str, int]] = None,
) -> List[int]:
    candidate_ids = _filter_ids(available_ids, entries, node.filters)
    node_cap = _cap_from_pool(len(candidate_ids), node.pool_fraction, node.pool_count)
    feasible_max = _compute_max_output(node, candidate_ids, entries)
    feasible_max = min(feasible_max, node_cap)

    if target_count is None:
        target_count = feasible_max
    if target_count > feasible_max:
        raise ValueError(
            f"Node {node.name}: requested {target_count} entries but only {feasible_max} are feasible."
        )

    if counts_report is not None:
        counts_report[node.name] = target_count

    if not node.children:
        return _sample_ids(candidate_ids, target_count, rng)

    child_counts = _allocate_child_counts(node, target_count)
    selected: List[int] = []
    remaining_pool = list(candidate_ids)

    for child in node.children:
        child_target = child_counts.get(child.name, 0)
        if child_target == 0:
            continue

        child_selected = _select_node(
            child,
            remaining_pool,
            entries,
            rng,
            target_count=child_target,
            counts_report=counts_report,
        )
        selected.extend(child_selected)
        chosen = set(child_selected)
        remaining_pool = [idx for idx in remaining_pool if idx not in chosen]

    if len(selected) != target_count:
        raise ValueError(
            f"Node {node.name}: selected {len(selected)} entries, expected {target_count}."
        )

    return selected


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("memory.index", "entries.json", "idmap.json"):
        path = output_dir / filename
        if path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{path} already exists. Pass overwrite=true in the YAML config to replace it."
                )
            path.unlink()


def build_report(
    output_dir: Path,
    selected_entries: Sequence["MemoryEntry"],
    skipped_reasons: Dict[str, int],
    dedup_stats: Dict[str, int],
    counts_report: Dict[str, int],
) -> Dict[str, Any]:
    success_count = sum(1 for entry in selected_entries if entry.is_success)
    failure_count = len(selected_entries) - success_count
    target_histogram: Dict[str, int] = {}
    for entry in selected_entries:
        key = f"{float(entry.target_value):.6f}"
        target_histogram[key] = target_histogram.get(key, 0) + 1

    return {
        "output_dir": str(output_dir),
        "selected_entries": len(selected_entries),
        "success_entries": success_count,
        "failure_entries": failure_count,
        "target_histogram": target_histogram,
        "skipped_reasons": skipped_reasons,
        "dedup_stats": dedup_stats,
        "quota_counts": counts_report,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)

    from failure_aware_dual_memory.component.memory import MemoryStore
    from failure_aware_dual_memory.component.validator.composition_validator import CompositionValidator

    input_dir = Path(config["input_dir"]).resolve()
    output_dir = Path(config["output_dir"]).resolve()
    overwrite = bool(config.get("overwrite", False))
    report_json = config.get("report_json")
    random_seed = int(config.get("random_seed", 42))
    dedup_mode = str(config.get("dedup_mode", "semantic"))
    use_onehot = bool(config.get("use_onehot", True))
    use_magpie = bool(config.get("use_magpie", False))

    base_filters = dict(config.get("base_filters", {}) or {})
    selection_payload = dict(config.get("selection", {}) or {})
    selection_root = build_node_spec(selection_payload, default_name="root")
    if selection_root.output_count is not None or selection_root.output_fraction is not None:
        raise ValueError("Root selection node must not define output_count or output_fraction.")

    entries_path = input_dir / "entries.json"
    if not entries_path.exists():
        raise FileNotFoundError(f"Missing entries.json in input directory: {input_dir}")

    probe_dir = output_dir / ".subset_probe"
    probe_store = MemoryStore(
        storage_dir=str(probe_dir),
        use_onehot=use_onehot,
        use_magpie=use_magpie,
    )
    probe_store.clear()
    if probe_dir.exists():
        probe_dir.rmdir()

    raw_entries = _load_entries(entries_path)
    validator = CompositionValidator()
    cleaned_entries: List["MemoryEntry"] = []
    skipped_reasons: Dict[str, int] = {}

    for raw in raw_entries:
        entry, reason = _clean_entry(raw, validator, probe_store)
        if entry is None:
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue
        cleaned_entries.append(entry)

    deduped_entries, dedup_stats = _dedupe_entries(cleaned_entries, dedup_mode)
    base_ids = list(range(len(deduped_entries)))
    base_ids = _filter_ids(base_ids, deduped_entries, base_filters)

    rng = random.Random(random_seed)
    counts_report: Dict[str, int] = {}
    selected_ids = _select_node(
        selection_root,
        base_ids,
        deduped_entries,
        rng,
        target_count=None,
        counts_report=counts_report,
    )

    selected_entries = [deduped_entries[idx] for idx in selected_ids]

    _prepare_output_dir(output_dir, overwrite)
    output_store = MemoryStore(
        storage_dir=str(output_dir),
        use_onehot=use_onehot,
        use_magpie=use_magpie,
    )
    output_store.clear()

    batch_size = int(config.get("rebuild_batch_size", 512))
    for start in range(0, len(selected_entries), batch_size):
        batch = selected_entries[start : start + batch_size]
        output_store.add_batch(batch)
    output_store.save()

    report = build_report(
        output_dir=output_dir,
        selected_entries=selected_entries,
        skipped_reasons=skipped_reasons,
        dedup_stats=dedup_stats,
        counts_report=counts_report,
    )

    print(json.dumps(report, indent=2, sort_keys=True))

    if report_json:
        report_path = Path(report_json).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
