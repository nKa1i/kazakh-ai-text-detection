"""
Kaz-RAID: Dataset Generator across 28 Experimental Conditions.

Generates:
1. Clean baseline condition (1 condition)
2. 9 adversarial attack operators across 4 linguistic tiers at 3 perturbation rates:
   epsilon in {0.05, 0.10, 0.20} (9 x 3 = 27 conditions)
Total: 28 conditions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from kaz_raid import get_all_operators


def generate_benchmark_splits(
    records: List[Dict[str, Any]],
    rates: Optional[List[float]] = None,
    selected_attacks: Optional[List[str]] = None,
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate adversarial benchmark conditions for input records.

    Args:
        records: List of sample dicts containing at least 'id' and 'text'.
        rates: Perturbation rates, defaults to [0.05, 0.10, 0.20].
        selected_attacks: List of attack operator names to include (default: all 9).
        seed: Random seed for deterministic perturbation.

    Returns:
        Dict mapping condition names to perturbed record lists.
    """
    if rates is None:
        rates = [0.05, 0.10, 0.20]

    all_operators = get_all_operators()
    if selected_attacks is not None:
        all_operators = {k: v for k, v in all_operators.items() if k in selected_attacks}

    splits: Dict[str, List[Dict[str, Any]]] = {}

    # 1. Clean Condition
    clean_records: List[Dict[str, Any]] = []
    for rec in records:
        r = dict(rec)
        r["original_text"] = rec.get("text", "")
        r["text"] = rec.get("text", "")
        r["condition"] = "clean"
        r["attack_name"] = "clean"
        r["perturbation_rate"] = 0.0
        r["tier"] = "clean"
        clean_records.append(r)
    splits["clean"] = clean_records

    # 2. Perturbed Conditions (Attack x Rate)
    for op_name, op in all_operators.items():
        for rate in rates:
            cond_key = f"{op_name}_rate_{rate:.2f}"
            cond_records: List[Dict[str, Any]] = []
            for rec in records:
                r = dict(rec)
                orig_text = rec.get("text", "")
                r["original_text"] = orig_text
                # Unique deterministic seed per sample
                rec_id = str(rec.get("id", ""))
                sample_seed = (seed + abs(hash(rec_id))) % (2**31 - 1)
                r["text"] = op.perturb(orig_text, rate=rate, seed=sample_seed)
                r["condition"] = cond_key
                r["attack_name"] = op_name
                r["perturbation_rate"] = float(rate)
                r["tier"] = getattr(op, "tier", "unknown")
                cond_records.append(r)
            splits[cond_key] = cond_records

    return splits


def main():
    parser = argparse.ArgumentParser(description="Generate Kaz-RAID 28-condition evaluation benchmark.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/kazakh_aigc_paired_2k.json",
        help="Input JSON path containing test records"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/kaz_raid_eval_2k.json",
        help="Output JSON path to save all 28 conditions"
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.20],
        help="Perturbation rates"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional subset sample size for fast testing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic benchmark generation"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: input file {input_path} does not exist.")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if args.sample_size and args.sample_size < len(records):
        records = records[:args.sample_size]

    print(f"Generating Kaz-RAID benchmark for {len(records)} records across rates {args.rates}...")
    splits = generate_benchmark_splits(records, rates=args.rates, seed=args.seed)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    total_samples = sum(len(v) for v in splits.values())
    print(f"[Done] Generated {len(splits)} conditions ({total_samples} total instances) saved to {output_path}")


if __name__ == "__main__":
    main()
