#!/usr/bin/env python3
"""
Loss mask coverage analysis.

For each training sample, report what fraction of tokens the loss_mask
activates on. If harmful and benign masks cover the same ~100% of tokens,
then selectivity at the mask level is impossible — the loss treats every
token identically. If harmful masks target injection tokens and benign
masks target tool-call tokens, there's structure the model can exploit.

Usage:
    python scripts/diagnostics/lossmask_coverage.py \
        --harmful-lossmasks data/lossmasks/harmful.jsonl \
        --benign-lossmasks data/lossmasks/benign.jsonl

Output: per-sample and aggregate statistics about mask coverage.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


def analyze_masks(lossmask_path: Path, label: str):
    """Analyze loss mask coverage for a set of samples."""
    coverages = []
    mask_lengths = []
    total_ones = 0
    total_tokens = 0

    with open(lossmask_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            mask = row.get("loss_mask", [])
            if not mask:
                continue

            n_tokens = len(mask)
            n_active = sum(1 for m in mask if m > 0)
            coverage = n_active / n_tokens if n_tokens > 0 else 0

            coverages.append(coverage)
            mask_lengths.append(n_tokens)
            total_ones += n_active
            total_tokens += n_tokens

    if not coverages:
        print(f"  [{label}] No samples found")
        return

    coverages.sort()
    n = len(coverages)
    avg = sum(coverages) / n
    median = coverages[n // 2]
    p10 = coverages[int(n * 0.1)]
    p90 = coverages[int(n * 0.9)]
    zero_count = sum(1 for c in coverages if c == 0)
    full_count = sum(1 for c in coverages if c > 0.95)

    print(f"  [{label}] {n} samples")
    print(f"    Coverage: mean={avg:.1%}, median={median:.1%}, "
          f"p10={p10:.1%}, p90={p90:.1%}")
    print(f"    Zero-mask samples: {zero_count}/{n} ({zero_count/n:.1%})")
    print(f"    >95% coverage: {full_count}/{n} ({full_count/n:.1%})")
    print(f"    Total active tokens: {total_ones}/{total_tokens} "
          f"({total_ones/total_tokens:.1%})")
    avg_len = sum(mask_lengths) / n
    print(f"    Avg sequence length: {avg_len:.0f} tokens")


def main():
    parser = argparse.ArgumentParser(description="Loss mask coverage analysis")
    parser.add_argument("--harmful-lossmasks", type=Path, nargs="+", required=True)
    parser.add_argument("--benign-lossmasks", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    print("=" * 60)
    print("Loss Mask Coverage Analysis")
    print("=" * 60)

    print("\nHARMFUL masks:")
    for p in args.harmful_lossmasks:
        analyze_masks(p, f"harmful: {p.name}")

    print("\nBENIGN masks:")
    for p in args.benign_lossmasks:
        analyze_masks(p, f"benign: {p.name}")

    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("  - If both harmful and benign have >90% coverage:")
    print("    → Loss treats most tokens identically → selectivity at mask")
    print("      level is impossible. Need feature-level (SRMU) selectivity.")
    print("  - If harmful targets specific tokens (injection span):")
    print("    → Good — loss focuses on the right tokens.")
    print("  - If benign has 0% coverage (injection_aware policy):")
    print("    → Expected — benign loss uses attention_mask, not loss_mask.")
    print("=" * 60)


if __name__ == "__main__":
    main()
