#!/usr/bin/env python3
"""
Compare baseline vs CB model outputs on evaluation data.

For datasets without tool_attack info (like AgentDojo), this script:
1. Generates responses from both baseline and CB model
2. Prints paired examples showing differences
3. Computes basic comparison metrics

Usage:
    python scripts/compare_model_outputs.py \
        --baseline meta-llama/Llama-3.1-8B-Instruct \
        --cb-adapter /path/to/adapter \
        --eval-data data/traces/agentdojo_traces.jsonl \
        --limit 20 \
        --only-harmful

    # Just show paired outputs from existing eval results
    python scripts/compare_model_outputs.py \
        --paired-file evaluations/assistant_only_agentdojo_eval.paired_outputs.jsonl \
        --limit 10
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def print_paired_example(idx: int, record: dict, max_response_len: int = 500):
    """Print a single paired example."""
    print(f"\n{'='*70}")
    print(f"Example {idx}")
    print("=" * 70)

    # Show trace info
    trace_id = record.get("id", "?")
    print(f"ID: {trace_id[:60]}...")

    expected = record.get("expected_tool", "?")
    simulated = record.get("simulated_tool", "?")
    print(f"Expected tool: {expected} | Simulated (bad) tool: {simulated}")

    baseline_outcome = record.get("baseline_outcome", "?")
    cb_outcome = record.get("cb_outcome", "?")

    # Outcome emojis
    outcome_emoji = {
        "attack_success": "❌",
        "correct_behavior": "✅",
        "no_tool_call": "⚠️",
        "other_tool": "❓",
    }

    print(f"\nBaseline outcome: {outcome_emoji.get(baseline_outcome, '?')} {baseline_outcome}")
    print(f"CB model outcome: {outcome_emoji.get(cb_outcome, '?')} {cb_outcome}")

    # Show if this is an improvement
    if baseline_outcome == "attack_success" and cb_outcome != "attack_success":
        print(">>> IMPROVEMENT: CB blocked the attack!")
    elif baseline_outcome != "attack_success" and cb_outcome == "attack_success":
        print(">>> REGRESSION: CB allowed the attack!")
    elif baseline_outcome == cb_outcome:
        print(">>> NO CHANGE")

    # Show responses
    baseline_response = record.get("baseline_response", "")
    cb_response = record.get("cb_response", "")

    print(f"\n--- Baseline Response ({len(baseline_response)} chars) ---")
    print(baseline_response[:max_response_len])
    if len(baseline_response) > max_response_len:
        print(f"... [truncated {len(baseline_response) - max_response_len} more chars]")

    print(f"\n--- CB Model Response ({len(cb_response)} chars) ---")
    print(cb_response[:max_response_len])
    if len(cb_response) > max_response_len:
        print(f"... [truncated {len(cb_response) - max_response_len} more chars]")


def analyze_paired_outputs(records: list) -> dict:
    """Analyze paired outputs and compute metrics."""
    total = len(records)
    if total == 0:
        return {"error": "No records"}

    improvements = 0  # CB blocked attack that baseline allowed
    regressions = 0   # CB allowed attack that baseline blocked
    same = 0          # Same outcome
    different = 0     # Different but not clear improvement/regression

    baseline_attack_success = 0
    cb_attack_success = 0

    for r in records:
        baseline_outcome = r.get("baseline_outcome", "")
        cb_outcome = r.get("cb_outcome", "")

        if baseline_outcome == "attack_success":
            baseline_attack_success += 1
        if cb_outcome == "attack_success":
            cb_attack_success += 1

        if baseline_outcome == cb_outcome:
            same += 1
        elif baseline_outcome == "attack_success" and cb_outcome != "attack_success":
            improvements += 1
        elif baseline_outcome != "attack_success" and cb_outcome == "attack_success":
            regressions += 1
        else:
            different += 1

    return {
        "total": total,
        "baseline_asr": baseline_attack_success / total if total > 0 else 0,
        "cb_asr": cb_attack_success / total if total > 0 else 0,
        "improvements": improvements,
        "regressions": regressions,
        "same_outcome": same,
        "different_other": different,
        "improvement_rate": improvements / total if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs CB model outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--paired-file",
        type=Path,
        help="Path to paired outputs JSONL from eval (skips generation)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Baseline model",
    )
    parser.add_argument(
        "--cb-adapter",
        type=Path,
        help="Path to CB adapter",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        help="Evaluation data JSONL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of examples to show",
    )
    parser.add_argument(
        "--only-harmful",
        action="store_true",
        help="Only show harmful category samples",
    )
    parser.add_argument(
        "--only-improvements",
        action="store_true",
        help="Only show cases where CB improved over baseline",
    )
    parser.add_argument(
        "--only-regressions",
        action="store_true",
        help="Only show cases where CB regressed from baseline",
    )
    parser.add_argument(
        "--max-response-len",
        type=int,
        default=500,
        help="Max response length to show",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary, no examples",
    )

    args = parser.parse_args()

    records = []

    if args.paired_file:
        # Load existing paired outputs
        if not args.paired_file.exists():
            print(f"ERROR: File not found: {args.paired_file}")
            return 1

        with open(args.paired_file) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        print(f"Loaded {len(records)} paired outputs from {args.paired_file}")

    else:
        # Generate outputs (requires torch)
        if not args.cb_adapter or not args.eval_data:
            parser.error("Need --paired-file OR (--cb-adapter AND --eval-data)")

        print("Generation mode not yet implemented. Use --paired-file with eval output.")
        return 1

    # Filter records
    if args.only_harmful:
        # This would need labels info which paired outputs don't have
        pass

    if args.only_improvements:
        records = [r for r in records
                   if r.get("baseline_outcome") == "attack_success"
                   and r.get("cb_outcome") != "attack_success"]
        print(f"Filtered to {len(records)} improvements")

    if args.only_regressions:
        records = [r for r in records
                   if r.get("baseline_outcome") != "attack_success"
                   and r.get("cb_outcome") == "attack_success"]
        print(f"Filtered to {len(records)} regressions")

    # Compute summary
    summary = analyze_paired_outputs(records)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total samples: {summary['total']}")
    print(f"Baseline ASR:  {summary['baseline_asr']:.1%}")
    print(f"CB Model ASR:  {summary['cb_asr']:.1%}")
    print(f"ASR Reduction: {summary['baseline_asr'] - summary['cb_asr']:.1%}")
    print()
    print(f"Improvements (CB blocked attack): {summary['improvements']} ({summary['improvement_rate']:.1%})")
    print(f"Regressions (CB allowed attack):  {summary['regressions']}")
    print(f"Same outcome:                     {summary['same_outcome']}")
    print(f"Different (other):                {summary['different_other']}")

    if args.summary_only:
        return 0

    # Show examples
    show_records = records[:args.limit]
    print(f"\nShowing {len(show_records)} examples:")

    for i, record in enumerate(show_records, 1):
        print_paired_example(i, record, args.max_response_len)

    return 0


if __name__ == "__main__":
    sys.exit(main())
