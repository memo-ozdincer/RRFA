#!/usr/bin/env python3
"""Split AgentDojo traces/renders/lossmasks into CB and retain sets.

CB Training Split Logic:
------------------------
- CB Set (harmful): Attack present AND model complied (attack_succeeded=True)
  → Use for circuit breaker / rerouting loss

- Retain Set: Attack not present OR model resisted (attack_succeeded=False)
  → Use for retain loss (capability preservation)
  → Includes: benign traces + correctly resisted attacks

This ensures we only apply CB loss to traces where the model exhibited
bad behavior, not traces where it correctly resisted the attack.

The traces file contains the labels fields:
- labels.attack_succeeded: True if model was compromised
- labels.attack_present: True if injection/attack was in the trace
- labels.category: "harmful" | "resisted" | "benign"

Renders and lossmasks are split by looking up trace_id in the traces labels.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Literal, NamedTuple


class TraceInfo(NamedTuple):
    """Information about a trace for splitting purposes."""
    attack_succeeded: bool
    attack_present: bool
    category: str  # "harmful", "resisted", "benign"


def load_trace_info(traces_path: Path) -> Dict[str, TraceInfo]:
    """
    Load trace_id -> TraceInfo mapping from traces file.

    Handles both old format (is_harmful boolean) and new format
    (attack_succeeded, attack_present, category).
    """
    info_map = {}
    with open(traces_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            trace_id = row.get("id") or row.get("trace_id")
            if not trace_id:
                continue

            trace_labels = row.get("labels", {})

            # New format: attack_succeeded field
            if "attack_succeeded" in trace_labels:
                attack_succeeded = bool(trace_labels.get("attack_succeeded", False))
                attack_present = bool(trace_labels.get("attack_present", False))
                category = trace_labels.get("category", "benign")
            # Backward compatibility: old is_harmful format
            elif "is_harmful" in trace_labels:
                # Old format: is_harmful was based on attack presence, not success
                # Treat is_harmful=True as attack_succeeded for backward compat
                is_harmful = trace_labels.get("is_harmful", False)
                attack_succeeded = is_harmful
                attack_present = is_harmful
                category = "harmful" if is_harmful else "benign"
            # Legacy: category-only format
            elif "category" in trace_labels:
                category = trace_labels.get("category", "benign")
                attack_succeeded = category == "harmful"
                attack_present = category in ("harmful", "resisted")
            else:
                # No labels - assume benign
                attack_succeeded = False
                attack_present = False
                category = "benign"

            info_map[trace_id] = TraceInfo(
                attack_succeeded=attack_succeeded,
                attack_present=attack_present,
                category=category,
            )

    return info_map


def split_traces(
    input_path: Path,
    cb_out: Path,
    retain_out: Path,
    resisted_handling: Literal["retain", "separate", "exclude"] = "retain",
    resisted_out: Path = None,
) -> Dict[str, int]:
    """
    Split traces file by CB training logic.

    Args:
        input_path: Input traces JSONL
        cb_out: Output for CB set (attack_succeeded=True)
        retain_out: Output for retain set (benign + optionally resisted)
        resisted_handling: How to handle "resisted" traces:
            - "retain": Include in retain set (default, recommended)
            - "separate": Write to separate file
            - "exclude": Don't include in any output
        resisted_out: Output for resisted traces (only if resisted_handling="separate")

    Returns:
        Dict with counts: cb, retain, resisted, total
    """
    counts = {"cb": 0, "retain": 0, "resisted": 0, "total": 0}

    f_resisted = None
    if resisted_handling == "separate" and resisted_out:
        resisted_out.parent.mkdir(parents=True, exist_ok=True)
        f_resisted = open(resisted_out, 'w')

    with open(input_path) as f_in, \
         open(cb_out, 'w') as f_cb, \
         open(retain_out, 'w') as f_retain:

        for line in f_in:
            if not line.strip():
                continue

            counts["total"] += 1
            row = json.loads(line)
            trace_labels = row.get("labels", {})

            # Determine attack_succeeded using new logic
            if "attack_succeeded" in trace_labels:
                attack_succeeded = bool(trace_labels.get("attack_succeeded", False))
                attack_present = bool(trace_labels.get("attack_present", False))
            elif "is_harmful" in trace_labels:
                # Backward compat
                attack_succeeded = trace_labels.get("is_harmful", False)
                attack_present = attack_succeeded
            else:
                category = trace_labels.get("category", "benign")
                attack_succeeded = category == "harmful"
                attack_present = category in ("harmful", "resisted")

            # Split based on attack_succeeded
            if attack_succeeded:
                # CB set: model was compromised
                f_cb.write(line)
                counts["cb"] += 1
            elif attack_present:
                # Resisted: attack present but model refused
                counts["resisted"] += 1
                if resisted_handling == "retain":
                    f_retain.write(line)
                    counts["retain"] += 1
                elif resisted_handling == "separate" and f_resisted:
                    f_resisted.write(line)
                # else: exclude
            else:
                # Benign: no attack present
                f_retain.write(line)
                counts["retain"] += 1

    if f_resisted:
        f_resisted.close()

    return counts


def split_by_trace_id(
    input_path: Path,
    cb_out: Path,
    retain_out: Path,
    trace_info: Dict[str, TraceInfo],
    resisted_handling: Literal["retain", "separate", "exclude"] = "retain",
    resisted_out: Path = None,
) -> Dict[str, int]:
    """
    Split file by looking up trace_id in info mapping.

    Used for renders/lossmasks that reference traces by ID.
    """
    counts = {"cb": 0, "retain": 0, "resisted": 0, "missing": 0, "total": 0}

    f_resisted = None
    if resisted_handling == "separate" and resisted_out:
        resisted_out.parent.mkdir(parents=True, exist_ok=True)
        f_resisted = open(resisted_out, 'w')

    with open(input_path) as f_in, \
         open(cb_out, 'w') as f_cb, \
         open(retain_out, 'w') as f_retain:

        for line in f_in:
            if not line.strip():
                continue

            counts["total"] += 1
            row = json.loads(line)
            trace_id = row.get("trace_id")

            if trace_id is None or trace_id not in trace_info:
                counts["missing"] += 1
                # Default to retain for unknown traces
                f_retain.write(line)
                counts["retain"] += 1
                continue

            info = trace_info[trace_id]

            if info.attack_succeeded:
                f_cb.write(line)
                counts["cb"] += 1
            elif info.attack_present:
                counts["resisted"] += 1
                if resisted_handling == "retain":
                    f_retain.write(line)
                    counts["retain"] += 1
                elif resisted_handling == "separate" and f_resisted:
                    f_resisted.write(line)
            else:
                f_retain.write(line)
                counts["retain"] += 1

    if f_resisted:
        f_resisted.close()

    if counts["missing"] > 0:
        print(f"  WARNING: {counts['missing']} rows had no/unknown trace_id, defaulted to retain")

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Split AgentDojo data into CB and retain sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CB Training Split Logic:
  CB Set:     attack_succeeded=True (model was compromised)
  Retain Set: attack_succeeded=False OR no attack (correct behavior)

Examples:
  # Basic split (resisted traces go to retain)
  python split_agentdojo.py --traces data.jsonl --output-dir split/

  # Separate resisted traces for analysis
  python split_agentdojo.py --traces data.jsonl --output-dir split/ \\
      --resisted-handling separate

  # Exclude resisted traces entirely
  python split_agentdojo.py --traces data.jsonl --output-dir split/ \\
      --resisted-handling exclude
        """
    )
    parser.add_argument("--traces", type=Path, required=True, help="Input traces JSONL (required for labels)")
    parser.add_argument("--renders", type=Path, help="Input renders JSONL")
    parser.add_argument("--lossmasks", type=Path, help="Input lossmasks JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--prefix", type=str, default="agentdojo", help="Output filename prefix")
    parser.add_argument(
        "--resisted-handling",
        type=str,
        choices=["retain", "separate", "exclude"],
        default="retain",
        help="How to handle traces where attack was present but model resisted: "
             "'retain' (include in retain set, recommended), "
             "'separate' (write to separate file), "
             "'exclude' (don't include in any output)"
    )

    # Legacy compatibility
    parser.add_argument(
        "--legacy-harmful-naming",
        action="store_true",
        help="Use legacy naming (harmful/benign) instead of (cb/retain) for output files"
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output naming
    if args.legacy_harmful_naming:
        cb_suffix = "harmful"
        retain_suffix = "benign"
        resisted_suffix = "resisted"
    else:
        cb_suffix = "cb"
        retain_suffix = "retain"
        resisted_suffix = "resisted"

    # First, load trace labels (this is required)
    print(f"Loading trace labels from: {args.traces}")
    trace_info = load_trace_info(args.traces)

    # Count categories
    cb_count = sum(1 for v in trace_info.values() if v.attack_succeeded)
    resisted_count = sum(1 for v in trace_info.values() if v.attack_present and not v.attack_succeeded)
    benign_count = sum(1 for v in trace_info.values() if not v.attack_present)

    print(f"  Found {len(trace_info)} traces:")
    print(f"    CB (attack succeeded): {cb_count}")
    print(f"    Resisted (attack failed): {resisted_count}")
    print(f"    Benign (no attack): {benign_count}")
    print(f"  Resisted handling: {args.resisted_handling}")

    # Split traces
    if args.traces and args.traces.exists():
        cb_out = args.output_dir / f"{args.prefix}_traces_{cb_suffix}.jsonl"
        retain_out = args.output_dir / f"{args.prefix}_traces_{retain_suffix}.jsonl"
        resisted_out = args.output_dir / f"{args.prefix}_traces_{resisted_suffix}.jsonl" if args.resisted_handling == "separate" else None

        print(f"\nSplitting traces: {args.traces}")
        counts = split_traces(
            args.traces, cb_out, retain_out,
            resisted_handling=args.resisted_handling,
            resisted_out=resisted_out,
        )
        print(f"  CB: {counts['cb']} -> {cb_out}")
        print(f"  Retain: {counts['retain']} -> {retain_out}")
        if args.resisted_handling == "separate":
            print(f"  Resisted: {counts['resisted']} -> {resisted_out}")

    # Split renders using trace_id lookup
    if args.renders and args.renders.exists():
        cb_out = args.output_dir / f"{args.prefix}_renders_{cb_suffix}.jsonl"
        retain_out = args.output_dir / f"{args.prefix}_renders_{retain_suffix}.jsonl"
        resisted_out = args.output_dir / f"{args.prefix}_renders_{resisted_suffix}.jsonl" if args.resisted_handling == "separate" else None

        print(f"\nSplitting renders: {args.renders}")
        counts = split_by_trace_id(
            args.renders, cb_out, retain_out, trace_info,
            resisted_handling=args.resisted_handling,
            resisted_out=resisted_out,
        )
        print(f"  CB: {counts['cb']} -> {cb_out}")
        print(f"  Retain: {counts['retain']} -> {retain_out}")
        if args.resisted_handling == "separate":
            print(f"  Resisted: {counts['resisted']} -> {resisted_out}")

    # Split lossmasks using trace_id lookup
    if args.lossmasks and args.lossmasks.exists():
        cb_out = args.output_dir / f"{args.prefix}_lossmasks_{cb_suffix}.jsonl"
        retain_out = args.output_dir / f"{args.prefix}_lossmasks_{retain_suffix}.jsonl"
        resisted_out = args.output_dir / f"{args.prefix}_lossmasks_{resisted_suffix}.jsonl" if args.resisted_handling == "separate" else None

        print(f"\nSplitting lossmasks: {args.lossmasks}")
        counts = split_by_trace_id(
            args.lossmasks, cb_out, retain_out, trace_info,
            resisted_handling=args.resisted_handling,
            resisted_out=resisted_out,
        )
        print(f"  CB: {counts['cb']} -> {cb_out}")
        print(f"  Retain: {counts['retain']} -> {retain_out}")
        if args.resisted_handling == "separate":
            print(f"  Resisted: {counts['resisted']} -> {resisted_out}")

    print("\n" + "=" * 60)
    print("CB TRAINING DATA SPLIT COMPLETE")
    print("=" * 60)
    print(f"\nUse these files with --mode mixed:")
    if args.legacy_harmful_naming:
        print(f"  --harmful-renders {args.output_dir}/{args.prefix}_renders_harmful.jsonl")
        print(f"  --harmful-lossmasks {args.output_dir}/{args.prefix}_lossmasks_harmful.jsonl")
        print(f"  --benign-renders {args.output_dir}/{args.prefix}_renders_benign.jsonl")
        print(f"  --benign-lossmasks {args.output_dir}/{args.prefix}_lossmasks_benign.jsonl")
    else:
        print(f"  --harmful-renders {args.output_dir}/{args.prefix}_renders_cb.jsonl")
        print(f"  --harmful-lossmasks {args.output_dir}/{args.prefix}_lossmasks_cb.jsonl")
        print(f"  --benign-renders {args.output_dir}/{args.prefix}_renders_retain.jsonl")
        print(f"  --benign-lossmasks {args.output_dir}/{args.prefix}_lossmasks_retain.jsonl")


if __name__ == "__main__":
    main()
