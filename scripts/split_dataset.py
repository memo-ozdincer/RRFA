#!/usr/bin/env python3
"""
Generic Dataset Splitter for CB Training

Splits any dataset into CB and retain sets based on:
1. Existing labels (attack_succeeded field)
2. Judge-based evaluation (for unlabeled datasets)
3. Configuration from dataset_config.yaml

This is a unified replacement for dataset-specific split scripts.

Usage:
    # Split using existing labels
    python scripts/split_dataset.py --traces data.jsonl --output-dir split/

    # Split using judge evaluation (for unlabeled data)
    python scripts/split_dataset.py --traces data.jsonl --output-dir split/ \\
        --use-judge --judge-provider openai

    # Split with dataset config
    python scripts/split_dataset.py --traces data.jsonl --output-dir split/ \\
        --config configs/dataset_config.yaml --dataset agentdojo

    # Split specific files with renders/lossmasks
    python scripts/split_dataset.py \\
        --traces traces.jsonl \\
        --renders renders.jsonl \\
        --lossmasks lossmasks.jsonl \\
        --output-dir split/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, NamedTuple, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class TraceInfo(NamedTuple):
    """Information about a trace for splitting purposes."""
    attack_succeeded: bool
    attack_present: bool
    category: str  # "harmful", "resisted", "benign"


def load_dataset_config(config_path: Path, dataset_name: str) -> Dict[str, Any]:
    """Load dataset configuration from YAML file."""
    if not HAS_YAML:
        raise ImportError("PyYAML required for config loading. Install with: pip install pyyaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    datasets = config.get("datasets", {})
    if dataset_name not in datasets:
        available = list(datasets.keys())
        raise ValueError(f"Dataset '{dataset_name}' not found in config. Available: {available}")

    return datasets[dataset_name]


def get_nested_field(obj: Dict[str, Any], field_path: str) -> Any:
    """Get a nested field from a dictionary using dot notation."""
    parts = field_path.split(".")
    value = obj
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def determine_trace_info_from_labels(trace: Dict[str, Any]) -> TraceInfo:
    """
    Determine trace info from existing labels.

    Supports multiple formats:
    - New format: labels.attack_succeeded, labels.attack_present
    - Old format: labels.is_harmful
    - Category-only: labels.category
    """
    labels = trace.get("labels", {})

    # New format with explicit fields
    if "attack_succeeded" in labels:
        attack_succeeded = bool(labels.get("attack_succeeded", False))
        attack_present = bool(labels.get("attack_present", attack_succeeded))
        category = labels.get("category", "harmful" if attack_succeeded else "benign")
        return TraceInfo(attack_succeeded, attack_present, category)

    # Old format: is_harmful
    if "is_harmful" in labels:
        is_harmful = labels.get("is_harmful", False)
        return TraceInfo(
            attack_succeeded=is_harmful,
            attack_present=is_harmful,
            category="harmful" if is_harmful else "benign",
        )

    # Category-only format
    if "category" in labels:
        category = labels.get("category", "benign")
        attack_succeeded = category == "harmful"
        attack_present = category in ("harmful", "resisted")
        return TraceInfo(attack_succeeded, attack_present, category)

    # No labels - assume benign
    return TraceInfo(
        attack_succeeded=False,
        attack_present=False,
        category="benign",
    )


def determine_trace_info_with_config(
    trace: Dict[str, Any],
    config: Dict[str, Any],
) -> TraceInfo:
    """Determine trace info using dataset configuration."""
    labels_config = config.get("labels", {})

    # Check attack presence
    attack_present_field = labels_config.get("attack_present_field")
    attack_present_check = labels_config.get("attack_present_check", "is_not_null")

    attack_present = False
    if attack_present_check == "always_true":
        attack_present = True
    elif attack_present_check == "is_not_null" and attack_present_field:
        attack_present = get_nested_field(trace, attack_present_field) is not None
    elif attack_present_check == "use_injection_detection":
        # Would need injection detection here
        # Fall back to label-based
        pass

    # Determine attack success
    attack_succeeded_source = labels_config.get("attack_succeeded_source", "field")
    attack_succeeded = False

    if attack_succeeded_source == "field":
        field = labels_config.get("attack_succeeded_field", "labels.attack_succeeded")
        attack_succeeded = bool(get_nested_field(trace, field))
    elif attack_succeeded_source == "derived":
        logic = labels_config.get("attack_succeeded_logic", "")
        if "NOT metadata.security" in logic:
            security = get_nested_field(trace, "metadata.security")
            attack_succeeded = attack_present and not security

    # Determine category
    if attack_present:
        if attack_succeeded:
            category = "harmful"
        else:
            category = "resisted"
    else:
        category = "benign"

    return TraceInfo(attack_succeeded, attack_present, category)


def load_trace_info(
    traces_path: Path,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, TraceInfo]:
    """
    Load trace_id -> TraceInfo mapping from traces file.

    Args:
        traces_path: Path to traces JSONL file
        config: Optional dataset config for determining labels

    Returns:
        Dict mapping trace_id to TraceInfo
    """
    info_map = {}

    with open(traces_path) as f:
        for line in f:
            if not line.strip():
                continue

            trace = json.loads(line)
            trace_id = trace.get("id") or trace.get("trace_id")
            if not trace_id:
                continue

            if config:
                info = determine_trace_info_with_config(trace, config)
            else:
                info = determine_trace_info_from_labels(trace)

            info_map[trace_id] = info

    return info_map


def split_file(
    input_path: Path,
    cb_out: Path,
    retain_out: Path,
    trace_info: Optional[Dict[str, TraceInfo]],
    resisted_handling: Literal["retain", "separate", "exclude"] = "retain",
    resisted_out: Optional[Path] = None,
    is_traces_file: bool = False,
) -> Dict[str, int]:
    """
    Split a file into CB and retain sets.

    Args:
        input_path: Input JSONL file
        cb_out: Output path for CB set
        retain_out: Output path for retain set
        trace_info: Optional mapping of trace_id -> TraceInfo (for renders/lossmasks)
        resisted_handling: How to handle resisted traces
        resisted_out: Output path for resisted set (if separate)
        is_traces_file: If True, extract labels from each row directly

    Returns:
        Counts dictionary
    """
    counts = {"cb": 0, "retain": 0, "resisted": 0, "missing": 0, "total": 0}

    # Ensure output directories exist
    cb_out.parent.mkdir(parents=True, exist_ok=True)
    retain_out.parent.mkdir(parents=True, exist_ok=True)

    f_resisted = None
    if resisted_handling == "separate" and resisted_out:
        resisted_out.parent.mkdir(parents=True, exist_ok=True)
        f_resisted = open(resisted_out, "w")

    with open(input_path) as f_in, \
         open(cb_out, "w") as f_cb, \
         open(retain_out, "w") as f_retain:

        for line in f_in:
            if not line.strip():
                continue

            counts["total"] += 1
            row = json.loads(line)

            # Get trace info
            if is_traces_file:
                # For traces files, extract directly from the row
                info = determine_trace_info_from_labels(row)
            else:
                # For renders/lossmasks, look up by trace_id
                trace_id = row.get("trace_id")
                if trace_id is None or (trace_info and trace_id not in trace_info):
                    counts["missing"] += 1
                    # Default to retain for unknown traces
                    f_retain.write(line)
                    counts["retain"] += 1
                    continue
                info = trace_info[trace_id] if trace_info else TraceInfo(False, False, "benign")

            # Split based on category
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
                # else: exclude
            else:
                f_retain.write(line)
                counts["retain"] += 1

    if f_resisted:
        f_resisted.close()

    if counts["missing"] > 0:
        print(f"  WARNING: {counts['missing']} rows had no/unknown trace_id, defaulted to retain")

    return counts


def split_with_judge(
    traces_path: Path,
    output_dir: Path,
    prefix: str,
    judge_provider: str = "openai",
    judge_model: Optional[str] = None,
    resisted_handling: str = "retain",
    renders_path: Optional[Path] = None,
    lossmasks_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    """
    Split traces using judge-based evaluation.

    First evaluates all traces with a judge to add attack_succeeded labels,
    then splits based on those labels.
    """
    from src.evaluation.judge import create_judge, JudgeResult

    # Create judge
    judge = create_judge(
        provider=judge_provider,
        model=judge_model,
        cache_enabled=True,
    )

    print(f"Using judge: {judge.model} ({judge_provider})")

    # First pass: evaluate traces and build info map
    trace_info: Dict[str, TraceInfo] = {}
    labeled_traces: List[Dict[str, Any]] = []

    with open(traces_path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue

            if limit and i >= limit:
                break

            trace = json.loads(line)
            trace_id = trace.get("id", f"trace_{i}")

            # Evaluate with judge
            result = judge.evaluate(trace)

            # Create trace info
            if result.injection_text is None:
                info = TraceInfo(False, False, "benign")
            elif result.attack_succeeded:
                info = TraceInfo(True, True, "harmful")
            else:
                info = TraceInfo(False, True, "resisted")

            trace_info[trace_id] = info

            # Add labels to trace
            if "labels" not in trace:
                trace["labels"] = {}
            trace["labels"]["attack_succeeded"] = result.attack_succeeded
            trace["labels"]["attack_present"] = result.injection_text is not None
            trace["labels"]["category"] = info.category
            trace["labels"]["judge_confidence"] = result.confidence
            trace["labels"]["judge_model"] = result.judge_model

            labeled_traces.append(trace)

            if (i + 1) % 50 == 0:
                print(f"  Evaluated {i + 1} traces...")

    # Write labeled traces to temp file, then split
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = output_dir / f"{prefix}_labeled.jsonl"

    with open(labeled_path, "w") as f:
        for trace in labeled_traces:
            f.write(json.dumps(trace) + "\n")

    # Now split using the labeled traces
    cb_suffix = "cb"
    retain_suffix = "retain"

    # Split traces
    traces_cb = output_dir / f"{prefix}_traces_{cb_suffix}.jsonl"
    traces_retain = output_dir / f"{prefix}_traces_{retain_suffix}.jsonl"
    traces_resisted = output_dir / f"{prefix}_traces_resisted.jsonl" if resisted_handling == "separate" else None

    counts = split_file(
        labeled_path,
        traces_cb,
        traces_retain,
        None,  # We'll read from the labeled file directly
        resisted_handling=resisted_handling,
        resisted_out=traces_resisted,
        is_traces_file=True,
    )

    print(f"\nTraces split:")
    print(f"  CB: {counts['cb']} -> {traces_cb}")
    print(f"  Retain: {counts['retain']} -> {traces_retain}")
    if resisted_handling == "separate":
        print(f"  Resisted: {counts['resisted']} -> {traces_resisted}")

    # Split renders if provided
    if renders_path and renders_path.exists():
        renders_cb = output_dir / f"{prefix}_renders_{cb_suffix}.jsonl"
        renders_retain = output_dir / f"{prefix}_renders_{retain_suffix}.jsonl"
        renders_resisted = output_dir / f"{prefix}_renders_resisted.jsonl" if resisted_handling == "separate" else None

        r_counts = split_file(
            renders_path,
            renders_cb,
            renders_retain,
            trace_info,
            resisted_handling=resisted_handling,
            resisted_out=renders_resisted,
        )
        print(f"\nRenders split:")
        print(f"  CB: {r_counts['cb']} -> {renders_cb}")
        print(f"  Retain: {r_counts['retain']} -> {renders_retain}")

    # Split lossmasks if provided
    if lossmasks_path and lossmasks_path.exists():
        lossmasks_cb = output_dir / f"{prefix}_lossmasks_{cb_suffix}.jsonl"
        lossmasks_retain = output_dir / f"{prefix}_lossmasks_{retain_suffix}.jsonl"
        lossmasks_resisted = output_dir / f"{prefix}_lossmasks_resisted.jsonl" if resisted_handling == "separate" else None

        l_counts = split_file(
            lossmasks_path,
            lossmasks_cb,
            lossmasks_retain,
            trace_info,
            resisted_handling=resisted_handling,
            resisted_out=lossmasks_resisted,
        )
        print(f"\nLossmasks split:")
        print(f"  CB: {l_counts['cb']} -> {lossmasks_cb}")
        print(f"  Retain: {l_counts['retain']} -> {lossmasks_retain}")

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into CB and retain sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input files
    parser.add_argument("--traces", type=Path, required=True, help="Input traces JSONL")
    parser.add_argument("--renders", type=Path, help="Input renders JSONL (optional)")
    parser.add_argument("--lossmasks", type=Path, help="Input lossmasks JSONL (optional)")

    # Output
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--prefix", type=str, default="dataset", help="Output filename prefix")

    # Configuration
    parser.add_argument("--config", type=Path, help="Dataset config YAML file")
    parser.add_argument("--dataset", type=str, help="Dataset name (required if using --config)")

    # Judge-based evaluation
    parser.add_argument(
        "--use-judge",
        action="store_true",
        help="Use LLM judge to evaluate traces without labels",
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        choices=["openai", "anthropic", "local"],
        default="openai",
        help="Judge provider (default: openai)",
    )
    parser.add_argument("--judge-model", type=str, help="Judge model")

    # Split options
    parser.add_argument(
        "--resisted-handling",
        type=str,
        choices=["retain", "separate", "exclude"],
        default="retain",
        help="How to handle resisted traces",
    )

    # Legacy options
    parser.add_argument(
        "--legacy-naming",
        action="store_true",
        help="Use legacy naming (harmful/benign instead of cb/retain)",
    )

    # Misc
    parser.add_argument("--limit", type=int, help="Limit traces to process (for testing)")

    args = parser.parse_args()

    # Validate arguments
    if args.config and not args.dataset:
        parser.error("--dataset required when using --config")

    if not args.traces.exists():
        print(f"ERROR: Traces file not found: {args.traces}")
        return 1

    print("=" * 60)
    print("CB TRAINING DATA SPLIT")
    print("=" * 60)
    print(f"\nInput traces: {args.traces}")
    print(f"Output directory: {args.output_dir}")

    # Load config if provided
    config = None
    if args.config:
        print(f"Using config: {args.config} (dataset: {args.dataset})")
        config = load_dataset_config(args.config, args.dataset)

    # Use judge if requested
    if args.use_judge:
        print(f"\nUsing judge-based evaluation ({args.judge_provider})")
        counts = split_with_judge(
            args.traces,
            args.output_dir,
            args.prefix,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            resisted_handling=args.resisted_handling,
            renders_path=args.renders,
            lossmasks_path=args.lossmasks,
            limit=args.limit,
        )
    else:
        # Standard label-based split
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Determine suffixes
        if args.legacy_naming:
            cb_suffix = "harmful"
            retain_suffix = "benign"
        else:
            cb_suffix = "cb"
            retain_suffix = "retain"

        # Load trace info
        print(f"\nLoading trace labels...")
        trace_info = load_trace_info(args.traces, config)

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
        traces_cb = args.output_dir / f"{args.prefix}_traces_{cb_suffix}.jsonl"
        traces_retain = args.output_dir / f"{args.prefix}_traces_{retain_suffix}.jsonl"
        traces_resisted = args.output_dir / f"{args.prefix}_traces_resisted.jsonl" if args.resisted_handling == "separate" else None

        print(f"\nSplitting traces: {args.traces}")
        counts = split_file(
            args.traces,
            traces_cb,
            traces_retain,
            None,
            resisted_handling=args.resisted_handling,
            resisted_out=traces_resisted,
            is_traces_file=True,
        )
        print(f"  CB: {counts['cb']} -> {traces_cb}")
        print(f"  Retain: {counts['retain']} -> {traces_retain}")
        if args.resisted_handling == "separate":
            print(f"  Resisted: {counts['resisted']} -> {traces_resisted}")

        # Split renders
        if args.renders and args.renders.exists():
            renders_cb = args.output_dir / f"{args.prefix}_renders_{cb_suffix}.jsonl"
            renders_retain = args.output_dir / f"{args.prefix}_renders_{retain_suffix}.jsonl"
            renders_resisted = args.output_dir / f"{args.prefix}_renders_resisted.jsonl" if args.resisted_handling == "separate" else None

            print(f"\nSplitting renders: {args.renders}")
            r_counts = split_file(
                args.renders,
                renders_cb,
                renders_retain,
                trace_info,
                resisted_handling=args.resisted_handling,
                resisted_out=renders_resisted,
            )
            print(f"  CB: {r_counts['cb']} -> {renders_cb}")
            print(f"  Retain: {r_counts['retain']} -> {renders_retain}")

        # Split lossmasks
        if args.lossmasks and args.lossmasks.exists():
            lossmasks_cb = args.output_dir / f"{args.prefix}_lossmasks_{cb_suffix}.jsonl"
            lossmasks_retain = args.output_dir / f"{args.prefix}_lossmasks_{retain_suffix}.jsonl"
            lossmasks_resisted = args.output_dir / f"{args.prefix}_lossmasks_resisted.jsonl" if args.resisted_handling == "separate" else None

            print(f"\nSplitting lossmasks: {args.lossmasks}")
            l_counts = split_file(
                args.lossmasks,
                lossmasks_cb,
                lossmasks_retain,
                trace_info,
                resisted_handling=args.resisted_handling,
                resisted_out=lossmasks_resisted,
            )
            print(f"  CB: {l_counts['cb']} -> {lossmasks_cb}")
            print(f"  Retain: {l_counts['retain']} -> {lossmasks_retain}")

    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
