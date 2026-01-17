#!/usr/bin/env python3
"""
Stage 2: Merge Data for Circuit Breaker Training

This script merges all harmful and retain data sources into a single
training file with the target Dr:Ds ratio (default 5:1).

Data Sources:
  Ds (harmful/shut-down):
    - Fujitsu B4 harmful samples
    - AgentDojo injection failures
    - Existing CB harmful data
    
  Dr (retain/safe):
    - Adversarial-safe samples (model resisted injection)
    - AgentDojo successful resistance samples
    - UltraChat general conversation
    - TAU2 customer service traces
    - XSTest borderline cases

Usage:
    python scripts/cb_data_generation/merge_stage2_data.py \
        --output data/circuit_breakers/stage2_training.jsonl \
        --dr-ratio 5.0
"""

import argparse
import json
import logging
import random
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


# =============================================================================
# Data Loading
# =============================================================================

def load_jsonl(path: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load JSONL file, return list of samples."""
    samples = []
    
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return samples
    
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {i+1} in {path.name}: {e}")
                continue
    
    return samples


def validate_sample(sample: Dict[str, Any]) -> bool:
    """Validate a sample has required fields."""
    if not isinstance(sample, dict):
        return False
    
    # Must have messages or context
    has_context = "messages" in sample or "context" in sample
    
    # Must have response
    has_response = "assistant_raw" in sample or "completion" in sample
    
    return has_context and has_response


def normalize_sample(sample: Dict[str, Any], split: str, source: str) -> Dict[str, Any]:
    """Normalize sample to standard format."""
    # Handle messages/context field
    if "messages" not in sample and "context" in sample:
        sample["messages"] = sample.pop("context")
    
    # Handle assistant_raw/completion field
    if "assistant_raw" not in sample and "completion" in sample:
        sample["assistant_raw"] = sample.pop("completion")
    
    # Ensure metadata exists
    if "metadata" not in sample:
        sample["metadata"] = {}
    
    # Set split and source
    sample["metadata"]["split"] = split
    sample["metadata"]["source"] = source

    # Ensure labels exists
    if "labels" not in sample:
        sample["labels"] = {}

    # Ensure labels split is set
    if "split" not in sample["labels"]:
        sample["labels"]["split"] = split

    # Tool-call marker
    assistant_raw = sample.get("assistant_raw", "")
    sample["metadata"]["has_tool_calls"] = "<|python_tag|>" in assistant_raw

    # Ensure tools field for tool-call samples
    if sample["metadata"]["has_tool_calls"] and not sample.get("tools"):
        if source in ("agentdojo", "agentdojo_failures", "agentdojo_resisted"):
            sample["tools"] = "agentdojo_native"
        elif source in ("tau2", "tau2_traces"):
            sample["tools"] = "tau2_native"
        elif source in ("adversarial_safe", "b4_advsafe"):
            sample["tools"] = "b4_standard_v1"
        else:
            # Fallback to B4 schema when tool calls exist
            sample["tools"] = "b4_standard_v1"
    
    return sample


def deduplicate_samples(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Deduplicate samples by id (preferred) or content hash."""
    seen = set()
    unique = []
    dupes = 0

    for s in samples:
        sample_id = s.get("id")
        if not sample_id:
            payload = json.dumps(
                {
                    "messages": s.get("messages"),
                    "assistant_raw": s.get("assistant_raw"),
                },
                sort_keys=True,
            )
            sample_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
            s["id"] = f"auto_{sample_id}"

        if sample_id in seen:
            dupes += 1
            continue
        seen.add(sample_id)
        unique.append(s)

    return unique, dupes


def load_samples_from_dir(
    data_dir: Path,
    split: str,
    max_per_source: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Load and normalize all JSONL files in a directory."""
    all_samples: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}

    if not data_dir.exists():
        logger.warning(f"Directory not found: {data_dir}")
        return all_samples, source_counts

    for path in sorted(data_dir.glob("*.jsonl")):
        source_name = path.stem
        samples = load_jsonl(path, max_per_source)
        valid_samples = []
        for s in samples:
            if validate_sample(s):
                normalized = normalize_sample(s, split, source_name)
                valid_samples.append(normalized)
        all_samples.extend(valid_samples)
        source_counts[source_name] = len(valid_samples)
        logger.info(f"  {source_name}: {len(valid_samples)} samples")

    return all_samples, source_counts


# =============================================================================
# Data Collection
# =============================================================================

def collect_harmful_data(base_dir: Path, max_per_source: Optional[int] = None) -> Tuple[List[Dict], Dict[str, int]]:
    """Collect all harmful (Ds) data sources."""
    all_samples = []
    source_counts = {}
    
    # Potential paths for harmful data
    harmful_paths = [
        # Stage 2 generated data
        (base_dir / "data/circuit_breakers/harmful/fujitsu_b4_harmful.jsonl", "b4_harmful"),
        (base_dir / "data/circuit_breakers/harmful/agentdojo_failures.jsonl", "agentdojo_failures"),
        
        # Existing CB data
        (base_dir / "data/circuit_breakers/cb_training_harmful.jsonl", "cb_harmful"),
        (base_dir / "data/cb_mvp/cb_training_harmful.jsonl", "cb_harmful_mvp"),
        
        # MVP data
        (base_dir / "data/cb_mvp/harmful.jsonl", "mvp_harmful"),
        
        # Agent-harm data
        (base_dir / "data/agent_harm/agent_harm_harmful.jsonl", "agent_harm"),
    ]
    
    for path, source_name in harmful_paths:
        if path.exists():
            samples = load_jsonl(path, max_per_source)
            
            # Normalize and validate
            valid_samples = []
            for s in samples:
                if validate_sample(s):
                    normalized = normalize_sample(s, "harmful", source_name)
                    valid_samples.append(normalized)
            
            all_samples.extend(valid_samples)
            source_counts[source_name] = len(valid_samples)
            logger.info(f"  {source_name}: {len(valid_samples)} samples")
    
    return all_samples, source_counts


def collect_retain_data(base_dir: Path, max_per_source: Optional[int] = None) -> Tuple[List[Dict], Dict[str, int]]:
    """Collect all retain (Dr) data sources with priority weighting."""
    all_samples = []
    source_counts = {}
    
    # Priority order for retain data (critical first)
    retain_sources = [
        # CRITICAL: Adversarial-safe samples (model resisted injection)
        {
            "path": base_dir / "data/circuit_breakers/retain/adversarial_safe.jsonl",
            "name": "adversarial_safe",
            "priority": "critical",
            "weight": 2.0,
        },
        # AgentDojo successful resistance
        {
            "path": base_dir / "data/circuit_breakers/retain/agentdojo_resisted.jsonl",
            "name": "agentdojo_resisted",
            "priority": "high",
            "weight": 1.5,
        },
        # TAU2 customer service
        {
            "path": base_dir / "data/circuit_breakers/retain/tau2_traces.jsonl",
            "name": "tau2",
            "priority": "high",
            "weight": 1.2,
        },
        # XSTest borderline
        {
            "path": base_dir / "data/circuit_breakers/retain/xstest_borderline.jsonl",
            "name": "xstest",
            "priority": "medium",
            "weight": 1.0,
        },
        # UltraChat general
        {
            "path": base_dir / "data/circuit_breakers/retain/ultrachat_subset.jsonl",
            "name": "ultrachat",
            "priority": "medium",
            "weight": 1.0,
        },
        # Existing retain data
        {
            "path": base_dir / "data/circuit_breakers/cb_training_retain.jsonl",
            "name": "cb_retain",
            "priority": "low",
            "weight": 0.8,
        },
        {
            "path": base_dir / "data/cb_mvp/retain.jsonl",
            "name": "mvp_retain",
            "priority": "low",
            "weight": 0.8,
        },
    ]
    
    for source in retain_sources:
        path = source["path"]
        source_name = source["name"]
        weight = source["weight"]
        
        if path.exists():
            samples = load_jsonl(path, max_per_source)
            
            # Normalize and validate
            valid_samples = []
            for s in samples:
                if validate_sample(s):
                    normalized = normalize_sample(s, "retain", source_name)
                    # Add weight to metadata
                    normalized["metadata"]["weight"] = weight
                    normalized["metadata"]["priority"] = source["priority"]
                    if "priority_class" not in normalized.get("labels", {}):
                        normalized.setdefault("labels", {})["priority_class"] = source_name
                    valid_samples.append(normalized)
            
            all_samples.extend(valid_samples)
            source_counts[source_name] = len(valid_samples)
            logger.info(f"  {source_name}: {len(valid_samples)} samples (weight={weight})")
    
    return all_samples, source_counts


# =============================================================================
# Data Merging
# =============================================================================

def merge_with_ratio(
    harmful_samples: List[Dict],
    retain_samples: List[Dict],
    dr_ratio: float = 5.0,
    shuffle: bool = True,
    seed: int = 42,
) -> List[Dict]:
    """
    Merge harmful and retain samples with target ratio.
    
    Args:
        harmful_samples: List of Ds samples
        retain_samples: List of Dr samples
        dr_ratio: Target Dr:Ds ratio
        shuffle: Whether to shuffle the merged data
        seed: Random seed for shuffling
    
    Returns:
        Merged list of samples
    """
    n_harmful = len(harmful_samples)
    n_retain = len(retain_samples)
    
    target_retain = int(n_harmful * dr_ratio)
    
    logger.info(f"\nMerging with Dr:Ds ratio = {dr_ratio}")
    logger.info(f"  Harmful (Ds): {n_harmful}")
    logger.info(f"  Retain (Dr): {n_retain}")
    logger.info(f"  Target retain: {target_retain}")
    
    # Adjust retain samples
    if n_retain > target_retain:
        # Prioritize by weight
        sorted_retain = sorted(
            retain_samples,
            key=lambda x: x.get("metadata", {}).get("weight", 1.0),
            reverse=True
        )
        retain_samples = sorted_retain[:target_retain]
        logger.info(f"  Using top {target_retain} retain samples by priority")
    elif n_retain < target_retain:
        # Oversample by weight to reach target ratio
        logger.warning(f"  Not enough retain samples. Have {n_retain}, need {target_retain}")

        if n_retain > 0:
            rng = random.Random(seed)
            weights = [s.get("metadata", {}).get("weight", 1.0) for s in retain_samples]
            # Sample with replacement to reach target size
            oversampled = []
            while len(oversampled) < target_retain:
                s = rng.choices(retain_samples, weights=weights, k=1)[0].copy()
                base_id = s.get("id", "auto")
                s["id"] = f"{base_id}_dup{len(oversampled)}"
                oversampled.append(s)

            retain_samples = oversampled[:target_retain]
            logger.info(f"  Oversampled to {len(retain_samples)} retain samples")
    
    # Merge
    merged = harmful_samples + retain_samples
    
    # Add index and compute final ratios
    actual_harmful = sum(1 for s in merged if s.get("metadata", {}).get("split") == "harmful")
    actual_retain = sum(1 for s in merged if s.get("metadata", {}).get("split") == "retain")
    actual_ratio = actual_retain / max(1, actual_harmful)
    
    logger.info(f"\nFinal dataset:")
    logger.info(f"  Harmful: {actual_harmful}")
    logger.info(f"  Retain: {actual_retain}")
    logger.info(f"  Actual ratio: {actual_ratio:.2f}")
    
    # Shuffle
    if shuffle:
        random.seed(seed)
        random.shuffle(merged)
        logger.info("  Shuffled merged dataset")
    
    return merged


def split_train_eval(
    samples: List[Dict[str, Any]],
    eval_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split samples into train/eval sets."""
    if eval_fraction <= 0:
        return samples, []

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n_eval = int(len(samples) * eval_fraction)
    eval_idx = set(indices[:n_eval])

    train_samples = [s for i, s in enumerate(samples) if i not in eval_idx]
    eval_samples = [s for i, s in enumerate(samples) if i in eval_idx]

    return train_samples, eval_samples


def validate_merge(
    harmful_samples: List[Dict[str, Any]],
    retain_samples: List[Dict[str, Any]],
    merged_samples: List[Dict[str, Any]],
    min_ratio: float = 4.0,
    min_adv_safe: int = 400,
) -> Tuple[bool, List[str]]:
    """Validate merged dataset against Stage 2 gates."""
    errors = []

    # Ratio check
    harmful_count = sum(1 for s in merged_samples if s.get("labels", {}).get("split") == "harmful")
    retain_count = sum(1 for s in merged_samples if s.get("labels", {}).get("split") == "retain")
    ratio = retain_count / max(1, harmful_count)
    if ratio < min_ratio:
        errors.append(f"Dr:Ds ratio {ratio:.2f} < {min_ratio}")

    # Adversarial-safe count check
    adv_safe = sum(
        1 for s in retain_samples
        if s.get("labels", {}).get("is_adversarial_safe") is True
    )
    if adv_safe < min_adv_safe:
        errors.append(f"Adversarial-safe count {adv_safe} < {min_adv_safe}")

    # Duplicate IDs check
    ids = [s.get("id") for s in merged_samples if s.get("id")]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate IDs found after merge")

    # Tools presence check for tool samples
    for s in merged_samples:
        labels = s.get("labels", {})
        priority_class = labels.get("priority_class")
        split = labels.get("split")
        tools = s.get("tools")
        tool_optional = (
            split == "retain" and
            priority_class == "general_conversation"
        )
        if not tool_optional and (tools is None or tools == ""):
            errors.append(f"Missing tools for sample {s.get('id')}")
            break

    return len(errors) == 0, errors


def compute_statistics(samples: List[Dict]) -> Dict[str, Any]:
    """Compute statistics for the merged dataset."""
    stats = {
        "total": len(samples),
        "by_split": {},
        "by_source": {},
        "has_tool_calls": 0,
        "avg_messages": 0,
    }
    
    total_messages = 0
    
    for s in samples:
        split = s.get("metadata", {}).get("split", "unknown")
        source = s.get("metadata", {}).get("source", "unknown")
        has_tools = s.get("metadata", {}).get("has_tool_calls", False)
        
        # Count by split
        stats["by_split"][split] = stats["by_split"].get(split, 0) + 1
        
        # Count by source
        stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
        
        # Count tool calls
        if has_tools:
            stats["has_tool_calls"] += 1
        
        # Count messages
        messages = s.get("messages", [])
        total_messages += len(messages)
    
    stats["avg_messages"] = total_messages / max(1, len(samples))
    
    return stats


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge Stage 2 training data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "data/circuit_breakers/stage2/train.jsonl",
        help="Output path for merged training data",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=None,
        help="Output path for eval split JSONL",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.0,
        help="Fraction of merged data to reserve for eval",
    )
    parser.add_argument(
        "--harmful-dir",
        type=Path,
        default=None,
        help="Directory containing harmful JSONL files",
    )
    parser.add_argument(
        "--retain-dir",
        type=Path,
        default=None,
        help="Directory containing retain JSONL files",
    )
    parser.add_argument(
        "--dr-ratio",
        type=float,
        default=5.0,
        help="Target Dr:Ds ratio (default: 5.0)",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="Maximum samples per source (for testing)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle the merged data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Output path for statistics JSON",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run Stage 2 merge validation gates",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("STAGE 2 DATA MERGE")
    logger.info("=" * 60)
    
    # Collect harmful data
    logger.info("\nCollecting harmful (Ds) data:")
    if args.harmful_dir:
        harmful_samples, harmful_counts = load_samples_from_dir(
            args.harmful_dir, "harmful", args.max_per_source
        )
    else:
        harmful_samples, harmful_counts = collect_harmful_data(
            BASE_DIR, args.max_per_source
        )
    
    # Collect retain data
    logger.info("\nCollecting retain (Dr) data:")
    if args.retain_dir:
        retain_samples, retain_counts = load_samples_from_dir(
            args.retain_dir, "retain", args.max_per_source
        )
    else:
        retain_samples, retain_counts = collect_retain_data(
            BASE_DIR, args.max_per_source
        )
    
    if not harmful_samples:
        logger.error("No harmful samples found! Run data generation scripts first.")
        return 1
    
    if not retain_samples:
        logger.error("No retain samples found! Run data generation scripts first.")
        return 1
    
    # Deduplicate
    harmful_samples, harmful_dupes = deduplicate_samples(harmful_samples)
    retain_samples, retain_dupes = deduplicate_samples(retain_samples)
    if harmful_dupes > 0 or retain_dupes > 0:
        logger.info(f"Deduplicated harmful: {harmful_dupes} duplicates removed")
        logger.info(f"Deduplicated retain: {retain_dupes} duplicates removed")

    # Merge with ratio
    merged = merge_with_ratio(
        harmful_samples,
        retain_samples,
        dr_ratio=args.dr_ratio,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )

    # Train/eval split
    train_samples, eval_samples = split_train_eval(
        merged,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    
    # Compute statistics
    stats = compute_statistics(train_samples)
    stats["harmful_sources"] = harmful_counts
    stats["retain_sources"] = retain_counts
    stats["dr_ratio_target"] = args.dr_ratio
    stats["eval_fraction"] = args.eval_fraction
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Save merged data (train)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"\nSaved {len(train_samples)} samples to {args.output}")

    # Save eval data
    if args.eval_output and eval_samples:
        args.eval_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.eval_output, "w", encoding="utf-8") as f:
            for sample in eval_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(eval_samples)} eval samples to {args.eval_output}")
    
    # Save statistics
    stats_output = args.stats_output
    if stats_output is None and args.output.parent.name == "stage2":
        stats_output = args.output.parent / "stats.json"
    if stats_output:
        with open(stats_output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_output}")

    # Validate merge gates
    if args.validate:
        ok, errors = validate_merge(
            harmful_samples,
            retain_samples,
            train_samples,
            min_ratio=4.0,
            min_adv_safe=400,
        )
        if not ok:
            logger.error("Merge validation failed:")
            for err in errors:
                logger.error(f"  - {err}")
            return 1
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total samples: {stats['total']}")
    logger.info(f"By split:")
    for split, count in stats["by_split"].items():
        pct = 100 * count / stats["total"]
        logger.info(f"  {split}: {count} ({pct:.1f}%)")
    logger.info(f"By source:")
    for source, count in sorted(stats["by_source"].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats["total"]
        logger.info(f"  {source}: {count} ({pct:.1f}%)")
    logger.info(f"Samples with tool calls: {stats['has_tool_calls']}")
    logger.info(f"Avg messages per sample: {stats['avg_messages']:.1f}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
