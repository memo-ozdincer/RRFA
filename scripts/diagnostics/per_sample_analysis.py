#!/usr/bin/env python3
"""
D1: Per-sample error categorization on eval outputs.

Reads eval results (next_tool_prediction JSON) and the original traces.
For each sample, reports: domain, attack type, CB output mode, baseline
confidence, and whether the CB successfully defended.

Usage:
    python scripts/diagnostics/per_sample_analysis.py \
        --eval-dir /path/to/eval/next_tool_prediction/ \
        --traces data/traces/agentdojo_truncated.jsonl \
        --output results/per_sample_analysis.json

Expects eval dir to contain:
    - baseline_results.json (or similar) with per-sample predictions
    - cb_results.json with per-sample CB predictions
"""

import argparse
import json
import logging
import re
from pathlib import Path
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def classify_output(text: str) -> str:
    """Classify a model output into modes."""
    if not text or not text.strip():
        return "empty"

    # Check for common gibberish patterns
    stripped = text.strip()

    # Token repetition: <|python_tag|> repeated, "assistant" repeated
    if stripped.count("<|python_tag|>") > 2:
        return "token_repetition"
    if "assistantassistant" in stripped.lower():
        return "token_repetition"

    # Character gibberish: mostly punctuation/brackets
    non_alnum = sum(1 for c in stripped if not c.isalnum() and c != ' ')
    if len(stripped) > 10 and non_alnum / len(stripped) > 0.7:
        return "char_gibberish"

    # Try to parse as JSON tool call
    try:
        # Try direct JSON parse
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "name" in parsed:
            return "valid_tool_call"
    except (json.JSONDecodeError, ValueError):
        pass

    # Look for JSON-like structure
    if '{"name"' in stripped or '{"function"' in stripped:
        try:
            # Find JSON object
            match = re.search(r'\{.*\}', stripped, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if "name" in parsed:
                    return "garbled_json"
        except (json.JSONDecodeError, ValueError):
            return "garbled_json"

    # Coherent text (refusal or conversation)
    words = stripped.split()
    if len(words) > 3:
        return "text_response"

    return "other"


def load_traces_by_id(path: Path) -> dict:
    """Load traces indexed by ID."""
    traces = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            tid = d.get("id", "")
            traces[tid] = d
    return traces


def main():
    parser = argparse.ArgumentParser(description="Per-sample error analysis (D1)")
    parser.add_argument("--eval-dir", type=Path, required=True,
                        help="Directory with eval result JSONs")
    parser.add_argument("--traces", type=Path, default=None,
                        help="Original traces JSONL for metadata (domain, attack type)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Find eval result files
    eval_files = list(args.eval_dir.glob("*.json"))
    if not eval_files:
        logger.error("No JSON files found in %s", args.eval_dir)
        return

    logger.info("Found %d eval files in %s", len(eval_files), args.eval_dir)

    # Load trace metadata if available
    trace_meta = {}
    if args.traces and args.traces.exists():
        logger.info("Loading trace metadata from %s", args.traces)
        trace_meta = load_traces_by_id(args.traces)
        logger.info("  Loaded %d traces", len(trace_meta))

    # Process each eval file
    all_results = []
    for eval_file in sorted(eval_files):
        logger.info("Processing %s", eval_file.name)
        with open(eval_file) as f:
            eval_data = json.load(f)

        # Handle different eval output formats
        samples = []
        if isinstance(eval_data, list):
            samples = eval_data
        elif isinstance(eval_data, dict):
            # Might have "results" key or "samples" key
            samples = eval_data.get("results", eval_data.get("samples", []))
            if not samples and "metrics" in eval_data:
                # Summary file, not per-sample — skip
                logger.info("  Skipping summary file %s", eval_file.name)
                continue

        for sample in samples:
            trace_id = sample.get("trace_id", sample.get("id", ""))
            meta = trace_meta.get(trace_id, {})

            domain = meta.get("source", {}).get("subset", "unknown")
            category = meta.get("labels", {}).get("category", "unknown")
            attack_type = meta.get("task", {}).get("family", "unknown")

            # Get CB output and classify
            cb_output = sample.get("cb_output", sample.get("prediction", ""))
            bl_output = sample.get("baseline_output", sample.get("baseline", ""))

            cb_mode = classify_output(cb_output) if cb_output else "missing"
            bl_mode = classify_output(bl_output) if bl_output else "missing"

            # Get result labels
            cb_correct = sample.get("cb_correct", sample.get("correct", None))
            bl_correct = sample.get("baseline_correct", None)

            # Fujitsu-specific: tool pair info
            tool_attack = meta.get("tool_attack", {})
            expected_tool = tool_attack.get("expected_tool", "")
            observed_tool = tool_attack.get("observed_tool", "")

            result = {
                "trace_id": trace_id,
                "domain": domain,
                "category": category,
                "attack_type": attack_type,
                "cb_output_mode": cb_mode,
                "bl_output_mode": bl_mode,
                "cb_correct": cb_correct,
                "bl_correct": bl_correct,
                "eval_file": eval_file.name,
                "expected_tool": expected_tool,
                "observed_tool": observed_tool,
            }
            all_results.append(result)

    if not all_results:
        logger.error("No samples found in eval files")
        return

    # Analysis
    print("=" * 70)
    print(f"Per-Sample Error Analysis ({len(all_results)} samples)")
    print("=" * 70)

    # By category
    for cat in ["harmful", "benign"]:
        cat_results = [r for r in all_results if r["category"] == cat]
        if not cat_results:
            continue

        print(f"\n{'='*40}")
        print(f"  {cat.upper()} ({len(cat_results)} samples)")
        print(f"{'='*40}")

        # By domain
        domain_counts = Counter(r["domain"] for r in cat_results)
        print(f"\n  By Domain:")
        for domain, count in domain_counts.most_common():
            domain_results = [r for r in cat_results if r["domain"] == domain]
            correct = sum(1 for r in domain_results if r["cb_correct"])
            modes = Counter(r["cb_output_mode"] for r in domain_results)
            print(f"    {domain:>12}: {count} samples, "
                  f"correct={correct}/{count} ({correct/count:.0%})")
            for mode, mc in modes.most_common(3):
                print(f"      {mode}: {mc}")

        # By output mode
        mode_counts = Counter(r["cb_output_mode"] for r in cat_results)
        print(f"\n  By CB Output Mode:")
        for mode, count in mode_counts.most_common():
            print(f"    {mode:>20}: {count} ({count/len(cat_results):.0%})")

    # Gibberish analysis
    gibberish = [r for r in all_results
                 if r["cb_output_mode"] in ("char_gibberish", "token_repetition", "garbled_json")]
    if gibberish:
        print(f"\n{'='*40}")
        print(f"  GIBBERISH ANALYSIS ({len(gibberish)} samples)")
        print(f"{'='*40}")
        by_cat = Counter(r["category"] for r in gibberish)
        by_domain = Counter(r["domain"] for r in gibberish)
        by_mode = Counter(r["cb_output_mode"] for r in gibberish)
        print(f"  By category: {dict(by_cat)}")
        print(f"  By domain:   {dict(by_domain.most_common())}")
        print(f"  By mode:     {dict(by_mode)}")

    # Fujitsu tool-pair breakdown
    fujitsu = [r for r in all_results if r.get("expected_tool")]
    if fujitsu:
        print(f"\n{'='*40}")
        print(f"  FUJITSU TOOL-PAIR BREAKDOWN ({len(fujitsu)} samples)")
        print(f"{'='*40}")
        pairs = Counter(
            f"{r['expected_tool']} -> {r['observed_tool']}"
            for r in fujitsu
        )
        for pair, count in pairs.most_common():
            pair_results = [r for r in fujitsu
                           if f"{r['expected_tool']} -> {r['observed_tool']}" == pair]
            cb_defended = sum(1 for r in pair_results if r["cb_correct"])
            bl_attacked = sum(1 for r in pair_results if not r.get("bl_correct", True))
            modes = Counter(r["cb_output_mode"] for r in pair_results)
            print(f"\n    {pair}: {count} samples")
            print(f"      BL attacked: {bl_attacked}/{count}")
            print(f"      CB defended:  {cb_defended}/{count} ({cb_defended/count:.0%})")
            print(f"      CB output:    {dict(modes.most_common(3))}")

        # Summary
        total_defended = sum(1 for r in fujitsu if r["cb_correct"])
        total_gib = sum(1 for r in fujitsu
                        if r["cb_output_mode"] in ("char_gibberish", "token_repetition", "garbled_json"))
        print(f"\n    TOTAL: {total_defended}/{len(fujitsu)} defended ({total_defended/len(fujitsu):.0%})")
        print(f"    Gibberish: {total_gib}/{len(fujitsu)} ({total_gib/len(fujitsu):.0%})")

    # Save
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
