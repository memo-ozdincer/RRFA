#!/usr/bin/env python3
"""Generate a LaTeX results table from eval run directories.

Reads eval JSONs and paired_outputs from each run directory and produces a
booktabs-formatted LaTeX table suitable for paper inclusion.

Usage (explicit run dirs):
    python scripts/generate_paper_table.py \
        --run-dirs /path/to/run1 /path/to/run2 \
        --labels "CB-orig" "CB-v2" \
        --output results/paper_table.tex

Usage (auto-discover from sweep dir):
    python scripts/generate_paper_table.py \
        --sweep-dir /scratch/memoozd/cb-scratch/sweeps/ \
        --pattern "kl_restore_*" \
        --output results/paper_table.tex
"""

import argparse
import fnmatch
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Gibberish detection
# ---------------------------------------------------------------------------

def is_gibberish(text: str) -> bool:
    """Heuristic gibberish detection based on word repetition."""
    if not text or len(text) < 5:
        return False
    words = text.split()
    if len(words) < 3:
        return False
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < 0.2


def compute_gibberish_rate(paired_outputs_path: Path) -> float | None:
    """Compute gibberish rate from a paired_outputs JSONL file.

    Returns the fraction of CB model responses that are gibberish,
    or None if the file does not exist.
    """
    if not paired_outputs_path.exists():
        return None
    total = 0
    gib_count = 0
    with open(paired_outputs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            cb_response = sample.get("cb_response", "")
            if cb_response is None:
                cb_response = ""
            total += 1
            if is_gibberish(cb_response):
                gib_count += 1
    if total == 0:
        return None
    return gib_count / total


# ---------------------------------------------------------------------------
# Metric extraction from eval JSONs
# ---------------------------------------------------------------------------

# Eval file search paths (relative to run dir):
#   1. eval/next_tool_prediction/<name>   (newer layout)
#   2. eval/<name>                        (older layout)

EVAL_FILES = {
    "fujitsu": "fujitsu_eval.json",
    "agentdojo": "agentdojo_eval.json",
    "benign": "agentdojo_benign_eval.json",
}

PAIRED_FILES = {
    "agentdojo": "agentdojo_eval.paired_outputs.jsonl",
    "benign": "agentdojo_benign_eval.paired_outputs.jsonl",
}


def find_eval_file(run_dir: Path, name: str) -> Path | None:
    """Locate an eval JSON, trying both directory layouts."""
    candidates = [
        run_dir / "eval" / "next_tool_prediction" / name,
        run_dir / "eval" / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def find_paired_file(run_dir: Path, name: str) -> Path | None:
    candidates = [
        run_dir / "eval" / "next_tool_prediction" / name,
        run_dir / "eval" / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def safe_get(d: dict, *keys, default=None):
    """Nested dict access."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def extract_metrics(run_dir: Path) -> dict:
    """Extract all table metrics from a single run directory."""
    metrics = {}

    # --- Fujitsu ---
    fuj_path = find_eval_file(run_dir, EVAL_FILES["fujitsu"])
    if fuj_path:
        with open(fuj_path) as f:
            fuj = json.load(f)
        metrics["fuj_asr"] = safe_get(
            fuj, "cb_model", "tool_flip_asr", "attack_success_rate"
        )
        metrics["fuj_correct"] = safe_get(
            fuj, "cb_model", "tool_flip_asr", "correct_tool_call_rate"
        )
        # Baseline
        metrics["bl_fuj_asr"] = safe_get(
            fuj, "baseline", "tool_flip_asr", "attack_success_rate"
        )
        metrics["bl_fuj_correct"] = safe_get(
            fuj, "baseline", "tool_flip_asr", "correct_tool_call_rate"
        )
    else:
        metrics["fuj_asr"] = None
        metrics["fuj_correct"] = None
        metrics["bl_fuj_asr"] = None
        metrics["bl_fuj_correct"] = None

    # --- AgentDojo (harmful) ---
    ad_path = find_eval_file(run_dir, EVAL_FILES["agentdojo"])
    if ad_path:
        with open(ad_path) as f:
            ad = json.load(f)
        metrics["ad_mal"] = safe_get(
            ad, "cb_model", "generation_comparison", "malicious_tool_call_rate"
        )
        metrics["ad_resist"] = safe_get(
            ad, "cb_model", "generation_comparison", "harmful_resistance_rate"
        )
        # Baseline
        metrics["bl_ad_mal"] = safe_get(
            ad, "baseline", "generation_comparison", "malicious_tool_call_rate"
        )
        metrics["bl_ad_resist"] = safe_get(
            ad, "baseline", "generation_comparison", "harmful_resistance_rate"
        )
    else:
        metrics["ad_mal"] = None
        metrics["ad_resist"] = None
        metrics["bl_ad_mal"] = None
        metrics["bl_ad_resist"] = None

    # --- AgentDojo (benign) ---
    ben_path = find_eval_file(run_dir, EVAL_FILES["benign"])
    if ben_path:
        with open(ben_path) as f:
            ben = json.load(f)
        metrics["ben_correct"] = safe_get(
            ben, "cb_model", "generation_comparison", "correct_tool_call_rate"
        )
        metrics["ben_notool"] = safe_get(
            ben, "cb_model", "generation_comparison", "no_tool_call_rate"
        )
        # Baseline
        metrics["bl_ben_correct"] = safe_get(
            ben, "baseline", "generation_comparison", "correct_tool_call_rate"
        )
        metrics["bl_ben_notool"] = safe_get(
            ben, "baseline", "generation_comparison", "no_tool_call_rate"
        )
    else:
        metrics["ben_correct"] = None
        metrics["ben_notool"] = None
        metrics["bl_ben_correct"] = None
        metrics["bl_ben_notool"] = None

    # --- Gibberish rate from paired outputs ---
    ad_paired = find_paired_file(run_dir, PAIRED_FILES["agentdojo"])
    metrics["ad_gib"] = compute_gibberish_rate(ad_paired) if ad_paired else None

    ben_paired = find_paired_file(run_dir, PAIRED_FILES["benign"])
    metrics["ben_gib"] = compute_gibberish_rate(ben_paired) if ben_paired else None

    # --- Real defense = AD Resist - AD Gib ---
    if metrics["ad_resist"] is not None and metrics["ad_gib"] is not None:
        metrics["real_def"] = max(0.0, metrics["ad_resist"] - metrics["ad_gib"])
    else:
        metrics["real_def"] = None

    # --- Run config metadata ---
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            metrics["config"] = json.load(f)
    else:
        metrics["config"] = {}

    return metrics


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def fmt_pct(val, digits: int = 1) -> str:
    """Format a 0-1 float as a percentage string, or '--' if None."""
    if val is None:
        return "--"
    return f"{val * 100:.{digits}f}"


def generate_latex_table(rows: list[dict], labels: list[str]) -> str:
    """Generate a booktabs LaTeX table from extracted metrics."""
    # Column spec
    col_spec = "l" + "r" * 8
    header_row = (
        r"Method & Fuj ASR$\downarrow$ & Fuj Corr$\uparrow$ "
        r"& AD Mal$\downarrow$ & AD Resist$\uparrow$ & AD Gib$\downarrow$ "
        r"& Ben Corr$\uparrow$ & Ben NoTool$\downarrow$ & Real Def$\uparrow$"
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Evaluation results across methods. "
        r"All values are percentages. Real Def = AD Resist $-$ AD Gib.}",
        r"\label{tab:results}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_row + r" \\",
        r"\midrule",
    ]

    # Check if we have baseline data from any run to add a Baseline row
    baseline_added = False
    for m in rows:
        if m.get("bl_fuj_asr") is not None:
            bl_row = (
                f"Baseline & {fmt_pct(m['bl_fuj_asr'])} & {fmt_pct(m['bl_fuj_correct'])}"
                f" & {fmt_pct(m['bl_ad_mal'])} & {fmt_pct(m['bl_ad_resist'])} & --"
                f" & {fmt_pct(m['bl_ben_correct'])} & {fmt_pct(m['bl_ben_notool'])}"
                f" & --"
                r" \\"
            )
            lines.append(bl_row)
            lines.append(r"\midrule")
            baseline_added = True
            break

    # Data rows
    for label, m in zip(labels, rows):
        # Escape underscores in label for LaTeX
        safe_label = label.replace("_", r"\_")
        row = (
            f"{safe_label}"
            f" & {fmt_pct(m['fuj_asr'])}"
            f" & {fmt_pct(m['fuj_correct'])}"
            f" & {fmt_pct(m['ad_mal'])}"
            f" & {fmt_pct(m['ad_resist'])}"
            f" & {fmt_pct(m['ad_gib'])}"
            f" & {fmt_pct(m['ben_correct'])}"
            f" & {fmt_pct(m['ben_notool'])}"
            f" & {fmt_pct(m['real_def'])}"
            r" \\"
        )
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Run directory discovery
# ---------------------------------------------------------------------------

def discover_run_dirs(sweep_dir: Path, pattern: str) -> list[Path]:
    """Find run directories matching a glob pattern under a sweep directory."""
    dirs = []
    for child in sorted(sweep_dir.iterdir()):
        if child.is_dir() and fnmatch.fnmatch(child.name, pattern):
            # Must have an eval subdirectory
            if (child / "eval").is_dir():
                dirs.append(child)
    if not dirs:
        print(f"Warning: no matching run dirs found in {sweep_dir} "
              f"with pattern '{pattern}'", file=sys.stderr)
    return dirs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX results table from eval run directories."
    )
    parser.add_argument(
        "--run-dirs", nargs="+", type=Path, default=None,
        help="Explicit list of run directories to include.",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Display labels for each run dir (must match --run-dirs count).",
    )
    parser.add_argument(
        "--sweep-dir", type=Path, default=None,
        help="Auto-discover run dirs under this sweep directory.",
    )
    parser.add_argument(
        "--pattern", default="*",
        help="Glob pattern for auto-discovery (default: '*').",
    )
    parser.add_argument(
        "--output", default="results/paper_table.tex",
        help="Output path for the LaTeX table.",
    )
    args = parser.parse_args()

    # Resolve run directories
    if args.run_dirs:
        run_dirs = args.run_dirs
    elif args.sweep_dir:
        run_dirs = discover_run_dirs(args.sweep_dir, args.pattern)
        if not run_dirs:
            sys.exit(1)
    else:
        print("Error: provide either --run-dirs or --sweep-dir.", file=sys.stderr)
        sys.exit(1)

    # Resolve labels
    if args.labels:
        if len(args.labels) != len(run_dirs):
            print(
                f"Error: {len(args.labels)} labels provided but "
                f"{len(run_dirs)} run dirs found.",
                file=sys.stderr,
            )
            sys.exit(1)
        labels = args.labels
    else:
        # Derive labels from directory names or run_config
        labels = []
        for rd in run_dirs:
            config_path = rd / "run_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                labels.append(cfg.get("run_name", rd.name))
            else:
                labels.append(rd.name)

    # Extract metrics for each run
    all_metrics = []
    for rd in run_dirs:
        if not rd.exists():
            print(f"Warning: run dir does not exist: {rd}", file=sys.stderr)
            all_metrics.append({k: None for k in [
                "fuj_asr", "fuj_correct", "ad_mal", "ad_resist", "ad_gib",
                "ben_correct", "ben_notool", "real_def",
                "bl_fuj_asr", "bl_fuj_correct", "bl_ad_mal", "bl_ad_resist",
                "bl_ben_correct", "bl_ben_notool", "ben_gib", "config",
            ]})
            continue
        print(f"Processing: {rd.name}")
        all_metrics.append(extract_metrics(rd))

    # Generate LaTeX
    latex = generate_latex_table(all_metrics, labels)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex + "\n")

    # Print summary to stdout
    print()
    print("=" * 70)
    print("Paper Table Summary")
    print("=" * 70)
    header = f"{'Method':<25} {'FujASR':>7} {'FujCor':>7} {'ADMal':>7} {'ADRes':>7} {'ADGib':>7} {'BenCor':>7} {'BenNT':>7} {'RealD':>7}"
    print(header)
    print("-" * 70)

    for label, m in zip(labels, all_metrics):
        row = (
            f"{label:<25}"
            f" {fmt_pct(m['fuj_asr']):>7}"
            f" {fmt_pct(m['fuj_correct']):>7}"
            f" {fmt_pct(m['ad_mal']):>7}"
            f" {fmt_pct(m['ad_resist']):>7}"
            f" {fmt_pct(m['ad_gib']):>7}"
            f" {fmt_pct(m['ben_correct']):>7}"
            f" {fmt_pct(m['ben_notool']):>7}"
            f" {fmt_pct(m['real_def']):>7}"
        )
        print(row)

    print()
    print(f"LaTeX table written to: {output_path}")
    print()
    print("--- LaTeX output ---")
    print(latex)


if __name__ == "__main__":
    main()
