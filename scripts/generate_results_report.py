#!/usr/bin/env python3
"""
Generate results_report.txt for a lossmask fix sweep.

Focused comparison table + post_injection vs cb_full_sequence analysis.
Complements generate_sweep_summary.py (which has ranking + individual run summaries).

Usage:
    python scripts/generate_results_report.py --sweep-dir /path/to/sweep

    # Also include lossmask coverage stats from diagnostics/
    python scripts/generate_results_report.py --sweep-dir /path/to/sweep --include-mask-stats
"""

import argparse
import json
import sys
from pathlib import Path


def fmt(v):
    if v == -1 or v is None:
        return "   n/a"
    return f"{v:7.0%}"


def collect_run(run_dir: Path) -> dict | None:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = json.load(f)

    row = {
        "run": run_dir.name,
        "policy": config.get("policy", "?"),
        "layers": config.get("layers_short", "?"),
    }

    # Fujitsu
    fuj = run_dir / "eval" / "fujitsu_eval.json"
    if fuj.exists():
        with open(fuj) as f:
            r = json.load(f)
        cb = r.get("cb_model", {}).get("tool_flip_asr", {})
        row["fuj_asr"] = cb.get("attack_success_rate", -1)
        row["fuj_correct"] = cb.get("correct_tool_call_rate", -1)
        row["fuj_no_tool"] = cb.get("no_tool_call_rate", -1)

    # AgentDojo harmful
    adh = run_dir / "eval" / "agentdojo_eval.json"
    if adh.exists():
        with open(adh) as f:
            r = json.load(f)
        cb = r.get("cb_model", {}).get("generation_comparison", {})
        row["ad_resist"] = cb.get("harmful_resistance_rate", -1)
        row["ad_malicious"] = cb.get("malicious_tool_call_rate", -1)

    # AgentDojo benign (THE KEY METRIC)
    adb = run_dir / "eval" / "agentdojo_benign_eval.json"
    if adb.exists():
        with open(adb) as f:
            r = json.load(f)
        cb = r.get("cb_model", {}).get("generation_comparison", {})
        row["ad_benign_correct"] = cb.get("correct_tool_call_rate", -1)
        row["ad_benign_no_tool"] = cb.get("no_tool_call_rate", -1)

    return row


def collect_mask_stats(diag_dir: Path) -> list[dict]:
    """Parse lossmask diagnostic outputs if present."""
    stats = []
    if not diag_dir.exists():
        return stats

    for mask_file in sorted(diag_dir.glob("*_lossmasks.jsonl")):
        policy = mask_file.stem.replace("_lossmasks", "")
        total = 0
        zero_count = 0
        ratios = []

        with open(mask_file) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                total += 1
                s = d.get("stats", {})
                ratio = s.get("mask_ratio", 0)
                if ratio == 0:
                    zero_count += 1
                else:
                    ratios.append(ratio)

        stats.append({
            "policy": policy,
            "total": total,
            "zero_mask": zero_count,
            "nonzero": len(ratios),
            "avg_coverage": sum(ratios) / len(ratios) if ratios else 0,
            "min_coverage": min(ratios) if ratios else 0,
            "max_coverage": max(ratios) if ratios else 0,
        })

    return stats


def generate_report(sweep_dir: Path, include_mask_stats: bool = False) -> str:
    lines = []

    lines.append("LOSSMASK FIX VALIDATION RESULTS")
    lines.append("=" * 96)
    lines.append(f"Sweep: {sweep_dir}")
    lines.append("")

    # Mask coverage stats
    if include_mask_stats:
        diag_dir = sweep_dir / "diagnostics"
        mask_stats = collect_mask_stats(diag_dir)
        if mask_stats:
            lines.append("LOSSMASK COVERAGE (pre-training diagnostics)")
            lines.append("-" * 80)
            lines.append(f"{'Policy':<35} {'Total':>6} {'Zero':>6} {'Nonzero':>8} {'Avg Cov':>8} {'Range':>15}")
            lines.append("-" * 80)
            for s in mask_stats:
                rng = f"{s['min_coverage']:.0%}-{s['max_coverage']:.0%}" if s['nonzero'] else "n/a"
                lines.append(
                    f"{s['policy']:<35} {s['total']:>6} {s['zero_mask']:>6} "
                    f"{s['nonzero']:>8} {s['avg_coverage']:>7.0%} {rng:>15}"
                )
            lines.append("")

    # Collect runs
    runs = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name in ("diagnostics",):
            continue
        row = collect_run(run_dir)
        if row:
            runs.append(row)

    if not runs:
        lines.append("No completed runs found.")
        return "\n".join(lines)

    # Comparison table
    hdr = f"{'Run':<45} {'Fuj ASR':>8} {'Fuj Cor':>8} {'AD Res':>8} {'AD Mal':>8} {'BEN COR':>8} {'BEN NT':>8}"
    sep = "-" * len(hdr)
    lines.append(hdr)
    lines.append(sep)

    for r in runs:
        lines.append(
            f"{r['run']:<45} {fmt(r.get('fuj_asr'))} {fmt(r.get('fuj_correct'))} "
            f"{fmt(r.get('ad_resist'))} {fmt(r.get('ad_malicious'))} "
            f"{fmt(r.get('ad_benign_correct'))} {fmt(r.get('ad_benign_no_tool'))}"
        )

    lines.append(sep)
    lines.append("")

    # Best configs
    best_benign = max(runs, key=lambda r: r.get("ad_benign_correct", -1))
    best_resist = max(runs, key=lambda r: r.get("ad_resist", -1))
    lines.append(f"Best benign_correct: {best_benign['run']} ({best_benign.get('ad_benign_correct', 0):.0%})")
    lines.append(f"Best harmful_resist: {best_resist['run']} ({best_resist.get('ad_resist', 0):.0%})")
    lines.append("")

    # Key comparison: post_injection vs cb_full_sequence
    pi_runs = [r for r in runs if "post_injection" in r["policy"]]
    cb_runs = [r for r in runs if r["policy"] == "cb_full_sequence"]
    if pi_runs and cb_runs:
        lines.append("POST-INJECTION vs CB_FULL_SEQUENCE")
        lines.append("-" * 50)
        pi_avg = sum(r.get("ad_benign_correct", 0) for r in pi_runs) / len(pi_runs)
        cb_avg = sum(r.get("ad_benign_correct", 0) for r in cb_runs) / len(cb_runs)
        lines.append(f"Avg benign_correct — post_injection policies: {pi_avg:.0%}")
        lines.append(f"Avg benign_correct — cb_full_sequence:        {cb_avg:.0%}")
        if pi_avg > cb_avg:
            lines.append(f">>> post_injection FIX WORKS: +{pi_avg - cb_avg:.0%} benign_correct <<<")
        elif pi_avg == cb_avg == 0:
            lines.append("Both still 0% — issue may be elsewhere (layers, loss, steps?)")
        else:
            lines.append("cb_full_sequence is better — unexpected, investigate.")

        # Also compare harmful resistance
        pi_resist = sum(r.get("ad_resist", 0) for r in pi_runs) / len(pi_runs)
        cb_resist = sum(r.get("ad_resist", 0) for r in cb_runs) / len(cb_runs)
        lines.append(f"Avg harmful_resist — post_injection policies: {pi_resist:.0%}")
        lines.append(f"Avg harmful_resist — cb_full_sequence:        {cb_resist:.0%}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate lossmask fix results report")
    parser.add_argument("--sweep-dir", required=True, help="Path to sweep directory")
    parser.add_argument("--include-mask-stats", action="store_true",
                        help="Include lossmask coverage stats from diagnostics/")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"ERROR: Sweep directory not found: {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    report = generate_report(sweep_dir, include_mask_stats=args.include_mask_stats)

    output_path = sweep_dir / "results_report.txt"
    with open(output_path, "w") as f:
        f.write(report)

    # Also print to stdout
    print(report)
    print(f"\nWritten to: {output_path}")


if __name__ == "__main__":
    main()
