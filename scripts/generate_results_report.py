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


def _find_eval_dirs(run_dir: Path) -> list[tuple[str, Path]]:
    """Find eval directories, supporting both old (eval/) and new (eval/{mode}/) layouts."""
    eval_dir = run_dir / "eval"
    if not eval_dir.exists():
        return []

    # Check for eval mode subdirectories first (new layout)
    mode_dirs = []
    for mode in ("full_trace", "next_tool_prediction"):
        subdir = eval_dir / mode
        if subdir.is_dir() and any(subdir.glob("*.json")):
            mode_dirs.append((mode, subdir))

    if mode_dirs:
        return mode_dirs

    # Fall back to old layout (JSONs directly in eval/)
    if any(eval_dir.glob("*.json")):
        return [("", eval_dir)]

    return []


def _collect_from_eval_dir(eval_dir: Path) -> dict:
    """Extract metrics from a single eval directory."""
    metrics = {}

    # Fujitsu
    fuj = eval_dir / "fujitsu_eval.json"
    if fuj.exists():
        with open(fuj) as f:
            r = json.load(f)
        cb = r.get("cb_model", {}).get("tool_flip_asr", {})
        metrics["fuj_asr"] = cb.get("attack_success_rate", -1)
        metrics["fuj_correct"] = cb.get("correct_tool_call_rate", -1)
        metrics["fuj_no_tool"] = cb.get("no_tool_call_rate", -1)

    # AgentDojo harmful
    adh = eval_dir / "agentdojo_eval.json"
    if adh.exists():
        with open(adh) as f:
            r = json.load(f)
        cb = r.get("cb_model", {}).get("generation_comparison", {})
        metrics["ad_resist"] = cb.get("harmful_resistance_rate", -1)
        metrics["ad_malicious"] = cb.get("malicious_tool_call_rate", -1)

    # AgentDojo benign (THE KEY METRIC)
    adb = eval_dir / "agentdojo_benign_eval.json"
    if adb.exists():
        with open(adb) as f:
            r = json.load(f)
        cb = r.get("cb_model", {}).get("generation_comparison", {})
        metrics["ad_benign_correct"] = cb.get("correct_tool_call_rate", -1)
        metrics["ad_benign_no_tool"] = cb.get("no_tool_call_rate", -1)

    return metrics


def collect_run(run_dir: Path) -> list[dict]:
    """Collect run data. Returns a list of rows (one per eval mode, or one for legacy layout)."""
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return []

    with open(config_path) as f:
        config = json.load(f)

    base = {
        "run": run_dir.name,
        "policy": config.get("policy", "?"),
        "layers": config.get("layers_short", "?"),
        "loss_mode": config.get("loss_mode", "?"),
    }

    eval_dirs = _find_eval_dirs(run_dir)
    if not eval_dirs:
        return [base]  # Return config-only row

    rows = []
    for mode_label, eval_dir in eval_dirs:
        row = dict(base)
        if mode_label:
            row["eval_mode"] = mode_label
        row.update(_collect_from_eval_dir(eval_dir))
        rows.append(row)

    return rows


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
        rows = collect_run(run_dir)
        runs.extend(rows)

    if not runs:
        lines.append("No completed runs found.")
        return "\n".join(lines)

    # Detect if we have eval modes
    has_eval_modes = any(r.get("eval_mode") for r in runs)
    eval_modes = sorted(set(r.get("eval_mode", "") for r in runs if r.get("eval_mode")))

    if has_eval_modes and len(eval_modes) > 1:
        # Multi-eval-mode table: one section per eval mode
        for em in eval_modes:
            em_label = "FT" if em == "full_trace" else "NTP" if em == "next_tool_prediction" else em
            em_runs = [r for r in runs if r.get("eval_mode") == em]
            lines.append(f"EVAL MODE: {em} ({em_label})")
            _append_comparison_table(lines, em_runs)
            lines.append("")

        # Side-by-side comparison
        lines.append("EVAL MODE COMPARISON (FT vs NTP)")
        lines.append("-" * 95)
        ft_map = {r["run"]: r for r in runs if r.get("eval_mode") == "full_trace"}
        ntp_map = {r["run"]: r for r in runs if r.get("eval_mode") == "next_tool_prediction"}
        lines.append(
            f"{'Run':<30} | {'FT Ben%':>7} {'NTP Ben%':>8} | "
            f"{'FT ASR':>6} {'NTP ASR':>7} | {'FT Res':>6} {'NTP Res':>7}"
        )
        lines.append("-" * 95)
        for run_name in sorted(set(r["run"] for r in runs)):
            ft = ft_map.get(run_name, {})
            ntp = ntp_map.get(run_name, {})
            lines.append(
                f"{run_name:<30} | "
                f"{fmt(ft.get('ad_benign_correct')):>7} {fmt(ntp.get('ad_benign_correct')):>8} | "
                f"{fmt(ft.get('fuj_asr')):>6} {fmt(ntp.get('fuj_asr')):>7} | "
                f"{fmt(ft.get('ad_resist')):>6} {fmt(ntp.get('ad_resist')):>7}"
            )
        lines.append("")
    else:
        # Single eval mode or legacy layout
        _append_comparison_table(lines, runs)

    # Best configs (use NTP if available, else all runs)
    best_pool = [r for r in runs if r.get("eval_mode") == "next_tool_prediction"] or runs
    best_benign = max(best_pool, key=lambda r: r.get("ad_benign_correct", -1))
    best_resist = max(best_pool, key=lambda r: r.get("ad_resist", -1))
    em_suffix = f" ({best_benign.get('eval_mode', '')})" if best_benign.get("eval_mode") else ""
    lines.append(f"Best benign_correct: {best_benign['run']}{em_suffix} ({best_benign.get('ad_benign_correct', 0):.0%})")
    em_suffix = f" ({best_resist.get('eval_mode', '')})" if best_resist.get("eval_mode") else ""
    lines.append(f"Best harmful_resist: {best_resist['run']}{em_suffix} ({best_resist.get('ad_resist', 0):.0%})")
    lines.append("")

    # Key comparison: post_injection vs cb_full_sequence (using best_pool)
    pi_runs = [r for r in best_pool if "post_injection" in r.get("policy", "")]
    cb_runs = [r for r in best_pool if r.get("policy") == "cb_full_sequence"]
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

        pi_resist = sum(r.get("ad_resist", 0) for r in pi_runs) / len(pi_runs)
        cb_resist = sum(r.get("ad_resist", 0) for r in cb_runs) / len(cb_runs)
        lines.append(f"Avg harmful_resist — post_injection policies: {pi_resist:.0%}")
        lines.append(f"Avg harmful_resist — cb_full_sequence:        {cb_resist:.0%}")
        lines.append("")

    # Key comparison: loss modes (legacy_cb vs triplet_full)
    lcb_runs = [r for r in best_pool if r.get("loss_mode") == "legacy_cb"]
    trip_runs = [r for r in best_pool if r.get("loss_mode") == "triplet_full"]
    if lcb_runs and trip_runs:
        lines.append("LEGACY_CB vs TRIPLET_FULL")
        lines.append("-" * 50)
        lcb_asr = sum(r.get("fuj_asr", 0) for r in lcb_runs) / len(lcb_runs)
        trip_asr = sum(r.get("fuj_asr", 0) for r in trip_runs) / len(trip_runs)
        lines.append(f"Avg Fujitsu ASR — legacy_cb:     {lcb_asr:.0%}")
        lines.append(f"Avg Fujitsu ASR — triplet_full:  {trip_asr:.0%}")
        if trip_asr < lcb_asr:
            lines.append(f">>> triplet_full better by {lcb_asr - trip_asr:.0%} ASR reduction <<<")
        lcb_ben = sum(r.get("ad_benign_correct", 0) for r in lcb_runs) / len(lcb_runs)
        trip_ben = sum(r.get("ad_benign_correct", 0) for r in trip_runs) / len(trip_runs)
        lines.append(f"Avg benign_correct — legacy_cb:     {lcb_ben:.0%}")
        lines.append(f"Avg benign_correct — triplet_full:  {trip_ben:.0%}")
        lines.append("")

    return "\n".join(lines)


def _append_comparison_table(lines: list[str], runs: list[dict]) -> None:
    """Append a comparison table for a set of runs."""
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
