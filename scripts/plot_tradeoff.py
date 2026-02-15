#!/usr/bin/env python3
"""
Safety vs Capability Tradeoff Visualization.

Reads sweep summary.csv and per-run eval JSONs to produce:
1. Safety vs Capability scatter plot (per dataset)
2. Pareto frontier highlighting
3. Aggregate tradeoff curves across alpha values

Usage:
    python scripts/plot_tradeoff.py --sweep-dir /path/to/sweep_dir
    python scripts/plot_tradeoff.py --sweep-dir /path/to/sweep_dir --output plots/
    python scripts/plot_tradeoff.py --csv /path/to/summary.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for cluster
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_summary_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Load sweep summary CSV."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_run_evals(sweep_dir: Path) -> List[Dict[str, Any]]:
    """Load detailed eval results from each run directory."""
    results = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("a"):
            continue

        entry: Dict[str, Any] = {"run_name": run_dir.name}

        # Parse run name
        import re
        m = re.match(r"a([\d.]+)_l([\d_]+)_(.*)", run_dir.name)
        if m:
            entry["alpha"] = float(m.group(1))
            entry["layers"] = m.group(2).replace("_", ",")
            entry["policy"] = m.group(3)

        # Fujitsu
        fujitsu_path = run_dir / "eval" / "fujitsu_eval.json"
        if fujitsu_path.exists():
            with open(fujitsu_path) as f:
                data = json.load(f)
            bl = data.get("baseline", {})
            cb = data.get("cb_model", {})
            entry["fujitsu_baseline_asr"] = bl.get("tool_flip_asr", {}).get("attack_success_rate")
            entry["fujitsu_cb_asr"] = cb.get("tool_flip_asr", {}).get("attack_success_rate")
            entry["fujitsu_capability"] = cb.get("capability_retention", {}).get("capability_retention")

        # LLMail
        llmail_path = run_dir / "eval" / "llmail_eval.json"
        if llmail_path.exists():
            with open(llmail_path) as f:
                data = json.load(f)
            bl = data.get("baseline", {})
            cb = data.get("cb_model", {})
            # Prefer llmail_attack metrics
            if "llmail_attack" in cb:
                entry["llmail_baseline_asr"] = bl.get("llmail_attack", {}).get("attack_success_rate")
                entry["llmail_cb_asr"] = cb.get("llmail_attack", {}).get("attack_success_rate")
                entry["llmail_refusal_rate"] = cb.get("llmail_attack", {}).get("refusal_rate")
            else:
                entry["llmail_baseline_asr"] = bl.get("tool_flip_asr", {}).get("attack_success_rate")
                entry["llmail_cb_asr"] = cb.get("tool_flip_asr", {}).get("attack_success_rate")
            if "llmail_usefulness" in cb:
                entry["llmail_usefulness"] = cb["llmail_usefulness"].get("usefulness_rate")

        # AgentDojo
        agentdojo_path = run_dir / "eval" / "agentdojo_eval.json"
        if agentdojo_path.exists():
            with open(agentdojo_path) as f:
                data = json.load(f)
            entry["agentdojo_diff"] = data.get("output_comparison", {}).get("difference_rate")

        results.append(entry)
    return results


def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "N/A" or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def pareto_frontier(points: List[tuple]) -> List[tuple]:
    """
    Compute 2D Pareto frontier.
    Points are (safety, capability) where higher is better for both.
    Returns points on the frontier sorted by safety.
    """
    sorted_pts = sorted(points, key=lambda p: p[0])
    frontier = []
    max_cap = float("-inf")
    for pt in sorted_pts:
        if pt[1] >= max_cap:
            frontier.append(pt)
            max_cap = pt[1]
    return frontier


def print_text_table(runs: List[Dict[str, Any]]) -> None:
    """Print a text-based tradeoff table when matplotlib is unavailable."""
    print("\n" + "=" * 100)
    print("SAFETY vs CAPABILITY TRADEOFF TABLE")
    print("=" * 100)

    header = (
        f"{'Run':<35} {'Alpha':>6} "
        f"{'Fuj ASR':>8} {'Fuj Cap':>8} "
        f"{'LLM ASR':>8} {'LLM Usef':>9} "
        f"{'AD Diff':>8}"
    )
    print(header)
    print("-" * 100)

    for r in sorted(runs, key=lambda x: x.get("alpha", 0)):
        alpha = r.get("alpha", "?")
        fuj_asr = r.get("fujitsu_cb_asr")
        fuj_cap = r.get("fujitsu_capability")
        llm_asr = r.get("llmail_cb_asr")
        llm_use = r.get("llmail_usefulness")
        ad_diff = r.get("agentdojo_diff")

        fuj_asr_s = f"{fuj_asr:.1%}" if fuj_asr is not None else "N/A"
        fuj_cap_s = f"{fuj_cap:.1%}" if fuj_cap is not None else "N/A"
        llm_asr_s = f"{llm_asr:.1%}" if llm_asr is not None else "N/A"
        llm_use_s = f"{llm_use:.1%}" if llm_use is not None else "N/A"
        ad_diff_s = f"{ad_diff:.1%}" if ad_diff is not None else "N/A"

        print(
            f"{r.get('run_name', '?'):<35} {alpha:>6} "
            f"{fuj_asr_s:>8} {fuj_cap_s:>8} "
            f"{llm_asr_s:>8} {llm_use_s:>9} "
            f"{ad_diff_s:>8}"
        )

    # Tradeoff summary
    print("\n--- TRADEOFF ANALYSIS ---")
    print("Safety = 1 - ASR (higher is better)")
    print("Capability = Usefulness / Capability Retention (higher is better)")
    print()

    for dataset, asr_key, cap_key in [
        ("Fujitsu", "fujitsu_cb_asr", "fujitsu_capability"),
        ("LLMail", "llmail_cb_asr", "llmail_usefulness"),
    ]:
        points = []
        for r in runs:
            asr = r.get(asr_key)
            cap = r.get(cap_key)
            if asr is not None and cap is not None:
                safety = 1.0 - asr
                points.append((safety, cap, r.get("run_name", "?")))

        if not points:
            print(f"  {dataset}: No data available")
            continue

        print(f"  {dataset} (safety, capability):")
        for s, c, name in sorted(points, key=lambda x: -x[0]):
            print(f"    {name:<35} safety={s:.1%}  capability={c:.1%}")

        # Pareto
        frontier = pareto_frontier([(s, c) for s, c, _ in points])
        print(f"  Pareto frontier ({len(frontier)} points):")
        for s, c in frontier:
            matching = [n for ss, cc, n in points if abs(ss - s) < 1e-9 and abs(cc - c) < 1e-9]
            print(f"    safety={s:.1%}  capability={c:.1%}  ({matching[0] if matching else '?'})")
        print()


def plot_tradeoffs(runs: List[Dict[str, Any]], output_dir: Path) -> None:
    """Generate matplotlib tradeoff plots."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot generation")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data per alpha for color coding
    alphas = sorted(set(r.get("alpha", 0) for r in runs))
    alpha_colors = {a: plt.cm.viridis(i / max(len(alphas) - 1, 1))
                    for i, a in enumerate(alphas)}

    # --- Plot 1: Fujitsu Safety vs Capability ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for r in runs:
        asr = r.get("fujitsu_cb_asr")
        cap = r.get("fujitsu_capability")
        alpha = r.get("alpha", 0)
        if asr is not None and cap is not None:
            safety = 1.0 - asr
            color = alpha_colors.get(alpha, "gray")
            ax.scatter(safety, cap, c=[color], s=60, edgecolors="black", linewidth=0.5,
                       label=f"alpha={alpha}" if alpha not in [r2.get("alpha") for r2 in runs[:runs.index(r)]] else "")

    # Pareto frontier
    fuj_points = [(1 - r["fujitsu_cb_asr"], r["fujitsu_capability"])
                  for r in runs if r.get("fujitsu_cb_asr") is not None and r.get("fujitsu_capability") is not None]
    if fuj_points:
        frontier = pareto_frontier(fuj_points)
        if len(frontier) > 1:
            fx, fy = zip(*frontier)
            ax.plot(fx, fy, "r--", alpha=0.5, label="Pareto frontier")

    ax.set_xlabel("Safety (1 - Attack Success Rate)")
    ax.set_ylabel("Capability Retention")
    ax.set_title("Fujitsu: Safety vs Capability Tradeoff")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fujitsu_tradeoff.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'fujitsu_tradeoff.png'}")

    # --- Plot 2: LLMail Safety vs Usefulness ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    has_llmail = False
    for r in runs:
        asr = r.get("llmail_cb_asr")
        use = r.get("llmail_usefulness")
        alpha = r.get("alpha", 0)
        if asr is not None and use is not None:
            has_llmail = True
            safety = 1.0 - asr
            color = alpha_colors.get(alpha, "gray")
            ax.scatter(safety, use, c=[color], s=60, edgecolors="black", linewidth=0.5)

    if has_llmail:
        llm_points = [(1 - r["llmail_cb_asr"], r["llmail_usefulness"])
                      for r in runs if r.get("llmail_cb_asr") is not None and r.get("llmail_usefulness") is not None]
        frontier = pareto_frontier(llm_points)
        if len(frontier) > 1:
            fx, fy = zip(*frontier)
            ax.plot(fx, fy, "r--", alpha=0.5, label="Pareto frontier")

        ax.set_xlabel("Safety (1 - LLMail Attack ASR)")
        ax.set_ylabel("Usefulness Rate")
        ax.set_title("LLMail: Safety vs Usefulness Tradeoff")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "llmail_tradeoff.png", dpi=150)
        print(f"  Saved: {output_dir / 'llmail_tradeoff.png'}")
    plt.close(fig)

    # --- Plot 3: Combined multi-panel ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Fujitsu ASR reduction vs alpha
    for r in runs:
        bl = r.get("fujitsu_baseline_asr")
        cb = r.get("fujitsu_cb_asr")
        alpha = r.get("alpha")
        if bl is not None and cb is not None and alpha is not None:
            reduction = bl - cb
            axes[0].scatter(alpha, reduction, c=[alpha_colors.get(alpha, "gray")],
                            s=40, edgecolors="black", linewidth=0.5)
    axes[0].set_xlabel("Alpha (alpha_max)")
    axes[0].set_ylabel("ASR Reduction (baseline - CB)")
    axes[0].set_title("Fujitsu: ASR Reduction vs Alpha")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[0].axhline(y=0, color="red", linestyle="--", alpha=0.3)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: LLMail ASR vs alpha
    for r in runs:
        asr = r.get("llmail_cb_asr")
        alpha = r.get("alpha")
        if asr is not None and alpha is not None:
            axes[1].scatter(alpha, asr, c=[alpha_colors.get(alpha, "gray")],
                            s=40, edgecolors="black", linewidth=0.5)
    axes[1].set_xlabel("Alpha (alpha_max)")
    axes[1].set_ylabel("LLMail Attack ASR")
    axes[1].set_title("LLMail: Attack ASR vs Alpha")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].grid(True, alpha=0.3)

    # Panel 3: AgentDojo diff rate vs alpha
    for r in runs:
        diff = r.get("agentdojo_diff")
        alpha = r.get("alpha")
        if diff is not None and alpha is not None:
            axes[2].scatter(alpha, diff, c=[alpha_colors.get(alpha, "gray")],
                            s=40, edgecolors="black", linewidth=0.5)
    axes[2].set_xlabel("Alpha (alpha_max)")
    axes[2].set_ylabel("AgentDojo Output Difference Rate")
    axes[2].set_title("AgentDojo: Behavior Change vs Alpha")
    axes[2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "combined_metrics.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'combined_metrics.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot safety vs capability tradeoff from sweep results",
    )
    parser.add_argument(
        "--sweep-dir", type=Path, default=None,
        help="Path to sweep directory (reads per-run eval JSONs)",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Path to summary.csv (alternative to --sweep-dir)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory for plots (default: <sweep-dir>/plots/)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip matplotlib plots, only print text table",
    )
    args = parser.parse_args()

    if not args.sweep_dir and not args.csv:
        parser.error("Provide either --sweep-dir or --csv")

    runs: List[Dict[str, Any]] = []

    if args.sweep_dir:
        if not args.sweep_dir.exists():
            print(f"Error: sweep directory not found: {args.sweep_dir}")
            sys.exit(1)
        runs = load_run_evals(args.sweep_dir)
        print(f"Loaded {len(runs)} runs from {args.sweep_dir}")
    elif args.csv:
        rows = load_summary_csv(args.csv)
        for row in rows:
            entry: Dict[str, Any] = {"run_name": f"a{row.get('alpha', '?')}_l{row.get('layers', '?').replace(',', '_')}_{row.get('policy', '?')}"}
            entry["alpha"] = _safe_float(row.get("alpha"))
            entry["layers"] = row.get("layers", "")
            entry["policy"] = row.get("policy", "")
            entry["fujitsu_baseline_asr"] = _safe_float(row.get("fujitsu_baseline_asr"))
            entry["fujitsu_cb_asr"] = _safe_float(row.get("fujitsu_cb_asr"))
            entry["llmail_baseline_asr"] = _safe_float(row.get("llmail_baseline_asr"))
            entry["llmail_cb_asr"] = _safe_float(row.get("llmail_cb_asr"))
            entry["llmail_usefulness"] = _safe_float(row.get("llmail_usefulness"))
            entry["agentdojo_diff"] = _safe_float(row.get("agentdojo_diff"))
            runs.append(entry)
        print(f"Loaded {len(runs)} rows from {args.csv}")

    if not runs:
        print("No data found.")
        sys.exit(1)

    # Always print text table
    print_text_table(runs)

    # Generate plots if matplotlib is available and not disabled
    if not args.no_plot and HAS_MATPLOTLIB:
        output_dir = args.output or (args.sweep_dir / "plots" if args.sweep_dir else Path("plots"))
        print(f"\nGenerating plots in {output_dir}...")
        plot_tradeoffs(runs, output_dir)
    elif not HAS_MATPLOTLIB:
        print("\nmatplotlib not installed - skipping plot generation.")
        print("Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
