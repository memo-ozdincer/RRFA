#!/usr/bin/env python3
"""
Generate publication-quality figures for the RRFA paper.

This script reads sweep results and generates:
1. Loss Mask Policy Comparison (Grouped Bar Chart)
2. Alpha Sensitivity Curves (Line Plot)
3. Pareto Frontier Plots (Scatter with Frontier)

Usage:
    python scripts/plot_publication_figures.py --sweep-dir /path/to/sweep_dir --output paper/figures/
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Reuse loading logic from plot_tradeoff.py
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

def plot_policy_comparison(runs: List[Dict[str, Any]], output_dir: Path):
    """Generate grouped bar chart for policy comparison."""
    # Aggregate data: Average ASR reduction per policy across all alphas/layers
    policies = sorted(list(set(r.get("policy", "unknown") for r in runs)))
    
    data = {
        "Policy": [],
        "Dataset": [],
        "ASR Reduction": []
    }
    
    for r in runs:
        policy = r.get("policy", "unknown")
        
        # Fujitsu
        if r.get("fujitsu_baseline_asr") is not None and r.get("fujitsu_cb_asr") is not None:
            red = r["fujitsu_baseline_asr"] - r["fujitsu_cb_asr"]
            data["Policy"].append(policy)
            data["Dataset"].append("Fujitsu B4")
            data["ASR Reduction"].append(red)
            
        # LLMail
        if r.get("llmail_baseline_asr") is not None and r.get("llmail_cb_asr") is not None:
            red = r["llmail_baseline_asr"] - r["llmail_cb_asr"]
            data["Policy"].append(policy)
            data["Dataset"].append("LLMail")
            data["ASR Reduction"].append(red)

    if not data["Policy"]:
        print("No data for policy comparison plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Create barplot
    ax = sns.barplot(
        data=data, 
        x="Dataset", 
        y="ASR Reduction", 
        hue="Policy",
        errorbar="sd", # Show standard deviation
        palette="viridis",
        alpha=0.9
    )
    
    ax.set_title("Impact of Loss Mask Policy on Attack Success Rate Reduction", fontsize=14)
    ax.set_ylabel("ASR Reduction (Higher is Better)", fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.legend(title="Loss Mask Policy", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / "policy_comparison.png", dpi=300)
    plt.close()
    print(f"Saved {output_dir / 'policy_comparison.png'}")

def plot_alpha_sensitivity(runs: List[Dict[str, Any]], output_dir: Path):
    """Generate line plot for alpha sensitivity."""
    # Plot ASR Reduction vs Alpha, grouped by Policy
    
    data = {
        "Alpha": [],
        "ASR Reduction": [],
        "Policy": [],
        "Dataset": []
    }
    
    for r in runs:
        alpha = r.get("alpha")
        policy = r.get("policy", "unknown")
        if alpha is None: continue
        
        # Fujitsu
        if r.get("fujitsu_baseline_asr") is not None and r.get("fujitsu_cb_asr") is not None:
            red = r["fujitsu_baseline_asr"] - r["fujitsu_cb_asr"]
            data["Alpha"].append(alpha)
            data["ASR Reduction"].append(red)
            data["Policy"].append(policy)
            data["Dataset"].append("Fujitsu B4")

    if not data["Alpha"]:
        print("No data for alpha sensitivity plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Line plot
    ax = sns.lineplot(
        data=data,
        x="Alpha",
        y="ASR Reduction",
        hue="Policy",
        style="Policy",
        markers=True,
        dashes=False,
        palette="viridis"
    )
    
    ax.set_title("Alpha Sensitivity: ASR Reduction on Fujitsu B4", fontsize=14)
    ax.set_ylabel("ASR Reduction (Higher is Better)", fontsize=12)
    ax.set_xlabel("Alpha (Rerouting Strength)", fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    plt.tight_layout()
    plt.savefig(output_dir / "alpha_sensitivity.png", dpi=300)
    plt.close()
    print(f"Saved {output_dir / 'alpha_sensitivity.png'}")

def pareto_frontier(points):
    """Find Pareto frontier points."""
    sorted_pts = sorted(points, key=lambda p: p[0])
    frontier = []
    max_cap = float("-inf")
    for pt in sorted_pts:
        if pt[1] >= max_cap:
            frontier.append(pt)
            max_cap = pt[1]
    return frontier

def plot_pareto_frontier(runs: List[Dict[str, Any]], output_dir: Path):
    """Generate clean Pareto frontier plot."""
    # Fujitsu Safety vs Capability
    points = []
    for r in runs:
        if r.get("fujitsu_cb_asr") is not None and r.get("fujitsu_capability") is not None:
            safety = 1.0 - r["fujitsu_cb_asr"]
            cap = r["fujitsu_capability"]
            policy = r.get("policy", "unknown")
            points.append((safety, cap, policy))
            
    if not points:
        print("No data for Pareto plot.")
        return
        
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    
    # Scatter all points
    safeties, caps, policies = zip(*points)
    sns.scatterplot(x=safeties, y=caps, hue=policies, palette="viridis", s=100, alpha=0.7, edgecolor="black")
    
    # Draw frontier
    frontier_pts = pareto_frontier([(s, c) for s, c, _ in points])
    if len(frontier_pts) > 1:
        fx, fy = zip(*frontier_pts)
        plt.plot(fx, fy, "r--", linewidth=2, label="Pareto Frontier")
        
    plt.title("Safety vs. Capability Trade-off (Fujitsu B4)", fontsize=14)
    plt.xlabel("Safety (1 - ASR)", fontsize=12)
    plt.ylabel("Capability Retention", fontsize=12)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(output_dir / "pareto_frontier.png", dpi=300)
    plt.close()
    print(f"Saved {output_dir / 'pareto_frontier.png'}")

def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--sweep-dir", type=Path, required=True, help="Path to sweep directory")
    parser.add_argument("--output", type=Path, default=Path("paper/figures"), help="Output directory")
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib/seaborn not installed.")
        sys.exit(1)
        
    if not args.sweep_dir.exists():
        print(f"Error: {args.sweep_dir} does not exist.")
        sys.exit(1)
        
    args.output.mkdir(parents=True, exist_ok=True)
    
    runs = load_run_evals(args.sweep_dir)
    print(f"Loaded {len(runs)} runs.")
    
    plot_policy_comparison(runs, args.output)
    plot_alpha_sensitivity(runs, args.output)
    plot_pareto_frontier(runs, args.output)

if __name__ == "__main__":
    main()
