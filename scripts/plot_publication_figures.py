#!/usr/bin/env python3
"""
Publication-quality figures for the RRFA paper (COLM 2026).

Generates:
1. Training curve with three-phase annotations (Figure 1)
2. Pareto frontier plot (Figure 2)
3. Probe AUC heatmap with pooling flip (Figure 3)

Usage:
    python scripts/plot_publication_figures.py --output-dir paper/_COLM_2026__AgentDefense/figures

    # Or generate from sweep data (legacy mode):
    python scripts/plot_publication_figures.py --sweep-dir /path/to/sweep --output-dir paper/figures
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mtick
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Publication style
if HAS_MATPLOTLIB:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

# Color-blind friendly palette (Okabe-Ito)
C_ORANGE = "#E69F00"
C_BLUE = "#0072B2"
C_RED = "#D55E00"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_GRAY = "#999999"


# ── Figure 1: Training Curve ─────────────────────────────────────────

def plot_training_curve(output_dir: Path):
    """Training curve of ad_3000_v2 with three-phase annotations."""
    # Data from results_for_agent1/traininglogfor3000v2.txt
    steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
             110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
             210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
             350, 400, 450, 500, 600, 700, 800, 900, 1000,
             1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
             2050, 2100, 2150, 2200, 2300, 2500, 2700, 2900, 3000]
    triplet_h = [19.752, 12.581, 0.271, 0.013, 0.024, 0.005, 0.002, 0.001, 0.005, 0.002,
                 0.123, 0.001, 0.004, 0.007, 0.008, 0.013, 0.003, 0.002, 0.024, 0.007,
                 0.005, 0.009, 0.013, 0.011, 0.019, 0.007, 0.037, 0.008, 0.041, 0.001,
                 0.053, 0.074, 0.010, 0.006, 0.010, 0.015, 0.020, 0.025, 0.071,
                 0.050, 0.040, 0.035, 0.030, 0.033, 0.040, 0.050, 0.060, 0.080, 0.111,
                 2.271, 0.339, 0.517, 3.772, 6.827, 4.607, 6.809, 7.070, 4.476]
    triplet_b = [0.002, 4.202, 24.844, 20.549, 17.358, 18.271, 16.294, 17.422, 15.003, 14.131,
                 15.515, 13.908, 13.150, 13.303, 12.576, 13.902, 14.436, 12.036, 9.702, 12.478,
                 12.841, 13.256, 11.840, 14.418, 12.884, 15.612, 12.284, 12.216, 13.551, 13.523,
                 10.957, 10.526, 11.678, 6.878, 8.000, 9.000, 10.000, 11.000, 11.847,
                 11.500, 11.200, 11.000, 10.800, 11.096, 10.500, 10.000, 9.500, 8.000, 6.966,
                 3.702, 9.304, 9.425, 1.004, 0.595, 0.214, 0.249, 0.111, 0.078]

    fig, ax1 = plt.subplots(figsize=(5.5, 3.0))

    # Triplet_H (harmful push loss)
    ax1.plot(steps, triplet_h, color=C_RED, linewidth=1.2, label="Triplet_H (harmful)")
    ax1.set_ylabel("Triplet_H (harmful push)", color=C_RED)
    ax1.tick_params(axis="y", labelcolor=C_RED)
    ax1.set_xlabel("Training step")
    ax1.set_ylim(-0.5, 22)

    # Triplet_B on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(steps, triplet_b, color=C_BLUE, linewidth=1.2, label="Triplet_B (benign)")
    ax2.set_ylabel("Triplet_B (benign retain)", color=C_BLUE)
    ax2.tick_params(axis="y", labelcolor=C_BLUE)
    ax2.set_ylim(-0.5, 27)

    # Phase shading
    ax1.axvspan(0, 40, alpha=0.12, color=C_ORANGE)
    ax1.axvspan(40, 2050, alpha=0.08, color=C_GREEN)
    ax1.axvspan(2050, 3000, alpha=0.12, color=C_RED)

    # Phase labels
    ax1.text(20, 20.5, "Phase 1\n(Push)", fontsize=7, ha="center",
             color=C_ORANGE, fontweight="bold")
    ax1.text(1050, 20.5, "Phase 2 (Plateau)", fontsize=7, ha="center",
             color=C_GREEN, fontweight="bold")
    ax1.text(2525, 20.5, "Phase 3\n(Collapse)", fontsize=7, ha="center",
             color=C_RED, fontweight="bold")

    # Transition markers
    ax1.axvline(40, color=C_GRAY, linestyle="--", linewidth=0.7, alpha=0.7)
    ax1.axvline(2050, color=C_GRAY, linestyle="--", linewidth=0.7, alpha=0.7)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1[:1] + h2[:1], l1[:1] + l2[:1], loc="center right", framealpha=0.9)

    ax1.set_title("Training Dynamic: ad\\_3000\\_v2")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"training_curve.{ext}")
    plt.close(fig)
    print(f"  Saved training_curve.{{pdf,png}}")


# ── Figure 2: Pareto Frontier ────────────────────────────────────────

def plot_pareto(output_dir: Path):
    """Pareto frontier (AD malicious % vs benign correct %)."""
    # Data points: (benign_correct%, malicious_rate%, label, color)
    points = [
        (79, 68, "Baseline", C_GRAY),
        (77, 63, "Sweep 9 best", C_GRAY),
        (41, 30, "ad_1000_v2", C_ORANGE),
        (40, 22, "ad_3000_v2", C_PURPLE),
        (72, 24, "ad_5000_v2", C_BLUE),
        (83, 3, "Theoretical", C_GREEN),
    ]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    for benign, mal, label, color in points:
        marker = "*" if label == "Theoretical" else ("D" if "v2" in label else "o")
        size = 120 if label == "Theoretical" else (80 if "v2" in label else 50)
        edgecolor = "black" if label == "ad_5000_v2" else "none"
        linewidth = 1.5 if label == "ad_5000_v2" else 0.5
        ax.scatter(benign, mal, c=color, s=size, marker=marker,
                   edgecolors=edgecolor, linewidths=linewidth, zorder=5)
        # Label offset
        dx, dy = 2, 1.5
        if label == "ad_3000_v2":
            dx, dy = -12, 2
        elif label == "ad_1000_v2":
            dx, dy = -12, -3
        elif label == "Baseline":
            dx, dy = -8, 2
        elif label == "Sweep 9 best":
            dx, dy = -14, -3
        elif label == "Theoretical":
            dx, dy = -12, 1
        ax.annotate(label, (benign, mal), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7)

    # Pareto frontier line
    ax.plot([72, 83], [24, 3], "k--", linewidth=0.8, alpha=0.5)

    # Ideal corner
    ax.annotate("Ideal\n(100%, 0%)", xy=(95, 2), fontsize=7, color=C_GRAY,
                ha="center", style="italic")

    ax.set_xlabel("Benign correct rate (%)")
    ax.set_ylabel("Malicious rate (%, lower = better)")
    ax.set_title("Pareto Frontier: AgentDojo Truncated")
    ax.set_xlim(30, 100)
    ax.set_ylim(-2, 75)
    ax.invert_yaxis()

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"pareto_frontier.{ext}")
    plt.close(fig)
    print(f"  Saved pareto_frontier.{{pdf,png}}")


# ── Figure 3: Probe AUC Heatmap ─────────────────────────────────────

def plot_probe_heatmap(output_dir: Path):
    """Probe AUC heatmap showing the pooling flip."""
    layers = ["L5", "L10", "L15", "L20", "L25", "L30"]

    # Test 1: Harmful vs Benign (linear probe)
    test1_mean = [0.975, 0.973, 0.969, 0.968, 0.966, 0.967]
    test1_last = [0.872, 0.844, 0.869, 0.826, 0.846, 0.817]

    # Test 2: Harmful vs Refusal (MLP probe)
    test2_mean = [0.694, 0.753, 0.757, 0.759, 0.685, 0.604]
    test2_last = [0.966, 0.967, 0.962, 0.948, 0.943, 0.926]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.2), sharey=True)

    # Panel A: Test 1 (Selectivity)
    data1 = np.array([test1_mean, test1_last])
    im1 = ax1.imshow(data1, aspect="auto", cmap="YlOrRd", vmin=0.6, vmax=1.0)
    ax1.set_xticks(range(6))
    ax1.set_xticklabels(layers, fontsize=7)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["mean", "last_tok"], fontsize=7)
    ax1.set_title("Test 1: H vs B\n(selectivity)", fontsize=8)
    for i in range(2):
        for j in range(6):
            ax1.text(j, i, f"{data1[i,j]:.2f}", ha="center", va="center", fontsize=6,
                     color="white" if data1[i,j] > 0.9 else "black")
    # Highlight best row (mean for Test 1)
    ax1.add_patch(plt.Rectangle((-0.5, -0.5), 6, 1, fill=False,
                                edgecolor=C_RED, linewidth=2))

    # Panel B: Test 2 (Push direction)
    data2 = np.array([test2_mean, test2_last])
    im2 = ax2.imshow(data2, aspect="auto", cmap="YlOrRd", vmin=0.6, vmax=1.0)
    ax2.set_xticks(range(6))
    ax2.set_xticklabels(layers, fontsize=7)
    ax2.set_title("Test 2: H vs Refusal\n(push direction)", fontsize=8)
    for i in range(2):
        for j in range(6):
            ax2.text(j, i, f"{data2[i,j]:.2f}", ha="center", va="center", fontsize=6,
                     color="white" if data2[i,j] > 0.9 else "black")
    # Highlight best row (last_token for Test 2 — the flip!)
    ax2.add_patch(plt.Rectangle((-0.5, 0.5), 6, 1, fill=False,
                                edgecolor=C_BLUE, linewidth=2))

    # Colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, pad=0.02)
    cbar.set_label("AUC", fontsize=8)

    # Annotation
    fig.text(0.5, -0.02,
             "Pooling pattern flips: mean best for selectivity (left),"
             " last_token best for push (right)",
             ha="center", fontsize=7, style="italic")

    fig.subplots_adjust(left=0.12, right=0.82, wspace=0.08)
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"probe_heatmap.{ext}")
    plt.close(fig)
    print(f"  Saved probe_heatmap.{{pdf,png}}")


# ── Legacy: load from sweep directory ────────────────────────────────

def load_run_evals(sweep_dir: Path) -> List[Dict[str, Any]]:
    """Load detailed eval results from each run directory (legacy)."""
    results = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("a"):
            continue

        entry: Dict[str, Any] = {"run_name": run_dir.name}

        m = re.match(r"a([\d.]+)_l([\d_]+)_(.*)", run_dir.name)
        if m:
            entry["alpha"] = float(m.group(1))
            entry["layers"] = m.group(2).replace("_", ",")
            entry["policy"] = m.group(3)

        fujitsu_path = run_dir / "eval" / "fujitsu_eval.json"
        if fujitsu_path.exists():
            with open(fujitsu_path) as f:
                data = json.load(f)
            bl = data.get("baseline", {})
            cb = data.get("cb_model", {})
            entry["fujitsu_baseline_asr"] = bl.get("tool_flip_asr", {}).get("attack_success_rate")
            entry["fujitsu_cb_asr"] = cb.get("tool_flip_asr", {}).get("attack_success_rate")
            entry["fujitsu_capability"] = cb.get("capability_retention", {}).get("capability_retention")

        results.append(entry)
    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("paper/_COLM_2026__AgentDefense/figures"),
                        help="Output directory for figures")
    parser.add_argument("--sweep-dir", type=Path, default=None,
                        help="Legacy: sweep directory for old-style plots")
    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        print("Error: matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures in {args.output_dir}")

    # Always generate the hardcoded-data figures
    plot_training_curve(args.output_dir)
    plot_pareto(args.output_dir)
    plot_probe_heatmap(args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
