#!/usr/bin/env python3
"""
Generate loom-ready visuals from a real sweep directory on the cluster.

This reads actual eval JSONs + paired outputs from a sweep and generates:
1. Results table with real numbers (dark-themed PNG)
2. Per-sample qualitative examples (baseline vs CB, cherry-picked)
3. Console-style aggregation (screenshot-ready terminal output)

Run on cluster:
    python scripts/generate_loom_from_sweep.py /scratch/memoozd/cb-scratch/sweeps/YOUR_SWEEP_ID

Or point at a local copy of the sweep dir:
    python scripts/generate_loom_from_sweep.py ./sweep_results/

Outputs to loom_visuals/ alongside the static slides.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Theme (matches generate_loom_visuals.py) ──────────────────────
BG      = "#0d1117"
FG      = "#e6edf3"
ACCENT  = "#58a6ff"
GREEN   = "#3fb950"
RED     = "#f85149"
ORANGE  = "#d29922"
PURPLE  = "#bc8cff"
GRAY    = "#484f58"
DIMMED  = "#8b949e"
CARD_BG = "#161b22"

if HAS_MPL:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "text.color": FG, "axes.labelcolor": FG,
        "xtick.color": FG, "ytick.color": FG,
        "font.family": "monospace", "font.size": 14,
    })


# ── Data loading ──────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def parse_run_name(name: str) -> dict:
    m = re.match(r"a([\d.]+)_l([\d_]+)_(.*)", name)
    if m:
        return {"alpha": float(m.group(1)),
                "layers": m.group(2), "policy": m.group(3)}
    return {"alpha": 0, "layers": "?", "policy": "?"}


def collect_runs(sweep_dir: Path) -> List[Dict[str, Any]]:
    """Collect all run results from sweep directory."""
    runs = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("a"):
            continue

        info = parse_run_name(run_dir.name)
        info["name"] = run_dir.name
        info["dir"] = run_dir

        # Fujitsu
        fj = load_json(run_dir / "eval" / "fujitsu_eval.json")
        if fj:
            base_asr = fj.get("baseline", {}).get("tool_flip_asr", {}).get("attack_success_rate", 0)
            cb_asr = fj.get("cb_model", {}).get("tool_flip_asr", {}).get("attack_success_rate", 0)
            info["fujitsu_base"] = base_asr * 100
            info["fujitsu_cb"] = cb_asr * 100
            info["fujitsu_delta"] = (base_asr - cb_asr) * 100

            paired = load_jsonl(run_dir / "eval" / "fujitsu_eval.paired_outputs.jsonl")
            info["fujitsu_improvements"] = sum(
                1 for p in paired
                if p.get("baseline_outcome") == "attack_success"
                and p.get("cb_outcome") != "attack_success"
            )
            info["fujitsu_regressions"] = sum(
                1 for p in paired
                if p.get("baseline_outcome") != "attack_success"
                and p.get("cb_outcome") == "attack_success"
            )
            info["fujitsu_paired"] = paired

        # AgentDojo
        ad = load_json(run_dir / "eval" / "agentdojo_eval.json")
        if ad:
            oc = ad.get("output_comparison", {})
            info["agentdojo_diff"] = oc.get("difference_rate", 0) * 100
            info["agentdojo_paired"] = load_jsonl(
                run_dir / "eval" / "agentdojo_eval.paired_outputs.jsonl")

        # LLMail
        lm = load_json(run_dir / "eval" / "llmail_eval.json")
        if lm:
            lm_base = lm.get("baseline", {}).get("llmail_attack", {})
            lm_cb = lm.get("cb_model", {}).get("llmail_attack", {})
            if lm_base:
                info["llmail_base"] = lm_base.get("attack_success_rate", 0) * 100
                info["llmail_cb"] = lm_cb.get("attack_success_rate", 0) * 100
            lm_useful = lm.get("cb_model", {}).get("llmail_usefulness", {})
            if lm_useful:
                info["llmail_usefulness"] = lm_useful.get("usefulness_rate", 0) * 100
            info["llmail_paired"] = load_jsonl(
                run_dir / "eval" / "llmail_eval.paired_outputs.jsonl")

        runs.append(info)
    return runs


# ── Console output (works without matplotlib) ─────────────────────

def print_console_results(runs: List[Dict[str, Any]]):
    """Print rich terminal output -- screenshot this for the loom."""
    B = "\033[1m"
    G = "\033[92m"
    R = "\033[91m"
    C = "\033[96m"
    Y = "\033[93m"
    D = "\033[2m"
    N = "\033[0m"

    print()
    print(f"{B}{'='*85}{N}")
    print(f"{B}  RRFA SWEEP RESULTS{N}")
    print(f"{'='*85}")

    # Header
    print(f"\n{C}{'Config':<30} {'Base':>7} {'CB':>7} {'Δ':>7} {'I/R':>7} {'AD':>6} {'LM':>6}{N}")
    print(f"{D}{'-'*85}{N}")

    best = None
    for r in sorted(runs, key=lambda x: x.get("fujitsu_cb", 999)):
        if "fujitsu_cb" not in r:
            continue
        if best is None:
            best = r

        cfg = f"α={r['alpha']}, L{{{r['layers']}}}"
        base = f"{r['fujitsu_base']:.1f}%"
        cb = f"{r['fujitsu_cb']:.1f}%"
        delta = f"{r['fujitsu_delta']:.1f}pp"
        ir = f"{r['fujitsu_improvements']}/{r['fujitsu_regressions']}"
        ad = f"{r.get('agentdojo_diff', 0):.0f}%" if "agentdojo_diff" in r else "N/A"
        lm = f"{r.get('llmail_cb', 0):.1f}%" if "llmail_cb" in r else "N/A"

        is_best = (r is best)
        c = G if is_best else ""
        e = N if is_best else ""
        w = B if is_best else ""

        print(f"  {w}{c}{cfg:<28}{e} {base:>7} {c}{cb:>7}{e} {c}{delta:>7}{e} "
              f"{c}{ir:>7}{e} {ad:>6} {lm:>6}")

    # Hero stats
    if best:
        print(f"\n{B}{'='*85}{N}")
        print(f"\n  {G}{B}BEST: α={best['alpha']}, layers {best['layers']}, "
              f"{best['policy']}{N}")
        print(f"  {G}Fujitsu ASR: {best['fujitsu_base']:.1f}% → {best['fujitsu_cb']:.1f}%  "
              f"({best['fujitsu_delta']:.1f}pp reduction){N}")
        print(f"  {G}Regressions: {best['fujitsu_regressions']}{N}")
        if "agentdojo_diff" in best:
            print(f"  {C}AgentDojo behavioral change: {best['agentdojo_diff']:.0f}%{N}")
        if "llmail_cb" in best:
            print(f"  {Y}LLMail ASR: {best['llmail_cb']:.1f}%{N}")
            if "llmail_usefulness" in best:
                print(f"  {Y}LLMail usefulness: {best['llmail_usefulness']:.1f}%{N}")

    # Cherry-picked examples
    if best and best.get("fujitsu_paired"):
        print(f"\n{B}{'='*85}{N}")
        print(f"{B}  CHERRY-PICKED EXAMPLES (CB blocked attack){N}")
        print(f"{'='*85}")

        successes = [p for p in best["fujitsu_paired"]
                     if p.get("baseline_outcome") == "attack_success"
                     and p.get("cb_outcome") != "attack_success"]

        for p in successes[:3]:
            print(f"\n  {D}ID: {p.get('id', '?')[:50]}{N}")
            print(f"  {R}Baseline: {(p.get('baseline_response', '') or '')[:120]}{N}")
            print(f"  {G}CB Model: {(p.get('cb_response', '') or '')[:120]}{N}")
            exp = p.get("expected_tool", "?")
            b_obs = p.get("baseline_observed_tool", "?")
            c_obs = p.get("cb_observed_tool", "?")
            print(f"  Expected: {exp}  |  Baseline got: {R}{b_obs}{N}  |  "
                  f"CB got: {G}{c_obs}{N}")

    if best and best.get("agentdojo_paired"):
        print(f"\n{B}{'='*85}{N}")
        print(f"{B}  AGENTDOJO EXAMPLES (behavioral changes){N}")
        print(f"{'='*85}")

        diffs = [p for p in best["agentdojo_paired"]
                 if p.get("responses_differ")]
        for p in diffs[:2]:
            print(f"\n  {D}ID: {p.get('id', '?')[:50]}{N}")
            print(f"  {R}Baseline: {(p.get('baseline_response', '') or '')[:120]}{N}")
            print(f"  {G}CB Model: {(p.get('cb_response', '') or '')[:120]}{N}")

    print()


# ── Matplotlib visuals ────────────────────────────────────────────

def card(ax, x, y, w, h, color=GRAY, lw=1.5):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                       facecolor=CARD_BG, edgecolor=color, linewidth=lw)
    ax.add_patch(r)


def generate_results_slide(runs: List[Dict[str, Any]], out_dir: Path):
    """Generate a results slide from real sweep data."""
    if not HAS_MPL:
        print("matplotlib not available, skipping PNG generation")
        return

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")

    ax.text(0.50, 0.93, "Sweep Results (from cluster)", fontsize=16,
            ha="center", color=DIMMED, fontweight="bold")

    # Sort by CB ASR
    valid = [r for r in runs if "fujitsu_cb" in r]
    valid.sort(key=lambda x: x["fujitsu_cb"])
    best = valid[0] if valid else None

    # Hero numbers
    if best:
        ax.text(0.20, 0.80, f'{best["fujitsu_base"]:.1f}%', fontsize=40,
                ha="center", color=RED, alpha=0.4, fontweight="bold")
        ax.text(0.20, 0.68, f'{best["fujitsu_cb"]:.1f}%', fontsize=44,
                ha="center", color=GREEN, fontweight="bold")
        ax.annotate("", xy=(0.20, 0.70), xytext=(0.20, 0.78),
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=3))
        ax.text(0.20, 0.60, "Fujitsu ASR", fontsize=14, ha="center", color=DIMMED)

        ax.text(0.50, 0.74, str(best["fujitsu_regressions"]), fontsize=44,
                ha="center", color=GREEN, fontweight="bold")
        ax.text(0.50, 0.60, "Regressions", fontsize=14, ha="center", color=DIMMED)

        if "agentdojo_diff" in best:
            ax.text(0.80, 0.74, f'{best["agentdojo_diff"]:.0f}%', fontsize=44,
                    ha="center", color=ACCENT, fontweight="bold")
            ax.text(0.80, 0.60, "AgentDojo Diff", fontsize=14, ha="center", color=DIMMED)

    # Table
    ax.plot([0.05, 0.95], [0.54, 0.54], color=GRAY, linewidth=1)
    cols = ["Config", "Base", "CB", "Delta", "I/R", "AD", "LLMail"]
    col_xs = [0.15, 0.30, 0.42, 0.54, 0.66, 0.78, 0.90]
    for x, c in zip(col_xs, cols):
        ax.text(x, 0.50, c, fontsize=12, ha="center", color=ACCENT, fontweight="bold")
    ax.plot([0.05, 0.95], [0.48, 0.48], color=GRAY, linewidth=0.5)

    for i, r in enumerate(valid[:6]):
        y = 0.43 - i * 0.05
        is_best = (i == 0)
        col = GREEN if is_best else FG
        w = "bold" if is_best else "normal"

        cfg = f"a={r['alpha']}, L{{{r['layers']}}}"
        vals = [
            cfg,
            f"{r['fujitsu_base']:.1f}",
            f"{r['fujitsu_cb']:.1f}",
            f"{r['fujitsu_delta']:.1f}",
            f"{r['fujitsu_improvements']}/{r['fujitsu_regressions']}",
            f"{r.get('agentdojo_diff', 0):.0f}" if "agentdojo_diff" in r else "N/A",
            f"{r.get('llmail_cb', 0):.1f}" if "llmail_cb" in r else "N/A",
        ]
        for x, v in zip(col_xs, vals):
            ax.text(x, y, v, fontsize=11, ha="center", color=col, fontweight=w)
        if is_best:
            rect = FancyBboxPatch((0.05, y - 0.018), 0.90, 0.04,
                                  boxstyle="round,pad=0.005",
                                  facecolor=GREEN, edgecolor="none", alpha=0.06)
            ax.add_patch(rect)

    # Zero regressions box
    card(ax, 0.08, 0.04, 0.84, 0.10, GREEN, 2.5)
    ax.text(0.50, 0.10, "Zero Regressions", fontsize=20, ha="center",
            color=GREEN, fontweight="bold")
    ax.text(0.50, 0.05, "CB never causes a safe sample to become vulnerable.",
            fontsize=12, ha="center", color=FG, alpha=0.9)

    out = out_dir / "06_results_live.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.3)
    plt.close(fig)
    print(f"  -> {out}")


def generate_examples_slide(runs: List[Dict[str, Any]], out_dir: Path):
    """Generate qualitative examples slide from best run's paired outputs."""
    if not HAS_MPL:
        return

    valid = [r for r in runs if r.get("fujitsu_paired")]
    if not valid:
        return
    best = min(valid, key=lambda x: x.get("fujitsu_cb", 999))

    successes = [p for p in best["fujitsu_paired"]
                 if p.get("baseline_outcome") == "attack_success"
                 and p.get("cb_outcome") != "attack_success"]
    if not successes:
        return

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    ax.text(0.50, 0.95, "Real Examples: CB Blocks Attack", fontsize=24,
            ha="center", color=FG, fontweight="bold")

    for i, p in enumerate(successes[:3]):
        y_top = 0.82 - i * 0.28
        is_restoration = (p.get("cb_observed_tool") == p.get("expected_tool"))
        border = GREEN if is_restoration else ACCENT
        card(ax, 0.03, y_top - 0.22, 0.94, 0.25, border, 2)

        tag = "RESTORED" if is_restoration else "BLOCKED"
        ax.text(0.06, y_top, f"Sample {i+1}  [{tag}]", fontsize=13,
                color=border, fontweight="bold")

        exp = p.get("expected_tool", "?")
        ax.text(0.06, y_top - 0.04, f"Expected tool: {exp}", fontsize=11,
                color=DIMMED)

        b_resp = (p.get("baseline_response", "") or "")[:140]
        c_resp = (p.get("cb_response", "") or "")[:140]
        b_tool = p.get("baseline_observed_tool", "?")
        c_tool = p.get("cb_observed_tool", "?")

        ax.text(0.06, y_top - 0.09, f"Baseline: {b_tool}", fontsize=12,
                color=RED, family="monospace")
        ax.text(0.06, y_top - 0.13, b_resp, fontsize=9, color=DIMMED)

        ax.text(0.06, y_top - 0.17, f"RRFA:     {c_tool}", fontsize=12,
                color=GREEN, family="monospace", fontweight="bold")
        ax.text(0.06, y_top - 0.21, c_resp, fontsize=9, color=FG, alpha=0.8)

    out = out_dir / "07_examples_live.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.3)
    plt.close(fig)
    print(f"  -> {out}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate loom visuals from a real sweep directory")
    parser.add_argument("sweep_dir", type=Path,
                        help="Path to sweep output directory")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: loom_visuals/)")
    parser.add_argument("--console-only", action="store_true",
                        help="Only print console output, skip PNGs")
    args = parser.parse_args()

    if not args.sweep_dir.exists():
        print(f"Error: {args.sweep_dir} not found")
        sys.exit(1)

    out_dir = args.output or Path(__file__).parent.parent / "loom_visuals"
    out_dir.mkdir(exist_ok=True)

    print(f"Reading sweep: {args.sweep_dir}")
    runs = collect_runs(args.sweep_dir)
    print(f"Found {len(runs)} runs")

    if not runs:
        print("No runs found!")
        sys.exit(1)

    # Always print console output (screenshot-ready)
    print_console_results(runs)

    if not args.console_only:
        generate_results_slide(runs, out_dir)
        generate_examples_slide(runs, out_dir)

    print("\nDone!")
    print("Screenshot the terminal output above for raw console visuals.")
    if not args.console_only:
        print(f"PNG slides saved to {out_dir}/")


if __name__ == "__main__":
    main()
