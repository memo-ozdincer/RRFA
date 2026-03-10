#!/usr/bin/env python3
"""
Analyze LoRA weight norms per layer from a saved PEFT checkpoint.

Identifies which layers have large LoRA perturbations (potential
collateral damage layers) vs CB target layers where perturbation
is expected.

For each of the 32 Llama layers and each module (q/k/v/o/gate/up/down),
computes ||B @ A||_F (effective LoRA perturbation norm) as well as
||A||_F and ||B||_F individually.

Outputs:
  - Formatted table to stdout
  - Bar plot of total LoRA norm per layer (PDF + PNG)
  - JSON summary with per-layer, per-module norms

Usage:
  python scripts/analysis/lora_norms.py \
      --adapter /path/to/lora_adapter \
      --output results/lora_norms \
      --cb-layers 10 20
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Module types in order for the table
MODULE_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MODULE_SHORT = ["q", "k", "v", "o", "gate", "up", "down"]

NUM_LAYERS = 32


def extract_lora_weights(
    adapter_path: str,
) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
    """Load LoRA adapter and extract A/B weight matrices per layer per module.

    Returns:
        {layer_idx: {module_name: {"A": tensor, "B": tensor}}}

    Works CPU-only -- just loads the state dict, no model forward pass.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoConfig

    # Try to load just the adapter state dict directly for speed
    adapter_dir = Path(adapter_path)

    # Look for adapter weights file
    weights_file = None
    for candidate in ["adapter_model.safetensors", "adapter_model.bin"]:
        p = adapter_dir / candidate
        if p.exists():
            weights_file = p
            break

    if weights_file is None:
        logger.error("No adapter weights found in %s", adapter_path)
        sys.exit(1)

    logger.info("Loading adapter weights from %s", weights_file)

    if weights_file.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(weights_file), device="cpu")
    else:
        state_dict = torch.load(str(weights_file), map_location="cpu")

    # Parse the state dict keys to extract per-layer, per-module weights
    # Expected key patterns:
    #   base_model.model.model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_A.default.weight
    #   base_model.model.model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_B.default.weight
    #   base_model.model.model.layers.{i}.mlp.{gate,up,down}_proj.lora_A.default.weight
    #   base_model.model.model.layers.{i}.mlp.{gate,up,down}_proj.lora_B.default.weight
    #
    # Some adapters omit "default":
    #   base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight

    import re

    layer_data: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    # Flexible regex to handle both "lora_A.default.weight" and "lora_A.weight"
    pattern = re.compile(
        r"(?:base_model\.model\.)?model\.layers\.(\d+)\."
        r"(?:self_attn|mlp)\."
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\."
        r"lora_(A|B)(?:\.default)?\.weight"
    )

    matched = 0
    for key, tensor in state_dict.items():
        m = pattern.search(key)
        if m:
            layer_idx = int(m.group(1))
            module_name = m.group(2)
            ab = m.group(3)  # "A" or "B"
            layer_data[layer_idx][module_name][ab] = tensor.float()
            matched += 1

    logger.info(
        "Parsed %d LoRA weight matrices across %d layers",
        matched, len(layer_data),
    )

    if matched == 0:
        logger.warning("No LoRA weights matched expected pattern.")
        logger.info("Available keys (first 10):")
        for k in list(state_dict.keys())[:10]:
            logger.info("  %s", k)

    return dict(layer_data)


def compute_norms(
    layer_data: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Compute per-layer, per-module norms.

    Returns:
        {layer_idx: {module_name: {"A_norm": float, "B_norm": float, "BA_norm": float}}}
    """
    norms = {}
    for layer_idx in range(NUM_LAYERS):
        norms[layer_idx] = {}
        modules = layer_data.get(layer_idx, {})
        for mod in MODULE_NAMES:
            if mod in modules and "A" in modules[mod] and "B" in modules[mod]:
                A = modules[mod]["A"]  # [r, d_in]
                B = modules[mod]["B"]  # [d_out, r]
                BA = B @ A             # [d_out, d_in]
                norms[layer_idx][mod] = {
                    "A_norm": torch.norm(A, p="fro").item(),
                    "B_norm": torch.norm(B, p="fro").item(),
                    "BA_norm": torch.norm(BA, p="fro").item(),
                }
            else:
                norms[layer_idx][mod] = {
                    "A_norm": 0.0,
                    "B_norm": 0.0,
                    "BA_norm": 0.0,
                }
    return norms


def print_table(
    norms: Dict[int, Dict[str, Dict[str, float]]],
    cb_layers: List[int],
) -> Tuple[List[float], float]:
    """Print a formatted table and return (total_norms, median)."""
    print()
    print("=" * 120)
    print("LoRA Effective Perturbation Norms  ||B @ A||_F  per Layer per Module")
    print("=" * 120)

    # Header
    hdr_parts = [f"{'Layer':>6s}", f"{'CB':>3s}"]
    for short in MODULE_SHORT:
        hdr_parts.append(f"{short:>9s}")
    hdr_parts.append(f"{'TOTAL':>10s}")
    hdr_parts.append(f"{'FLAG':>8s}")
    print("  ".join(hdr_parts))
    print("-" * 120)

    total_norms = []
    for layer_idx in range(NUM_LAYERS):
        is_cb = layer_idx in cb_layers
        cb_mark = " *" if is_cb else "  "
        parts = [f"{layer_idx:>6d}", f"{cb_mark:>3s}"]

        layer_total = 0.0
        for mod in MODULE_NAMES:
            val = norms[layer_idx][mod]["BA_norm"]
            layer_total += val
            parts.append(f"{val:>9.4f}")

        parts.append(f"{layer_total:>10.4f}")
        total_norms.append(layer_total)

        # Flag placeholder (filled after we know median)
        parts.append("")
        print("  ".join(parts))

    # Compute median of non-CB layers for flagging
    non_cb_totals = [
        total_norms[i] for i in range(NUM_LAYERS) if i not in cb_layers and total_norms[i] > 0
    ]
    if non_cb_totals:
        median_val = sorted(non_cb_totals)[len(non_cb_totals) // 2]
    else:
        median_val = 0.0

    # Reprint with flags
    print()
    print("=" * 120)
    print("LoRA Effective Perturbation Norms  ||B @ A||_F  per Layer per Module")
    print("CB target layers marked with *. Flagged layers exceed non-CB median.")
    print("=" * 120)

    hdr_parts = [f"{'Layer':>6s}", f"{'CB':>3s}"]
    for short in MODULE_SHORT:
        hdr_parts.append(f"{short:>9s}")
    hdr_parts.append(f"{'TOTAL':>10s}")
    hdr_parts.append(f"{'FLAG':>12s}")
    print("  ".join(hdr_parts))
    print("-" * 120)

    flagged_layers = []
    for layer_idx in range(NUM_LAYERS):
        is_cb = layer_idx in cb_layers
        cb_mark = " *" if is_cb else "  "
        parts = [f"{layer_idx:>6d}", f"{cb_mark:>3s}"]

        layer_total = 0.0
        for mod in MODULE_NAMES:
            val = norms[layer_idx][mod]["BA_norm"]
            layer_total += val
            parts.append(f"{val:>9.4f}")

        parts.append(f"{layer_total:>10.4f}")

        flag = ""
        if not is_cb and layer_total > median_val and layer_total > 0:
            flag = "COLLATERAL"
            flagged_layers.append(layer_idx)
        elif is_cb:
            flag = "CB-TARGET"

        parts.append(f"{flag:>12s}")
        print("  ".join(parts))

    print("-" * 120)
    print(f"  Non-CB median: {median_val:.4f}")
    if flagged_layers:
        print(f"  Flagged collateral layers: {flagged_layers}")
    else:
        print("  No collateral damage layers detected.")
    print()

    return total_norms, median_val


def plot_norms(
    norms: Dict[int, Dict[str, Dict[str, float]]],
    total_norms: List[float],
    cb_layers: List[int],
    median_val: float,
    output_dir: Path,
):
    """Save a bar plot of total LoRA norm per layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=(14, 5))

    layers = list(range(NUM_LAYERS))
    colors = []
    for i in layers:
        if i in cb_layers:
            colors.append("#2196F3")  # blue for CB target
        elif total_norms[i] > median_val and total_norms[i] > 0:
            colors.append("#F44336")  # red for collateral
        else:
            colors.append("#9E9E9E")  # gray for normal

    bars = ax.bar(layers, total_norms, color=colors, edgecolor="white", linewidth=0.5)

    # Median line (non-CB)
    ax.axhline(y=median_val, color="#FF9800", linestyle="--", linewidth=1.2,
               label=f"Non-CB median = {median_val:.4f}")

    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel(r"$\| B \cdot A \|_F$ (total across modules)", fontsize=12)
    ax.set_title("LoRA Effective Perturbation Norm per Layer", fontsize=14, fontweight="bold")
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8, rotation=45)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="CB target layer"),
        Patch(facecolor="#F44336", label="Collateral (> median)"),
        Patch(facecolor="#9E9E9E", label="Normal"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save PDF and PNG
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "lora_norms_per_layer.pdf"
    png_path = output_dir / "lora_norms_per_layer.png"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", pdf_path)
    logger.info("Saved plot: %s", png_path)

    # Also make a stacked bar chart by module type
    fig2, ax2 = plt.subplots(figsize=(14, 6))

    bottom = [0.0] * NUM_LAYERS
    module_colors = {
        "q_proj": "#1565C0",
        "k_proj": "#1976D2",
        "v_proj": "#42A5F5",
        "o_proj": "#90CAF9",
        "gate_proj": "#2E7D32",
        "up_proj": "#66BB6A",
        "down_proj": "#A5D6A7",
    }

    for mod in MODULE_NAMES:
        vals = [norms[i][mod]["BA_norm"] for i in range(NUM_LAYERS)]
        ax2.bar(layers, vals, bottom=bottom, color=module_colors[mod],
                label=mod, edgecolor="white", linewidth=0.3)
        bottom = [b + v for b, v in zip(bottom, vals)]

    # Mark CB layers
    for cb_l in cb_layers:
        ax2.axvline(x=cb_l, color="#F44336", linestyle=":", linewidth=1.0, alpha=0.7)

    ax2.set_xlabel("Transformer Layer", fontsize=12)
    ax2.set_ylabel(r"$\| B \cdot A \|_F$", fontsize=12)
    ax2.set_title("LoRA Perturbation by Module Type", fontsize=14, fontweight="bold")
    ax2.set_xticks(layers)
    ax2.set_xticklabels([str(l) for l in layers], fontsize=8, rotation=45)
    ax2.legend(fontsize=9, ncol=4, loc="upper right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    pdf2 = output_dir / "lora_norms_by_module.pdf"
    png2 = output_dir / "lora_norms_by_module.png"
    fig2.savefig(pdf2, dpi=150, bbox_inches="tight")
    fig2.savefig(png2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Saved plot: %s", pdf2)
    logger.info("Saved plot: %s", png2)


def save_json_summary(
    norms: Dict[int, Dict[str, Dict[str, float]]],
    total_norms: List[float],
    cb_layers: List[int],
    median_val: float,
    output_dir: Path,
):
    """Save a JSON summary of all norms."""
    summary = {
        "cb_layers": cb_layers,
        "non_cb_median": median_val,
        "layers": {},
    }
    for layer_idx in range(NUM_LAYERS):
        layer_info = {
            "is_cb_target": layer_idx in cb_layers,
            "total_BA_norm": total_norms[layer_idx],
            "flagged_collateral": (
                layer_idx not in cb_layers
                and total_norms[layer_idx] > median_val
                and total_norms[layer_idx] > 0
            ),
            "modules": norms[layer_idx],
        }
        summary["layers"][str(layer_idx)] = layer_info

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "lora_norms_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved JSON summary: %s", json_path)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LoRA weight norms per layer from a saved checkpoint"
    )
    parser.add_argument(
        "--adapter", type=str, required=True,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--output", type=str, default="results/lora_norms",
        help="Output directory for plots and summary (default: results/lora_norms)",
    )
    parser.add_argument(
        "--cb-layers", type=int, nargs="+", default=[10, 20],
        help="CB target layers to highlight (default: 10 20)",
    )
    args = parser.parse_args()

    adapter_path = args.adapter
    output_dir = Path(args.output)
    cb_layers = args.cb_layers

    logger.info("Adapter: %s", adapter_path)
    logger.info("CB layers: %s", cb_layers)
    logger.info("Output: %s", output_dir)

    # 1. Extract weights
    layer_data = extract_lora_weights(adapter_path)

    if not layer_data:
        logger.error("No LoRA weights found. Exiting.")
        sys.exit(1)

    # 2. Compute norms
    norms = compute_norms(layer_data)

    # 3. Print table
    total_norms, median_val = print_table(norms, cb_layers)

    # 4. Plot
    plot_norms(norms, total_norms, cb_layers, median_val, output_dir)

    # 5. JSON summary
    save_json_summary(norms, total_norms, cb_layers, median_val, output_dir)

    # Final summary
    active_layers = [i for i in range(NUM_LAYERS) if total_norms[i] > 0]
    collateral = [
        i for i in range(NUM_LAYERS)
        if i not in cb_layers and total_norms[i] > median_val and total_norms[i] > 0
    ]
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Active LoRA layers: {len(active_layers)} / {NUM_LAYERS}")
    print(f"  CB target layers: {cb_layers}")
    print(f"  Non-CB median ||B@A||_F: {median_val:.4f}")
    if collateral:
        print(f"  Collateral damage layers: {collateral}")
        max_collateral = max(collateral, key=lambda i: total_norms[i])
        print(f"  Worst collateral: layer {max_collateral} "
              f"(norm={total_norms[max_collateral]:.4f}, "
              f"{total_norms[max_collateral]/median_val:.1f}x median)")
    else:
        print("  No collateral damage detected.")
    print(f"  Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
