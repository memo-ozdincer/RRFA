#!/usr/bin/env python3
"""
D3: LoRA weight norm per layer.

Computes ||A_l @ B_l||_F for each LoRA layer in a checkpoint.
Layers with large norms far from CB targets (10, 20) are learning
global model modifications beyond what CB requires — "collateral damage"
layers. This identifies where the adapter is making unnecessary changes.

Usage:
    python scripts/diagnostics/lora_weight_norms.py \
        --adapter /path/to/lora/adapter \
        --cb-layers 10 20 \
        --output results/lora_norms.json \
        --plot results/lora_norms.png
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LoRA weight norm analysis (D3)")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter dir")
    parser.add_argument("--cb-layers", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()

    import torch
    from safetensors.torch import load_file

    # Load adapter weights
    adapter_path = Path(args.adapter)
    weight_files = list(adapter_path.glob("adapter_model*.safetensors"))
    if not weight_files:
        weight_files = list(adapter_path.glob("adapter_model*.bin"))

    if not weight_files:
        logger.error("No adapter weight files found in %s", adapter_path)
        return

    logger.info("Loading adapter weights from %s", weight_files[0])
    if weight_files[0].suffix == ".safetensors":
        state_dict = load_file(weight_files[0])
    else:
        state_dict = torch.load(weight_files[0], map_location="cpu")

    # Parse LoRA weight names to extract layer indices
    # Typical names: base_model.model.model.layers.10.self_attn.q_proj.lora_A.weight
    layer_norms = defaultdict(lambda: {"modules": {}, "total_norm": 0.0})

    for name, param in state_dict.items():
        # Extract layer index
        parts = name.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    continue

        if layer_idx is None:
            continue

        # Extract module name and A/B
        is_a = "lora_A" in name
        is_b = "lora_B" in name
        if not (is_a or is_b):
            continue

        # Get module path (e.g., self_attn.q_proj)
        module_parts = []
        for i, p in enumerate(parts):
            if p in ("self_attn", "mlp"):
                # Grab module type and proj name
                module_parts = parts[i:i+2] if i + 1 < len(parts) else [p]
                break
        module_name = ".".join(module_parts) if module_parts else "unknown"

        key = f"layer_{layer_idx}_{module_name}"
        ab = "A" if is_a else "B"
        layer_norms[key][ab] = param.float()
        layer_norms[key]["layer"] = layer_idx
        layer_norms[key]["module"] = module_name

    # Compute ||A @ B||_F for each module, aggregate per layer
    per_layer = defaultdict(float)
    per_layer_modules = defaultdict(list)
    module_details = []

    for key, info in layer_norms.items():
        if "A" not in info or "B" not in info:
            continue

        A = info["A"]
        B = info["B"]
        layer_idx = info["layer"]
        module_name = info["module"]

        # LoRA: output = x @ A^T @ B^T, so effective weight delta = B @ A
        # A: (r, in), B: (out, r) → delta: (out, in)
        delta = B @ A
        norm = torch.norm(delta).item()

        per_layer[layer_idx] += norm
        per_layer_modules[layer_idx].append((module_name, norm))
        module_details.append({
            "layer": layer_idx,
            "module": module_name,
            "norm": norm,
            "A_shape": list(A.shape),
            "B_shape": list(B.shape),
        })

    # Print results
    print("=" * 70)
    print("LoRA Weight Norms Per Layer")
    print("=" * 70)
    print(f"{'Layer':>6} {'Total ||BA||_F':>14} {'CB Target':>10}  Modules")
    print("-" * 70)

    max_norm = max(per_layer.values()) if per_layer else 1.0
    for layer_idx in sorted(per_layer.keys()):
        norm = per_layer[layer_idx]
        is_cb = "  ←" if layer_idx in args.cb_layers else ""
        bar = "█" * int(30 * norm / max_norm)
        modules = per_layer_modules[layer_idx]
        top_module = max(modules, key=lambda x: x[1])[0] if modules else ""
        print(f"  {layer_idx:>4}  {norm:>12.4f}  {is_cb:>10}  {bar}  {top_module}")

    # Summary
    cb_norm = sum(per_layer.get(l, 0) for l in args.cb_layers)
    total_norm = sum(per_layer.values())
    non_cb_norm = total_norm - cb_norm

    print()
    print(f"CB layer norm:     {cb_norm:.4f} ({cb_norm/total_norm:.1%} of total)")
    print(f"Non-CB layer norm: {non_cb_norm:.4f} ({non_cb_norm/total_norm:.1%} of total)")
    print(f"Ratio:             {non_cb_norm/cb_norm:.2f}x")
    print()
    if non_cb_norm / total_norm > 0.5:
        print("⚠  More than 50% of LoRA weight change is OUTSIDE CB target layers.")
        print("   This suggests significant 'collateral damage' — the adapter is")
        print("   modifying the model globally, not just at CB-relevant layers.")
    else:
        print("✓  Majority of weight change concentrated at CB target layers.")

    # Save
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "per_layer": {str(k): v for k, v in sorted(per_layer.items())},
            "cb_layers": args.cb_layers,
            "cb_norm": cb_norm,
            "non_cb_norm": non_cb_norm,
            "total_norm": total_norm,
            "modules": module_details,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info("Saved to %s", args.output)

    # Plot
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            layers = sorted(per_layer.keys())
            norms = [per_layer[l] for l in layers]
            colors = ["red" if l in args.cb_layers else "steelblue" for l in layers]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(layers, norms, color=colors, alpha=0.8)
            ax.set_xlabel("Layer")
            ax.set_ylabel("||B @ A||_F")
            ax.set_title("LoRA Weight Norms Per Layer (red = CB target)")
            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
            logger.info("Plot saved to %s", args.plot)
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
