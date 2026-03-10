#!/usr/bin/env python3
"""
Zero out LoRA weights for specified layers in an existing adapter.

Creates a new adapter directory with the ablated weights. Use for
testing whether distant-from-CB layers are "collateral damage" or
load-bearing, without retraining.

Usage:
    # Zero out layers 0-7 and 23-31 (keep only 8-22):
    python scripts/ablate_lora_layers.py \
        --adapter /path/to/adapter \
        --keep-layers 8-22 \
        --output /path/to/ablated_adapter

    # Zero out everything except CB target layers:
    python scripts/ablate_lora_layers.py \
        --adapter /path/to/adapter \
        --keep-layers 10,20 \
        --output /path/to/ablated_adapter
"""

import argparse
import re
import shutil
from pathlib import Path


def parse_layer_spec(spec: str) -> set:
    """Parse '8-22' or '10,20' or '8-12,18-22' into a set of ints."""
    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            layers.update(range(int(lo), int(hi) + 1))
        else:
            layers.add(int(part))
    return layers


def main():
    parser = argparse.ArgumentParser(description="Ablate LoRA layers by zeroing weights")
    parser.add_argument("--adapter", type=Path, required=True, help="Source adapter dir")
    parser.add_argument("--keep-layers", type=str, required=True,
                        help="Layers to KEEP (e.g., '8-22' or '10,20'). All others zeroed.")
    parser.add_argument("--output", type=Path, required=True, help="Output adapter dir")
    args = parser.parse_args()

    import torch
    from safetensors.torch import load_file, save_file

    keep = parse_layer_spec(args.keep_layers)
    print(f"Keeping layers: {sorted(keep)}")

    # Find weight file
    weight_files = list(args.adapter.glob("adapter_model*.safetensors"))
    use_safetensors = bool(weight_files)
    if not weight_files:
        weight_files = list(args.adapter.glob("adapter_model*.bin"))
    if not weight_files:
        print(f"ERROR: No adapter weights in {args.adapter}")
        return

    # Load
    wf = weight_files[0]
    print(f"Loading: {wf}")
    if use_safetensors:
        state_dict = load_file(wf)
    else:
        state_dict = torch.load(wf, map_location="cpu")

    # Zero out layers not in keep set
    zeroed_layers = set()
    kept_layers = set()
    for name in list(state_dict.keys()):
        match = re.search(r"layers\.(\d+)\.", name)
        if match:
            layer_idx = int(match.group(1))
            if layer_idx not in keep:
                state_dict[name] = torch.zeros_like(state_dict[name])
                zeroed_layers.add(layer_idx)
            else:
                kept_layers.add(layer_idx)

    print(f"Kept {len(kept_layers)} layers: {sorted(kept_layers)}")
    print(f"Zeroed {len(zeroed_layers)} layers: {sorted(zeroed_layers)}")

    # Copy adapter dir and save modified weights
    args.output.mkdir(parents=True, exist_ok=True)
    for f in args.adapter.iterdir():
        if f.name.startswith("adapter_model"):
            continue  # We'll write the modified one
        dst = args.output / f.name
        if not dst.exists():
            shutil.copy2(f, dst)

    # Save
    out_path = args.output / wf.name
    if use_safetensors:
        save_file(state_dict, out_path)
    else:
        torch.save(state_dict, out_path)
    print(f"Saved ablated adapter to: {args.output}")

    # Compute norms for verification
    from collections import defaultdict
    layer_norms = defaultdict(float)
    for name, param in state_dict.items():
        match = re.search(r"layers\.(\d+)\.", name)
        if match:
            layer_idx = int(match.group(1))
            layer_norms[layer_idx] += param.float().norm().item()

    print("\nPost-ablation norms:")
    for layer_idx in sorted(layer_norms.keys()):
        status = "KEPT" if layer_idx in keep else "ZERO"
        print(f"  Layer {layer_idx:>2}: {layer_norms[layer_idx]:.4f}  [{status}]")


if __name__ == "__main__":
    main()
