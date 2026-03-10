#!/usr/bin/env python3
"""
D5: Distance distribution analysis between CB-enabled and CB-disabled representations.

Computes dl2rc distance at CB target layers for harmful and benign eval samples.
The overlap between harmful and benign distance distributions IS the selectivity
problem visualized. A threshold on this distance gives a free inference-time
classifier: large distance = "CB pushed hard" = "probably harmful."

Usage (on cluster with GPU):
    python scripts/diagnostics/distance_distributions.py \
        --model /path/to/base/model \
        --adapter /path/to/lora/adapter \
        --harmful-traces data/traces/agentdojo_truncated.jsonl \
        --benign-traces data/traces/agentdojo_truncated.jsonl \
        --layers 10 20 \
        --output results/distance_distributions.json \
        --plot results/distance_distributions.png

Output:
    - JSON with per-sample distances (layer, category, distance)
    - Optional plot: overlapping histograms of harmful vs benign distances
    - Threshold analysis: optimal distance cutoff for classifier
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_traces(path: Path, category: str, max_samples: int = 200):
    """Load traces filtered by category."""
    traces = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("labels", {}).get("category", "") == category:
                traces.append(d)
                if len(traces) >= max_samples:
                    break
    return traces


def render_trace(trace, tokenizer, max_length):
    """Render trace to input_ids."""
    messages = trace.get("messages", [])
    if not messages:
        return None

    chat_messages = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("system", "user", "assistant", "tool") and content:
            mapped_role = "user" if role == "tool" else role
            chat_messages.append({"role": mapped_role, "content": str(content)})

    if not chat_messages:
        return None

    try:
        text = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=False)
        encoded = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=False)
        return encoded
    except Exception:
        return None


def compute_dl2rc(a, b):
    """dl2rc distance: L2 + 10*ReLU(1 - cos_sim)"""
    import torch
    import torch.nn.functional as F
    l2 = torch.norm(a - b, dim=-1)
    cos_dist = 1.0 - F.cosine_similarity(a, b, dim=-1, eps=1e-8)
    return l2 + 10.0 * F.relu(cos_dist)


def main():
    parser = argparse.ArgumentParser(description="Distance distributions (D5)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter dir")
    parser.add_argument("--harmful-traces", type=Path, required=True)
    parser.add_argument("--benign-traces", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Load model
    logger.info("Loading base model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    logger.info("Loading adapter: %s", args.adapter)
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()
    device = next(model.parameters()).device

    # Load traces
    harmful = load_traces(args.harmful_traces, "harmful", args.max_samples)
    benign = load_traces(args.benign_traces, "benign", args.max_samples)
    logger.info("Loaded %d harmful, %d benign traces", len(harmful), len(benign))

    results = []

    for category, traces in [("harmful", harmful), ("benign", benign)]:
        for i, trace in enumerate(traces):
            encoded = render_trace(trace, tokenizer, args.max_length)
            if encoded is None:
                continue

            input_ids = encoded["input_ids"].to(device)
            attn_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                # CB-enabled (adapter active)
                model.enable_adapter_layers()
                out_enabled = model(input_ids=input_ids, attention_mask=attn_mask,
                                    output_hidden_states=True, use_cache=False)

                # CB-disabled (adapter off)
                model.disable_adapter_layers()
                out_disabled = model(input_ids=input_ids, attention_mask=attn_mask,
                                     output_hidden_states=True, use_cache=False)

                model.enable_adapter_layers()

            seq_len = int(attn_mask.sum().item())
            last_pos = seq_len - 1

            for layer in args.layers:
                hs_idx = layer + 1
                if hs_idx >= len(out_enabled.hidden_states):
                    continue

                h_enabled = out_enabled.hidden_states[hs_idx][0, last_pos, :]
                h_disabled = out_disabled.hidden_states[hs_idx][0, last_pos, :]

                dist = compute_dl2rc(h_enabled.unsqueeze(0), h_disabled.unsqueeze(0))

                results.append({
                    "trace_id": trace.get("id", f"{category}_{i}"),
                    "category": category,
                    "domain": trace.get("source", {}).get("subset", "unknown"),
                    "layer": layer,
                    "distance": float(dist.item()),
                    "seq_len": seq_len,
                })

            if (i + 1) % 20 == 0:
                logger.info("  %s: %d/%d done", category, i + 1, len(traces))

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %d distance measurements to %s", len(results), args.output)

    # Analysis
    for layer in args.layers:
        h_dists = [r["distance"] for r in results if r["category"] == "harmful" and r["layer"] == layer]
        b_dists = [r["distance"] for r in results if r["category"] == "benign" and r["layer"] == layer]

        if not h_dists or not b_dists:
            continue

        h_arr = np.array(h_dists)
        b_arr = np.array(b_dists)

        print(f"\nLayer {layer}:")
        print(f"  Harmful: mean={h_arr.mean():.2f}, std={h_arr.std():.2f}, "
              f"median={np.median(h_arr):.2f}")
        print(f"  Benign:  mean={b_arr.mean():.2f}, std={b_arr.std():.2f}, "
              f"median={np.median(b_arr):.2f}")
        print(f"  Separation: {h_arr.mean() - b_arr.mean():.2f} "
              f"(harmful should be LARGER if CB pushed)")

        # Find optimal threshold
        all_dists = np.concatenate([h_arr, b_arr])
        all_labels = np.concatenate([np.ones(len(h_arr)), np.zeros(len(b_arr))])

        best_acc = 0
        best_thresh = 0
        for thresh in np.linspace(all_dists.min(), all_dists.max(), 100):
            preds = (all_dists > thresh).astype(int)
            acc = (preds == all_labels).mean()
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        print(f"  Optimal threshold: {best_thresh:.2f} (accuracy={best_acc:.1%})")
        print(f"  → At this threshold: "
              f"harmful recall={(h_arr > best_thresh).mean():.1%}, "
              f"benign preserved={(b_arr <= best_thresh).mean():.1%}")

    # Plot if requested
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, len(args.layers), figsize=(6 * len(args.layers), 4))
            if len(args.layers) == 1:
                axes = [axes]

            for ax, layer in zip(axes, args.layers):
                h_dists = [r["distance"] for r in results
                           if r["category"] == "harmful" and r["layer"] == layer]
                b_dists = [r["distance"] for r in results
                           if r["category"] == "benign" and r["layer"] == layer]

                ax.hist(h_dists, bins=30, alpha=0.6, label="Harmful", color="red")
                ax.hist(b_dists, bins=30, alpha=0.6, label="Benign", color="blue")
                ax.axvline(best_thresh, color="black", linestyle="--", label=f"Threshold={best_thresh:.1f}")
                ax.set_xlabel("dl2rc Distance (CB-enabled vs CB-disabled)")
                ax.set_ylabel("Count")
                ax.set_title(f"Layer {layer}")
                ax.legend()

            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
            logger.info("Plot saved to %s", args.plot)
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
