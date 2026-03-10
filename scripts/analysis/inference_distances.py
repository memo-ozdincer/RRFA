#!/usr/bin/env python3
"""
Compute dl2rc distances between adapter-enabled and adapter-disabled
representations at inference time for each eval sample.

Produces distance distributions for harmful vs benign, overlap
statistics, and optimal threshold analysis.

For each trace:
  1. Forward pass with adapter ENABLED  -> hidden states at target layers
  2. Forward pass with adapter DISABLED -> hidden states at target layers
  3. Compute dl2rc distance at last-token position

Outputs:
  - Overlapping histograms: harmful vs benign distance distributions
  - ROC curve
  - CSV of per-sample distances
  - Threshold analysis: detection rate vs false positive rate

Reuses data loading patterns from probe_separability.py.

Usage:
  python scripts/analysis/inference_distances.py \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --adapter /path/to/lora_adapter \
      --traces data/traces/agentdojo_truncated.jsonl \
      --layers 10 20 \
      --max-samples 200 \
      --output results/inference_distances
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Data loading (from probe_separability.py) ─────────────────────────


def load_traces_split(
    path: Path, max_samples: int = 0
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load traces and split by labels.category."""
    harmful, benign = [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            trace = json.loads(line)
            cat = trace.get("labels", {}).get("category", "")
            if cat == "harmful":
                harmful.append(trace)
            elif cat == "benign":
                benign.append(trace)

    if max_samples > 0:
        harmful = harmful[:max_samples]
        benign = benign[:max_samples]

    return harmful, benign


# ── Rendering ─────────────────────────────────────────────────────────


def render_trace(
    trace: Dict[str, Any], tokenizer, max_length: int
) -> Optional[Dict]:
    """Render a trace to input_ids using chat template."""
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
        text = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=False
        )
        encoded = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
        }
    except Exception as e:
        logger.debug("Failed to render trace %s: %s", trace.get("id", "?"), e)
        return None


# ── Distance function ─────────────────────────────────────────────────


def dl2rc(a, b):
    """dl2rc distance: L2 + 10*ReLU(cos_dist).

    Args:
        a, b: (hidden_dim,) tensors

    Returns:
        scalar tensor
    """
    import torch
    import torch.nn.functional as F

    l2 = torch.norm(a - b, p=2)
    cos_dist = 1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
    return l2 + 10.0 * F.relu(cos_dist).squeeze()


# ── Hidden state extraction with adapter toggle ──────────────────────


def extract_distances_for_sample(
    model,
    input_ids,
    attention_mask,
    target_layers: List[int],
    device,
) -> Dict[int, float]:
    """Run forward pass with adapter ON and OFF, compute dl2rc distances.

    Uses PEFT's disable_adapter() context manager for the frozen pass.

    Returns:
        {layer_idx: dl2rc_distance} at the last-token position.
    """
    import torch

    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    seq_len = int(attention_mask.sum().item())
    last_pos = seq_len - 1

    distances = {}

    with torch.no_grad():
        # Forward with adapter ENABLED
        outputs_on = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Forward with adapter DISABLED
        with model.disable_adapter():
            outputs_off = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        for layer_idx in target_layers:
            hs_idx = layer_idx + 1  # hidden_states[0] is embeddings
            if hs_idx >= len(outputs_on.hidden_states):
                continue
            if hs_idx >= len(outputs_off.hidden_states):
                continue

            h_on = outputs_on.hidden_states[hs_idx][0, last_pos, :].float()
            h_off = outputs_off.hidden_states[hs_idx][0, last_pos, :].float()
            distances[layer_idx] = dl2rc(h_on, h_off).item()

    return distances


# ── Statistics ────────────────────────────────────────────────────────


def compute_overlap_stats(
    harmful_dists: np.ndarray,
    benign_dists: np.ndarray,
) -> Dict[str, Any]:
    """Compute overlap statistics between harmful and benign distributions.

    Returns dict with KL divergence, optimal threshold, detection rate, FPR.
    """
    # Estimate KL divergence via histogram binning
    all_vals = np.concatenate([harmful_dists, benign_dists])
    n_bins = min(50, max(10, len(all_vals) // 5))
    bin_edges = np.linspace(all_vals.min() - 1e-8, all_vals.max() + 1e-8, n_bins + 1)

    h_counts, _ = np.histogram(harmful_dists, bins=bin_edges, density=True)
    b_counts, _ = np.histogram(benign_dists, bins=bin_edges, density=True)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    h_prob = h_counts / (h_counts.sum() + eps) + eps
    b_prob = b_counts / (b_counts.sum() + eps) + eps

    # KL(harmful || benign)
    kl_hb = float(np.sum(h_prob * np.log(h_prob / b_prob)))
    # KL(benign || harmful)
    kl_bh = float(np.sum(b_prob * np.log(b_prob / h_prob)))

    # Find threshold that maximizes balanced accuracy
    thresholds = np.linspace(all_vals.min(), all_vals.max(), 500)
    best_threshold = 0.0
    best_balanced_acc = 0.0
    best_tpr = 0.0
    best_fpr = 0.0

    # Harmful samples should have LARGER distances (adapter pushes them away)
    # So: predict "harmful" if distance > threshold
    for t in thresholds:
        tpr = (harmful_dists > t).mean()  # true positive rate (harmful detection)
        fpr = (benign_dists > t).mean()   # false positive rate
        tnr = 1.0 - fpr
        balanced_acc = 0.5 * (tpr + tnr)
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_threshold = float(t)
            best_tpr = float(tpr)
            best_fpr = float(fpr)

    # Also compute full ROC for AUC
    tprs = []
    fprs = []
    for t in thresholds:
        tpr = float((harmful_dists > t).mean())
        fpr = float((benign_dists > t).mean())
        tprs.append(tpr)
        fprs.append(fpr)

    # Sort by FPR for AUC computation
    sorted_pairs = sorted(zip(fprs, tprs))
    sorted_fprs = [p[0] for p in sorted_pairs]
    sorted_tprs = [p[1] for p in sorted_pairs]
    auc = float(np.trapz(sorted_tprs, sorted_fprs))

    return {
        "kl_harmful_benign": kl_hb,
        "kl_benign_harmful": kl_bh,
        "optimal_threshold": best_threshold,
        "detection_rate_at_optimal": best_tpr,
        "fpr_at_optimal": best_fpr,
        "balanced_accuracy": best_balanced_acc,
        "auc": auc,
        "harmful_mean": float(harmful_dists.mean()),
        "harmful_std": float(harmful_dists.std()),
        "benign_mean": float(benign_dists.mean()),
        "benign_std": float(benign_dists.std()),
        "roc_fprs": sorted_fprs,
        "roc_tprs": sorted_tprs,
    }


# ── Plotting ──────────────────────────────────────────────────────────


def plot_distributions(
    harmful_dists: np.ndarray,
    benign_dists: np.ndarray,
    stats: Dict[str, Any],
    layer_label: str,
    output_dir: Path,
):
    """Create publication-quality histogram and ROC curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left panel: overlapping histograms ──
    ax = axes[0]
    n_bins = min(40, max(15, (len(harmful_dists) + len(benign_dists)) // 8))
    all_vals = np.concatenate([harmful_dists, benign_dists])
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)

    ax.hist(harmful_dists, bins=bin_edges, alpha=0.6, color="#D32F2F",
            label=f"Harmful (n={len(harmful_dists)})", density=True, edgecolor="white")
    ax.hist(benign_dists, bins=bin_edges, alpha=0.6, color="#1976D2",
            label=f"Benign (n={len(benign_dists)})", density=True, edgecolor="white")

    # Threshold line
    ax.axvline(x=stats["optimal_threshold"], color="#FF9800", linestyle="--",
               linewidth=2.0, label=f"Threshold = {stats['optimal_threshold']:.2f}")

    ax.set_xlabel("dl2rc Distance (adapter ON vs OFF)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Distance Distribution ({layer_label})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add stats annotation
    stats_text = (
        f"Detection: {stats['detection_rate_at_optimal']:.0%}\n"
        f"FPR: {stats['fpr_at_optimal']:.0%}\n"
        f"AUC: {stats['auc']:.3f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    # ── Right panel: ROC curve ──
    ax2 = axes[1]
    ax2.plot(stats["roc_fprs"], stats["roc_tprs"], color="#D32F2F", linewidth=2.0,
             label=f"ROC (AUC = {stats['auc']:.3f})")
    ax2.plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--", linewidth=1.0, label="Random")

    # Mark optimal threshold point
    ax2.scatter([stats["fpr_at_optimal"]], [stats["detection_rate_at_optimal"]],
                color="#FF9800", s=100, zorder=5, edgecolors="black", linewidth=1.5)
    ax2.annotate(
        f"({stats['fpr_at_optimal']:.2f}, {stats['detection_rate_at_optimal']:.2f})",
        xy=(stats["fpr_at_optimal"], stats["detection_rate_at_optimal"]),
        xytext=(stats["fpr_at_optimal"] + 0.1, stats["detection_rate_at_optimal"] - 0.1),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("True Positive Rate (Detection)", fontsize=11)
    ax2.set_title(f"ROC Curve ({layer_label})", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, loc="lower right")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_aspect("equal")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(alpha=0.2)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = layer_label.replace(" ", "_").replace(",", "_")
    pdf_path = output_dir / f"distance_distribution_{safe_label}.pdf"
    png_path = output_dir / f"distance_distribution_{safe_label}.png"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", pdf_path)


def save_csv(
    records: List[Dict[str, Any]],
    target_layers: List[int],
    output_dir: Path,
):
    """Save per-sample distances as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_sample_distances.csv"

    fieldnames = ["sample_idx", "category", "trace_id"]
    for layer in target_layers:
        fieldnames.append(f"dl2rc_layer{layer}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    logger.info("Saved CSV: %s (%d rows)", csv_path, len(records))


def main():
    parser = argparse.ArgumentParser(
        description="Compute dl2rc distances between adapter ON/OFF representations"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Base model path (HuggingFace model ID or local path)",
    )
    parser.add_argument(
        "--adapter", type=str, required=True,
        help="LoRA adapter path",
    )
    parser.add_argument(
        "--traces", type=Path, required=True,
        help="Traces JSONL file (split by labels.category)",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[10, 20],
        help="Target layers for distance computation (default: 10 20)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=200,
        help="Max samples per category (default: 200)",
    )
    parser.add_argument(
        "--max-length", type=int, default=2048,
        help="Max sequence length for tokenization (default: 2048)",
    )
    parser.add_argument(
        "--output", type=str, default="results/inference_distances",
        help="Output directory (default: results/inference_distances)",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Model dtype (default: bfloat16)",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    output_dir = Path(args.output)
    target_layers = args.layers

    # ── Load data ──
    logger.info("Loading traces from %s", args.traces)
    harmful_traces, benign_traces = load_traces_split(args.traces, args.max_samples)
    logger.info("Harmful: %d, Benign: %d", len(harmful_traces), len(benign_traces))

    if len(harmful_traces) < 5 or len(benign_traces) < 5:
        logger.error(
            "Too few samples: %d harmful, %d benign (need >= 5 each)",
            len(harmful_traces), len(benign_traces),
        )
        sys.exit(1)

    # ── Load model + adapter ──
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    logger.info("Loading base model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Loading adapter: %s", args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    device = next(model.parameters()).device
    logger.info("Model loaded on device: %s", device)

    # ── Extract distances ──
    logger.info("Computing dl2rc distances at layers %s ...", target_layers)

    records = []  # per-sample records for CSV
    harmful_dists_by_layer: Dict[int, List[float]] = {l: [] for l in target_layers}
    benign_dists_by_layer: Dict[int, List[float]] = {l: [] for l in target_layers}

    for category, traces, dist_dict in [
        ("harmful", harmful_traces, harmful_dists_by_layer),
        ("benign", benign_traces, benign_dists_by_layer),
    ]:
        skipped = 0
        for i, trace in enumerate(traces):
            rendered = render_trace(trace, tokenizer, args.max_length)
            if rendered is None:
                skipped += 1
                continue

            distances = extract_distances_for_sample(
                model,
                rendered["input_ids"],
                rendered["attention_mask"],
                target_layers,
                device,
            )

            if not distances:
                skipped += 1
                continue

            # Store distances
            record = {
                "sample_idx": len(records),
                "category": category,
                "trace_id": trace.get("id", f"{category}_{i}"),
            }
            for layer_idx, dist_val in distances.items():
                dist_dict[layer_idx].append(dist_val)
                record[f"dl2rc_layer{layer_idx}"] = f"{dist_val:.6f}"
            records.append(record)

            if (i + 1) % 25 == 0:
                logger.info("  %s: %d/%d done", category, i + 1, len(traces))

        logger.info(
            "  %s: %d extracted, %d skipped",
            category, len(traces) - skipped, skipped,
        )

    # ── Analyze each layer ──
    print()
    print("=" * 80)
    print("Inference Distance Analysis: dl2rc(adapter_ON, adapter_OFF)")
    print("=" * 80)

    for layer_idx in target_layers:
        harmful_arr = np.array(harmful_dists_by_layer[layer_idx])
        benign_arr = np.array(benign_dists_by_layer[layer_idx])

        if len(harmful_arr) < 2 or len(benign_arr) < 2:
            logger.warning("Layer %d: too few samples, skipping", layer_idx)
            continue

        stats = compute_overlap_stats(harmful_arr, benign_arr)

        # Print summary
        print(f"\n  Layer {layer_idx}:")
        print(f"    Harmful distances: mean={stats['harmful_mean']:.4f}, "
              f"std={stats['harmful_std']:.4f} (n={len(harmful_arr)})")
        print(f"    Benign distances:  mean={stats['benign_mean']:.4f}, "
              f"std={stats['benign_std']:.4f} (n={len(benign_arr)})")
        print(f"    KL(harmful||benign) = {stats['kl_harmful_benign']:.4f}")
        print(f"    AUC = {stats['auc']:.3f}")
        print(f"    Optimal threshold = {stats['optimal_threshold']:.4f}")
        print(f"      -> Detection rate (TPR) = {stats['detection_rate_at_optimal']:.1%}")
        print(f"      -> False positive rate   = {stats['fpr_at_optimal']:.1%}")
        print(f"      -> Balanced accuracy     = {stats['balanced_accuracy']:.1%}")

        # Interpretation
        if stats["auc"] >= 0.85:
            verdict = "STRONG SEPARATION -- adapter changes harmful reps substantially more"
        elif stats["auc"] >= 0.70:
            verdict = "MODERATE SEPARATION -- adapter shows some preference for harmful reps"
        elif stats["auc"] >= 0.55:
            verdict = "WEAK SEPARATION -- adapter barely distinguishes harmful from benign"
        else:
            verdict = "NO SEPARATION -- adapter affects harmful and benign equally"
        print(f"    Verdict: {verdict}")

        # Plot
        layer_label = f"layer{layer_idx}"
        plot_distributions(harmful_arr, benign_arr, stats, layer_label, output_dir)

    # ── Combined distance (mean across layers) ──
    if len(target_layers) > 1:
        # Compute mean distance across all target layers per sample
        harmful_combined = []
        benign_combined = []

        min_harmful = min(len(harmful_dists_by_layer[l]) for l in target_layers)
        min_benign = min(len(benign_dists_by_layer[l]) for l in target_layers)

        for i in range(min_harmful):
            mean_d = np.mean([harmful_dists_by_layer[l][i] for l in target_layers])
            harmful_combined.append(mean_d)
        for i in range(min_benign):
            mean_d = np.mean([benign_dists_by_layer[l][i] for l in target_layers])
            benign_combined.append(mean_d)

        harmful_combined = np.array(harmful_combined)
        benign_combined = np.array(benign_combined)

        if len(harmful_combined) >= 2 and len(benign_combined) >= 2:
            combined_stats = compute_overlap_stats(harmful_combined, benign_combined)
            layer_str = "+".join(str(l) for l in target_layers)

            print(f"\n  Combined (mean of layers {layer_str}):")
            print(f"    Harmful: mean={combined_stats['harmful_mean']:.4f}, "
                  f"std={combined_stats['harmful_std']:.4f}")
            print(f"    Benign:  mean={combined_stats['benign_mean']:.4f}, "
                  f"std={combined_stats['benign_std']:.4f}")
            print(f"    AUC = {combined_stats['auc']:.3f}")
            print(f"    Optimal threshold = {combined_stats['optimal_threshold']:.4f}")
            print(f"      -> Detection = {combined_stats['detection_rate_at_optimal']:.1%}, "
                  f"FPR = {combined_stats['fpr_at_optimal']:.1%}")

            plot_distributions(
                harmful_combined, benign_combined, combined_stats,
                f"combined_layers_{layer_str}", output_dir,
            )

    # ── Save CSV ──
    save_csv(records, target_layers, output_dir)

    # ── Save JSON summary ──
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "traces": str(args.traces),
        "target_layers": target_layers,
        "n_harmful": len(harmful_traces),
        "n_benign": len(benign_traces),
        "per_layer": {},
    }
    for layer_idx in target_layers:
        harmful_arr = np.array(harmful_dists_by_layer[layer_idx])
        benign_arr = np.array(benign_dists_by_layer[layer_idx])
        if len(harmful_arr) >= 2 and len(benign_arr) >= 2:
            s = compute_overlap_stats(harmful_arr, benign_arr)
            # Remove ROC arrays from JSON (too large)
            summary["per_layer"][str(layer_idx)] = {
                k: v for k, v in s.items() if k not in ("roc_fprs", "roc_tprs")
            }

    json_path = output_dir / "distance_analysis_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary: %s", json_path)

    print()
    print("=" * 80)
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
