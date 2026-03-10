#!/usr/bin/env python3
"""
Extract a linear probe's weight vector as an importance mask for
SRMU-style feature-selective circuit breaker training.

Trains a logistic regression probe on hidden states at a target layer,
then saves the weight vector (direction) and derived importance mask
as a .pt file that can be loaded during CB training.

Usage:
  # From single file with harmful/benign labels
  python scripts/extract_probe_direction.py \
      --traces data/traces/agentdojo_truncated.jsonl \
      --model /path/to/model --layer 10 --output probe_layer10.pt

  # From explicit class-0 / class-1 files
  python scripts/extract_probe_direction.py \
      --class0 data/traces/agentdojo_refusal_synthetic.jsonl --class0-label benign \
      --class1 data/traces/agentdojo_truncated.jsonl --class1-label harmful \
      --model /path/to/model --layer 10 --output probe_layer10.pt

Output (.pt):
  {
      "direction": tensor(4096,),      # unit-normalized probe weight vector
      "importance": tensor(4096,),     # |w_i| / max(|w_i|) in [0, 1]
      "intercept": float,
      "metadata": {
          "auc": float,
          "accuracy": float,
          "layer": int,
          "pooling": str,
          "n_harmful": int,
          "n_benign": int,
          "probe_type": "logistic_regression",
          "scaler_mean": tensor,
          "scaler_scale": tensor,
      }
  }
"""

import argparse
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


# ── Data loading (copied from probe_separability.py) ─────────────────


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


def load_traces_flat(
    path: Path, category_filter: Optional[str], max_samples: int = 0
) -> List[Dict[str, Any]]:
    """Load traces, optionally filtering by category."""
    traces = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            trace = json.loads(line)
            if category_filter:
                cat = trace.get("labels", {}).get("category", "")
                if cat != category_filter:
                    continue
            traces.append(trace)
            if max_samples > 0 and len(traces) >= max_samples:
                break
    return traces


# ── Rendering ────────────────────────────────────────────────────────


def render_trace(trace: Dict[str, Any], tokenizer, max_length: int) -> Optional[Dict]:
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


# ── Hidden state extraction ──────────────────────────────────────────


def extract_hidden_states(
    model,
    input_ids,
    attention_mask,
    target_layer: int,
    device,
    pooling: str = "last_token",
) -> Optional[np.ndarray]:
    """Forward pass, extract hidden state at a single layer with specified pooling."""
    import torch

    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    seq_len = attention_mask.sum().item()
    hs_idx = target_layer + 1
    if hs_idx >= len(outputs.hidden_states):
        return None

    hs = outputs.hidden_states[hs_idx]  # [1, T, H]

    if pooling == "last_token":
        last_pos = int(seq_len) - 1
        vec = hs[0, last_pos, :]
    elif pooling == "last_quarter":
        start = max(0, int(seq_len * 0.75))
        end = int(seq_len)
        vec = hs[0, start:end, :].mean(dim=0)
    else:  # "mean"
        mask = attention_mask.float().unsqueeze(-1)
        vec = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        vec = vec[0]

    return vec.cpu().float().numpy()


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Extract probe direction vector for importance-weighted CB loss"
    )

    # Data sources — two modes
    data_grp = parser.add_argument_group("Data (mode 1: single file)")
    data_grp.add_argument("--traces", type=Path, help="Traces JSONL (split by labels.category)")

    pair_grp = parser.add_argument_group("Data (mode 2: explicit class-0 / class-1)")
    pair_grp.add_argument("--class0", type=Path, help="Class-0 (benign) traces JSONL")
    pair_grp.add_argument("--class0-label", type=str, default=None,
                          help="Filter class-0 by labels.category")
    pair_grp.add_argument("--class1", type=Path, help="Class-1 (harmful) traces JSONL")
    pair_grp.add_argument("--class1-label", type=str, default=None,
                          help="Filter class-1 by labels.category")

    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--layer", type=int, default=10, help="Target layer (default: 10)")
    parser.add_argument("--pooling", type=str, default="last_token",
                        choices=["mean", "last_token", "last_quarter"])
    parser.add_argument("--output", type=Path, required=True, help="Output .pt file")
    parser.add_argument("--max-samples", type=int, default=200, help="Max per class")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── Load data ──
    if args.class0 and args.class1:
        class0_traces = load_traces_flat(args.class0, args.class0_label, args.max_samples)
        class1_traces = load_traces_flat(args.class1, args.class1_label, args.max_samples)
        logger.info("Class-0 (benign): %d from %s", len(class0_traces), args.class0.name)
        logger.info("Class-1 (harmful): %d from %s", len(class1_traces), args.class1.name)
    elif args.traces:
        class1_traces, class0_traces = load_traces_split(args.traces, args.max_samples)
        logger.info("Harmful: %d, Benign: %d", len(class1_traces), len(class0_traces))
    else:
        parser.error("Provide either --traces or both --class0 and --class1")

    if len(class0_traces) < 10 or len(class1_traces) < 10:
        logger.error("Too few samples: %d class-0, %d class-1", len(class0_traces), len(class1_traces))
        sys.exit(1)

    # ── Load model ──
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    # ── Extract hidden states ──
    logger.info("Extracting layer=%d pooling=%s ...", args.layer, args.pooling)

    features = []
    labels = []
    skipped = 0

    for label, traces in [(1, class1_traces), (0, class0_traces)]:
        for i, trace in enumerate(traces):
            rendered = render_trace(trace, tokenizer, args.max_length)
            if rendered is None:
                skipped += 1
                continue

            vec = extract_hidden_states(
                model, rendered["input_ids"], rendered["attention_mask"],
                args.layer, device, pooling=args.pooling,
            )
            if vec is None:
                skipped += 1
                continue

            features.append(vec)
            labels.append(label)

            if (i + 1) % 50 == 0:
                tag = "harmful" if label == 1 else "benign"
                logger.info("  %s: %d/%d done", tag, i + 1, len(traces))

    X = np.stack(features)
    y = np.array(labels)
    n_harmful = int((y == 1).sum())
    n_benign = int((y == 0).sum())
    logger.info("Extracted %d samples (%d harmful, %d benign, %d skipped)",
                len(y), n_harmful, n_benign, skipped)

    # ── Train probe ──
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.model_selection import cross_val_predict

    logger.info("Training logistic regression probe...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    probe = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    probe.fit(X_scaled, y)

    # Evaluate with cross-val predictions
    y_prob = cross_val_predict(
        LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        X_scaled, y, cv=5, method="predict_proba",
    )[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    auc = roc_auc_score(y, y_prob)
    acc = accuracy_score(y, y_pred)
    logger.info("Probe AUC=%.3f, Accuracy=%.1f%%", auc, acc * 100)

    # ── Extract direction and importance ──
    w = probe.coef_[0]  # (H,) numpy array
    w_tensor = torch.from_numpy(w).float()

    # Normalize to unit vector
    direction = w_tensor / w_tensor.norm()

    # Importance: |w_i| / max(|w_i|) in [0, 1]
    abs_w = w_tensor.abs()
    importance = abs_w / (abs_w.max() + 1e-8)

    # Stats
    sparsity = (importance < 0.1).sum().item() / importance.numel()
    top100_mean = importance.topk(100).values.mean().item()
    logger.info("Importance mask: dim=%d, sparsity(<%0.1f)=%.1f%%, top-100 mean=%.3f",
                importance.shape[0], 0.1, sparsity * 100, top100_mean)

    # ── Save ──
    args.output.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "direction": direction,
        "importance": importance,
        "intercept": float(probe.intercept_[0]),
        "metadata": {
            "auc": float(auc),
            "accuracy": float(acc),
            "layer": args.layer,
            "pooling": args.pooling,
            "n_harmful": n_harmful,
            "n_benign": n_benign,
            "probe_type": "logistic_regression",
            "scaler_mean": torch.from_numpy(scaler.mean_).float(),
            "scaler_scale": torch.from_numpy(scaler.scale_).float(),
        },
    }

    torch.save(save_dict, args.output)
    logger.info("Saved to %s", args.output)

    # Summary
    print(f"\n{'='*60}")
    print(f"Probe Direction Extracted Successfully")
    print(f"{'='*60}")
    print(f"  Layer: {args.layer}")
    print(f"  Pooling: {args.pooling}")
    print(f"  AUC: {auc:.3f}")
    print(f"  Accuracy: {acc:.1%}")
    print(f"  Direction dim: {direction.shape[0]}")
    print(f"  Importance sparsity (<0.1): {sparsity:.1%}")
    print(f"  Top-100 importance mean: {top100_mean:.3f}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")
    print(f"\nTo use in training:")
    print(f"  python src/training/train_schema.py ... --importance-mask {args.output}")


if __name__ == "__main__":
    main()
