#!/usr/bin/env python3
"""
Representation probe diagnostic for circuit breaker training.

Tests whether the base model's hidden states can distinguish between
different trace categories. Supports multiple pooling strategies,
probe types, and comparison modes.

Usage:
  # Harmful vs benign (original test)
  python scripts/probe_separability.py \
      --traces data/traces/agentdojo_truncated.jsonl \
      --model /path/to/model --layers 5 10 15 20 25 30

  # Harmful vs refusal (same injection context, different response)
  python scripts/probe_separability.py \
      --class0 data/traces/agentdojo_truncated.jsonl --class0-label harmful \
      --class1 data/traces/agentdojo_refusal_synthetic.jsonl --class1-label benign \
      --model /path/to/model --layers 5 10 15 20 25 30

  # All layers, last-token + MLP
  python scripts/probe_separability.py \
      --traces data/traces/agentdojo_truncated.jsonl \
      --model /path/to/model --layers $(seq 0 31) \
      --pooling last_token --probe mlp
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


# ── Data loading ──────────────────────────────────────────────────────────


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


# ── Rendering ─────────────────────────────────────────────────────────────


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
            # tool messages get mapped to user for chat template compatibility
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


# ── Hidden state extraction ───────────────────────────────────────────────


def extract_hidden_states(
    model,
    input_ids,
    attention_mask,
    target_layers: List[int],
    device,
    pooling: str = "mean",
) -> Dict[int, np.ndarray]:
    """Forward pass, extract hidden states with specified pooling."""
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
    mask = attention_mask.float().unsqueeze(-1)  # [1, T, 1]

    result = {}
    for layer_idx in target_layers:
        hs_idx = layer_idx + 1
        if hs_idx >= len(outputs.hidden_states):
            continue
        hs = outputs.hidden_states[hs_idx]  # [1, T, H]

        if pooling == "last_token":
            # Last non-padding token — most relevant for next-token prediction
            last_pos = int(seq_len) - 1
            vec = hs[0, last_pos, :]
        elif pooling == "last_quarter":
            # Mean of last 25% of tokens — captures response region
            start = max(0, int(seq_len * 0.75))
            end = int(seq_len)
            vec = hs[0, start:end, :].mean(dim=0)
        else:  # "mean"
            vec = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
            vec = vec[0]

        result[layer_idx] = vec.cpu().float().numpy()

    return result


# ── Probes ────────────────────────────────────────────────────────────────


def run_probe(
    features: np.ndarray,
    labels: np.ndarray,
    probe_type: str = "linear",
) -> Dict[str, float]:
    """Train a probe and report metrics via 5-fold CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aucs = [], []
    tp_all, fp_all, tn_all, fn_all = 0, 0, 0, 0

    for train_idx, test_idx in skf.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if probe_type == "mlp":
            clf = MLPClassifier(
                hidden_layer_sizes=(256, 64),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
            )
        else:
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        try:
            aucs.append(roc_auc_score(y_test, y_prob))
        except ValueError:
            pass

        for yt, yp in zip(y_test, y_pred):
            if yt == 1 and yp == 1:
                tp_all += 1
            elif yt == 0 and yp == 1:
                fp_all += 1
            elif yt == 0 and yp == 0:
                tn_all += 1
            else:
                fn_all += 1

    acc = np.mean(accs)
    auc = np.mean(aucs) if aucs else 0.0
    harmful_recall = tp_all / max(tp_all + fn_all, 1)
    benign_recall = tn_all / max(tn_all + fp_all, 1)

    return {
        "accuracy": acc,
        "auc": auc,
        "harmful_recall": harmful_recall,
        "benign_recall": benign_recall,
    }


# ── Printing ──────────────────────────────────────────────────────────────


def print_table(
    layers: List[int],
    all_features: Dict[int, List[np.ndarray]],
    labels_arr: np.ndarray,
    probe_type: str,
    title: str,
):
    """Run probes and print a results table."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    n1 = int((labels_arr == 1).sum())
    n0 = int((labels_arr == 0).sum())
    print(f"  Samples: {n1} class-1, {n0} class-0")
    print(f"  Chance baseline: {max(labels_arr.mean(), 1 - labels_arr.mean()):.1%}")
    print(f"  Probe: {probe_type}")
    print()
    hdr = f"  {'Layer':>7s}  {'Accuracy':>9s}  {'AUC':>6s}  {'C1 Recall':>10s}  {'C0 Recall':>10s}  {'Verdict':>10s}"
    print(hdr)
    print(f"  {'-----':>7s}  {'---------':>9s}  {'------':>6s}  {'----------':>10s}  {'----------':>10s}  {'----------':>10s}")

    for layer_idx in layers:
        if layer_idx not in all_features or not all_features[layer_idx]:
            continue
        features = np.stack(all_features[layer_idx])
        result = run_probe(features, labels_arr, probe_type)

        if result["auc"] >= 0.85:
            verdict = "SEPARABLE"
        elif result["auc"] >= 0.70:
            verdict = "WEAK"
        elif result["auc"] >= 0.55:
            verdict = "MARGINAL"
        else:
            verdict = "NO SIGNAL"

        print(
            f"  {layer_idx:>7d}  {result['accuracy']:>9.1%}  {result['auc']:>6.3f}  "
            f"{result['harmful_recall']:>10.1%}  {result['benign_recall']:>10.1%}  "
            f"{verdict:>10s}"
        )

    # Concatenated
    if len(layers) > 1:
        avail = [l for l in layers if l in all_features and all_features[l]]
        if avail:
            all_concat = np.concatenate([np.stack(all_features[l]) for l in avail], axis=1)
            result = run_probe(all_concat, labels_arr, probe_type)
            verdict = (
                "SEPARABLE" if result["auc"] >= 0.85
                else "WEAK" if result["auc"] >= 0.70
                else "MARGINAL" if result["auc"] >= 0.55
                else "NO SIGNAL"
            )
            tag = "all"
            print(
                f"  {tag:>7s}  {result['accuracy']:>9.1%}  {result['auc']:>6.3f}  "
                f"{result['harmful_recall']:>10.1%}  {result['benign_recall']:>10.1%}  "
                f"{verdict:>10s}"
            )


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Representation probe diagnostic")

    # Data sources — two modes
    data_grp = parser.add_argument_group("Data (mode 1: single file with harmful/benign labels)")
    data_grp.add_argument("--traces", type=Path, help="Traces JSONL (split by labels.category)")

    pair_grp = parser.add_argument_group("Data (mode 2: explicit class-0 and class-1 files)")
    pair_grp.add_argument("--class0", type=Path, help="Class-0 traces JSONL")
    pair_grp.add_argument("--class0-label", type=str, default=None,
                          help="Filter class-0 file by labels.category (e.g. 'benign')")
    pair_grp.add_argument("--class1", type=Path, help="Class-1 traces JSONL")
    pair_grp.add_argument("--class1-label", type=str, default=None,
                          help="Filter class-1 file by labels.category (e.g. 'harmful')")

    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--layers", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--max-samples", type=int, default=200, help="Max per class")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--pooling", type=str, nargs="+",
                        default=["mean", "last_token", "last_quarter"],
                        choices=["mean", "last_token", "last_quarter"],
                        help="Pooling strategies to test")
    parser.add_argument("--probe", type=str, nargs="+",
                        default=["linear", "mlp"],
                        choices=["linear", "mlp"],
                        help="Probe types to test")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── Load data ──
    if args.class0 and args.class1:
        class0_traces = load_traces_flat(args.class0, args.class0_label, args.max_samples)
        class1_traces = load_traces_flat(args.class1, args.class1_label, args.max_samples)
        logger.info("Class-0: %d traces from %s", len(class0_traces), args.class0.name)
        logger.info("Class-1: %d traces from %s", len(class1_traces), args.class1.name)
    elif args.traces:
        class1_traces, class0_traces = load_traces_split(args.traces, args.max_samples)
        logger.info("Harmful (class-1): %d, Benign (class-0): %d", len(class1_traces), len(class0_traces))
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

    # ── Extract hidden states for each pooling strategy ──
    # We do ONE forward pass per sample and extract all poolings from the same outputs.
    # But since pooling happens inside extract_hidden_states and we want to avoid
    # re-running forward passes, we extract per-pooling in sequence (forward pass is cached
    # by the model when inputs are the same — but in practice we just re-run, it's fast enough).

    for pooling in args.pooling:
        logger.info("Extracting with pooling=%s...", pooling)

        all_features: Dict[int, List[np.ndarray]] = {l: [] for l in args.layers}
        all_labels: List[int] = []
        skipped = 0

        for label, traces in [(1, class1_traces), (0, class0_traces)]:
            for i, trace in enumerate(traces):
                rendered = render_trace(trace, tokenizer, args.max_length)
                if rendered is None:
                    skipped += 1
                    continue

                hs = extract_hidden_states(
                    model, rendered["input_ids"], rendered["attention_mask"],
                    args.layers, device, pooling=pooling,
                )

                if not hs:
                    skipped += 1
                    continue

                for layer_idx, vec in hs.items():
                    all_features[layer_idx].append(vec)
                all_labels.append(label)

                if (i + 1) % 50 == 0:
                    tag = "class-1" if label == 1 else "class-0"
                    logger.info("  %s: %d/%d done", tag, i + 1, len(traces))

        labels_arr = np.array(all_labels)
        logger.info(
            "Extracted %d samples (%d class-1, %d class-0, %d skipped) pooling=%s",
            len(all_labels), sum(all_labels), len(all_labels) - sum(all_labels),
            skipped, pooling,
        )

        for probe_type in args.probe:
            title = f"pooling={pooling} | probe={probe_type}"
            print_table(args.layers, all_features, labels_arr, probe_type, title)

    print()
    print("Interpretation:")
    print("  SEPARABLE (AUC>=0.85): CB rerouting CAN work at this layer")
    print("  WEAK (AUC 0.70-0.85): Some signal, but CB may struggle")
    print("  MARGINAL (AUC 0.55-0.70): Barely above chance")
    print("  NO SIGNAL (AUC<0.55): CB cannot work at this layer")
    print()


if __name__ == "__main__":
    main()
