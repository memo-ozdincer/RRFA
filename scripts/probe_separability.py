#!/usr/bin/env python3
"""
Linear probe diagnostic: Can the base model distinguish AD harmful vs benign
at the representation level?

Extracts hidden states from specified layers, trains logistic regression,
reports accuracy/AUC. If a linear classifier can't separate them, no amount
of CB training will work at those layers.

Usage (on cluster with GPU):
  python scripts/probe_separability.py \
      --traces data/traces/agentdojo_truncated.jsonl \
      --model /path/to/model \
      --layers 5 10 15 20 25 \
      --max-samples 200 \
      --max-length 2048

Or via sbatch:
  sbatch slurm/pipeline/probe_separability.sbatch
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


def load_traces(
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

    logger.info("Loaded %d harmful, %d benign traces", len(harmful), len(benign))
    return harmful, benign


def render_trace(trace: Dict[str, Any], tokenizer, max_length: int) -> Optional[Dict]:
    """Render a trace to input_ids using chat template."""
    messages = trace.get("messages", [])
    if not messages:
        return None

    # Convert to chat format
    chat_messages = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("system", "user", "assistant") and content:
            chat_messages.append({"role": role, "content": str(content)})

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


def extract_hidden_states(
    model,
    input_ids,
    attention_mask,
    target_layers: List[int],
    device: str,
) -> Dict[int, np.ndarray]:
    """Forward pass, extract mean-pooled hidden states at target layers."""
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

    # Mean pool over non-padding tokens
    mask = attention_mask.float().unsqueeze(-1)  # [1, T, 1]
    result = {}
    for layer_idx in target_layers:
        hs_idx = layer_idx + 1  # hidden_states[0] is embedding
        if hs_idx >= len(outputs.hidden_states):
            continue
        hs = outputs.hidden_states[hs_idx]  # [1, T, H]
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)  # [1, H]
        result[layer_idx] = pooled[0].cpu().float().numpy()

    return result


def run_probe(
    features: np.ndarray,
    labels: np.ndarray,
    layer_name: str,
) -> Dict[str, float]:
    """Train logistic regression and report metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    # 5-fold CV for more reliable estimate on small data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aucs = [], []
    tp_all, fp_all, tn_all, fn_all = 0, 0, 0, 0

    for train_idx, test_idx in skf.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        try:
            aucs.append(roc_auc_score(y_test, y_prob))
        except ValueError:
            pass

        # Confusion matrix elements
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


def main():
    parser = argparse.ArgumentParser(description="Linear probe on hidden states")
    parser.add_argument("--traces", type=Path, required=True, help="Traces JSONL")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30],
        help="Layers to probe",
    )
    parser.add_argument("--max-samples", type=int, default=200, help="Max per class")
    parser.add_argument("--max-length", type=int, default=2048, help="Max seq length")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load data
    harmful, benign = load_traces(args.traces, args.max_samples)
    if len(harmful) < 10 or len(benign) < 10:
        logger.error("Too few samples: %d harmful, %d benign", len(harmful), len(benign))
        sys.exit(1)

    # Load model
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

    # Render and extract
    logger.info("Extracting hidden states at layers %s...", args.layers)
    all_features: Dict[int, List[np.ndarray]] = {l: [] for l in args.layers}
    all_labels: List[int] = []
    skipped = 0

    for label, traces in [(1, harmful), (0, benign)]:
        for i, trace in enumerate(traces):
            rendered = render_trace(trace, tokenizer, args.max_length)
            if rendered is None:
                skipped += 1
                continue

            hs = extract_hidden_states(
                model, rendered["input_ids"], rendered["attention_mask"],
                args.layers, device,
            )

            if not hs:
                skipped += 1
                continue

            for layer_idx, vec in hs.items():
                all_features[layer_idx].append(vec)
            all_labels.append(label)

            if (i + 1) % 50 == 0:
                tag = "harmful" if label == 1 else "benign"
                logger.info("  %s: %d/%d done", tag, i + 1, len(traces))

    logger.info(
        "Extracted %d samples (%d harmful, %d benign, %d skipped)",
        len(all_labels),
        sum(all_labels),
        len(all_labels) - sum(all_labels),
        skipped,
    )

    labels_arr = np.array(all_labels)

    # Run probes
    print()
    print("=" * 70)
    print("LINEAR PROBE RESULTS — Can base model distinguish harmful vs benign?")
    print("=" * 70)
    print(f"  Samples: {sum(labels_arr == 1)} harmful, {sum(labels_arr == 0)} benign")
    print(f"  Chance baseline: {max(labels_arr.mean(), 1 - labels_arr.mean()):.1%}")
    print()
    print(f"  {'Layer':>7s}  {'Accuracy':>9s}  {'AUC':>6s}  {'Harm Recall':>12s}  {'Ben Recall':>11s}  {'Verdict':>10s}")
    print(f"  {'-----':>7s}  {'---------':>9s}  {'------':>6s}  {'------------':>12s}  {'-----------':>11s}  {'----------':>10s}")

    for layer_idx in args.layers:
        features = np.stack(all_features[layer_idx])
        result = run_probe(features, labels_arr, f"layer_{layer_idx}")

        # Interpret
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
            f"{result['harmful_recall']:>12.1%}  {result['benign_recall']:>11.1%}  "
            f"{verdict:>10s}"
        )

    # Also try concatenating all layers
    print()
    all_concat = np.concatenate(
        [np.stack(all_features[l]) for l in args.layers], axis=1
    )
    result = run_probe(all_concat, labels_arr, "all_layers")
    verdict = (
        "SEPARABLE" if result["auc"] >= 0.85
        else "WEAK" if result["auc"] >= 0.70
        else "MARGINAL" if result["auc"] >= 0.55
        else "NO SIGNAL"
    )
    layers_str = "+".join(str(l) for l in args.layers)
    print(
        f"  {layers_str:>7s}  {result['accuracy']:>9.1%}  {result['auc']:>6.3f}  "
        f"{result['harmful_recall']:>12.1%}  {result['benign_recall']:>11.1%}  "
        f"{verdict:>10s}"
    )

    print()
    print("Interpretation:")
    print("  SEPARABLE (AUC≥0.85): CB rerouting CAN work at this layer")
    print("  WEAK (AUC 0.70-0.85): Some signal, but CB may struggle")
    print("  MARGINAL (AUC 0.55-0.70): Barely above chance")
    print("  NO SIGNAL (AUC<0.55): CB cannot work at this layer — change layers or approach")
    print()


if __name__ == "__main__":
    main()
