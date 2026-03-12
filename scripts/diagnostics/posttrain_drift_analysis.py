#!/usr/bin/env python3
"""
Post-training drift analysis for circuit breaker models.

Measures how much benign trace representations have drifted from the frozen
model after CB training. Answers: does the push loss corrupt context
representations, and if so, where?

Key measurements:
1. Per-token drift on benign traces: ||rep_trained - rep_frozen||_2
   Split by token region: system prompt, user message, tool output, assistant response
2. Per-token cosine shift: how much has the direction changed?
3. Context vs response drift ratio: is context drifting despite not being pushed?

Usage:
    python scripts/diagnostics/posttrain_drift_analysis.py \
        --model /path/to/base/model \
        --adapter /path/to/lora/adapter \
        --benign-traces data/traces/agentdojo_truncated.jsonl \
        --layers 5 10 15 20 25 30 \
        --max-samples 50 \
        --output results/posttrain_drift.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_traces(path: Path, category: str = "benign", max_samples: int = 100) -> List[Dict]:
    traces = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            cat = d.get("labels", {}).get("category", "")
            if category == "all" or cat == category:
                traces.append(d)
                if len(traces) >= max_samples:
                    break
    return traces


def render_trace(trace, tokenizer, max_length=4096, tools=None):
    messages = trace.get("messages", [])
    if not messages:
        return None
    chat_messages = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("system", "user", "assistant", "tool"):
            chat_messages.append({
                "role": role if role != "tool" else "user",
                "content": str(content) if content else "",
            })
    if not chat_messages:
        return None
    try:
        text = tokenizer.apply_chat_template(chat_messages, tools=tools, tokenize=False,
                                              add_generation_prompt=False)
        return tokenizer(text, return_tensors="pt", max_length=max_length,
                         truncation=True, padding=False)
    except Exception:
        return None


def classify_token_regions(_trace, tokenizer, encoded) -> Dict[str, Tuple[int, int]]:
    """Classify token positions into regions: system, user, tool_output, assistant_response."""
    text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
    regions = {}

    # Find system region
    sys_start = text.find("<|start_header_id|>system<|end_header_id|>")
    user_start = text.find("<|start_header_id|>user<|end_header_id|>")
    if sys_start >= 0 and user_start > sys_start:
        sys_toks = len(tokenizer.encode(text[:sys_start], add_special_tokens=False))
        user_toks = len(tokenizer.encode(text[:user_start], add_special_tokens=False))
        regions["system"] = (sys_toks, user_toks)

    # Find last assistant region
    asst_header = "<|start_header_id|>assistant<|end_header_id|>"
    last_asst = text.rfind(asst_header)
    if last_asst >= 0:
        asst_toks = len(tokenizer.encode(text[:last_asst + len(asst_header)], add_special_tokens=False))
        total_toks = encoded["input_ids"].size(1)
        regions["assistant_response"] = (asst_toks, total_toks)

    # Everything between system end and assistant start is "context" (user + tool outputs)
    if "system" in regions and "assistant_response" in regions:
        regions["context"] = (regions["system"][1], regions["assistant_response"][0])

    return regions


@torch.no_grad()
def extract_reps(model, encoded, target_layers, device):
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, use_cache=False, return_dict=True)
    result = {}
    for layer in target_layers:
        hs_idx = layer + 1
        if 0 <= hs_idx < len(outputs.hidden_states):
            result[layer] = outputs.hidden_states[hs_idx].cpu()
    del outputs
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="Post-training drift analysis")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True, help="LoRA adapter path")
    parser.add_argument("--benign-traces", type=Path, required=True)
    parser.add_argument("--harmful-traces", type=Path, default=None,
                        help="Optional: also measure drift on harmful traces")
    parser.add_argument("--tool-schema", type=Path, default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge-adapter", action="store_true", default=False,
                        help="Merge adapter before analysis (faster but can't toggle)")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map="auto",
    )

    logger.info("Loading adapter: %s", args.adapter)
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    tools = None
    if args.tool_schema and args.tool_schema.exists():
        with open(args.tool_schema) as f:
            tools = json.load(f).get("tools", [])

    benign_traces = load_traces(args.benign_traces, "benign", args.max_samples)
    logger.info("Loaded %d benign traces", len(benign_traces))

    harmful_traces = []
    if args.harmful_traces and args.harmful_traces.exists():
        harmful_traces = load_traces(args.harmful_traces, "harmful", args.max_samples)
        logger.info("Loaded %d harmful traces", len(harmful_traces))

    # Collect drift stats per layer per region
    drift_stats = {
        layer: {
            "system": {"l2": [], "cos_shift": []},
            "context": {"l2": [], "cos_shift": []},
            "assistant_response": {"l2": [], "cos_shift": []},
            "all_tokens": {"l2": [], "cos_shift": []},
        }
        for layer in args.layers
    }
    harmful_drift_stats = {layer: {"all_tokens": {"l2": [], "cos_shift": []}} for layer in args.layers}

    for trace_idx, trace in enumerate(benign_traces):
        encoded = render_trace(trace, tokenizer, args.max_length, tools)
        if encoded is None:
            continue

        regions = classify_token_regions(trace, tokenizer, encoded)

        # Get trained model reps (adapter enabled)
        model.enable_adapter_layers()
        trained_reps = extract_reps(model, encoded, args.layers, device)

        # Get frozen model reps (adapter disabled)
        model.disable_adapter_layers()
        frozen_reps = extract_reps(model, encoded, args.layers, device)

        model.enable_adapter_layers()

        for layer in args.layers:
            if layer not in trained_reps or layer not in frozen_reps:
                continue

            t_rep = trained_reps[layer][0]  # [T, H]
            f_rep = frozen_reps[layer][0]   # [T, H]
            T = min(t_rep.size(0), f_rep.size(0))

            # Per-token L2 drift and cosine shift
            l2_drift = torch.norm(t_rep[:T] - f_rep[:T], p=2, dim=-1)  # [T]
            cos_shift = 1.0 - F.cosine_similarity(t_rep[:T], f_rep[:T], dim=-1)  # [T]

            # Record per-region
            drift_stats[layer]["all_tokens"]["l2"].append(l2_drift.mean().item())
            drift_stats[layer]["all_tokens"]["cos_shift"].append(cos_shift.mean().item())

            for region_name, (start, end) in regions.items():
                end = min(end, T)
                start = min(start, end)
                if end > start:
                    drift_stats[layer][region_name]["l2"].append(l2_drift[start:end].mean().item())
                    drift_stats[layer][region_name]["cos_shift"].append(cos_shift[start:end].mean().item())

        if (trace_idx + 1) % 10 == 0:
            logger.info("  benign: %d/%d processed", trace_idx + 1, len(benign_traces))

    # Harmful traces drift (to compare push effect)
    for trace_idx, trace in enumerate(harmful_traces):
        encoded = render_trace(trace, tokenizer, args.max_length, tools)
        if encoded is None:
            continue

        model.enable_adapter_layers()
        trained_reps = extract_reps(model, encoded, args.layers, device)
        model.disable_adapter_layers()
        frozen_reps = extract_reps(model, encoded, args.layers, device)
        model.enable_adapter_layers()

        for layer in args.layers:
            if layer not in trained_reps or layer not in frozen_reps:
                continue
            t_rep = trained_reps[layer][0]
            f_rep = frozen_reps[layer][0]
            T = min(t_rep.size(0), f_rep.size(0))
            l2_drift = torch.norm(t_rep[:T] - f_rep[:T], p=2, dim=-1)
            cos_shift = 1.0 - F.cosine_similarity(t_rep[:T], f_rep[:T], dim=-1)
            harmful_drift_stats[layer]["all_tokens"]["l2"].append(l2_drift.mean().item())
            harmful_drift_stats[layer]["all_tokens"]["cos_shift"].append(cos_shift.mean().item())

        if (trace_idx + 1) % 10 == 0:
            logger.info("  harmful: %d/%d processed", trace_idx + 1, len(harmful_traces))

    # Summarize
    results = {"config": {"model": args.model, "adapter": args.adapter, "layers": args.layers}}

    benign_summary = {}
    for layer in args.layers:
        layer_data = {}
        for region, stats in drift_stats[layer].items():
            l2_vals = stats["l2"]
            cos_vals = stats["cos_shift"]
            if l2_vals:
                layer_data[region] = {
                    "mean_l2_drift": float(np.mean(l2_vals)),
                    "std_l2_drift": float(np.std(l2_vals)),
                    "mean_cos_shift": float(np.mean(cos_vals)),
                    "std_cos_shift": float(np.std(cos_vals)),
                    "n": len(l2_vals),
                }
        benign_summary[str(layer)] = layer_data
    results["benign_drift"] = benign_summary

    if harmful_traces:
        harmful_summary = {}
        for layer in args.layers:
            l2_vals = harmful_drift_stats[layer]["all_tokens"]["l2"]
            cos_vals = harmful_drift_stats[layer]["all_tokens"]["cos_shift"]
            if l2_vals:
                harmful_summary[str(layer)] = {
                    "mean_l2_drift": float(np.mean(l2_vals)),
                    "mean_cos_shift": float(np.mean(cos_vals)),
                    "n": len(l2_vals),
                }
        results["harmful_drift"] = harmful_summary

    # Three-tier retain mask recommendation
    recommendations = {}
    for layer in args.layers:
        ld = benign_summary.get(str(layer), {})
        sys_drift = ld.get("system", {}).get("mean_l2_drift", 0)
        ctx_drift = ld.get("context", {}).get("mean_l2_drift", 0)
        resp_drift = ld.get("assistant_response", {}).get("mean_l2_drift", 0)

        if resp_drift > 0:
            ratio = ctx_drift / resp_drift if resp_drift > 0.001 else 0
            recommendations[str(layer)] = {
                "context_to_response_drift_ratio": float(ratio),
                "recommendation": (
                    "three_tier_mask" if ratio > 0.1
                    else "response_only_mask"
                ),
                "suggested_weights": {
                    "system": 0.1 if ratio > 0.1 else 0.0,
                    "context": 0.2 if ratio > 0.1 else 0.0,
                    "response": 1.0,
                },
                "reasoning": (
                    f"Context drift is {ratio:.0%} of response drift. "
                    + ("Significant context drift detected — use three-tier mask."
                       if ratio > 0.1
                       else "Minimal context drift — response-only mask is safe.")
                ),
            }
    results["retain_mask_recommendations"] = recommendations

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)

    # Print summary
    print("\n" + "=" * 70)
    print("DRIFT ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\n{'Layer':<8} {'System L2':<14} {'Context L2':<14} {'Response L2':<14} {'Ctx/Resp Ratio':<16} {'Recommendation'}")
    print("-" * 80)
    for layer in args.layers:
        ld = benign_summary.get(str(layer), {})
        sys_d = ld.get("system", {}).get("mean_l2_drift", 0)
        ctx_d = ld.get("context", {}).get("mean_l2_drift", 0)
        resp_d = ld.get("assistant_response", {}).get("mean_l2_drift", 0)
        ratio = ctx_d / resp_d if resp_d > 0.001 else 0
        rec = recommendations.get(str(layer), {}).get("recommendation", "n/a")
        print(f"{layer:<8} {sys_d:<14.4f} {ctx_d:<14.4f} {resp_d:<14.4f} {ratio:<16.2%} {rec}")

    if harmful_traces:
        print(f"\n{'Layer':<8} {'Benign All L2':<16} {'Harmful All L2':<16} {'Push/Retain Ratio'}")
        print("-" * 60)
        for layer in args.layers:
            b_all = benign_summary.get(str(layer), {}).get("all_tokens", {}).get("mean_l2_drift", 0)
            h_all = results.get("harmful_drift", {}).get(str(layer), {}).get("mean_l2_drift", 0)
            ratio = h_all / b_all if b_all > 0.001 else 0
            print(f"{layer:<8} {b_all:<16.4f} {h_all:<16.4f} {ratio:<16.1f}x")


if __name__ == "__main__":
    main()
