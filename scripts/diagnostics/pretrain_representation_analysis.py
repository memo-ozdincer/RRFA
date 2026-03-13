#!/usr/bin/env python3
"""
Pre-training representation diagnostics for circuit breaker design.

Runs on the FROZEN base model (no adapter) using paired harmful/benign traces.
Answers three critical questions before any CB training:

1. Where do harmful and benign representations naturally diverge?
   (per-layer, per-token cosine similarity between paired traces)

2. How do four trace types relate geometrically?
   (A=injection+tricked, B=injection+refusal, C=no_injection+correct, D=no_injection+refusal)
   Computes cosine similarity matrix between all pairs at response token positions.

3. Is there a natural alarm signal at injection positions?
   (separation between harmful/benign at injection tokens, propagation into response,
    attention weight analysis)

Also checks whether ReLU(cos_sim) is clipping to zero — if frozen harmful reps
are already near-orthogonal at target layers, the push loss has no gradient.

Usage:
    python scripts/diagnostics/pretrain_representation_analysis.py \
        --model /path/to/model \
        --harmful-traces data/traces/agentdojo_truncated.jsonl \
        --benign-traces data/traces/agentdojo_truncated.jsonl \
        --contrastive-pairs data/traces/contrastive_pairs.jsonl \
        --refusal-traces data/traces/agentdojo_refusal_synthetic.jsonl \
        --layers 5 10 15 20 25 30 \
        --max-samples 50 \
        --output results/pretrain_diagnostics.json \
        --tool-schema configs/tool_schemas/agentdojo_v1.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── data loading ────────────────────────────────────────────────────────────

def load_traces(path: Path, category: Optional[str] = None, max_samples: int = 200) -> List[Dict]:
    traces = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if category is None or d.get("labels", {}).get("category", "") == category:
                traces.append(d)
                if len(traces) >= max_samples:
                    break
    return traces


def load_contrastive_pairs(path: Path) -> Dict[str, str]:
    """Load harmful_trace_id → benign_trace_id mapping."""
    mapping = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            mapping[d["harmful_trace_id"]] = d["benign_trace_id"]
    return mapping


def render_trace_to_inputs(trace: Dict, tokenizer, max_length: int = 4096,
                           tools: Optional[List] = None) -> Optional[Dict[str, torch.Tensor]]:
    """Render a trace to model inputs via apply_chat_template."""
    messages = trace.get("messages", [])
    if not messages:
        return None

    chat_messages = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")
        if role in ("system", "user", "assistant", "tool"):
            entry = {"role": role if role != "tool" else "user", "content": str(content) if content else ""}
            if role == "assistant" and tool_calls:
                entry["tool_calls"] = tool_calls
            chat_messages.append(entry)

    if not chat_messages:
        return None

    try:
        text = tokenizer.apply_chat_template(
            chat_messages, tools=tools, tokenize=False, add_generation_prompt=False
        )
        encoded = tokenizer(text, return_tensors="pt", max_length=max_length,
                            truncation=True, padding=False)
        return encoded
    except Exception as e:
        logger.debug("render failed: %s", e)
        return None


def find_injection_span(trace: Dict) -> Optional[Tuple[int, int]]:
    """Return (char_start, char_end) of injection in the trace messages."""
    hints = trace.get("signal_hints", {})
    span = hints.get("injection_char_span")
    if span:
        return (span.get("start", 0), span.get("end", 0))
    # Fallback: search for <INFORMATION> markers in tool messages
    for msg in trace.get("messages", []):
        content = str(msg.get("content", ""))
        start = content.find("<INFORMATION>")
        if start >= 0:
            end = content.find("</INFORMATION>")
            if end >= 0:
                return (start, end + len("</INFORMATION>"))
    return None


def find_response_token_range(trace: Dict, tokenizer, encoded: Dict[str, torch.Tensor]) -> Optional[Tuple[int, int]]:
    """Find the token range of the last assistant message (the response/decision)."""
    messages = trace.get("messages", [])
    # Find last assistant message
    last_asst_idx = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            last_asst_idx = i
    if last_asst_idx is None:
        return None

    # Reconstruct text up to the last assistant message to find token offset
    # Simpler approach: decode full sequence and find <|python_tag|> or assistant header
    input_ids = encoded["input_ids"][0]
    text = tokenizer.decode(input_ids, skip_special_tokens=False)

    # Find the last assistant header
    asst_header = "<|start_header_id|>assistant<|end_header_id|>"
    idx = text.rfind(asst_header)
    if idx < 0:
        return None

    # Convert char position to token position
    prefix = text[:idx + len(asst_header)]
    start_tok = len(tokenizer.encode(prefix, add_special_tokens=False))
    end_tok = len(input_ids)
    return (min(start_tok, end_tok - 1), end_tok)


# ─── core analysis ───────────────────────────────────────────────────────────

@torch.no_grad()
def extract_hidden_states(model, encoded: Dict[str, torch.Tensor], target_layers: List[int],
                          device: str = "cuda") -> Dict[int, torch.Tensor]:
    """Forward pass, return hidden states at target layers. Shape: [1, T, H]."""
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hs = outputs.hidden_states  # tuple of (num_layers + 1) tensors
    result = {}
    for layer_idx in target_layers:
        hs_idx = layer_idx + 1  # hidden_states[0] is embedding
        if 0 <= hs_idx < len(hs):
            result[layer_idx] = hs[hs_idx].cpu()
    del outputs
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def extract_attention_weights(model, encoded: Dict[str, torch.Tensor], target_layers: List[int],
                              device: str = "cuda") -> Dict[int, torch.Tensor]:
    """Forward pass, return attention weights at target layers. Shape: [1, num_heads, T, T]."""
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )
    attn = outputs.attentions  # tuple of (num_layers) tensors
    result = {}
    for layer_idx in target_layers:
        if 0 <= layer_idx < len(attn):
            result[layer_idx] = attn[layer_idx].cpu()
    del outputs
    torch.cuda.empty_cache()
    return result


def cosine_sim_per_token(rep_a: torch.Tensor, rep_b: torch.Tensor) -> torch.Tensor:
    """Per-token cosine similarity. Input: [1, T, H]. Output: [T]."""
    a = rep_a[0]  # [T, H]
    b = rep_b[0]  # [T, H]
    min_t = min(a.size(0), b.size(0))
    a, b = a[:min_t], b[:min_t]
    return F.cosine_similarity(a, b, dim=-1)


def l2_per_token(rep_a: torch.Tensor, rep_b: torch.Tensor) -> torch.Tensor:
    """Per-token L2 distance. Input: [1, T, H]. Output: [T]."""
    a = rep_a[0]
    b = rep_b[0]
    min_t = min(a.size(0), b.size(0))
    return torch.norm(a[:min_t] - b[:min_t], p=2, dim=-1)


def pool_tokens(rep: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Mean-pool token range. Input: [1, T, H]. Output: [H]."""
    return rep[0, start:end].mean(dim=0)


# ─── diagnostic 1: base model cosine at each layer ──────────────────────────

def diagnostic_1_base_cosine(
    model, harmful_traces, benign_traces, pair_mapping, tokenizer,
    target_layers, device, max_length, tools
) -> Dict:
    """Cosine similarity between paired harmful/benign traces at each layer."""
    results = {layer: [] for layer in target_layers}
    per_token_results = {layer: [] for layer in target_layers}

    benign_by_id = {t["id"]: t for t in benign_traces}
    n_pairs = 0

    # Pre-check: how many pairs can we form?
    matchable = sum(1 for h in harmful_traces if pair_mapping.get(h["id"]) in benign_by_id)
    if matchable == 0:
        logger.error("ID MISMATCH: 0 harmful traces match contrastive pairs + benign set")
        logger.error("  Sample harmful IDs: %s", [h["id"] for h in harmful_traces[:3]])
        logger.error("  Sample pair keys:   %s", list(pair_mapping.keys())[:3])
        logger.error("  Sample benign IDs:  %s", list(benign_by_id.keys())[:3])
        # Check for partial matches
        in_pairs = sum(1 for h in harmful_traces if h["id"] in pair_mapping)
        logger.error("  Harmful in pairs (ignoring benign): %d/%d", in_pairs, len(harmful_traces))
        if in_pairs > 0:
            sample_b_ids = [pair_mapping[h["id"]] for h in harmful_traces if h["id"] in pair_mapping][:3]
            logger.error("  Target benign IDs from pairs: %s", sample_b_ids)
            logger.error("  These benign IDs in benign set? %s",
                         [bid in benign_by_id for bid in sample_b_ids])
    else:
        logger.info("  %d/%d harmful traces have valid contrastive pairs", matchable, len(harmful_traces))

    for h_trace in harmful_traces:
        h_id = h_trace["id"]
        b_id = pair_mapping.get(h_id)
        if not b_id or b_id not in benign_by_id:
            continue
        b_trace = benign_by_id[b_id]

        h_enc = render_trace_to_inputs(h_trace, tokenizer, max_length, tools)
        b_enc = render_trace_to_inputs(b_trace, tokenizer, max_length, tools)
        if h_enc is None or b_enc is None:
            continue

        h_reps = extract_hidden_states(model, h_enc, target_layers, device)
        b_reps = extract_hidden_states(model, b_enc, target_layers, device)

        for layer in target_layers:
            if layer in h_reps and layer in b_reps:
                cos = cosine_sim_per_token(h_reps[layer], b_reps[layer])
                results[layer].append(cos.mean().item())
                per_token_results[layer].append(cos.tolist())

        n_pairs += 1
        if n_pairs % 10 == 0:
            logger.info("  diagnostic_1: %d pairs processed", n_pairs)

    summary = {}
    for layer in target_layers:
        vals = results[layer]
        if vals:
            summary[str(layer)] = {
                "mean_cosine": float(np.mean(vals)),
                "std_cosine": float(np.std(vals)),
                "min_cosine": float(np.min(vals)),
                "max_cosine": float(np.max(vals)),
                "n_pairs": len(vals),
            }

    return {"summary": summary, "per_token_profiles": per_token_results, "n_pairs": n_pairs}


# ─── diagnostic 2: four-way cosine matrix ────────────────────────────────────

def diagnostic_2_four_way(
    model, harmful_traces, benign_traces, refusal_traces,
    pair_mapping, tokenizer, target_layers, device, max_length, tools
) -> Dict:
    """Cosine similarity matrix between trace types at response token positions.

    A = injection + tricked (harmful)
    B = injection + refusal
    C = no injection + correct (benign)
    """
    benign_by_id = {t["id"]: t for t in benign_traces}
    refusal_by_id = {t["id"].replace("_refusal", ""): t for t in (refusal_traces or [])}

    pair_keys = ["A_B", "A_C", "B_C"]
    layer_results = {layer: {k: [] for k in pair_keys} for layer in target_layers}
    n_triplets = 0

    for h_trace in harmful_traces:
        h_id = h_trace["id"]
        b_id = pair_mapping.get(h_id)
        if not b_id or b_id not in benign_by_id:
            continue
        b_trace = benign_by_id[b_id]

        # Trace A: harmful (injection + tricked)
        a_enc = render_trace_to_inputs(h_trace, tokenizer, max_length, tools)
        # Trace C: benign (no injection + correct)
        c_enc = render_trace_to_inputs(b_trace, tokenizer, max_length, tools)
        if a_enc is None or c_enc is None:
            continue

        # Trace B: refusal (injection + refusal) — may not exist for all
        r_trace = refusal_by_id.get(h_id)
        b_enc = render_trace_to_inputs(r_trace, tokenizer, max_length, tools) if r_trace else None

        a_reps = extract_hidden_states(model, a_enc, target_layers, device)
        c_reps = extract_hidden_states(model, c_enc, target_layers, device)
        b_reps = extract_hidden_states(model, b_enc, target_layers, device) if b_enc else None

        # Find response token ranges
        a_resp = find_response_token_range(h_trace, tokenizer, a_enc)
        c_resp = find_response_token_range(b_trace, tokenizer, c_enc)
        b_resp = find_response_token_range(r_trace, tokenizer, b_enc) if r_trace and b_enc else None

        for layer in target_layers:
            if layer not in a_reps or layer not in c_reps:
                continue

            # Pool response tokens
            a_pool = pool_tokens(a_reps[layer], *(a_resp or (0, a_reps[layer].size(1))))
            c_pool = pool_tokens(c_reps[layer], *(c_resp or (0, c_reps[layer].size(1))))

            cos_ac = F.cosine_similarity(a_pool.unsqueeze(0), c_pool.unsqueeze(0)).item()
            layer_results[layer]["A_C"].append(cos_ac)

            if b_reps and layer in b_reps:
                b_pool = pool_tokens(b_reps[layer], *(b_resp or (0, b_reps[layer].size(1))))
                cos_ab = F.cosine_similarity(a_pool.unsqueeze(0), b_pool.unsqueeze(0)).item()
                cos_bc = F.cosine_similarity(b_pool.unsqueeze(0), c_pool.unsqueeze(0)).item()
                layer_results[layer]["A_B"].append(cos_ab)
                layer_results[layer]["B_C"].append(cos_bc)

        n_triplets += 1
        if n_triplets % 10 == 0:
            logger.info("  diagnostic_2: %d triplets processed", n_triplets)

    summary = {}
    for layer in target_layers:
        layer_summary = {}
        for pair_key in pair_keys:
            vals = layer_results[layer][pair_key]
            if vals:
                layer_summary[pair_key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }
        summary[str(layer)] = layer_summary

    return {"summary": summary, "n_triplets": n_triplets}


# ─── diagnostic 3: alarm signal analysis ─────────────────────────────────────

def diagnostic_3_alarm_signal(
    model, harmful_traces, benign_traces, pair_mapping, tokenizer,
    target_layers, device, max_length, tools
) -> Dict:
    """Check if injection positions are naturally distinguishable.

    Step 1: Separation at injection token positions between harmful/benign
    Step 2: Propagation — do injection reps predict response reps?
    Step 3: Attention patterns — how much do response tokens attend to injection?
    """
    benign_by_id = {t["id"]: t for t in benign_traces}

    separation = {layer: [] for layer in target_layers}
    propagation = {layer: [] for layer in target_layers}
    attention_to_injection = {layer: [] for layer in target_layers}
    relu_cos_stats = {layer: {"positive_frac": [], "mean_cos": []} for layer in target_layers}
    n_samples = 0

    for h_trace in harmful_traces:
        h_id = h_trace["id"]
        b_id = pair_mapping.get(h_id)
        if not b_id or b_id not in benign_by_id:
            continue
        b_trace = benign_by_id[b_id]

        h_enc = render_trace_to_inputs(h_trace, tokenizer, max_length, tools)
        b_enc = render_trace_to_inputs(b_trace, tokenizer, max_length, tools)
        if h_enc is None or b_enc is None:
            continue

        h_reps = extract_hidden_states(model, h_enc, target_layers, device)
        b_reps = extract_hidden_states(model, b_enc, target_layers, device)

        # Find injection and response token ranges in harmful trace
        h_resp = find_response_token_range(h_trace, tokenizer, h_enc)

        # Approximate injection token range: tokens between last tool output start
        # and the response. Use full text search for <INFORMATION> markers.
        h_text = tokenizer.decode(h_enc["input_ids"][0], skip_special_tokens=False)
        inj_char_start = h_text.find("<INFORMATION>")
        inj_char_end = h_text.find("</INFORMATION>")

        inj_tok_range = None
        if inj_char_start >= 0 and inj_char_end >= 0:
            # Approximate token positions from char positions
            prefix_inj_start = tokenizer.encode(h_text[:inj_char_start], add_special_tokens=False)
            prefix_inj_end = tokenizer.encode(h_text[:inj_char_end + len("</INFORMATION>")], add_special_tokens=False)
            inj_tok_range = (len(prefix_inj_start), len(prefix_inj_end))

        for layer in target_layers:
            if layer not in h_reps or layer not in b_reps:
                continue

            h_rep = h_reps[layer]  # [1, T_h, H]
            b_rep = b_reps[layer]  # [1, T_b, H]

            # ReLU(cos_sim) check: would the push loss have gradient?
            # Cosine between harmful and benign frozen reps at response positions
            if h_resp:
                resp_cos = cosine_sim_per_token(h_rep, b_rep)
                resp_slice = resp_cos[h_resp[0]:h_resp[1]] if h_resp[1] <= len(resp_cos) else resp_cos
                relu_cos_stats[layer]["positive_frac"].append(
                    (resp_slice > 0).float().mean().item()
                )
                relu_cos_stats[layer]["mean_cos"].append(resp_slice.mean().item())

            # Step 1: Separation at injection positions
            if inj_tok_range:
                i_start, i_end = inj_tok_range
                i_end = min(i_end, h_rep.size(1), b_rep.size(1))
                i_start = min(i_start, i_end)
                if i_end > i_start:
                    h_inj_pool = h_rep[0, i_start:i_end].mean(dim=0)
                    # Benign trace at equivalent positions (shifted due to no injection)
                    b_equiv_end = min(i_end, b_rep.size(1))
                    b_equiv_start = min(i_start, b_equiv_end)
                    if b_equiv_end > b_equiv_start:
                        b_equiv_pool = b_rep[0, b_equiv_start:b_equiv_end].mean(dim=0)
                        sep = torch.norm(h_inj_pool - b_equiv_pool).item()
                        separation[layer].append(sep)

            # Step 2: Propagation — cosine between injection and response reps
            if inj_tok_range and h_resp:
                i_start, i_end = inj_tok_range
                i_end = min(i_end, h_rep.size(1))
                r_start, r_end = h_resp
                r_end = min(r_end, h_rep.size(1))
                if i_end > i_start and r_end > r_start:
                    inj_pool = h_rep[0, i_start:i_end].mean(dim=0)
                    resp_pool = h_rep[0, r_start:r_end].mean(dim=0)
                    prop_cos = F.cosine_similarity(inj_pool.unsqueeze(0), resp_pool.unsqueeze(0)).item()
                    propagation[layer].append(prop_cos)

        # Step 3: Attention patterns (only for a subset — expensive)
        if n_samples < 20 and inj_tok_range and h_resp:
            try:
                h_attn = extract_attention_weights(model, h_enc, target_layers, device)
                for layer in target_layers:
                    if layer in h_attn:
                        attn = h_attn[layer][0]  # [num_heads, T, T]
                        i_start, i_end = inj_tok_range
                        r_start, r_end = h_resp
                        i_end = min(i_end, attn.size(-1))
                        r_end = min(r_end, attn.size(-2))
                        if i_end > i_start and r_end > r_start:
                            # Mean attention from response tokens to injection tokens
                            attn_slice = attn[:, r_start:r_end, i_start:i_end]
                            mean_attn = attn_slice.mean().item()
                            attention_to_injection[layer].append(mean_attn)
            except Exception as e:
                logger.debug("attention extraction failed: %s", e)

        n_samples += 1
        if n_samples % 10 == 0:
            logger.info("  diagnostic_3: %d samples processed", n_samples)

    summary = {}
    for layer in target_layers:
        layer_data = {}

        sep_vals = separation[layer]
        if sep_vals:
            layer_data["injection_separation"] = {
                "mean_l2": float(np.mean(sep_vals)),
                "std_l2": float(np.std(sep_vals)),
                "n": len(sep_vals),
            }

        prop_vals = propagation[layer]
        if prop_vals:
            layer_data["injection_to_response_propagation"] = {
                "mean_cosine": float(np.mean(prop_vals)),
                "std_cosine": float(np.std(prop_vals)),
                "n": len(prop_vals),
            }

        attn_vals = attention_to_injection[layer]
        if attn_vals:
            layer_data["attention_response_to_injection"] = {
                "mean_attention": float(np.mean(attn_vals)),
                "std_attention": float(np.std(attn_vals)),
                "n": len(attn_vals),
            }

        relu_pos = relu_cos_stats[layer]["positive_frac"]
        relu_mean = relu_cos_stats[layer]["mean_cos"]
        if relu_pos:
            layer_data["relu_cos_sim_at_response"] = {
                "mean_positive_frac": float(np.mean(relu_pos)),
                "mean_cosine": float(np.mean(relu_mean)),
                "note": "If positive_frac is low, ReLU(cos_sim) push loss has zero gradient",
            }

        summary[str(layer)] = layer_data

    return {"summary": summary, "n_samples": n_samples}


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-training representation diagnostics for CB design"
    )
    parser.add_argument("--model", type=str, required=True, help="Base model path")
    parser.add_argument("--harmful-traces", type=Path, required=True)
    parser.add_argument("--benign-traces", type=Path, required=True,
                        help="Benign traces (or same file as harmful, split by category)")
    parser.add_argument("--contrastive-pairs", type=Path, required=True,
                        help="JSONL mapping harmful_trace_id → benign_trace_id")
    parser.add_argument("--refusal-traces", type=Path, default=None,
                        help="Optional refusal traces for four-way comparison")
    parser.add_argument("--tool-schema", type=Path, default=None,
                        help="Tool schema JSON for apply_chat_template")
    parser.add_argument("--layers", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30],
                        help="Layers to analyze")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    # Imports that need GPU
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map="auto",
    )
    model.eval()

    # Load tool schema
    tools = None
    if args.tool_schema and args.tool_schema.exists():
        with open(args.tool_schema) as f:
            schema = json.load(f)
        tools = schema.get("tools", [])
        logger.info("Loaded %d tools from %s", len(tools), args.tool_schema)

    # Load data
    logger.info("Loading traces...")
    harmful_traces = load_traces(args.harmful_traces, "harmful", args.max_samples)
    benign_traces = load_traces(args.benign_traces, "benign", args.max_samples * 2)
    pair_mapping = load_contrastive_pairs(args.contrastive_pairs)
    logger.info("  harmful=%d benign=%d pairs=%d", len(harmful_traces), len(benign_traces), len(pair_mapping))

    refusal_traces = None
    if args.refusal_traces and args.refusal_traces.exists():
        refusal_traces = load_traces(args.refusal_traces, None, args.max_samples)
        logger.info("  refusal=%d", len(refusal_traces))

    all_results = {
        "config": {
            "model": args.model,
            "layers": args.layers,
            "max_samples": args.max_samples,
            "n_harmful": len(harmful_traces),
            "n_benign": len(benign_traces),
            "n_pairs": len(pair_mapping),
        }
    }

    # Diagnostic 1
    logger.info("=== Diagnostic 1: Base model cosine similarity ===")
    d1 = diagnostic_1_base_cosine(
        model, harmful_traces, benign_traces, pair_mapping, tokenizer,
        args.layers, device, args.max_length, tools
    )
    all_results["diagnostic_1_base_cosine"] = d1
    for layer, data in d1["summary"].items():
        logger.info("  Layer %s: mean_cos=%.4f std=%.4f (n=%d)",
                     layer, data["mean_cosine"], data["std_cosine"], data["n_pairs"])

    # Diagnostic 2
    logger.info("=== Diagnostic 2: Four-way cosine matrix ===")
    d2 = diagnostic_2_four_way(
        model, harmful_traces, benign_traces, refusal_traces,
        pair_mapping, tokenizer, args.layers, device, args.max_length, tools
    )
    all_results["diagnostic_2_four_way"] = d2
    for layer, data in d2["summary"].items():
        parts = []
        for k, v in data.items():
            parts.append(f"{k}={v['mean']:.4f}")
        logger.info("  Layer %s: %s", layer, " | ".join(parts))

    # Diagnostic 3
    logger.info("=== Diagnostic 3: Alarm signal analysis ===")
    d3 = diagnostic_3_alarm_signal(
        model, harmful_traces, benign_traces, pair_mapping, tokenizer,
        args.layers, device, args.max_length, tools
    )
    all_results["diagnostic_3_alarm_signal"] = d3
    for layer, data in d3["summary"].items():
        parts = []
        for k, v in data.items():
            if isinstance(v, dict):
                for vk, vv in v.items():
                    if isinstance(vv, float):
                        parts.append(f"{k}.{vk}={vv:.4f}")
        logger.info("  Layer %s: %s", layer, " | ".join(parts[:4]))

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s", args.output)

    # Print interpretation guide
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print()
    print("Diagnostic 1 (Base cosine):")
    print("  High cosine (>0.95) = harmful/benign reps nearly identical → easy to push apart")
    print("  Low cosine (<0.8)  = already divergent → alarm signal exists naturally")
    print()
    print("Diagnostic 2 (Four-way):")
    print("  cos(B,C) high = refusal and correct reps are geometrically close → stable training")
    print("  cos(A,B) low  = model distinguishes tricked from refusal → natural geometry to exploit")
    print("  cos(A,C) low  = core signal exists at this layer")
    print()
    print("Diagnostic 3 (Alarm signal):")
    print("  Large injection separation + weak propagation → alarm signal adds value")
    print("  Large separation + strong propagation → alarm redundant (response push sufficient)")
    print("  Low separation → no coherent alarm cluster, skip alarm signal")
    print()
    print("ReLU(cos_sim) check:")
    print("  If positive_frac < 0.5 at response tokens → push loss has dead gradient")
    print("  This would explain zero training effect / identical outputs")


if __name__ == "__main__":
    main()
