#!/usr/bin/env python3
"""
Lossmask Diagnostic Script
===========================
Joins render_v1 + lossmask_v1 and checks for correctness issues:

  1. Zero-mask samples (all-zero loss_mask) — these produce degenerate pooled
     representations (zero vector) in the triplet loss, providing no gradient signal.
  2. Alignment coverage check — if assistant spans exist in the render alignment
     but the mask is still all-zero, that indicates an ETL_B bug.
  3. Mask boundary accuracy — verifies that the 1s in the mask fall within the
     correct role spans (assistant, tool, etc.) as declared by the alignment.
  4. Length mismatch — loss_mask length vs input_ids length.
  5. Sample-weight zero — samples with sample_weight=0 contributing nothing.

Usage:
    # Use default data/ directory:
    python scripts/diagnose_lossmasks.py

    # Custom paths:
    python scripts/diagnose_lossmasks.py \\
        --renders  data/renders/render_v1.jsonl \\
        --lossmasks data/lossmasks/lossmask_v1.jsonl

    # Verbose: print examples of each problem type
    python scripts/diagnose_lossmasks.py --verbose
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
RENDERS_DIR = DATA_DIR / "renders"
LOSSMASKS_DIR = DATA_DIR / "lossmasks"


# =============================================================================
# IO
# =============================================================================

def _iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _load_pair(render_path: Path, mask_path: Path) -> List[Dict[str, Any]]:
    """Join renders and lossmasks by render_id."""
    renders = {r["render_id"]: r for r in _iter_jsonl(render_path)}
    samples = []
    missing = 0
    for m in _iter_jsonl(mask_path):
        rid = m.get("render_id")
        r = renders.get(rid)
        if r is None:
            missing += 1
            continue
        samples.append({"render": r, "mask": m})
    if missing:
        print(f"  [warn] {missing} lossmask rows had no matching render.")
    return samples


# =============================================================================
# Checks
# =============================================================================

def _check_pair(render: Dict, mask: Dict) -> Dict[str, Any]:
    """Run all checks on a single (render, lossmask) pair."""
    input_ids = render.get("input_ids", [])
    loss_mask = mask.get("loss_mask", [])
    attention_mask = render.get("attention_mask", [1] * len(input_ids))
    alignment = render.get("alignment") or {}
    policy_id = mask.get("policy_id", "unknown")
    sample_weight = mask.get("sample_weight", 1.0)

    seq_len = len(input_ids)
    mask_len = len(loss_mask)
    attn_len = len(attention_mask)

    nonzero = sum(1 for x in loss_mask if x > 0)
    is_zero_mask = nonzero == 0
    length_mismatch = mask_len != seq_len or attn_len != seq_len

    # Extract role spans from alignment
    message_spans = alignment.get("message_spans") or []
    assistant_spans = alignment.get("assistant_spans") or []
    tool_call_spans = alignment.get("tool_call_spans") or []  # used in visual print
    has_alignment = bool(message_spans)
    has_assistant_spans = bool(assistant_spans)

    # Zero-mask despite having assistant spans = ETL_B bug
    zero_with_assistant = is_zero_mask and has_assistant_spans

    # Coverage check: are all masked tokens inside "expected" role spans?
    # Policy-aware: full_sequence / cb_full_sequence are expected to mask widely.
    bad_coverage_tokens = 0
    WIDE_POLICIES = {"full_sequence", "cb_full_sequence", "custom"}
    if not is_zero_mask and has_alignment and policy_id not in WIDE_POLICIES:
        # For targeted policies (assistant_only, tool_calls_only, completion_only, etc.)
        # masked tokens should NOT fall outside non-system spans.
        non_sys_set = set()
        for span in message_spans:
            if span.get("role") != "system":
                non_sys_set.update(range(span["token_start"], span["token_end"]))

        for tok_i, v in enumerate(loss_mask):
            if v > 0 and tok_i not in non_sys_set:
                bad_coverage_tokens += 1

    # BOS token masked? (only a concern for targeted policies, not full_sequence)
    WIDE_POLICIES = {"full_sequence", "cb_full_sequence", "custom"}
    bos_masked = (loss_mask[0] > 0 if loss_mask else False) and policy_id not in WIDE_POLICIES

    # Tokens beyond attention_mask that are masked
    pad_masked = sum(
        1 for (a, lm) in zip(attention_mask, loss_mask)
        if a == 0 and lm > 0
    )

    return {
        "policy_id": policy_id,
        "seq_len": seq_len,
        "mask_len": mask_len,
        "nonzero": nonzero,
        "coverage_frac": nonzero / mask_len if mask_len else 0,
        "is_zero_mask": is_zero_mask,
        "zero_with_assistant": zero_with_assistant,
        "has_alignment": has_alignment,
        "has_assistant_spans": has_assistant_spans,
        "n_assistant_spans": len(assistant_spans),
        "length_mismatch": length_mismatch,
        "bos_masked": bos_masked,
        "pad_masked": pad_masked,
        "sample_weight": sample_weight,
        "bad_coverage_tokens": bad_coverage_tokens,
        "render_id": render.get("render_id"),
        "trace_id": mask.get("trace_id"),
    }


# =============================================================================
# Reporting
# =============================================================================

def _sep(char="=", width=70):
    print(char * width)


def _pct(n, total):
    if total == 0:
        return "0.0%"
    return f"{100 * n / total:.1f}%"


def report(samples: List[Dict], label: str, verbose: bool = False):
    results = [_check_pair(s["render"], s["mask"]) for s in samples]
    n = len(results)
    if n == 0:
        print(f"{label}: (empty)")
        return

    _sep()
    print(f"DATASET: {label}  [{n} samples]")
    _sep()

    # --- Policy breakdown ---
    by_policy = defaultdict(list)
    for r in results:
        by_policy[r["policy_id"]].append(r)

    print(f"\nPolicy breakdown:")
    for policy, items in sorted(by_policy.items()):
        pn = len(items)
        zero = sum(1 for x in items if x["is_zero_mask"])
        zero_with_aspan = sum(1 for x in items if x["zero_with_assistant"])
        avg_cov = sum(x["coverage_frac"] for x in items) / pn
        weight_zero = sum(1 for x in items if x["sample_weight"] == 0)
        print(f"  {policy:30s}  n={pn:5d}  zero_mask={zero:5d} ({_pct(zero,pn)})  "
              f"zero_but_has_aspan={zero_with_aspan:4d}  avg_coverage={avg_cov:.1%}  "
              f"weight_zero={weight_zero}")

    # --- Global stats ---
    n_zero = sum(1 for r in results if r["is_zero_mask"])
    n_mismatch = sum(1 for r in results if r["length_mismatch"])
    n_bos = sum(1 for r in results if r["bos_masked"])
    n_pad = sum(1 for r in results if r["pad_masked"] > 0)
    n_bad_cov = sum(1 for r in results if r["bad_coverage_tokens"] > 0)
    n_zero_w = sum(1 for r in results if r["sample_weight"] == 0)
    n_zero_aspan = sum(1 for r in results if r["zero_with_assistant"])

    print(f"\nGlobal issues:")
    print(f"  Zero loss masks:              {n_zero:5d} / {n} ({_pct(n_zero,n)})")
    print(f"    of which have asst spans:   {n_zero_aspan:5d}  [ETL_B alignment bug]")
    print(f"  Length mismatches:            {n_mismatch:5d} / {n} ({_pct(n_mismatch,n)})")
    print(f"  BOS token masked (bad):       {n_bos:5d} / {n} ({_pct(n_bos,n)})")
    print(f"  Pad tokens masked (bad):      {n_pad:5d} / {n} ({_pct(n_pad,n)})")
    print(f"  Tokens outside role spans:    {n_bad_cov:5d} / {n} ({_pct(n_bad_cov,n)})")
    print(f"  Sample weight = 0:            {n_zero_w:5d} / {n} ({_pct(n_zero_w,n)})")

    # --- Coverage histogram ---
    buckets = [0] * 6  # [0%, 0-10%, 10-30%, 30-50%, 50-80%, 80-100%]
    for r in results:
        f = r["coverage_frac"]
        if f == 0:
            buckets[0] += 1
        elif f < 0.10:
            buckets[1] += 1
        elif f < 0.30:
            buckets[2] += 1
        elif f < 0.50:
            buckets[3] += 1
        elif f < 0.80:
            buckets[4] += 1
        else:
            buckets[5] += 1

    print(f"\nMask coverage histogram (fraction of tokens with loss_mask>0):")
    labels = ["0% (zero)", "0-10%", "10-30%", "30-50%", "50-80%", "80-100%"]
    for lbl, cnt in zip(labels, buckets):
        bar = "█" * (cnt * 40 // max(1, n))
        print(f"  {lbl:15s} {cnt:5d}  {bar}")

    if verbose:
        # --- Examples: zero-mask with assistant span (ETL_B bug) ---
        bugs = [r for r in results if r["zero_with_assistant"]]
        if bugs:
            print(f"\n--- EXAMPLES: zero mask despite having assistant spans (first 5) ---")
            for r in bugs[:5]:
                print(f"  trace_id={r['trace_id']}  policy={r['policy_id']}  "
                      f"seq_len={r['seq_len']}  n_asspans={r['n_assistant_spans']}")

        # --- Examples: bad coverage ---
        bad = [r for r in results if r["bad_coverage_tokens"] > 0]
        if bad:
            print(f"\n--- EXAMPLES: tokens masked outside expected spans (first 5) ---")
            for r in bad[:5]:
                print(f"  trace_id={r['trace_id']}  policy={r['policy_id']}  "
                      f"bad_tokens={r['bad_coverage_tokens']}  seq_len={r['seq_len']}")

        # --- Examples: length mismatch ---
        mis = [r for r in results if r["length_mismatch"]]
        if mis:
            print(f"\n--- EXAMPLES: length mismatches (first 5) ---")
            for r in mis[:5]:
                print(f"  trace_id={r['trace_id']}  seq_len={r['seq_len']}  "
                      f"mask_len={r['mask_len']}")

        # --- Show a good and bad sample visually ---
        nonzero_samples = [s for s, r in zip(samples, results) if not r["is_zero_mask"]]
        zero_samples = [s for s, r in zip(samples, results) if r["is_zero_mask"] and r["has_assistant_spans"]]

        if nonzero_samples:
            print(f"\n--- GOOD SAMPLE (has nonzero mask): ---")
            _print_sample_visual(nonzero_samples[0])
        if zero_samples:
            print(f"\n--- BAD SAMPLE (zero mask despite assistant spans): ---")
            _print_sample_visual(zero_samples[0])


def _print_sample_visual(sample: Dict, max_tokens: int = 60):
    """Print a visual of token roles vs mask for a sample."""
    render = sample["render"]
    mask = sample["mask"]

    loss_mask = mask.get("loss_mask", [])
    alignment = render.get("alignment") or {}
    input_ids = render.get("input_ids", [])
    attention_mask = render.get("attention_mask", [1] * len(input_ids))

    # Build role map per token
    role_map = {}
    for span in (alignment.get("message_spans") or []):
        role = span.get("role", "?")
        for i in range(span["token_start"], span["token_end"]):
            role_map[i] = role

    print(f"  trace_id={mask.get('trace_id')}  policy={mask.get('policy_id')}  "
          f"seq_len={len(input_ids)}  nonzero={sum(1 for x in loss_mask if x>0)}")

    # Print compact per-token view
    n = min(len(input_ids), max_tokens)
    print(f"  Token# : Role       : LossMask : AttMask")
    for i in range(n):
        role = role_map.get(i, "-")
        lm = loss_mask[i] if i < len(loss_mask) else 0
        am = attention_mask[i] if i < len(attention_mask) else 0
        flag = ""
        if lm > 0 and role in ("system", "-"):
            flag = " [!!bad role]"
        if am == 0 and lm > 0:
            flag = " [!!pad masked]"
        print(f"  {i:6d} : {role:10s} : {lm:.2f}     : {am}{flag}")
    if len(input_ids) > max_tokens:
        print(f"  ... ({len(input_ids) - max_tokens} more tokens)")

    # Assistant span summary
    asst_spans = alignment.get("assistant_spans") or []
    print(f"  Assistant spans: {[(s['token_start'], s['token_end']) for s in asst_spans]}")
    tc_spans = alignment.get("tool_call_spans") or []
    if tc_spans:
        print(f"  Tool call spans: {[(s['token_start'], s['token_end'], s.get('tool_name','?')) for s in tc_spans]}")


# =============================================================================
# Entry point
# =============================================================================

def _discover_pairs(renders_dir: Path, masks_dir: Path):
    """Auto-discover matching render+lossmask pairs by stem name."""
    pairs = []
    for mask_path in sorted(masks_dir.glob("*.jsonl")):
        stem = mask_path.stem
        render_path = renders_dir / f"{stem}.jsonl"
        if render_path.exists():
            pairs.append((render_path, mask_path, stem))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--renders", type=Path, help="Renders JSONL (overrides auto-discovery)")
    parser.add_argument("--lossmasks", type=Path, help="Lossmasks JSONL (overrides auto-discovery)")
    parser.add_argument("--renders-dir", type=Path, default=RENDERS_DIR, help="Dir of render JSONL files")
    parser.add_argument("--lossmasks-dir", type=Path, default=LOSSMASKS_DIR, help="Dir of lossmask JSONL files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print examples of each issue type")
    args = parser.parse_args()

    if args.renders and args.lossmasks:
        pairs = [(args.renders, args.lossmasks, args.renders.stem)]
    else:
        pairs = _discover_pairs(args.renders_dir, args.lossmasks_dir)
        if not pairs:
            print(f"No matching render+lossmask pairs found in:\n"
                  f"  renders:   {args.renders_dir}\n"
                  f"  lossmasks: {args.lossmasks_dir}")
            sys.exit(1)

    for render_path, mask_path, label in pairs:
        print(f"\nLoading: {mask_path.name} + {render_path.name}")
        samples = _load_pair(render_path, mask_path)
        report(samples, label=label, verbose=args.verbose)

    _sep()
    print("\nSummary of actionable issues:")
    print("  1. zero_mask samples -> pooled_representations yields a ~zero vector.")
    print("     If loss_mask is all-zero and is not None, the trainer uses it for")
    print("     pooling (line: 'token_mask=harmful_loss_mask if harmful_loss_mask is not None")
    print("     else harmful_attention_mask'). A zero-mask sample effectively contributes")
    print("     no signal to the triplet loss and wastes compute.")
    print("  2. zero_with_assistant (ETL_B bug): alignment spans exist but mask is zero.")
    print("     This indicates _mask_assistant_only saw the alignment but wrote zeros.")
    print("     Likely cause: token_start == token_end (empty span) or span indexing mismatch.")
    print("  3. BOS/pad tokens masked: mask leaks into non-content positions.")
    print("  4. Length mismatch: loss_mask and input_ids differ in length -> truncation")
    print("     silently drops tail of the mask or pads with zeros in SchemaDataset.")


if __name__ == "__main__":
    main()
