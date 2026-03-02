#!/usr/bin/env python3
"""
Diagnose training data: dump full traces, loss masks, pooling masks,
decoded text with mask annotations, and statistics.

Usage:
    python scripts/diagnose_training_data.py \
        --tokenizer meta-llama/Llama-3.1-8B-Instruct \
        --harmful-renders renders/fujitsu_b4_ds.jsonl renders/agentdojo_ds.jsonl \
        --harmful-lossmasks lossmasks/fujitsu_b4_ds.jsonl lossmasks/agentdojo_ds.jsonl \
        --benign-renders renders/fujitsu_b4_dr.jsonl renders/agentdojo_dr.jsonl \
        --benign-lossmasks lossmasks/fujitsu_b4_dr.jsonl lossmasks/agentdojo_dr.jsonl \
        --output diagnostic.txt \
        --num-samples 3
"""

import argparse
import json
from pathlib import Path


def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_renders_and_masks(render_path: Path, lossmask_path: Path):
    renders = {row["render_id"]: row for row in iter_jsonl(render_path)}
    samples = []
    for mask_row in iter_jsonl(lossmask_path):
        render_id = mask_row.get("render_id")
        render = renders.get(render_id)
        if render is None:
            continue
        input_ids = render["input_ids"]
        samples.append({
            "input_ids": input_ids,
            "attention_mask": render.get("attention_mask", [1] * len(input_ids)),
            "loss_mask": mask_row["loss_mask"],
            "pooling_mask": mask_row.get("pooling_mask"),
            "trace_id": mask_row.get("trace_id"),
            "render_id": render_id,
            "policy_id": mask_row.get("policy_id"),
            "source_file": str(render_path),
        })
    return samples


def annotate_tokens(tokenizer, input_ids, loss_mask, pooling_mask=None):
    """Decode each token and annotate with mask status."""
    lines = []
    for i, tid in enumerate(input_ids):
        token_str = tokenizer.decode([tid], skip_special_tokens=False)
        # Escape newlines/tabs for display
        token_display = repr(token_str)[1:-1]  # strip outer quotes
        lm = loss_mask[i] if i < len(loss_mask) else 0
        pm = pooling_mask[i] if pooling_mask and i < len(pooling_mask) else "N/A"
        lines.append(f"  [{i:5d}] id={tid:8d}  loss={lm}  pool={pm}  |{token_display}|")
    return lines


def dump_sample(out, idx, sample, tokenizer, label):
    """Dump a single sample with full detail."""
    input_ids = sample["input_ids"]
    loss_mask = sample["loss_mask"]
    pooling_mask = sample.get("pooling_mask")
    attn_mask = sample.get("attention_mask", [1] * len(input_ids))

    out.write(f"\n{'='*100}\n")
    out.write(f"[{label} SAMPLE #{idx}]\n")
    out.write(f"{'='*100}\n")
    out.write(f"  trace_id:    {sample.get('trace_id', 'N/A')}\n")
    out.write(f"  render_id:   {sample.get('render_id', 'N/A')}\n")
    out.write(f"  policy_id:   {sample.get('policy_id', 'N/A')}\n")
    out.write(f"  source_file: {sample.get('source_file', 'N/A')}\n")
    out.write(f"  seq_length:  {len(input_ids)}\n")
    out.write(f"  attention_mask sum: {sum(attn_mask)} / {len(attn_mask)}\n")
    out.write(f"  loss_mask sum:      {sum(loss_mask)} / {len(loss_mask)}\n")
    if pooling_mask:
        out.write(f"  pooling_mask sum:   {sum(pooling_mask)} / {len(pooling_mask)}\n")
        out.write(f"  pooling != loss:    {sum(1 for a, b in zip(loss_mask, pooling_mask) if a != b)} tokens differ\n")
    else:
        out.write(f"  pooling_mask:       None (will use loss_mask)\n")
    out.write(f"\n")

    # === FULL DECODED TEXT ===
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    out.write(f"--- FULL DECODED TEXT (untruncated) ---\n")
    out.write(full_text)
    out.write(f"\n--- END FULL TEXT ---\n\n")

    # === LOSS MASK SEGMENTS ===
    # Show which portions are in the loss mask vs not
    out.write(f"--- LOSS MASK SEGMENTS ---\n")
    segments = []
    current_mask_val = loss_mask[0] if loss_mask else 0
    current_ids = []
    for i, tid in enumerate(input_ids):
        mv = loss_mask[i] if i < len(loss_mask) else 0
        if mv != current_mask_val:
            segments.append((current_mask_val, current_ids[:]))
            current_ids = []
            current_mask_val = mv
        current_ids.append(tid)
    if current_ids:
        segments.append((current_mask_val, current_ids[:]))

    for seg_idx, (mask_val, seg_ids) in enumerate(segments):
        seg_text = tokenizer.decode(seg_ids, skip_special_tokens=False)
        label_str = "LOSS=1 (in loss)" if mask_val else "LOSS=0 (masked out)"
        out.write(f"\n  [Segment {seg_idx}: {label_str}, {len(seg_ids)} tokens]\n")
        out.write(f"  {seg_text}\n")
    out.write(f"--- END LOSS MASK SEGMENTS ---\n\n")

    # === POOLING MASK SEGMENTS (if different from loss mask) ===
    if pooling_mask and pooling_mask != loss_mask:
        out.write(f"--- POOLING MASK SEGMENTS ---\n")
        segments = []
        current_mask_val = pooling_mask[0]
        current_ids = []
        for i, tid in enumerate(input_ids):
            mv = pooling_mask[i] if i < len(pooling_mask) else 0
            if mv != current_mask_val:
                segments.append((current_mask_val, current_ids[:]))
                current_ids = []
                current_mask_val = mv
            current_ids.append(tid)
        if current_ids:
            segments.append((current_mask_val, current_ids[:]))

        for seg_idx, (mask_val, seg_ids) in enumerate(segments):
            seg_text = tokenizer.decode(seg_ids, skip_special_tokens=False)
            label_str = "POOL=1 (in pooling)" if mask_val else "POOL=0 (masked out)"
            out.write(f"\n  [Segment {seg_idx}: {label_str}, {len(seg_ids)} tokens]\n")
            out.write(f"  {seg_text}\n")
        out.write(f"--- END POOLING MASK SEGMENTS ---\n\n")

    # === TOKEN-BY-TOKEN DUMP (first 200 + last 50 tokens) ===
    out.write(f"--- TOKEN-BY-TOKEN (first 200 + last 50) ---\n")
    token_lines = annotate_tokens(tokenizer, input_ids, loss_mask, pooling_mask)
    if len(token_lines) <= 250:
        out.write("\n".join(token_lines))
    else:
        out.write("\n".join(token_lines[:200]))
        out.write(f"\n  ... ({len(token_lines) - 250} tokens omitted) ...\n")
        out.write("\n".join(token_lines[-50:]))
    out.write(f"\n--- END TOKEN-BY-TOKEN ---\n")


def dump_stats(out, samples, label):
    """Dump aggregate statistics for a set of samples."""
    out.write(f"\n{'='*100}\n")
    out.write(f"STATISTICS: {label} ({len(samples)} samples)\n")
    out.write(f"{'='*100}\n")

    if not samples:
        out.write("  (no samples)\n")
        return

    seq_lens = [len(s["input_ids"]) for s in samples]
    loss_sums = [sum(s["loss_mask"]) for s in samples]
    loss_fracs = [sum(s["loss_mask"]) / max(len(s["loss_mask"]), 1) for s in samples]
    has_pooling = [s.get("pooling_mask") is not None for s in samples]

    out.write(f"  Total samples:    {len(samples)}\n")
    out.write(f"  Sequence lengths: min={min(seq_lens)}, max={max(seq_lens)}, "
              f"mean={sum(seq_lens)/len(seq_lens):.0f}\n")
    out.write(f"  Loss mask tokens: min={min(loss_sums)}, max={max(loss_sums)}, "
              f"mean={sum(loss_sums)/len(loss_sums):.1f}\n")
    out.write(f"  Loss mask frac:   min={min(loss_fracs):.3f}, max={max(loss_fracs):.3f}, "
              f"mean={sum(loss_fracs)/len(loss_fracs):.3f}\n")
    out.write(f"  Has pooling_mask: {sum(has_pooling)}/{len(samples)}\n")

    zero_loss = sum(1 for s in loss_sums if s == 0)
    if zero_loss > 0:
        out.write(f"  WARNING: {zero_loss} samples have ALL-ZERO loss masks!\n")

    # Source file breakdown
    sources = {}
    for s in samples:
        src = s.get("source_file", "unknown")
        sources[src] = sources.get(src, 0) + 1
    out.write(f"  Source files:\n")
    for src, count in sorted(sources.items()):
        out.write(f"    {count:5d} from {src}\n")

    # Policy breakdown
    policies = {}
    for s in samples:
        p = s.get("policy_id", "unknown")
        policies[p] = policies.get(p, 0) + 1
    out.write(f"  Policies:\n")
    for p, count in sorted(policies.items(), key=lambda x: -x[1]):
        out.write(f"    {count:5d}  {p}\n")

    # Check for loss_mask / input_ids length mismatches
    mismatches = sum(1 for s in samples if len(s["loss_mask"]) != len(s["input_ids"]))
    if mismatches > 0:
        out.write(f"  WARNING: {mismatches} samples have loss_mask length != input_ids length!\n")

    # Check for pooling_mask / input_ids length mismatches
    pool_mismatches = sum(
        1 for s in samples
        if s.get("pooling_mask") and len(s["pooling_mask"]) != len(s["input_ids"])
    )
    if pool_mismatches > 0:
        out.write(f"  WARNING: {pool_mismatches} samples have pooling_mask length != input_ids length!\n")


def main():
    parser = argparse.ArgumentParser(description="Diagnose training data")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer path")
    parser.add_argument("--harmful-renders", type=Path, nargs="+", required=True)
    parser.add_argument("--harmful-lossmasks", type=Path, nargs="+", required=True)
    parser.add_argument("--benign-renders", type=Path, nargs="+", required=True)
    parser.add_argument("--benign-lossmasks", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output diagnostic file")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of samples to dump per category")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Load all data
    harmful = []
    for rp, mp in zip(args.harmful_renders, args.harmful_lossmasks):
        harmful.extend(load_renders_and_masks(rp, mp))

    benign = []
    for rp, mp in zip(args.benign_renders, args.benign_lossmasks):
        benign.extend(load_renders_and_masks(rp, mp))

    with open(args.output, "w") as out:
        out.write("=" * 100 + "\n")
        out.write("TRAINING DATA DIAGNOSTIC\n")
        out.write("=" * 100 + "\n")
        out.write(f"Tokenizer: {args.tokenizer}\n")
        out.write(f"Harmful render files: {[str(p) for p in args.harmful_renders]}\n")
        out.write(f"Harmful lossmask files: {[str(p) for p in args.harmful_lossmasks]}\n")
        out.write(f"Benign render files: {[str(p) for p in args.benign_renders]}\n")
        out.write(f"Benign lossmask files: {[str(p) for p in args.benign_lossmasks]}\n")
        out.write(f"Total harmful: {len(harmful)}\n")
        out.write(f"Total benign:  {len(benign)}\n")
        out.write(f"Samples to dump: {args.num_samples} per category\n")
        out.write("\n")

        # Stats
        dump_stats(out, harmful, "HARMFUL (DS)")
        dump_stats(out, benign, "BENIGN (DR)")

        # Dump individual samples
        n = args.num_samples

        # Harmful: first N from each source file
        source_groups = {}
        for s in harmful:
            src = s.get("source_file", "unknown")
            source_groups.setdefault(src, []).append(s)

        for src, group in sorted(source_groups.items()):
            out.write(f"\n\n{'#'*100}\n")
            out.write(f"# HARMFUL SAMPLES FROM: {src}\n")
            out.write(f"# ({len(group)} total in this file)\n")
            out.write(f"{'#'*100}\n")
            for i, sample in enumerate(group[:n], 1):
                dump_sample(out, i, sample, tokenizer, f"HARMFUL [{Path(src).stem}]")

        # Benign: first N from each source file
        source_groups = {}
        for s in benign:
            src = s.get("source_file", "unknown")
            source_groups.setdefault(src, []).append(s)

        for src, group in sorted(source_groups.items()):
            out.write(f"\n\n{'#'*100}\n")
            out.write(f"# BENIGN SAMPLES FROM: {src}\n")
            out.write(f"# ({len(group)} total in this file)\n")
            out.write(f"{'#'*100}\n")
            for i, sample in enumerate(group[:n], 1):
                dump_sample(out, i, sample, tokenizer, f"BENIGN [{Path(src).stem}]")

        out.write(f"\n\n{'='*100}\n")
        out.write("END OF DIAGNOSTIC\n")
        out.write(f"{'='*100}\n")

    print(f"Diagnostic written to: {args.output}")


if __name__ == "__main__":
    main()
