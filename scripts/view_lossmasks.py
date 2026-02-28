#!/usr/bin/env python3
"""
Lossmask viewer — compare trace content vs rendered lossmask.

Prints each sample showing:
  - The original trace messages (role + text excerpt)
  - The render alignment spans (token ranges per role)
  - Which spans are masked and what fraction

Usage:
    python scripts/view_lossmasks.py                    # all datasets
    python scripts/view_lossmasks.py --dataset agentdojo
    python scripts/view_lossmasks.py --dataset fujitsu
    python scripts/view_lossmasks.py --dataset agentdojo --limit 20
    python scripts/view_lossmasks.py --nonzero-only     # skip zero-mask samples
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TRACE_FILES = {
    "agentdojo": DATA_DIR / "traces" / "agentdojo_augmented.jsonl",
    "fujitsu":   DATA_DIR / "traces" / "trace_v1.jsonl",
}
RENDERS_FILE  = DATA_DIR / "renders"  / "render_v1.jsonl"
MASKS_FILE    = DATA_DIR / "lossmasks" / "lossmask_v1.jsonl"

SEP = "─" * 72


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _nonzero_spans(mask):
    """Return list of (start, end) for contiguous nonzero runs."""
    spans, in_span, start = [], False, 0
    for i, v in enumerate(mask):
        if v > 0 and not in_span:
            start, in_span = i, True
        elif v == 0 and in_span:
            spans.append((start, i)); in_span = False
    if in_span:
        spans.append((start, len(mask)))
    return spans


def _excerpt(text, max_chars=120):
    text = (text or "").replace("\n", "↵ ")
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


def print_sample(trace, render, mask, idx):
    tid = mask.get("trace_id", "?")
    policy = mask.get("policy_id", "?")
    loss_mask = mask.get("loss_mask", [])
    input_ids = render.get("input_ids", [])
    alignment = render.get("alignment") or {}
    msg_spans   = alignment.get("message_spans") or []
    asst_spans  = alignment.get("assistant_spans") or []
    tc_spans    = alignment.get("tool_call_spans") or []

    nonzero = sum(1 for x in loss_mask if x > 0)
    total   = len(loss_mask)

    print(f"\n{'═'*72}")
    print(f"[{idx}] {tid}")
    print(f"     policy={policy}  seq_len={total}  masked={nonzero}/{total} ({100*nonzero//max(1,total)}%)")
    print(SEP)

    # --- Original trace messages ---
    messages = trace.get("messages", []) if trace else []
    if messages:
        print("TRACE MESSAGES:")
        for msg_i, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
            line = f"  [{msg_i}] {role:10s}: {_excerpt(content, 100)}"
            if tool_calls:
                names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                line += f"  +TOOL_CALL({', '.join(names)})"
            print(line)
    else:
        print("TRACE MESSAGES: (not found locally — trace file mismatch)")

    # --- Render alignment + mask coverage per span ---
    print(SEP)
    print("RENDER SPANS vs LOSS MASK:")
    print(f"  {'#':>3}  {'role':12s}  {'tokens':>12s}  {'masked':>12s}  {'coverage':>9s}  tool_name")
    for span in msg_spans:
        role  = span.get("role", "?")
        ts    = span.get("token_start", 0)
        te    = span.get("token_end", 0)
        n_tok = te - ts
        n_masked = sum(1 for x in loss_mask[ts:te] if x > 0)
        cov   = f"{100*n_masked//max(1,n_tok)}%" if n_tok else "n/a"
        midx  = span.get("message_index", "?")
        # Find tool calls in this span
        tc_names = [s.get("tool_name","?") for s in tc_spans if s.get("message_index") == midx]
        tc_str = f"[{', '.join(tc_names)}]" if tc_names else ""
        mark = "✓" if n_masked > 0 else "✗"
        print(f"  {midx!s:>3}  {role:12s}  [{ts:4d},{te:4d})={n_tok:4d}t  "
              f"masked={n_masked:4d}t  {cov:>8s}  {mark}  {tc_str}")

    # --- Nonzero mask spans ---
    mask_spans = _nonzero_spans(loss_mask)
    print(SEP)
    if mask_spans:
        print(f"NONZERO MASK SPANS ({len(mask_spans)} runs):")
        for s, e in mask_spans:
            # Which role does this span fall under?
            roles_in_span = set()
            for sp in msg_spans:
                if sp["token_start"] < e and sp["token_end"] > s:
                    roles_in_span.add(sp["role"])
            print(f"  [{s:4d},{e:4d})  len={e-s:4d}  roles={sorted(roles_in_span)}")
    else:
        print("NONZERO MASK SPANS: (none — zero mask)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["agentdojo", "fujitsu", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="Max samples to print (0=all)")
    ap.add_argument("--nonzero-only", action="store_true", help="Skip zero-mask samples")
    ap.add_argument("--renders",  type=Path, default=RENDERS_FILE)
    ap.add_argument("--lossmasks",type=Path, default=MASKS_FILE)
    ap.add_argument("--traces-agentdojo", type=Path, default=TRACE_FILES["agentdojo"],
                    help="default: agentdojo_augmented.jsonl (matches render_v1 IDs)")
    ap.add_argument("--traces-fujitsu",   type=Path, default=TRACE_FILES["fujitsu"])
    args = ap.parse_args()

    print("Loading renders…", end=" ", flush=True)
    renders = {r["render_id"]: r for r in _load_jsonl(args.renders)}
    print(f"{len(renders)}")

    print("Loading lossmasks…", end=" ", flush=True)
    masks = _load_jsonl(args.lossmasks)
    print(f"{len(masks)}")

    traces_by_id = {}
    for ds, path in [("agentdojo", args.traces_agentdojo), ("fujitsu", args.traces_fujitsu)]:
        if args.dataset not in ("all", ds):
            continue
        if path.exists():
            for t in _load_jsonl(path):
                traces_by_id[t["id"]] = t
    print(f"Loaded {len(traces_by_id)} traces")

    # Filter masks by dataset
    def _matches(tid):
        if args.dataset == "all":
            return True
        return args.dataset in tid

    printed = 0
    for i, m in enumerate(masks):
        tid = m.get("trace_id", "")
        if not _matches(tid):
            continue
        lm = m.get("loss_mask", [])
        if args.nonzero_only and all(x == 0 for x in lm):
            continue

        render = renders.get(m.get("render_id"), {})
        trace  = traces_by_id.get(tid)

        print_sample(trace, render, m, printed + 1)
        printed += 1

        if args.limit and printed >= args.limit:
            break

    print(f"\n{'═'*72}")
    print(f"Printed {printed} samples.")


if __name__ == "__main__":
    main()
