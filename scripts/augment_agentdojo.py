#!/usr/bin/env python3
"""AgentDojo multi-model ingestion + data augmentation.

Operations:
  ingest     - Re-run ETL_A on full spotlight extract (all 17 models)
  backfill   - Populate injection_char_span for traces missing it
  removal    - Harmful -> Benign via injection removal
  transplant - Benign -> Harmful via injection transplant
  taxonomy   - Tag traces with OWASP taxonomy categories

Usage:
  python scripts/augment_agentdojo.py \
      --raw-input data/agent_dojo/agentdojo_spotlight_extract.jsonl \
      --traces-input data/traces/agentdojo_traces.jsonl \
      --output data/traces/agentdojo_augmented.jsonl \
      --operations ingest,backfill,removal,transplant,taxonomy \
      --seed 42 --stats
"""

import argparse
import copy
import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas.trace import Trace  # noqa: F401 - used by ETL_A
from src.schemas.tools.ETL_A import (
    compile_injection_patterns,
    convert_agentdojo_record,
    detect_injection_in_messages,
)

logger = logging.getLogger(__name__)

INJECTION_RE = re.compile(r"<INFORMATION>.*?</INFORMATION>", re.DOTALL)

# OWASP LLM Top 10 mapping for AgentDojo traces
OWASP_TAGS = {
    "LLM01": "Prompt Injection",
    "LLM06": "Excessive Agency",
    "LLM08": "Vector & Embedding Weaknesses",
}


# ── Helpers ──────────────────────────────────────────────────────────────


def _iter_jsonl(path: Path):
    """Yield (line_number, record) from a JSONL file."""
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)


def _write_jsonl(traces: List[Dict], path: Path):
    """Write list of trace dicts to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    logger.info("Wrote %d traces to %s", len(traces), path)


def _dedup_key(rec: Dict) -> str:
    """Dedup key from raw record metadata (includes suite for benign tasks)."""
    meta = rec.get("metadata", {})
    return f"{meta.get('suite_name')}|{meta.get('user_task_id')}|{meta.get('injection_task_id')}|{meta.get('model')}"


def _make_derived_id(source_dataset: str, op: str, parent_id: str, salt: str = "") -> str:
    """Generate deterministic ID for derived traces."""
    content = f"{op}:{parent_id}:{salt}"
    h = hashlib.sha256(content.encode()).hexdigest()[:32]
    return f"trace_{source_dataset}_{h}"


# ── Operation: ingest ────────────────────────────────────────────────────


def op_ingest(
    raw_path: Path,
    injection_patterns: List[re.Pattern],
    stats: Dict[str, int],
) -> List[Dict]:
    """Re-ingest all raw records through ETL_A converter."""
    traces = []
    seen_keys: Set[str] = set()
    skipped_dup = 0

    for line_num, rec in _iter_jsonl(raw_path):
        key = _dedup_key(rec)
        if key in seen_keys:
            skipped_dup += 1
            continue
        seen_keys.add(key)

        trace = convert_agentdojo_record(
            rec, split="train", line_number=line_num,
            injection_patterns=injection_patterns,
        )
        if trace is None:
            stats["ingest_failed"] += 1
            continue

        td = trace.to_dict()
        # Mark provenance
        td.setdefault("raw_metadata", {}).setdefault("source_fields", {})
        td["raw_metadata"]["source_fields"]["augmentation_provenance"] = "original"

        traces.append(td)

    stats["ingested"] = len(traces)
    stats["ingest_dedup_skipped"] = skipped_dup
    return traces


# ── Operation: backfill ──────────────────────────────────────────────────


def op_backfill(
    traces: List[Dict],
    injection_patterns: List[re.Pattern],
    stats: Dict[str, int],
) -> List[Dict]:
    """Backfill injection_char_span for traces that are missing it."""
    filled = 0
    for t in traces:
        sh = t.get("signal_hints") or {}
        if sh.get("injection_char_span"):
            continue  # already has span

        # Run detection on raw messages
        raw_msgs = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in t.get("messages", [])
        ]
        first_span, all_spans = detect_injection_in_messages(raw_msgs, injection_patterns)

        if first_span:
            if "signal_hints" not in t or t["signal_hints"] is None:
                t["signal_hints"] = {}
            t["signal_hints"]["injection_char_span"] = {
                "message_index": first_span.message_index,
                "char_start": first_span.char_start,
                "char_end": first_span.char_end,
            }
            # Store all spans if multiple
            if len(all_spans) > 1:
                t.setdefault("raw_metadata", {}).setdefault("source_fields", {})
                t["raw_metadata"]["source_fields"]["all_injection_spans"] = [
                    {"message_index": s.message_index, "char_start": s.char_start, "char_end": s.char_end}
                    for s in all_spans
                ]
            filled += 1

    stats["backfill_filled"] = filled
    return traces


# ── Operation: removal ───────────────────────────────────────────────────


def op_removal(traces: List[Dict], stats: Dict[str, int]) -> List[Dict]:
    """Create benign traces by removing injections from harmful traces.

    For each harmful trace with a detectable <INFORMATION>...</INFORMATION>
    injection in a tool message:
    1. Remove the injection text from the tool response
    2. Truncate the conversation after the cleaned tool message
    3. Re-label as benign
    """
    new_traces = []

    for t in traces:
        cat = (t.get("labels") or {}).get("category", "")
        if cat not in ("harmful", "resisted"):
            continue

        messages = t.get("messages", [])
        # Find first tool message with injection
        injection_idx = None
        cleaned_content = None

        for i, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content") or ""
            match = INJECTION_RE.search(content)
            if match:
                injection_idx = i
                cleaned_content = content[:match.start()] + content[match.end():]
                cleaned_content = cleaned_content.strip()
                break

        if injection_idx is None or cleaned_content is None:
            continue

        # Build new trace: keep messages up to and including the cleaned tool msg
        new_msgs = []
        for j in range(injection_idx + 1):
            msg_copy = copy.deepcopy(messages[j])
            if j == injection_idx:
                msg_copy["content"] = cleaned_content
            new_msgs.append(msg_copy)

        parent_id = t.get("id", "unknown")
        new_id = _make_derived_id("agentdojo", "removal", parent_id)

        new_trace = copy.deepcopy(t)
        new_trace["id"] = new_id
        new_trace["messages"] = new_msgs
        new_trace["labels"] = {
            "category": "benign",
            "security_outcome": "safe",
            "attack_type": None,
            "attack_succeeded": False,
            "attack_present": False,
        }
        new_trace.setdefault("links", {})["parent_trace_ids"] = [parent_id]
        new_trace.setdefault("source", {})["tier"] = "derived"
        new_trace.setdefault("raw_metadata", {}).setdefault("source_fields", {})
        new_trace["raw_metadata"]["source_fields"]["augmentation_provenance"] = "removal"
        new_trace["training"] = {
            "sample_weight": 1.0,
            "loss_mask_policy": "assistant_only",
            "mixture": {"class_id": "agentdojo/benign"},
        }
        # Clear injection span since we removed it
        if "signal_hints" in new_trace:
            new_trace["signal_hints"]["injection_char_span"] = None

        new_traces.append(new_trace)

    stats["removal_created"] = len(new_traces)
    return new_traces


# ── Operation: transplant ────────────────────────────────────────────────


def _collect_injections(traces: List[Dict]) -> List[str]:
    """Collect unique injection texts from harmful traces."""
    injections = set()
    for t in traces:
        cat = (t.get("labels") or {}).get("category", "")
        if cat != "harmful":
            continue
        for msg in t.get("messages", []):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content") or ""
            for match in INJECTION_RE.finditer(content):
                injections.add(match.group(0))
    return list(injections)


def op_transplant(
    traces: List[Dict],
    rng: random.Random,
    stats: Dict[str, int],
) -> List[Dict]:
    """Create harmful traces by transplanting injections into benign tool messages.

    For each benign trace with at least one tool message:
    1. Pick a random injection text from harmful traces
    2. Insert it into the first tool message content
    3. Truncate after the injected tool message
    4. Re-label as harmful
    """
    injections = _collect_injections(traces)
    if not injections:
        logger.warning("No injections found for transplant operation")
        stats["transplant_created"] = 0
        return []

    new_traces = []
    benign_traces = [
        t for t in traces
        if (t.get("labels") or {}).get("category") == "benign"
    ]

    for t in benign_traces:
        messages = t.get("messages", [])
        # Find first tool message
        tool_idx = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "tool" and (msg.get("content") or "").strip():
                tool_idx = i
                break

        if tool_idx is None:
            continue

        # Pick random injection
        injection = rng.choice(injections)
        tool_content = messages[tool_idx].get("content") or ""

        # Insert injection at a random position in the tool content
        if len(tool_content) > 50:
            insert_pos = rng.randint(len(tool_content) // 4, 3 * len(tool_content) // 4)
        else:
            insert_pos = len(tool_content)

        injected_content = tool_content[:insert_pos] + "\n" + injection + "\n" + tool_content[insert_pos:]

        # Build new messages: up to and including injected tool msg
        new_msgs = []
        for j in range(tool_idx + 1):
            msg_copy = copy.deepcopy(messages[j])
            if j == tool_idx:
                msg_copy["content"] = injected_content
            new_msgs.append(msg_copy)

        parent_id = t.get("id", "unknown")
        new_id = _make_derived_id("agentdojo", "transplant", parent_id, injection[:50])

        new_trace = copy.deepcopy(t)
        new_trace["id"] = new_id
        new_trace["messages"] = new_msgs
        new_trace["labels"] = {
            "category": "harmful",
            "security_outcome": "unsafe",
            "attack_type": "prompt_injection",
            "attack_succeeded": True,
            "attack_present": True,
        }
        new_trace.setdefault("links", {})["parent_trace_ids"] = [parent_id]
        new_trace.setdefault("source", {})["tier"] = "derived"
        new_trace.setdefault("raw_metadata", {}).setdefault("source_fields", {})
        new_trace["raw_metadata"]["source_fields"]["augmentation_provenance"] = "transplant"
        new_trace["training"] = {
            "sample_weight": 1.0,
            "loss_mask_policy": "assistant_only",
            "mixture": {"class_id": "agentdojo/harmful"},
        }

        # Set injection span for the injected content
        inj_start = insert_pos + 1  # after \n
        inj_end = inj_start + len(injection)
        new_trace.setdefault("signal_hints", {})
        new_trace["signal_hints"]["injection_char_span"] = {
            "message_index": tool_idx,
            "char_start": inj_start,
            "char_end": inj_end,
        }

        new_traces.append(new_trace)

    stats["transplant_created"] = len(new_traces)
    return new_traces


# ── Operation: contrastive ───────────────────────────────────────────────


def op_contrastive(traces: List[Dict], stats: Dict[str, int]) -> List[Dict]:
    """Build contrastive pair mapping: harmful <-> benign (removal) traces.

    For each harmful trace that has a corresponding removal-provenance benign
    trace, create a mapping record linking both. The training pipeline uses
    these pairs to provide counterfactual signal: "when you see injection,
    still call the tool you would have called without it."
    """
    # Index removal traces by parent_trace_id
    removal_by_parent: Dict[str, str] = {}
    for t in traces:
        sf = (t.get("raw_metadata") or {}).get("source_fields") or {}
        if sf.get("augmentation_provenance") == "removal":
            parent_ids = (t.get("links") or {}).get("parent_trace_ids", [])
            for pid in parent_ids:
                removal_by_parent[pid] = t["id"]

    pairs = []
    for t in traces:
        cat = (t.get("labels") or {}).get("category", "")
        if cat not in ("harmful", "resisted"):
            continue
        tid = t.get("id", "")
        if tid in removal_by_parent:
            pairs.append({
                "harmful_trace_id": tid,
                "benign_trace_id": removal_by_parent[tid],
                "pair_type": "injection_removal",
            })

    stats["contrastive_pairs"] = len(pairs)
    return pairs


# ── Operation: taxonomy ──────────────────────────────────────────────────


def op_taxonomy(traces: List[Dict], stats: Dict[str, int]) -> List[Dict]:
    """Tag traces with OWASP LLM Top 10 taxonomy categories.

    Mapping:
    - LLM01 (Prompt Injection): all traces with attack_present=True
    - LLM06 (Excessive Agency): harmful traces where model called tools it shouldn't
    - LLM08 (Vector/Embedding Weaknesses): traces with injection in tool/retrieval content
    """
    tagged = 0
    for t in traces:
        tags = []
        labels = t.get("labels") or {}
        cat = labels.get("category", "benign")
        attack_present = labels.get("attack_present", False)

        # LLM01: Prompt Injection - any trace with attack present
        if attack_present or cat in ("harmful", "resisted"):
            tags.append("LLM01")

        # LLM06: Excessive Agency - harmful traces (model complied with injection)
        if cat == "harmful":
            tags.append("LLM06")

        # LLM08: Vector & Embedding Weaknesses - injection via tool response (RAG-like)
        sh = t.get("signal_hints") or {}
        if sh.get("injection_char_span"):
            span = sh["injection_char_span"]
            msg_idx = span.get("message_index", -1)
            msgs = t.get("messages", [])
            if 0 <= msg_idx < len(msgs) and msgs[msg_idx].get("role") == "tool":
                tags.append("LLM08")

        if tags:
            t.setdefault("raw_metadata", {}).setdefault("source_fields", {})
            t["raw_metadata"]["source_fields"]["taxonomy_tags"] = tags
            tagged += 1

    stats["taxonomy_tagged"] = tagged
    return traces


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="AgentDojo multi-model ingestion + data augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw-input", type=Path,
        default=Path("data/agent_dojo/agentdojo_spotlight_extract.jsonl"),
        help="Raw AgentDojo extract JSONL",
    )
    parser.add_argument(
        "--traces-input", type=Path, default=None,
        help="Existing traces JSONL (for backfill on existing data). "
             "If omitted, only ingest is used.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/traces/agentdojo_augmented.jsonl"),
        help="Output augmented traces JSONL",
    )
    parser.add_argument(
        "--operations", type=str,
        default="ingest,backfill,removal,transplant,taxonomy",
        help="Comma-separated operations to run (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stats", action="store_true", help="Print detailed stats")
    parser.add_argument(
        "--max-transplant", type=int, default=None,
        help="Cap on number of transplant traces (default: no cap)",
    )
    parser.add_argument(
        "--contrastive-output", type=Path, default=None,
        help="Output path for contrastive pair mapping JSONL (used by contrastive operation)",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    rng = random.Random(args.seed)
    ops = [o.strip() for o in args.operations.split(",")]
    injection_patterns = compile_injection_patterns()
    stats: Dict[str, int] = defaultdict(int)

    print("=" * 60)
    print("AGENTDOJO DATA AUGMENTATION")
    print("=" * 60)
    print(f"Operations: {ops}")
    print(f"Raw input: {args.raw_input}")
    print(f"Output: {args.output}")
    print()

    traces: List[Dict] = []

    # Step 1: Ingest
    if "ingest" in ops:
        print(">>> Running ingest...")
        traces = op_ingest(args.raw_input, injection_patterns, stats)
        print(f"    Ingested: {stats['ingested']} traces")
        if stats.get("ingest_dedup_skipped"):
            print(f"    Dedup skipped: {stats['ingest_dedup_skipped']}")
        if stats.get("ingest_failed"):
            print(f"    Failed: {stats['ingest_failed']}")
    elif args.traces_input and args.traces_input.exists():
        print(f">>> Loading existing traces from {args.traces_input}...")
        for _, rec in _iter_jsonl(args.traces_input):
            traces.append(rec)
        print(f"    Loaded: {len(traces)} traces")

    if not traces:
        print("ERROR: No traces to process. Provide --raw-input for ingest or --traces-input.")
        sys.exit(1)

    # Step 2: Backfill
    if "backfill" in ops:
        print(">>> Running backfill...")
        traces = op_backfill(traces, injection_patterns, stats)
        print(f"    Backfilled: {stats['backfill_filled']} traces")

    # Step 3: Removal (harmful -> benign)
    removal_traces = []
    if "removal" in ops:
        print(">>> Running removal...")
        removal_traces = op_removal(traces, stats)
        print(f"    Created: {stats['removal_created']} benign traces from injection removal")

    # Step 4: Transplant (benign -> harmful)
    transplant_traces = []
    if "transplant" in ops:
        print(">>> Running transplant...")
        transplant_traces = op_transplant(traces, rng, stats)
        if args.max_transplant and len(transplant_traces) > args.max_transplant:
            rng.shuffle(transplant_traces)
            transplant_traces = transplant_traces[:args.max_transplant]
            print(f"    Capped transplant to {args.max_transplant}")
        print(f"    Created: {stats['transplant_created']} harmful traces from injection transplant")

    # Step 4b: Contrastive pair mapping (must run after removal)
    if "contrastive" in ops:
        combined_for_pairs = traces + removal_traces + transplant_traces
        print(">>> Running contrastive pair mapping...")
        contrastive_pairs = op_contrastive(combined_for_pairs, stats)
        print(f"    Created: {stats['contrastive_pairs']} contrastive pairs")
        if contrastive_pairs and args.contrastive_output:
            args.contrastive_output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.contrastive_output, "w") as f:
                for pair in contrastive_pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            print(f"    Written to: {args.contrastive_output}")
        elif contrastive_pairs and not args.contrastive_output:
            print("    WARNING: --contrastive-output not specified, pairs not written")

    # Combine all traces
    all_traces = traces + removal_traces + transplant_traces

    # Step 5: Taxonomy
    if "taxonomy" in ops:
        print(">>> Running taxonomy tagging...")
        all_traces = op_taxonomy(all_traces, stats)
        print(f"    Tagged: {stats['taxonomy_tagged']} traces with OWASP categories")

    # Deduplicate by trace ID
    seen_ids: Set[str] = set()
    deduped = []
    for t in all_traces:
        tid = t.get("id", "")
        if tid in seen_ids:
            continue
        seen_ids.add(tid)
        deduped.append(t)

    stats["output_total"] = len(deduped)

    # Write output
    _write_jsonl(deduped, args.output)

    # Print stats
    if args.stats:
        print()
        print("=" * 60)
        print("AUGMENTATION STATS")
        print("=" * 60)

        # Category breakdown
        cats = Counter()
        provenances = Counter()
        models = Counter()
        suites = Counter()
        taxonomy_counts = Counter()

        for t in deduped:
            cat = (t.get("labels") or {}).get("category", "unknown")
            cats[cat] += 1

            sf = (t.get("raw_metadata") or {}).get("source_fields") or {}
            prov = sf.get("augmentation_provenance", "original")
            provenances[prov] += 1

            model = (t.get("source") or {}).get("model_id", "unknown")
            models[model] += 1

            suite = (t.get("source") or {}).get("subset", "unknown")
            suites[suite] += 1

            for tag in sf.get("taxonomy_tags", []):
                taxonomy_counts[tag] += 1

        print(f"\nTotal output: {len(deduped)} traces")
        print(f"\nBy category:")
        for k, v in sorted(cats.items()):
            print(f"  {k}: {v}")
        print(f"\nBy provenance:")
        for k, v in sorted(provenances.items()):
            print(f"  {k}: {v}")
        print(f"\nBy model ({len(models)} models):")
        for k, v in models.most_common():
            print(f"  {k}: {v}")
        print(f"\nBy suite:")
        for k, v in sorted(suites.items()):
            print(f"  {k}: {v}")
        if taxonomy_counts:
            print(f"\nOWASP taxonomy tags:")
            for tag, count in sorted(taxonomy_counts.items()):
                desc = OWASP_TAGS.get(tag, "")
                print(f"  {tag} ({desc}): {count}")

    print(f"\nDone. Output: {args.output} ({len(deduped)} traces)")


if __name__ == "__main__":
    main()
