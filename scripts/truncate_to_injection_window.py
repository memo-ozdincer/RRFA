#!/usr/bin/env python3
"""Truncate multi-turn traces to the injection decision window.

For traces with injection_char_span:
  Keep messages[0 .. first_post_injection_assistant] inclusive.
  Drops all follow-up tool calls and responses after the critical decision.

For benign traces (no injection):
  Keep messages[0 .. first_assistant_after_first_tool] inclusive.
  One user→assistant→tool→assistant cycle.

Harmful traces without injection_char_span are dropped (can't mask properly).

Usage:
  python scripts/truncate_to_injection_window.py \
      --input data/traces/agentdojo_augmented.jsonl \
      --output data/traces/agentdojo_truncated.jsonl
"""

import argparse
import json
import sys
from collections import Counter


def truncate_trace(trace: dict) -> dict | None:
    """Truncate a trace to the injection decision window. Returns None to drop."""
    messages = trace.get("messages", [])
    category = trace.get("labels", {}).get("category", "")
    span = (trace.get("signal_hints") or {}).get("injection_char_span")

    if span:
        # Has injection: keep up to first post-injection assistant
        inj_idx = span["message_index"]
        for j in range(inj_idx + 1, len(messages)):
            if messages[j]["role"] == "assistant":
                trace["messages"] = messages[: j + 1]
                return trace
        # No assistant after injection — keep as-is (rare edge case)
        return trace

    elif category == "harmful":
        # Harmful without span — can't mask properly, drop
        return None

    else:
        # Benign or resisted without span: keep first tool→assistant cycle
        for j in range(1, len(messages)):
            if messages[j]["role"] == "tool":
                # Find next assistant after this tool
                for k in range(j + 1, len(messages)):
                    if messages[k]["role"] == "assistant":
                        trace["messages"] = messages[: k + 1]
                        return trace
                # No assistant after tool — keep up to tool
                trace["messages"] = messages[: j + 1]
                return trace
        # No tool message at all — keep as-is
        return trace


def main():
    parser = argparse.ArgumentParser(description="Truncate traces to injection window")
    parser.add_argument("--input", required=True, help="Input traces JSONL")
    parser.add_argument("--output", required=True, help="Output truncated JSONL")
    args = parser.parse_args()

    stats = Counter()
    output_traces = []

    with open(args.input) as f:
        for line in f:
            trace = json.loads(line)
            orig_msgs = len(trace.get("messages", []))
            result = truncate_trace(trace)

            if result is None:
                stats["dropped"] += 1
                continue

            new_msgs = len(result["messages"])
            cat = result.get("labels", {}).get("category", "?")
            stats[f"kept_{cat}"] += 1
            if new_msgs < orig_msgs:
                stats["truncated"] += 1
            stats["total_msgs_before"] += orig_msgs
            stats["total_msgs_after"] += new_msgs
            output_traces.append(result)

    with open(args.output, "w") as f:
        for trace in output_traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    print(f"Input:     {sum(v for k, v in stats.items() if k.startswith('kept_')) + stats['dropped']} traces")
    print(f"Output:    {len(output_traces)} traces")
    print(f"Dropped:   {stats['dropped']} (harmful without injection span)")
    print(f"Truncated: {stats['truncated']} traces")
    if stats["total_msgs_before"]:
        print(f"Messages:  {stats['total_msgs_before']} → {stats['total_msgs_after']} "
              f"({100*stats['total_msgs_after']/stats['total_msgs_before']:.1f}%)")
    print(f"\nBy category:")
    for k, v in sorted(stats.items()):
        if k.startswith("kept_"):
            print(f"  {k[5:]}: {v}")


if __name__ == "__main__":
    main()
