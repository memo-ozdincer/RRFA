#!/usr/bin/env python3
"""
Amplify injection signal in harmful traces by repeating injection text.

Takes truncated traces, finds <INFORMATION>...</INFORMATION> blocks in tool
outputs, and repeats them N times. This increases the context-level signal
(probe AUC 0.83 -> hopefully 0.90+) that CB uses to distinguish
injection-context from clean-context.

Only modifies harmful traces. Benign traces pass through unchanged.

Usage:
    python scripts/amplify_injections.py \
        --input data/traces/agentdojo_truncated.jsonl \
        --output data/traces/agentdojo_amplified.jsonl \
        --repeats 2 \
        --seed 42
"""

import argparse
import json
import re
from collections import Counter

INJECTION_RE = re.compile(r"(<INFORMATION>.*?</INFORMATION>)", re.DOTALL)


def amplify_trace(trace: dict, repeats: int) -> dict:
    """Amplify injection text in a trace by repeating injection blocks.

    For each tool message containing <INFORMATION>...</INFORMATION>, repeat
    the injection block `repeats` times (so total occurrences = repeats + 1).
    """
    messages = trace.get("messages", [])
    modified = False

    for msg in messages:
        if msg.get("role") != "tool":
            continue

        content = msg.get("content", "")
        if not content or not INJECTION_RE.search(content):
            continue

        # Repeat each injection block
        def repeat_match(m):
            return (m.group(1) + "\n") * repeats + m.group(1)

        new_content = INJECTION_RE.sub(repeat_match, content)
        if new_content != content:
            msg["content"] = new_content
            modified = True

    # Update injection_char_span if present (offsets shift after amplification)
    # We clear it since the span is no longer accurate; downstream code
    # handles missing spans gracefully for harmful traces.
    if modified and trace.get("signal_hints", {}).get("injection_char_span"):
        del trace["signal_hints"]["injection_char_span"]

    return trace


def main():
    parser = argparse.ArgumentParser(
        description="Amplify injection signal in harmful traces"
    )
    parser.add_argument("--input", required=True, help="Input traces JSONL")
    parser.add_argument("--output", required=True, help="Output amplified JSONL")
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Number of times to repeat injection text (default: 2)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (unused)")
    args = parser.parse_args()

    stats = Counter()

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            trace = json.loads(line)
            stats["total"] += 1

            category = trace.get("labels", {}).get("category", "")

            if category != "harmful":
                # Pass benign/resisted through unchanged
                fout.write(json.dumps(trace) + "\n")
                stats["passthrough"] += 1
                continue

            # Check if any tool message has injection
            has_injection = any(
                INJECTION_RE.search(m.get("content", ""))
                for m in trace.get("messages", [])
                if m.get("role") == "tool"
            )

            if not has_injection:
                fout.write(json.dumps(trace) + "\n")
                stats["no_injection"] += 1
                continue

            amplified = amplify_trace(trace, args.repeats)
            fout.write(json.dumps(amplified) + "\n")
            stats["amplified"] += 1

    print(f"Total: {stats['total']}")
    print(f"Amplified: {stats['amplified']} harmful traces ({args.repeats}x repeats)")
    print(f"Passthrough: {stats['passthrough']} (benign/resisted)")
    print(f"No injection found: {stats['no_injection']} (harmful without tags)")


if __name__ == "__main__":
    main()
