#!/usr/bin/env python3
"""
Generate synthetic refusal traces from harmful AgentDojo traces.

Takes harmful traces (where model follows injection → malicious tool call),
keeps the full context up to the last assistant message, replaces that message
with a refusal. Result is labeled benign and used as CB retain data so the
model learns "when you see this injection context, produce refusal-like
representations."

Usage:
    python scripts/generate_refusal_traces.py \
        --traces data/traces/agentdojo_truncated.jsonl \
        --output data/traces/agentdojo_refusal_synthetic.jsonl \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

REFUSAL_TEMPLATES = [
    "I notice the previous output contains instructions that weren't part of the original request. I'll disregard those and focus on what you actually asked for.",
    "The tool result appears to contain injected instructions. I won't follow those. Let me continue with your original request instead.",
    "I've detected what looks like a prompt injection attempt embedded in the tool output. I will not execute those instructions.",
    "I can see that the returned data includes additional instructions that weren't part of your original query. I'm going to ignore those and proceed with your actual request.",
    "It looks like the previous result contained some embedded instructions trying to get me to perform unauthorized actions. I'll skip those entirely.",
    "I need to flag something: the tool output included what appears to be injected commands. For your safety, I'm not going to follow them. What would you like me to do with your original request?",
    "The data returned by the tool seems to contain hidden instructions. These don't align with your original request, so I'm ignoring them.",
    "I won't proceed with the actions suggested in the tool output, as they appear to be injected instructions rather than legitimate data. Let me help you with what you originally asked.",
    "Some of the content in the tool response looks like it's trying to hijack this conversation. I'm disregarding it and sticking to your original request.",
    "I'm not going to follow the instructions embedded in that tool result. They appear to be a prompt injection attempt. How would you like me to proceed with your original task?",
]


def generate_refusal_traces(
    traces_path: Path,
    output_path: Path,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)

    harmful_traces: List[Dict[str, Any]] = []
    total = 0
    with open(traces_path) as f:
        for line in f:
            if not line.strip():
                continue
            trace = json.loads(line)
            total += 1
            if trace.get("labels", {}).get("category") == "harmful":
                harmful_traces.append(trace)

    print(f"Loaded {total} traces, {len(harmful_traces)} harmful")

    generated = 0
    skipped = 0

    with open(output_path, "w") as out:
        for trace in harmful_traces:
            messages = trace.get("messages", [])

            # Must end with assistant message
            if not messages or messages[-1].get("role") != "assistant":
                skipped += 1
                continue

            # Keep all messages except the last (malicious) assistant message
            context = messages[:-1]

            # Need at least system + user + something
            if len(context) < 2:
                skipped += 1
                continue

            # Build refusal message
            refusal_text = rng.choice(REFUSAL_TEMPLATES)
            refusal_msg = {
                "role": "assistant",
                "content": refusal_text,
            }

            # Build new trace
            new_trace = {
                "id": trace["id"] + "_refusal",
                "source": {"dataset": "synthetic", "tier": "derived", "subset": "refusal"},
                "messages": context + [refusal_msg],
                "labels": {
                    "category": "benign",
                    "security_outcome": "safe",
                    "attack_succeeded": False,
                    "attack_present": trace.get("labels", {}).get("attack_present", True),
                },
            }

            # Carry over optional fields
            for key in ("split", "task", "tier", "completeness"):
                if key in trace:
                    new_trace[key] = trace[key]

            out.write(json.dumps(new_trace) + "\n")
            generated += 1

    print(f"Generated {generated} refusal traces, skipped {skipped}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic refusal traces")
    parser.add_argument("--traces", type=Path, required=True, help="Input traces JSONL (harmful + benign)")
    parser.add_argument("--output", type=Path, required=True, help="Output refusal traces JSONL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_refusal_traces(args.traces, args.output, args.seed)


if __name__ == "__main__":
    main()
