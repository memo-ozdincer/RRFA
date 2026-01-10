#!/usr/bin/env python3
"""Quick script to analyze training data structure."""
import json
import sys

with open("cb_training_batches.jsonl", "r") as f:
    batch = json.loads(f.readline())

print("BATCH KEYS:", list(batch.keys()))
print()

if "harmful" in batch and batch["harmful"]:
    s = batch["harmful"][0]
    print("HARMFUL SAMPLE 0 KEYS:", list(s.keys()))
    print()
    print("has_tool_calls:", s.get("has_tool_calls"))
    print("is_agentic:", s.get("is_agentic"))
    print("tool_attack type:", type(s.get("tool_attack")).__name__)
    if s.get("tool_attack"):
        ta = s["tool_attack"]
        if isinstance(ta, dict):
            print("  tool_attack keys:", list(ta.keys()))
            print("  combined_query (100c):", repr(ta.get("combined_query", "")[:100]))
    print()
    print("assistant_raw (200 chars):", repr(s.get("assistant_raw", "")[:200]))
    print()
    print("harmful_completion (200 chars):", repr(s.get("harmful_completion", "")[:200]))
    print()
    print("user_prompt (200 chars):", repr(s.get("user_prompt", "")[:200]))
    print()
    if "messages" in s:
        print("messages count:", len(s["messages"]))
        for i, m in enumerate(s["messages"][:3]):
            print(f"  msg[{i}] role={m.get('role')}, content[:100]={repr(m.get('content', '')[:100])}")

# Count different sample types
print("\n" + "="*60)
print("SAMPLE TYPE ANALYSIS ACROSS ALL BATCHES")
print("="*60)

stats = {
    "total": 0,
    "has_tool_calls": 0,
    "has_is_agentic": 0,
    "has_tool_attack": 0,
    "has_python_tag_in_completion": 0,
    "has_messages": 0,
    "has_assistant_raw": 0,
}

with open("cb_training_batches.jsonl", "r") as f:
    for line in f:
        if not line.strip():
            continue
        batch = json.loads(line)
        for s in batch.get("harmful", []) + batch.get("benign", []):
            stats["total"] += 1
            if s.get("has_tool_calls"):
                stats["has_tool_calls"] += 1
            if s.get("is_agentic"):
                stats["has_is_agentic"] += 1
            if s.get("tool_attack"):
                stats["has_tool_attack"] += 1
            completion = s.get("harmful_completion", "") or s.get("benign_completion", "") or s.get("assistant_raw", "")
            if "<|python_tag|>" in completion:
                stats["has_python_tag_in_completion"] += 1
            if s.get("messages"):
                stats["has_messages"] += 1
            if s.get("assistant_raw"):
                stats["has_assistant_raw"] += 1

for k, v in stats.items():
    pct = 100 * v / max(1, stats["total"]) if k != "total" else ""
    print(f"  {k}: {v}" + (f" ({pct:.1f}%)" if pct else ""))
