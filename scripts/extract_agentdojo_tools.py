#!/usr/bin/env python3
"""
Extract tool definitions from AgentDojo trace data.

Reads agentdojo_augmented.jsonl (or any trace JSONL) and builds an OpenAI-style
tool schema JSON by:
  1. Collecting all unique tool names from tool_calls and tool-role messages
  2. Inferring parameter names and types from observed arguments
  3. Outputting a schema compatible with load_tool_schema() in eval.py

Usage:
    python scripts/extract_agentdojo_tools.py \
        --traces data/traces/agentdojo_augmented.jsonl \
        --output configs/tool_schemas/agentdojo_v1.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set


def _infer_json_type(value: Any) -> str:
    """Map Python type to JSON Schema type."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def extract_tools(traces_path: Path) -> List[Dict[str, Any]]:
    """Extract tool definitions from trace JSONL."""
    # tool_name -> param_name -> set of observed types
    tool_params: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    tool_names: Set[str] = set()

    with open(traces_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages", [])

            for msg in messages:
                # Collect from tool_calls in assistant messages
                for tc in msg.get("tool_calls", []) or []:
                    fn = tc.get("function", {})
                    name = fn.get("name")
                    if not name:
                        continue
                    tool_names.add(name)

                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    if isinstance(args, dict):
                        for k, v in args.items():
                            tool_params[name][k].add(_infer_json_type(v))

                # Collect from tool-role messages (may reveal tools with no
                # observed call arguments)
                if msg.get("role") == "tool":
                    tn = msg.get("name")
                    if tn:
                        tool_names.add(tn)

    # Build OpenAI-style tool definitions
    tools: List[Dict[str, Any]] = []
    for name in sorted(tool_names):
        params = tool_params.get(name, {})
        properties: Dict[str, Any] = {}
        for param_name, type_set in sorted(params.items()):
            # Pick the most common/specific type; prefer string if ambiguous
            if len(type_set) == 1:
                json_type = next(iter(type_set))
            elif "string" in type_set:
                json_type = "string"
            else:
                json_type = sorted(type_set)[0]

            properties[param_name] = {
                "type": json_type,
                "description": param_name.replace("_", " ").capitalize(),
            }

        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": name.replace("_", " ").capitalize(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                },
            },
        }

        # Add required fields (all observed params are required by default)
        if properties:
            tool_def["function"]["parameters"]["required"] = sorted(properties.keys())

        tools.append(tool_def)

    return tools


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract tool definitions from AgentDojo trace data",
    )
    parser.add_argument(
        "--traces",
        type=Path,
        required=True,
        help="Input trace JSONL (e.g. data/traces/agentdojo_augmented.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output tool schema JSON (e.g. configs/tool_schemas/agentdojo_v1.json)",
    )
    args = parser.parse_args()

    if not args.traces.exists():
        print(f"ERROR: Traces file not found: {args.traces}", file=sys.stderr)
        sys.exit(1)

    tools = extract_tools(args.traces)

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$comment": "Auto-generated tool schema from AgentDojo traces — DO NOT EDIT manually",
        "version": "agentdojo_v1",
        "created": "2026-03-09",
        "description": "Tool definitions extracted from AgentDojo augmented traces",
        "tools": tools,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(schema, f, indent=2)
        f.write("\n")

    print(f"Extracted {len(tools)} tool definitions -> {args.output}")

    # Validation: verify all trace tool names are covered
    trace_names: Set[str] = set()
    with open(args.traces) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for msg in record.get("messages", []):
                for tc in msg.get("tool_calls", []) or []:
                    name = (tc.get("function") or {}).get("name")
                    if name:
                        trace_names.add(name)
                if msg.get("role") == "tool" and msg.get("name"):
                    trace_names.add(msg["name"])

    schema_names = {t["function"]["name"] for t in tools}
    missing = trace_names - schema_names
    if missing:
        print(f"WARNING: {len(missing)} tools in traces but NOT in schema: {missing}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Validation OK: all {len(trace_names)} trace tool names covered")


if __name__ == "__main__":
    main()
