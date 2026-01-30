#!/usr/bin/env python3
"""Split AgentDojo traces/renders/lossmasks into harmful and benign files.

AgentDojo data has both harmful and benign samples in the same file,
distinguished by labels.is_harmful. This script splits them so they
can be used with train_schema.py's mixed mode alongside Fujitsu DS/DR.
"""

import argparse
import json
from pathlib import Path


def split_jsonl(input_path: Path, harmful_out: Path, benign_out: Path, label_field: str = "labels.is_harmful"):
    """Split a JSONL file into harmful and benign based on a label field."""
    harmful_count = 0
    benign_count = 0

    with open(input_path) as f_in, \
         open(harmful_out, 'w') as f_harmful, \
         open(benign_out, 'w') as f_benign:

        for line in f_in:
            if not line.strip():
                continue

            row = json.loads(line)

            # Navigate nested field like "labels.is_harmful"
            is_harmful = row
            for key in label_field.split('.'):
                is_harmful = is_harmful.get(key, {}) if isinstance(is_harmful, dict) else None

            # Default to False if field not found
            if is_harmful is None:
                is_harmful = False

            if is_harmful:
                f_harmful.write(line)
                harmful_count += 1
            else:
                f_benign.write(line)
                benign_count += 1

    return harmful_count, benign_count


def main():
    parser = argparse.ArgumentParser(description="Split AgentDojo data into harmful/benign")
    parser.add_argument("--traces", type=Path, help="Input traces JSONL")
    parser.add_argument("--renders", type=Path, help="Input renders JSONL")
    parser.add_argument("--lossmasks", type=Path, help="Input lossmasks JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--prefix", type=str, default="agentdojo", help="Output filename prefix")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file type
    for file_type, input_path in [("traces", args.traces), ("renders", args.renders), ("lossmasks", args.lossmasks)]:
        if input_path and input_path.exists():
            # For renders/lossmasks, we need to match by trace_id with the traces file
            # But if traces exist, we split based on that

            harmful_out = args.output_dir / f"{args.prefix}_{file_type}_harmful.jsonl"
            benign_out = args.output_dir / f"{args.prefix}_{file_type}_benign.jsonl"

            print(f"Splitting {file_type}: {input_path}")
            harmful, benign = split_jsonl(input_path, harmful_out, benign_out)
            print(f"  Harmful: {harmful} -> {harmful_out}")
            print(f"  Benign: {benign} -> {benign_out}")

    print("\nDone! Use these files with --mode mixed:")
    print(f"  --harmful-renders {args.output_dir}/{args.prefix}_renders_harmful.jsonl")
    print(f"  --harmful-lossmasks {args.output_dir}/{args.prefix}_lossmasks_harmful.jsonl")
    print(f"  --benign-renders {args.output_dir}/{args.prefix}_renders_benign.jsonl")
    print(f"  --benign-lossmasks {args.output_dir}/{args.prefix}_lossmasks_benign.jsonl")


if __name__ == "__main__":
    main()
