#!/usr/bin/env python3
"""
DIAGNOSTIC: Inspect training batches to see actual data content.

This script prints actual samples from the training batches to diagnose
masking and data quality issues.

Usage:
    python scripts/circuit_breakers/inspect_training_batches.py \
        --batches /scratch/memoozd/cb_stage2_data/stage2/train_batches.jsonl \
        --n 3
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description="Inspect training batches")
    parser.add_argument("--batches", type=Path, required=True, help="Path to train_batches.jsonl")
    parser.add_argument("--n", type=int, default=3, help="Number of batches to inspect")
    parser.add_argument("--tokenize", action="store_true", help="Also show tokenization")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                        help="Model for tokenizer (only if --tokenize)")
    args = parser.parse_args()

    if not args.batches.exists():
        print(f"ERROR: File not found: {args.batches}")
        return 1

    # Load batches
    batches = []
    with open(args.batches, "r") as f:
        for line in f:
            if line.strip():
                batches.append(json.loads(line))

    print("=" * 80)
    print("TRAINING BATCH INSPECTION")
    print("=" * 80)
    print(f"File: {args.batches}")
    print(f"Total batches: {len(batches)}")
    print()

    # Aggregate stats
    total_harmful = 0
    harmful_with_python_tag = 0
    harmful_with_messages = 0
    harmful_with_assistant_raw = 0

    for batch in batches:
        for sample in batch.get("harmful", []):
            total_harmful += 1
            assistant_raw = sample.get("assistant_raw", "")
            messages = sample.get("messages", [])
            
            if "<|python_tag|>" in assistant_raw:
                harmful_with_python_tag += 1
            if messages:
                harmful_with_messages += 1
            if assistant_raw:
                harmful_with_assistant_raw += 1

    print("AGGREGATE STATS:")
    print(f"  Total harmful samples: {total_harmful}")
    print(f"  With messages[]: {harmful_with_messages} ({100*harmful_with_messages/max(1,total_harmful):.1f}%)")
    print(f"  With assistant_raw: {harmful_with_assistant_raw} ({100*harmful_with_assistant_raw/max(1,total_harmful):.1f}%)")
    print(f"  With <|python_tag|> in assistant_raw: {harmful_with_python_tag} ({100*harmful_with_python_tag/max(1,total_harmful):.1f}%)")
    print()

    # Load tokenizer if requested
    tokenizer = None
    if args.tokenize:
        try:
            from transformers import AutoTokenizer
            import os
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            tokenizer = AutoTokenizer.from_pretrained(
                args.model, token=hf_token, local_files_only=offline, trust_remote_code=True
            )
            print(f"Loaded tokenizer: {args.model}")
            python_tag_id = tokenizer.convert_tokens_to_ids("<|python_tag|>")
            end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
            print(f"  <|python_tag|> token ID: {python_tag_id}")
            print(f"  <|end_header_id|> token ID: {end_header_id}")
            print()
        except Exception as e:
            print(f"Could not load tokenizer: {e}")
            tokenizer = None

    # Inspect individual batches
    for i, batch in enumerate(batches[:args.n]):
        print("=" * 80)
        print(f"BATCH {i} (batch_id={batch.get('batch_id', '?')})")
        print("=" * 80)

        # Harmful samples
        harmful = batch.get("harmful", [])
        print(f"\n--- HARMFUL SAMPLES ({len(harmful)}) ---")
        
        for j, sample in enumerate(harmful):
            print(f"\n[Harmful {j}] ID: {sample.get('id', 'unknown')}")
            print(f"  Source: {sample.get('metadata', {}).get('source', '?')}")
            
            # Messages
            messages = sample.get("messages", [])
            print(f"  Messages: {len(messages)} turns")
            if messages:
                # Show first and last message
                if len(messages) > 0:
                    first = messages[0]
                    print(f"    [0] {first.get('role', '?')}: {str(first.get('content', ''))[:100]}...")
                if len(messages) > 1:
                    last = messages[-1]
                    print(f"    [{len(messages)-1}] {last.get('role', '?')}: {str(last.get('content', ''))[:100]}...")

            # Assistant raw
            assistant_raw = sample.get("assistant_raw", "")
            has_python_tag = "<|python_tag|>" in assistant_raw
            print(f"  assistant_raw length: {len(assistant_raw)} chars")
            print(f"  Has <|python_tag|>: {'YES ✓' if has_python_tag else 'NO ✗'}")
            
            if assistant_raw:
                # Show the actual content
                print(f"  assistant_raw content:")
                print(f"    >>> {assistant_raw[:300]}{'...' if len(assistant_raw) > 300 else ''}")
            
            # Labels
            labels = sample.get("labels", {})
            print(f"  Labels: split={labels.get('split')}, observed_tool={labels.get('observed_tool')}")

            # Tokenization analysis
            if tokenizer and has_python_tag:
                # Build the full text as the trainer would
                if sample.get("is_agentic") and messages:
                    # Append assistant_raw as final message
                    full_messages = list(messages) + [{"role": "assistant", "content": assistant_raw}]
                    try:
                        full_text = tokenizer.apply_chat_template(
                            full_messages, tokenize=False, add_generation_prompt=False
                        )
                    except:
                        full_text = assistant_raw
                else:
                    full_text = assistant_raw

                tokens = tokenizer.encode(full_text, add_special_tokens=True)
                print(f"  Tokenized length: {len(tokens)} tokens")
                
                # Find <|python_tag|> position
                python_tag_id = tokenizer.convert_tokens_to_ids("<|python_tag|>")
                if python_tag_id in tokens:
                    pos = tokens.index(python_tag_id)
                    print(f"  <|python_tag|> at position: {pos}/{len(tokens)} ({100*pos/len(tokens):.1f}% through)")
                    print(f"  Tokens after <|python_tag|>: {len(tokens) - pos}")
                    
                    # Show tokens around python_tag
                    start = max(0, pos - 3)
                    end = min(len(tokens), pos + 10)
                    context_tokens = tokens[start:end]
                    context_decoded = tokenizer.decode(context_tokens)
                    print(f"  Context around <|python_tag|>:")
                    print(f"    >>> {context_decoded}")

        # Benign samples (brief summary)
        benign = batch.get("benign", [])
        print(f"\n--- BENIGN SAMPLES ({len(benign)}) ---")
        for j, sample in enumerate(benign[:2]):  # Just show first 2
            print(f"  [Benign {j}] ID: {sample.get('id', 'unknown')[:50]}, Source: {sample.get('metadata', {}).get('source', '?')}")
            assistant_raw = sample.get("assistant_raw", "")
            has_python_tag = "<|python_tag|>" in assistant_raw
            print(f"    assistant_raw: {len(assistant_raw)} chars, <|python_tag|>: {'YES' if has_python_tag else 'NO'}")

    print()
    print("=" * 80)
    print("END OF INSPECTION")
    print("=" * 80)

    # Final diagnosis
    print()
    print("DIAGNOSIS:")
    if harmful_with_python_tag < total_harmful * 0.9:
        print(f"  ⚠️  Only {100*harmful_with_python_tag/max(1,total_harmful):.1f}% of harmful samples have <|python_tag|>")
        print("     This means the --require-tool-calls-harmful filter may not be working,")
        print("     or the source data doesn't have tool calls in assistant_raw.")
    else:
        print(f"  ✓  {100*harmful_with_python_tag/max(1,total_harmful):.1f}% of harmful samples have <|python_tag|>")

    return 0


if __name__ == "__main__":
    sys.exit(main())
