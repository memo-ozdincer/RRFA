#!/usr/bin/env python3
"""
Debug script to inspect what the model is actually generating.
Helps diagnose why tool calls are not being parsed.

Usage:
    python scripts/cb_data_generation/debug_tool_calling.py \
        --model mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated \
        --limit 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Debug tool calling format")
    parser.add_argument("--model", required=True, help="Model ID")
    parser.add_argument("--tool-schema", default="configs/tool_schemas/b4_standard_v1.json")
    parser.add_argument("--b4-data", default="data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl")
    parser.add_argument("--limit", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    
    args = parser.parse_args()
    
    # Load tool schema
    with open(args.tool_schema) as f:
        schema = json.load(f)
    
    tools = schema.get("tools", [])
    system_prompt = schema.get("system_prompt", "You are a helpful assistant with access to tools.")
    
    logger.info(f"Loaded {len(tools)} tools from schema")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Load sample data
    logger.info(f"Loading samples from: {args.b4_data}")
    samples = []
    with open(args.b4_data) as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            samples.append(json.loads(line))
    
    logger.info(f"Loaded {len(samples)} samples")
    
    # Test prompt formatting
    logger.info("\n" + "=" * 80)
    logger.info("TESTING PROMPT FORMATTING")
    logger.info("=" * 80)
    
    for i, sample in enumerate(samples[:2], 1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["combined_query"]},
        ]
        
        logger.info(f"\n--- Sample {i} ---")
        logger.info(f"Expected tool: {sample['expected_tool']}")
        logger.info(f"Attack simulates: {sample['simulated_tool']}")
        
        # Try formatting with tools
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            logger.info(f"\n✓ Chat template WITH tools succeeded")
            logger.info(f"Prompt length: {len(prompt)} chars")
            
            # Show first and last 500 chars
            if len(prompt) > 1000:
                logger.info(f"\nFirst 500 chars:\n{prompt[:500]}")
                logger.info(f"\n[...{len(prompt)-1000} chars omitted...]\n")
                logger.info(f"Last 500 chars:\n{prompt[-500:]}")
            else:
                logger.info(f"\nFull prompt:\n{prompt}")
                
        except TypeError as e:
            logger.warning(f"\n✗ Chat template WITH tools failed: {e}")
            
            # Try without tools
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                logger.info(f"\n✓ Chat template WITHOUT tools succeeded")
                logger.info(f"Prompt length: {len(prompt)} chars")
                logger.info(f"\nPrompt:\n{prompt}")
                
            except Exception as e2:
                logger.error(f"✗ Chat template WITHOUT tools also failed: {e2}")
    
    # Now test generation
    if args.backend == "vllm":
        logger.info("\n" + "=" * 80)
        logger.info("TESTING vLLM GENERATION")
        logger.info("=" * 80)
        
        try:
            from vllm import LLM, SamplingParams
            
            logger.info(f"Loading vLLM model: {args.model}")
            llm = LLM(
                model=args.model,
                tensor_parallel_size=1,
                dtype="bfloat16",
                trust_remote_code=True,
            )
            
            # Generate for first sample
            sample = samples[0]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sample["combined_query"]},
            ]
            
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=256,
                top_p=0.95,
            )
            
            logger.info("Generating response...")
            outputs = llm.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text
            
            logger.info("\n" + "=" * 80)
            logger.info("MODEL OUTPUT")
            logger.info("=" * 80)
            logger.info(response)
            logger.info("=" * 80)
            
            # Check for tool call markers
            markers = [
                "<|python_tag|>",
                "<|eom_id|>",
                "<|eot_id|>",
                "```",
                "Action:",
                "ToolCall:",
            ]
            
            logger.info("\nMarker detection:")
            for marker in markers:
                if marker in response:
                    logger.info(f"  ✓ Found: {marker}")
                else:
                    logger.info(f"  ✗ Missing: {marker}")
            
            # Try parsing
            logger.info("\nAttempting to parse tool call...")
            from generate_ds_mvp import extract_tool_call
            
            tool_call = extract_tool_call(response)
            if tool_call:
                logger.info(f"✓ Parsed tool call: {json.dumps(tool_call, indent=2)}")
            else:
                logger.error("✗ Could not parse any tool call from response")
                
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        logger.info("Use --backend vllm to test actual generation")


if __name__ == "__main__":
    main()
