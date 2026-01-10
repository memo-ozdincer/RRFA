#!/usr/bin/env python3
"""
Rebuild CB Training Batches with Proper Llama 3.1 Format

This script fixes the critical issue where training data was not in the correct
Llama 3.1 chat template format, causing:
1. Assistant boundary masking to fail (couldn't find <|start_header_id|>assistant)
2. Missing <|eot_id|> terminators on completions
3. Loss being applied to full sequence instead of completion-only

The fix:
1. For each sample, render using the ACTUAL Llama 3.1 tokenizer's chat template
2. Ensure tool calls use <|python_tag|> format
3. Add proper <|eot_id|> terminators

Usage:
    python scripts/cb_data_generation/rebuild_training_data.py \
        --input data/cb_mvp/cb_training_batches.jsonl \
        --output data/cb_mvp/cb_training_batches_fixed.jsonl \
        --model meta-llama/Llama-3.1-8B-Instruct

    # Or for the scratch directory:
    python scripts/cb_data_generation/rebuild_training_data.py \
        --input /scratch/memoozd/cb_mvp_data/cb_training_batches.jsonl \
        --output /scratch/memoozd/cb_mvp_data/cb_training_batches_fixed.jsonl
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Llama 3.1 Format Constants
# =============================================================================

# Special tokens for Llama 3.1
LLAMA_BOS = "<|begin_of_text|>"
LLAMA_HEADER_START = "<|start_header_id|>"
LLAMA_HEADER_END = "<|end_header_id|>"
LLAMA_EOT = "<|eot_id|>"
LLAMA_EOM = "<|eom_id|>"
LLAMA_PYTHON_TAG = "<|python_tag|>"


def format_llama31_message(role: str, content: str, is_tool_call: bool = False) -> str:
    """Format a single message in Llama 3.1 format."""
    header = f"{LLAMA_HEADER_START}{role}{LLAMA_HEADER_END}\n\n"
    
    if is_tool_call:
        # Tool call responses end with <|eom_id|> (expecting tool response)
        return f"{header}{content}{LLAMA_EOM}"
    else:
        # Regular messages end with <|eot_id|>
        return f"{header}{content}{LLAMA_EOT}"


def format_tool_call_llama31(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Format a tool call in Llama 3.1 format."""
    # Llama 3.1 tool call format
    return f'{LLAMA_PYTHON_TAG}{{"name": "{tool_name}", "parameters": {json.dumps(arguments)}}}'


def extract_tool_call_from_text(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Extract tool name and arguments from text that might contain a tool call."""
    # Check for python_tag format
    if LLAMA_PYTHON_TAG in text:
        content = text.split(LLAMA_PYTHON_TAG, 1)[1]
        # Remove end tokens
        for end_token in [LLAMA_EOM, LLAMA_EOT, "</s>"]:
            content = content.split(end_token)[0]
        content = content.strip()
        
        # Parse JSON
        try:
            data = json.loads(content)
            name = data.get("name")
            params = data.get("parameters", {})
            if name:
                return name, params
        except json.JSONDecodeError:
            pass
    
    # Check for function call syntax: tool_name({...})
    match = re.search(r'(\w+)\s*\(\s*(\{[^}]+\})\s*\)', text)
    if match:
        try:
            return match.group(1), json.loads(match.group(2))
        except json.JSONDecodeError:
            pass
    
    return None


def render_sample_llama31(
    sample: Dict[str, Any],
    is_harmful: bool,
    system_prompt: str = "You are a helpful assistant with access to tools.",
    tools_json: Optional[str] = None,
) -> str:
    """
    Render a sample in proper Llama 3.1 chat format.
    
    Returns the full text with:
    - System message with tool definitions
    - User message
    - Assistant response with tool call (if applicable)
    - Proper <|eot_id|>/<|eom_id|> terminators
    """
    parts = [LLAMA_BOS]
    
    # System message with tools
    full_system = system_prompt
    if tools_json:
        full_system += f"\n\nYou have access to the following tools:\n\n{tools_json}"
    parts.append(format_llama31_message("system", full_system))
    
    # Get user prompt
    user_prompt = sample.get("user_prompt", "")
    if not user_prompt:
        # Try combined_query from tool_attack
        tool_attack = sample.get("tool_attack", {})
        user_prompt = tool_attack.get("combined_query", "")
    if not user_prompt:
        # Last resort: messages
        messages = sample.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                user_prompt = msg.get("content", "")
                break
    
    if not user_prompt:
        logger.warning(f"No user prompt found in sample {sample.get('id', 'unknown')}")
        return ""
    
    parts.append(format_llama31_message("user", user_prompt))
    
    # Get assistant completion
    completion = sample.get("harmful_completion" if is_harmful else "benign_completion", "")
    if not completion:
        completion = sample.get("assistant_raw", "")
    
    if not completion:
        logger.warning(f"No completion found in sample {sample.get('id', 'unknown')}")
        return ""
    
    # Check if completion contains a tool call
    tool_call = extract_tool_call_from_text(completion)
    
    if tool_call:
        tool_name, tool_args = tool_call
        # Clean completion text (remove existing tool call syntax if any)
        clean_content = completion.split(LLAMA_PYTHON_TAG)[0].strip()
        if not clean_content:
            clean_content = ""
        
        # Format as tool call
        tool_call_str = format_tool_call_llama31(tool_name, tool_args)
        full_completion = f"{clean_content}\n\n{tool_call_str}" if clean_content else tool_call_str
        
        # Tool calls end with <|eom_id|>
        parts.append(format_llama31_message("assistant", full_completion, is_tool_call=True))
    else:
        # Regular completion with <|eot_id|>
        parts.append(format_llama31_message("assistant", completion))
    
    return "".join(parts)


def rebuild_batch(
    batch: Dict[str, Any],
    system_prompt: str,
    tools_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Rebuild a batch with proper Llama 3.1 formatted text fields."""
    new_batch = {
        "id": batch.get("id", "unknown"),
        "harmful": [],
        "benign": [],
    }
    
    # Process harmful samples
    for sample in batch.get("harmful", []):
        text = render_sample_llama31(sample, is_harmful=True, system_prompt=system_prompt, tools_json=tools_json)
        if text:
            new_sample = sample.copy()
            new_sample["text"] = text
            new_sample["is_agentic"] = sample.get("is_agentic", sample.get("has_tool_calls", False))
            new_batch["harmful"].append(new_sample)
    
    # Process benign samples
    for sample in batch.get("benign", []):
        text = render_sample_llama31(sample, is_harmful=False, system_prompt=system_prompt, tools_json=tools_json)
        if text:
            new_sample = sample.copy()
            new_sample["text"] = text
            new_sample["is_agentic"] = sample.get("is_agentic", sample.get("has_tool_calls", False))
            new_batch["benign"].append(new_sample)
    
    return new_batch


def load_tool_schema(schema_path: Optional[Path]) -> Tuple[str, Optional[str]]:
    """Load tool schema and return system prompt and tools JSON."""
    default_prompt = "You are a helpful assistant with access to tools."
    
    if not schema_path or not schema_path.exists():
        return default_prompt, None
    
    with open(schema_path, "r") as f:
        schema = json.load(f)
    
    system_prompt = schema.get("system_prompt", default_prompt)
    tools = schema.get("tools", [])
    tools_json = json.dumps(tools, indent=2) if tools else None
    
    return system_prompt, tools_json


def validate_sample(text: str) -> Dict[str, bool]:
    """Validate a rendered sample for proper Llama 3.1 format."""
    return {
        "has_bos": LLAMA_BOS in text,
        "has_system_header": f"{LLAMA_HEADER_START}system{LLAMA_HEADER_END}" in text,
        "has_user_header": f"{LLAMA_HEADER_START}user{LLAMA_HEADER_END}" in text,
        "has_assistant_header": f"{LLAMA_HEADER_START}assistant{LLAMA_HEADER_END}" in text,
        "has_terminator": LLAMA_EOT in text or LLAMA_EOM in text,
        "has_tool_call": LLAMA_PYTHON_TAG in text,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild CB training batches with proper Llama 3.1 format",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input cb_training_batches.jsonl file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "--tool-schema",
        type=Path,
        default=None,
        help="Path to tool schema JSON (optional)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate input file, don't rebuild",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of batches to process",
    )
    
    args = parser.parse_args()
    
    # Load tool schema
    system_prompt, tools_json = load_tool_schema(args.tool_schema)
    logger.info(f"System prompt length: {len(system_prompt)}")
    logger.info(f"Tools JSON: {'loaded' if tools_json else 'none'}")
    
    # Load input batches
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    batches = []
    with open(args.input, "r") as f:
        for line in f:
            if line.strip():
                batches.append(json.loads(line))
    
    logger.info(f"Loaded {len(batches)} batches from {args.input}")
    
    if args.limit:
        batches = batches[:args.limit]
        logger.info(f"Limited to {len(batches)} batches")
    
    # Validation stats
    stats = {
        "total_harmful": 0,
        "total_benign": 0,
        "has_text_before": 0,
        "has_text_after": 0,
        "has_terminator": 0,
        "has_tool_call": 0,
        "has_assistant_header": 0,
    }
    
    # Process batches
    output_batches = []
    for batch in batches:
        # Count before
        for sample in batch.get("harmful", []):
            stats["total_harmful"] += 1
            if sample.get("text"):
                stats["has_text_before"] += 1
        for sample in batch.get("benign", []):
            stats["total_benign"] += 1
            if sample.get("text"):
                stats["has_text_before"] += 1
        
        if args.validate_only:
            continue
        
        # Rebuild
        new_batch = rebuild_batch(batch, system_prompt, tools_json)
        
        # Validate and count after
        for sample in new_batch.get("harmful", []):
            text = sample.get("text", "")
            if text:
                stats["has_text_after"] += 1
                validation = validate_sample(text)
                if validation["has_terminator"]:
                    stats["has_terminator"] += 1
                if validation["has_tool_call"]:
                    stats["has_tool_call"] += 1
                if validation["has_assistant_header"]:
                    stats["has_assistant_header"] += 1
        
        for sample in new_batch.get("benign", []):
            text = sample.get("text", "")
            if text:
                stats["has_text_after"] += 1
                validation = validate_sample(text)
                if validation["has_terminator"]:
                    stats["has_terminator"] += 1
                if validation["has_assistant_header"]:
                    stats["has_assistant_header"] += 1
        
        output_batches.append(new_batch)
    
    # Print stats
    total_samples = stats["total_harmful"] + stats["total_benign"]
    logger.info("\n" + "=" * 60)
    logger.info("REBUILD STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total harmful samples: {stats['total_harmful']}")
    logger.info(f"Total benign samples: {stats['total_benign']}")
    logger.info(f"Had text BEFORE: {stats['has_text_before']} / {total_samples} ({100*stats['has_text_before']/max(1,total_samples):.1f}%)")
    
    if not args.validate_only:
        logger.info(f"Has text AFTER: {stats['has_text_after']} / {total_samples} ({100*stats['has_text_after']/max(1,total_samples):.1f}%)")
        logger.info(f"Has terminator: {stats['has_terminator']} / {stats['has_text_after']} ({100*stats['has_terminator']/max(1,stats['has_text_after']):.1f}%)")
        logger.info(f"Has assistant header: {stats['has_assistant_header']} / {stats['has_text_after']} ({100*stats['has_assistant_header']/max(1,stats['has_text_after']):.1f}%)")
        logger.info(f"Has tool call: {stats['has_tool_call']} / {stats['has_text_after']} ({100*stats['has_tool_call']/max(1,stats['has_text_after']):.1f}%)")
        
        # Write output
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for batch in output_batches:
                f.write(json.dumps(batch) + "\n")
        
        logger.info(f"\nWrote {len(output_batches)} batches to {args.output}")
        
        # Print example
        if output_batches and output_batches[0].get("harmful"):
            example = output_batches[0]["harmful"][0]
            logger.info("\n" + "=" * 60)
            logger.info("EXAMPLE RENDERED TEXT (first 1000 chars)")
            logger.info("=" * 60)
            logger.info(example.get("text", "")[:1000])
    
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
