#!/usr/bin/env python3
"""
MVP Circuit Breaker Set (Ds) Generator - Stage 1

Generate the Circuit Breaker Set with BEHAVIORAL FILTERING.
Unlike the full generate_ds.py, this script ONLY includes samples where
the model actually exhibits the harmful behavior (attack succeeds).

Key Stage 1 Principles:
1. Only include samples where observed_tool == simulated_tool (attack succeeded)
2. Enforce minimum yield rate (>10%) or fail
3. Use frozen tool schema (b4_standard_v1)
4. Store full messages array, not just prompt string
5. Validate Llama 3.1 format compliance

Usage:
    python scripts/cb_data_generation/generate_ds.py \
        --b4-data data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl \
        --tool-schema configs/tool_schemas/b4_standard_v1.json \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output data/cb_mvp/ds_stage1.jsonl \
        --min-yield 0.10

    # Dry run to check yield rate
    python scripts/cb_data_generation/generate_ds.py --dry-run --limit 100
"""

# Set cache directories BEFORE any other imports (for vLLM/flashinfer)
import os

# Set cache directories to cb-scratch/cache/
cache_root = "/scratch/memoozd/cb-scratch/cache"

# CRITICAL FIX: Override HOME to trick flashinfer that defaults to ~/.cache
os.environ["HOME"] = cache_root 

os.makedirs(os.path.join(cache_root, "vllm"), exist_ok=True)
os.makedirs(os.path.join(cache_root, "flashinfer"), exist_ok=True)
os.makedirs(os.path.join(cache_root, "xdg_cache"), exist_ok=True) # Ensure XDG dir exists

os.environ["VLLM_USAGE_STATS_DIR"] = os.path.join(cache_root, "vllm")
os.environ["FLASHINFER_WORKSPACE_DIR"] = os.path.join(cache_root, "flashinfer")
os.environ["XDG_CACHE_HOME"] = os.path.join(cache_root, "xdg_cache") # Re-assert XDG cache

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


# =============================================================================
# HuggingFace Token Resolution
# =============================================================================

def resolve_hf_token() -> Optional[str]:
    """Resolve HuggingFace token from environment."""
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )


# =============================================================================
# Tool Schema Loading
# =============================================================================

def load_tool_schema(schema_path: Path) -> Dict[str, Any]:
    """Load frozen tool schema from JSON file."""
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    logger.info(f"Loaded tool schema: {schema.get('version', 'unknown')}")
    return schema


def get_tools_list(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tools list from schema."""
    return schema.get("tools", [])


def get_system_prompt(schema: Dict[str, Any]) -> str:
    """Extract system prompt from schema."""
    return schema.get("system_prompt", "You are a helpful assistant with access to tools.")


# =============================================================================
# Data Source Loaders
# =============================================================================

def load_fujitsu_b4(
    path: Path,
    limit: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Load Fujitsu B4 orchestrator attack records.
    
    Note: We load ALL records (not just success_only), because we need to
    test which ones actually succeed with our target model.
    """
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Extract fields
            benign_query = record.get("benign_query", "")
            malicious_injection = record.get("malicious_injection", "")
            combined_query = record.get("combined_query", "")
            expected_tool = record.get("expected_tool", "")
            simulated_tool = record.get("simulated_tool", "")
            
            # Skip if missing required fields
            if not combined_query or not expected_tool or not simulated_tool:
                continue
            
            yield {
                "id": record.get("record_id", f"fujitsu_b4_{count}"),
                "source": "fujitsu_b4",
                "benign_query": benign_query,
                "malicious_injection": malicious_injection,
                "combined_query": combined_query,
                "expected_tool": expected_tool,
                "simulated_tool": simulated_tool,
                "metadata": {
                    "attack_id": record.get("attack_id"),
                    "category": record.get("category"),
                    "subtype": record.get("subtype"),
                    "judge_note": record.get("judge_note"),
                    "original_success": record.get("success", False),
                },
            }
            
            count += 1
            if limit and count >= limit:
                return


# =============================================================================
# vLLM Backend (for 4x H100 tensor parallelism)
# =============================================================================

class VLLMBackend:
    """vLLM backend for high-throughput batched inference with tensor parallelism."""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 16384,
        dtype: str = "bfloat16",
    ):
        from vllm import LLM, SamplingParams
        
        hf_token = resolve_hf_token()
        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        
        # Resolve to local path if in offline mode
        if offline_mode:
            resolved_path = resolve_local_model_path(model_path, hf_token)
            if resolved_path != model_path:
                logger.info(f"  Resolved to local path: {resolved_path}")
            model_path = resolved_path
        
        logger.info(f"Loading vLLM model: {model_path}")
        logger.info(f"  Tensor parallel size: {tensor_parallel_size}")
        
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=dtype,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.SamplingParams = SamplingParams
    
    def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float = 0.95,
    ) -> List[str]:
        """Generate responses for a batch of prompts."""
        sampling_params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            skip_special_tokens=False,  # Keep special tokens for tool call parsing
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def format_prompt_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
    ) -> str:
        """Format messages with tools using tokenizer's chat template."""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            # Fallback without tools
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )


# =============================================================================
# Model Loading and Generation
# =============================================================================

def resolve_local_model_path(model_id: str, hf_token: Optional[str] = None) -> str:
    """
    Resolve a HuggingFace model ID to its local cache path using snapshot_download.
    
    When in offline mode, we need to pass the actual local path instead of
    a Hub model ID to avoid API calls during model_info() checks.
    """
    from huggingface_hub import snapshot_download
    
    # If it's already a local path, return as-is
    if os.path.isdir(model_id):
        return model_id
    
    # Use snapshot_download with local_files_only=True to get cached path
    try:
        local_path = snapshot_download(
            repo_id=model_id,
            local_files_only=True,
            token=hf_token,
        )
        return local_path
    except Exception as e:
        logger.warning(f"Could not resolve local path for {model_id}: {e}")
        return model_id


def load_model_and_tokenizer(
    model_path: str,
    device: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load model and tokenizer for generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    hf_token = resolve_hf_token()
    
    # Check if we're in offline mode (compute nodes have no internet)
    offline_mode = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    
    logger.info(f"Loading model: {model_path}")
    
    # In offline mode, resolve Hub ID to local cache path to avoid API calls
    if offline_mode:
        resolved_path = resolve_local_model_path(model_path, hf_token)
        if resolved_path != model_path:
            logger.info(f"  Resolved to local path: {resolved_path}")
        model_path = resolved_path
        logger.info("  (offline mode - using cached files only)")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=hf_token,
        trust_remote_code=True,
        local_files_only=offline_mode,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        token=hf_token,
        local_files_only=offline_mode,
    )
    model.eval()
    
    return model, tokenizer


def generate_with_tools(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """
    Generate response with tool-calling capability.
    
    Returns the raw assistant output (including special tokens).
    """
    # Apply chat template with tools
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            input_text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            # Fallback if tools not supported in template
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        # Manual formatting
        parts = []
        for msg in messages:
            parts.append(f"{msg['role']}: {msg['content']}")
        input_text = "\n".join(parts) + "\nassistant:"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode the full output including special tokens for analysis
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract just the new tokens (assistant response)
    input_len = inputs['input_ids'].shape[1]
    response_tokens = outputs[0][input_len:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=False)
    
    return response.strip()


def extract_tool_call(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract tool call from assistant response.
    
    Handles multiple formats:
    - Llama 3.1: <|python_tag|>tool_name(args)<|eom_id|>
    - JSON: {"name": "...", "parameters": {...}}
    - Function call: tool_name({"arg": "value"})
    """
    # Try Llama 3.1 format first
    if "<|python_tag|>" in response:
        content = response.split("<|python_tag|>", 1)[1]
        # Remove end tokens
        for end_token in ["<|eom_id|>", "<|eot_id|>", "</s>", "<|end_of_text|>"]:
            content = content.split(end_token)[0]
        content = content.strip()
        
        # Try function call syntax: tool_name(args)
        match = re.match(r'(\w+)\s*\((.*)\)', content, re.DOTALL)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2).strip()
            
            # Try to parse arguments
            try:
                # Handle keyword arguments: query='value'
                if '=' in args_str and not args_str.startswith('{'):
                    # Convert keyword args to dict
                    args = {}
                    for part in re.findall(r"(\w+)\s*=\s*['\"]([^'\"]*)['\"]", args_str):
                        args[part[0]] = part[1]
                    return {"name": tool_name, "parameters": args}
                else:
                    # Try JSON parsing
                    args = json.loads(args_str) if args_str else {}
                    return {"name": tool_name, "parameters": args}
            except (json.JSONDecodeError, ValueError):
                return {"name": tool_name, "parameters": {"raw": args_str}}
        
        # Try JSON format
        try:
            data = json.loads(content)
            name = data.get("name") or data.get("function", {}).get("name")
            params = data.get("parameters") or data.get("arguments") or {}
            if name:
                return {"name": name, "parameters": params}
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON tool call anywhere
    json_match = re.search(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*\}', response)
    if json_match:
        try:
            # Find the full JSON object
            start = response.find('{', json_match.start())
            depth = 0
            end = start
            for i, c in enumerate(response[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            
            json_str = response[start:end]
            data = json.loads(json_str)
            name = data.get("name") or data.get("function", {}).get("name")
            params = data.get("parameters") or data.get("arguments") or {}
            if name:
                return {"name": name, "parameters": params}
        except (json.JSONDecodeError, ValueError):
            pass
    
    return None


def fix_assistant_raw_format(assistant_raw: str) -> str:
    """
    Fix common formatting issues in assistant_raw.
    
    Ensures that responses with <|python_tag|> have proper end tokens.
    This is applied to all generated responses before saving.
    
    Args:
        assistant_raw: The raw assistant response
    
    Returns:
        Fixed assistant_raw with proper formatting
    """
    if not assistant_raw:
        return assistant_raw
    
    # If response has <|python_tag|> but missing end token, add it
    if "<|python_tag|>" in assistant_raw:
        has_end_token = "<|eom_id|>" in assistant_raw or "<|eot_id|>" in assistant_raw
        if not has_end_token:
            # Add <|eom_id|> at the end
            assistant_raw = assistant_raw.rstrip() + "<|eom_id|>"
    
    return assistant_raw


def validate_llama_format(assistant_raw: str) -> Tuple[bool, str]:
    """Validate Llama 3.1 tool format."""
    if not assistant_raw:
        return False, "Empty assistant_raw"
    
    if "<|python_tag|>" not in assistant_raw:
        return False, "Missing <|python_tag|>"
    
    has_valid_end = any(
        assistant_raw.rstrip().endswith(end)
        for end in ["<|eom_id|>", "<|eot_id|>"]
    )
    if not has_valid_end:
        has_end_token = "<|eom_id|>" in assistant_raw or "<|eot_id|>" in assistant_raw
        if not has_end_token:
            return False, "Missing <|eom_id|> or <|eot_id|>"
    
    if "```" in assistant_raw:
        return False, "Contains markdown code block"
    
    for prefix in ["Action:", "ToolCall:", "Function:"]:
        if assistant_raw.strip().startswith(prefix):
            return False, f"Contains forbidden prefix: {prefix}"
    
    return True, ""


# =============================================================================
# MVP Ds Builder
# =============================================================================

def build_ds_mvp(
    b4_records: List[Dict[str, Any]],
    model,
    tokenizer,
    tools: List[Dict[str, Any]],
    system_prompt: str,
    temperature: float = 0.7,
    min_yield: float = 0.10,
    save_all: bool = False,
    verbose: bool = True,
    collect_examples: bool = False,
    n_successful: int = 10,
    n_correct: int = 5,
    n_no_tool: int = 5,
    n_other_tool: int = 5,
    n_format_errors: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build Ds from ONLY successful flips under actual runtime.
    
    Args:
        b4_records: List of B4 attack records
        model: The language model
        tokenizer: The tokenizer
        tools: Tool definitions
        system_prompt: System prompt to use
        temperature: Generation temperature
        min_yield: Minimum acceptable yield rate (default 10%)
        save_all: If True, save all records (not just successful flips)
        verbose: Print progress
    
    Returns:
        (ds_samples, stats) tuple
    """
    ds_samples = []
    all_samples = []  # For debugging
    
    stats = {
        "total": 0,
        "successful_flips": 0,
        "correct_behavior": 0,
        "no_tool_call": 0,
        "other_tool": 0,
        "format_errors": 0,
    }
    
    # Example collection
    examples = {
        "successful_flips": [],
        "correct_behavior": [],
        "no_tool_call": [],
        "other_tool": [],
        "format_errors": [],
    } if collect_examples else None
    
    iterator = tqdm(b4_records, desc="Building Ds MVP") if verbose else b4_records
    
    for record in iterator:
        stats["total"] += 1
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record["combined_query"]},
        ]
        
        # Generate response
        try:
            response = generate_with_tools(
                model, tokenizer, messages, tools,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Generation error for {record['id']}: {e}")
            stats["format_errors"] += 1
            continue
        
        # Extract tool call
        tool_call = extract_tool_call(response)
        observed_tool = tool_call["name"] if tool_call else None
        expected_tool = record["expected_tool"]
        simulated_tool = record["simulated_tool"]
        
        # Determine outcome
        if observed_tool is None:
            stats["no_tool_call"] += 1
            is_flip_success = False
            category = "no_tool_call"
        elif observed_tool == simulated_tool:
            stats["successful_flips"] += 1
            is_flip_success = True
            category = "successful_flips"
        elif observed_tool == expected_tool:
            stats["correct_behavior"] += 1
            is_flip_success = False
            category = "correct_behavior"
        else:
            stats["other_tool"] += 1
            is_flip_success = False
            category = "other_tool"
        
        # Fix formatting issues
        response = fix_assistant_raw_format(response)
        
        # Validate format
        is_valid, format_error = validate_llama_format(response)
        if not is_valid:
            stats["format_errors"] += 1
            if verbose and stats["format_errors"] <= 5:
                logger.warning(f"Format error: {format_error} in {record['id']}")
            if collect_examples and len(examples["format_errors"]) < n_format_errors:
                examples["format_errors"].append({
                    "id": record["id"],
                    "combined_query": record["combined_query"],
                    "assistant_raw": response,
                    "expected_tool": expected_tool,
                    "observed_tool": observed_tool,
                    "simulated_tool": simulated_tool,
                    "format_error": format_error,
                })
        
        # Build sample
        sample = {
            "id": record["id"],
            "messages": messages,
            "tools": "b4_standard_v1",  # Reference to frozen schema
            "assistant_raw": response,
            "tool_calls_structured": [tool_call] if tool_call else [],
            "labels": {
                "expected_tool": expected_tool,
                "observed_tool": observed_tool,
                "simulated_tool": simulated_tool,
                "is_flip_success": is_flip_success,
            },
            "metadata": {
                "split": "train",
                "source": "b4",
                "benign_query": record["benign_query"],
                "malicious_injection": record["malicious_injection"],
                "format_valid": is_valid,
                "format_error": format_error if not is_valid else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **record.get("metadata", {}),
            },
        }
        
        all_samples.append(sample)
        
        # Collect examples
        if collect_examples and is_valid:
            max_counts = {
                "successful_flips": n_successful,
                "correct_behavior": n_correct,
                "no_tool_call": n_no_tool,
                "other_tool": n_other_tool,
            }
            if category in max_counts and len(examples[category]) < max_counts[category]:
                examples[category].append({
                    "id": record["id"],
                    "combined_query": record["combined_query"],
                    "assistant_raw": response,
                    "expected_tool": expected_tool,
                    "observed_tool": observed_tool,
                    "simulated_tool": simulated_tool,
                    "format_error": None,
                })
        
        # ONLY include if attack succeeded
        if is_flip_success:
            ds_samples.append(sample)
    
    # Compute yield rate
    yield_rate = stats["successful_flips"] / stats["total"] if stats["total"] > 0 else 0
    stats["yield_rate"] = yield_rate
    
    # Log stats
    logger.info("")
    logger.info("=" * 60)
    logger.info("DS MVP BUILD STATS")
    logger.info("=" * 60)
    logger.info(f"Total records processed: {stats['total']}")
    logger.info(f"Successful flips (Ds):   {stats['successful_flips']} ({yield_rate:.1%})")
    logger.info(f"Correct behavior:        {stats['correct_behavior']}")
    logger.info(f"No tool call:            {stats['no_tool_call']}")
    logger.info(f"Other tool:              {stats['other_tool']}")
    logger.info(f"Format errors:           {stats['format_errors']}")
    logger.info(f"Yield rate:              {yield_rate:.1%}")
    logger.info(f"Min yield threshold:     {min_yield:.1%}")
    logger.info("=" * 60)
    
    # Check yield threshold
    if yield_rate < min_yield:
        error_msg = (
            f"Ds yield too low ({yield_rate:.1%} < {min_yield:.1%}). "
            f"Cannot train tool-flip breaker from model that rarely flips. "
            f"Consider: higher temperature, different generator model, or attack prompt tuning."
        )
        logger.error(error_msg)
        stats["error"] = error_msg
    
    # Add examples to stats
    if collect_examples:
        stats["examples"] = examples
    
    return (all_samples if save_all else ds_samples), stats


def build_ds_mvp_vllm(
    b4_records: List[Dict[str, Any]],
    vllm_backend: "VLLMBackend",
    tools: List[Dict[str, Any]],
    system_prompt: str,
    batch_size: int = 32,
    temperature: float = 0.7,
    max_tokens: int = 256,
    min_yield: float = 0.10,
    save_all: bool = False,
    verbose: bool = True,
    collect_examples: bool = False,
    n_successful: int = 10,
    n_correct: int = 5,
    n_no_tool: int = 5,
    n_other_tool: int = 5,
    n_format_errors: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build Ds using vLLM backend with batched generation.
    
    Optimized for multi-GPU (tensor parallel) inference.
    """
    ds_samples = []
    all_samples = []
    
    stats = {
        "total": 0,
        "successful_flips": 0,
        "correct_behavior": 0,
        "no_tool_call": 0,
        "other_tool": 0,
        "format_errors": 0,
    }
    
    # Example collection
    examples = {
        "successful_flips": [],
        "correct_behavior": [],
        "no_tool_call": [],
        "other_tool": [],
        "format_errors": [],
    } if collect_examples else None
    
    # Prepare all prompts first
    logger.info(f"Preparing {len(b4_records)} prompts for batched generation...")
    all_prompts = []
    for record in b4_records:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record["combined_query"]},
        ]
        prompt = vllm_backend.format_prompt_with_tools(messages, tools)
        all_prompts.append(prompt)
    
    # Process in batches
    num_batches = (len(all_prompts) + batch_size - 1) // batch_size
    all_responses = []
    
    logger.info(f"Generating responses in {num_batches} batches (batch_size={batch_size})...")
    
    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_prompts))
        batch_prompts = all_prompts[start_idx:end_idx]
        
        try:
            batch_responses = vllm_backend.generate_batch(
                batch_prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            all_responses.extend(batch_responses)
        except Exception as e:
            logger.error(f"Batch {batch_idx} generation error: {e}")
            all_responses.extend([""] * len(batch_prompts))
    
    # Process results
    logger.info("Processing generated responses...")
    for i, (record, response) in enumerate(tqdm(zip(b4_records, all_responses), 
                                                   total=len(b4_records),
                                                   desc="Building Ds MVP")):
        stats["total"] += 1
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record["combined_query"]},
        ]
        
        # Extract tool call
        tool_call = extract_tool_call(response)
        observed_tool = tool_call["name"] if tool_call else None
        expected_tool = record["expected_tool"]
        simulated_tool = record["simulated_tool"]
        
        # Fix formatting issues
        response = fix_assistant_raw_format(response)
        
        # Determine outcome
        if observed_tool is None:
            stats["no_tool_call"] += 1
            is_flip_success = False
            category = "no_tool_call"
        elif observed_tool == simulated_tool:
            stats["successful_flips"] += 1
            is_flip_success = True
            category = "successful_flips"
        elif observed_tool == expected_tool:
            stats["correct_behavior"] += 1
            is_flip_success = False
            category = "correct_behavior"
        else:
            stats["other_tool"] += 1
            is_flip_success = False
            category = "other_tool"
        
        # Validate format
        is_valid, format_error = validate_llama_format(response)
        if not is_valid:
            stats["format_errors"] += 1
            if verbose and stats["format_errors"] <= 5:
                logger.warning(f"Format error: {format_error} in {record['id']}")
            if collect_examples and len(examples["format_errors"]) < n_format_errors:
                examples["format_errors"].append({
                    "id": record["id"],
                    "combined_query": record["combined_query"],
                    "assistant_raw": response,
                    "expected_tool": expected_tool,
                    "observed_tool": observed_tool,
                    "simulated_tool": simulated_tool,
                    "format_error": format_error,
                })
        
        # Build sample
        sample = {
            "id": record["id"],
            "messages": messages,
            "tools": "b4_standard_v1",
            "assistant_raw": response,
            "tool_calls_structured": [tool_call] if tool_call else [],
            "labels": {
                "expected_tool": expected_tool,
                "observed_tool": observed_tool,
                "simulated_tool": simulated_tool,
                "is_flip_success": is_flip_success,
            },
            "metadata": {
                "split": "train",
                "source": "b4",
                "benign_query": record["benign_query"],
                "malicious_injection": record["malicious_injection"],
                "format_valid": is_valid,
                "format_error": format_error if not is_valid else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **record.get("metadata", {}),
            },
        }
        
        all_samples.append(sample)
        
        # Collect examples
        if collect_examples and is_valid:
            max_counts = {
                "successful_flips": n_successful,
                "correct_behavior": n_correct,
                "no_tool_call": n_no_tool,
                "other_tool": n_other_tool,
            }
            if category in max_counts and len(examples[category]) < max_counts[category]:
                examples[category].append({
                    "id": record["id"],
                    "combined_query": record["combined_query"],
                    "assistant_raw": response,
                    "expected_tool": expected_tool,
                    "observed_tool": observed_tool,
                    "simulated_tool": simulated_tool,
                    "format_error": None,
                })
        
        if is_flip_success:
            ds_samples.append(sample)
    
    # Compute yield rate
    yield_rate = stats["successful_flips"] / stats["total"] if stats["total"] > 0 else 0
    stats["yield_rate"] = yield_rate
    
    # Log stats
    logger.info("")
    logger.info("=" * 60)
    logger.info("DS MVP BUILD STATS (vLLM)")
    logger.info("=" * 60)
    logger.info(f"Total records processed: {stats['total']}")
    logger.info(f"Successful flips (Ds):   {stats['successful_flips']} ({yield_rate:.1%})")
    logger.info(f"Correct behavior:        {stats['correct_behavior']}")
    logger.info(f"No tool call:            {stats['no_tool_call']}")
    logger.info(f"Other tool:              {stats['other_tool']}")
    logger.info(f"Format errors:           {stats['format_errors']}")
    logger.info(f"Yield rate:              {yield_rate:.1%}")
    logger.info(f"Min yield threshold:     {min_yield:.1%}")
    logger.info("=" * 60)
    
    if yield_rate < min_yield:
        error_msg = (
            f"Ds yield too low ({yield_rate:.1%} < {min_yield:.1%}). "
            f"Cannot train tool-flip breaker from model that rarely flips."
        )
        logger.error(error_msg)
        stats["error"] = error_msg
    
    # Add examples to stats
    if collect_examples:
        stats["examples"] = examples
    
    return (all_samples if save_all else ds_samples), stats


# =============================================================================
# Example Reporting
# =============================================================================

def print_examples_report(examples: Dict[str, List[Dict[str, Any]]], truncate: bool = True) -> None:
    """Print a formatted report of collected examples."""
    max_len = 200 if truncate else None
    
    category_names = {
        "successful_flips": "Successful Tool Flips (Ds targets)",
        "correct_behavior": "Correct Behavior (non-successful Ds)",
        "no_tool_call": "No Tool Call",
        "other_tool": "Other Tool",
        "format_errors": "Format Errors",
    }
    
    for category, name in category_names.items():
        if category not in examples or not examples[category]:
            continue
        
        print("\n" + "=" * 80)
        print(f"{name} ({len(examples[category])} examples)")
        print("=" * 80)
        
        for i, ex in enumerate(examples[category], 1):
            print(f"\n--- Example {i} ---")
            print(f"ID: {ex['id']}")
            print(f"Expected: {ex['expected_tool']} | Observed: {ex['observed_tool']} | Simulated: {ex['simulated_tool']}")
            
            query = ex['combined_query']
            if max_len and len(query) > max_len:
                query = query[:max_len] + "..."
            print(f"\nQuery:\n{query}")
            
            response = ex['assistant_raw']
            if max_len and len(response) > max_len:
                response = response[:max_len] + "..."
            print(f"\nAssistant Response:\n{response}")
            
            if ex.get('format_error'):
                print(f"\nFormat Error: {ex['format_error']}")
        
        print("\n" + "=" * 80)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate MVP Circuit Breaker Set (Ds) with behavioral filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Data paths
    parser.add_argument(
        "--b4-data",
        type=Path,
        default=BASE_DIR / "data" / "fujitsu" / "orchestrator_attacks_combined_deduplicated.jsonl",
        help="Path to Fujitsu B4 data",
    )
    parser.add_argument(
        "--tool-schema",
        type=Path,
        default=BASE_DIR / "configs" / "tool_schemas" / "b4_standard_v1.json",
        help="Path to frozen tool schema",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "data" / "cb_mvp" / "ds_stage1.jsonl",
        help="Output path for Ds",
    )
    parser.add_argument(
        "--output-ids",
        type=Path,
        default=None,
        help="Output path for training IDs (for held-out split)",
    )
    
    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["transformers", "vllm"],
        default="transformers",
        help="Backend: 'transformers' (single GPU) or 'vllm' (multi-GPU tensor parallel)",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (e.g., 4 for 4x H100)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Max model length for vLLM (default: 16384)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for vLLM generation (ignored for transformers)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (transformers backend only)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model dtype",
    )
    
    # Generation
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (higher = more varied outputs)",
    )
    parser.add_argument(
        "--min-yield",
        type=float,
        default=0.10,
        help="Minimum acceptable yield rate (default: 10%%)",
    )
    
    # Options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to process",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save all records, not just successful flips (for debugging)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and show stats without generating",
    )
    parser.add_argument(
        "--fail-on-low-yield",
        action="store_true",
        help="Exit with code 1 if yield < min-yield",
    )
    
    # Example collection
    parser.add_argument(
        "--print-examples",
        action="store_true",
        help="Print example datapoints from each category",
    )
    parser.add_argument(
        "--examples-out",
        type=Path,
        default=None,
        help="Output path for examples JSON (default: <output>.examples.json)",
    )
    parser.add_argument(
        "--n-successful",
        type=int,
        default=10,
        help="Number of successful flip examples to collect (default: 10)",
    )
    parser.add_argument(
        "--n-correct",
        type=int,
        default=5,
        help="Number of correct behavior examples to collect (default: 5)",
    )
    parser.add_argument(
        "--n-no-tool",
        type=int,
        default=5,
        help="Number of no tool call examples to collect (default: 5)",
    )
    parser.add_argument(
        "--n-other-tool",
        type=int,
        default=5,
        help="Number of other tool examples to collect (default: 5)",
    )
    parser.add_argument(
        "--n-format-errors",
        type=int,
        default=5,
        help="Number of format error examples to collect (default: 5)",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Do not truncate example outputs when printing",
    )
    
    args = parser.parse_args()
    
    # Load tool schema
    if not args.tool_schema.exists():
        logger.error(f"Tool schema not found: {args.tool_schema}")
        return 1
    
    schema = load_tool_schema(args.tool_schema)
    tools = get_tools_list(schema)
    system_prompt = get_system_prompt(schema)
    
    # Load B4 data
    if not args.b4_data.exists():
        logger.error(f"B4 data not found: {args.b4_data}")
        return 1
    
    logger.info(f"Loading B4 data from {args.b4_data}...")
    b4_records = list(load_fujitsu_b4(args.b4_data, limit=args.limit))
    logger.info(f"Loaded {len(b4_records)} B4 records")
    
    if args.dry_run:
        logger.info("DRY RUN - showing sample records:")
        for record in b4_records[:3]:
            logger.info(f"  {record['id']}")
            logger.info(f"    expected: {record['expected_tool']}, simulated: {record['simulated_tool']}")
            logger.info(f"    query: {record['combined_query'][:80]}...")
        logger.info(f"Would write to: {args.output}")
        return 0
    
    # Load model based on backend
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    
    if args.backend == "vllm":
        logger.info(f"Using vLLM backend with tensor_parallel={args.tensor_parallel}")
        vllm_backend = VLLMBackend(
            model_path=args.model,
            tensor_parallel_size=args.tensor_parallel,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
        )
        
        # Build Ds MVP with vLLM (batched)
        ds_samples, stats = build_ds_mvp_vllm(
            b4_records=b4_records,
            vllm_backend=vllm_backend,
            tools=tools,
            system_prompt=system_prompt,
            batch_size=args.batch_size,
            temperature=args.temperature,
            min_yield=args.min_yield,
            save_all=args.save_all,
            collect_examples=args.print_examples or args.examples_out is not None,
            n_successful=args.n_successful,
            n_correct=args.n_correct,
            n_no_tool=args.n_no_tool,
            n_other_tool=args.n_other_tool,
            n_format_errors=args.n_format_errors,
        )
    else:
        logger.info("Using transformers backend")
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            device=args.device,
            torch_dtype=dtype_map[args.dtype],
        )
        
        # Build Ds MVP with transformers
        ds_samples, stats = build_ds_mvp(
            b4_records=b4_records,
            model=model,
            tokenizer=tokenizer,
            tools=tools,
            system_prompt=system_prompt,
            temperature=args.temperature,
            min_yield=args.min_yield,
            save_all=args.save_all,
            collect_examples=args.print_examples or args.examples_out is not None,
            n_successful=args.n_successful,
            n_correct=args.n_correct,
            n_no_tool=args.n_no_tool,
            n_other_tool=args.n_other_tool,
            n_format_errors=args.n_format_errors,
        )
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in ds_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logger.info(f"Wrote {len(ds_samples)} samples to {args.output}")
    
    # Write IDs file
    if args.output_ids or True:  # Always write IDs
        ids_path = args.output_ids or args.output.with_suffix(".ids.txt")
        with open(ids_path, "w", encoding="utf-8") as f:
            for sample in ds_samples:
                f.write(sample["id"] + "\n")
        logger.info(f"Wrote IDs to {ids_path}")
    
    # Write stats (without examples to keep it clean)
    stats_for_file = {k: v for k, v in stats.items() if k != "examples"}
    stats_path = args.output.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_for_file, f, indent=2)
    logger.info(f"Wrote stats to {stats_path}")
    
    # Handle examples
    if "examples" in stats and stats["examples"]:
        examples = stats["examples"]
        
        # Print examples if requested
        if args.print_examples:
            print_examples_report(examples, truncate=not args.no_truncate)
        
        # Write examples to JSON file
        examples_path = args.examples_out or args.output.with_suffix(".examples.json")
        with open(examples_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote examples to {examples_path}")
    
    # Check yield threshold
    if args.fail_on_low_yield and stats.get("yield_rate", 0) < args.min_yield:
        logger.error("Yield rate below threshold - exiting with code 1")
        return 1
    
    # Clean up
    if args.backend == "vllm":
        del vllm_backend
    else:
        del model
    torch.cuda.empty_cache()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
