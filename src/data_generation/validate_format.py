#!/usr/bin/env python3
"""
Data Format Validation Script - Batch Format

Validates that training data files conform to the batch format output by
rebuild_training_data.py, which is expected by the training script.

**Batch Format** - JSONL file where each line is:
{
  "id": "batch_0",
  "harmful": [
    {
      "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>...",
      "is_agentic": true,
      "has_tool_call_in_text": true,
      ...other original fields...
    }
  ],
  "benign": [
    {
      "text": "<fully rendered conversation>",
      "is_agentic": true,
      "has_tool_call_in_text": true,
      ...
    }
  ]
}

Checks:
1. Each line is a valid JSON batch with "id", "harmful", "benign" keys
2. Each sample in harmful/benign has "text" field (pre-rendered Llama 3.1 format)
3. Text fields contain proper Llama 3.1 special tokens
4. No markdown wrappers or forbidden prefixes
5. Tool call formatting is correct (for agentic samples)

Usage:
    python src/data_generation/validate_format.py \
        --data data/cb_mvp/cb_training_batches.jsonl \
        --strict
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


# =============================================================================
# Llama 3.1 Format Constants
# =============================================================================

LLAMA_BOS = "<|begin_of_text|>"
LLAMA_HEADER_START = "<|start_header_id|>"
LLAMA_HEADER_END = "<|end_header_id|>"
LLAMA_EOT = "<|eot_id|>"
LLAMA_EOM = "<|eom_id|>"
LLAMA_PYTHON_TAG = "<|python_tag|>"


# =============================================================================
# Validation Result Classes
# =============================================================================

class SampleValidationResult:
    """Result of validating a single sample within a batch."""
    
    def __init__(self, sample_id: str, sample_type: str):
        self.sample_id = sample_id
        self.sample_type = sample_type  # "harmful" or "benign"
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def add_error(self, message: str):
        self.errors.append(message)
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


class BatchValidationResult:
    """Result of validating a batch."""
    
    def __init__(self, batch_id: str):
        self.batch_id = batch_id
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.sample_results: List[SampleValidationResult] = []
    
    def add_error(self, message: str):
        self.errors.append(message)
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def add_sample_result(self, result: SampleValidationResult):
        self.sample_results.append(result)
    
    @property
    def is_valid(self) -> bool:
        if len(self.errors) > 0:
            return False
        return all(r.is_valid for r in self.sample_results)
    
    @property
    def all_errors(self) -> List[str]:
        """All errors including from samples."""
        errors = list(self.errors)
        for r in self.sample_results:
            for e in r.errors:
                errors.append(f"[{r.sample_type}:{r.sample_id}] {e}")
        return errors
    
    @property
    def all_warnings(self) -> List[str]:
        """All warnings including from samples."""
        warnings = list(self.warnings)
        for r in self.sample_results:
            for w in r.warnings:
                warnings.append(f"[{r.sample_type}:{r.sample_id}] {w}")
        return warnings


# =============================================================================
# Sample Validators
# =============================================================================

def validate_sample_text(sample: Dict[str, Any], result: SampleValidationResult):
    """Validate the pre-rendered text field of a sample."""
    text = sample.get("text")
    
    if text is None:
        result.add_error("Missing 'text' field")
        return
    
    if not isinstance(text, str):
        result.add_error(f"'text' should be string, got {type(text).__name__}")
        return
    
    if not text.strip():
        result.add_error("Empty 'text' field")
        return
    
    # Check for Llama 3.1 special tokens
    if not text.startswith(LLAMA_BOS):
        result.add_error(f"Text should start with {LLAMA_BOS}")
    
    if LLAMA_HEADER_START not in text:
        result.add_error(f"Missing {LLAMA_HEADER_START} in text")
    
    if LLAMA_HEADER_END not in text:
        result.add_error(f"Missing {LLAMA_HEADER_END} in text")
    
    # Check for required roles
    if "system" not in text:
        result.add_warning("No 'system' role found in text")
    
    if "user" not in text:
        result.add_error("No 'user' role found in text")
    
    if "assistant" not in text:
        result.add_error("No 'assistant' role found in text")
    
    # Check for proper end token
    has_eom = LLAMA_EOM in text
    has_eot = LLAMA_EOT in text
    
    if not (has_eom or has_eot):
        result.add_warning("Missing end tokens (<|eom_id|> or <|eot_id|>)")
    
    # Check text ends with an end token (not critical but good)
    stripped = text.rstrip()
    if not (stripped.endswith(LLAMA_EOM) or stripped.endswith(LLAMA_EOT)):
        result.add_warning("Text does not end with <|eom_id|> or <|eot_id|>")
    
    # Forbidden patterns
    if "```" in text:
        # Check if it's in the content vs in tool call JSON
        # Markdown in tool call JSON args is OK
        before_python_tag = text.split(LLAMA_PYTHON_TAG)[0] if LLAMA_PYTHON_TAG in text else text
        if "```" in before_python_tag:
            result.add_error("Contains markdown code block (```) outside tool call")
    
    # Forbidden prefixes in assistant response
    if f"{LLAMA_HEADER_END}\n\nAction:" in text:
        result.add_error("Contains forbidden 'Action:' prefix in assistant response")
    if f"{LLAMA_HEADER_END}\n\nToolCall:" in text:
        result.add_error("Contains forbidden 'ToolCall:' prefix in assistant response")


def validate_sample_agentic_flags(sample: Dict[str, Any], result: SampleValidationResult):
    """Validate agentic-related flags on a sample."""
    text = sample.get("text", "")
    is_agentic = sample.get("is_agentic", False)
    has_tool_call = sample.get("has_tool_call_in_text", False)
    
    # Check consistency between flags and actual text
    has_python_tag_in_text = LLAMA_PYTHON_TAG in text
    
    if has_tool_call and not has_python_tag_in_text:
        result.add_error("has_tool_call_in_text=true but no <|python_tag|> in text")
    
    if has_python_tag_in_text and not has_tool_call:
        result.add_warning("Has <|python_tag|> in text but has_tool_call_in_text=false")
    
    # If marked as agentic, should have tool call
    if is_agentic and not has_python_tag_in_text:
        result.add_warning("is_agentic=true but no tool call in text")


def validate_sample_tool_call_format(sample: Dict[str, Any], result: SampleValidationResult):
    """Validate tool call JSON format within the text."""
    text = sample.get("text", "")
    
    if LLAMA_PYTHON_TAG not in text:
        return  # No tool call to validate
    
    # Extract content after python tag
    parts = text.split(LLAMA_PYTHON_TAG, 1)
    if len(parts) < 2:
        return
    
    after_tag = parts[1]
    
    # Remove end tokens to get JSON
    for end_token in [LLAMA_EOM, LLAMA_EOT, "</s>", "<|end"]:
        after_tag = after_tag.split(end_token)[0]
    
    json_content = after_tag.strip()
    
    if not json_content:
        result.add_error("Empty content after <|python_tag|>")
        return
    
    # Try to parse as JSON
    try:
        data = json.loads(json_content)
        
        # Should be a dict with "name" and "parameters"
        if not isinstance(data, dict):
            result.add_error(f"Tool call should be dict, got {type(data).__name__}")
            return
        
        if "name" not in data:
            result.add_error("Tool call JSON missing 'name' field")
        
        if "parameters" not in data and "arguments" not in data:
            result.add_warning("Tool call JSON missing 'parameters' field")
        
    except json.JSONDecodeError as e:
        result.add_error(f"Invalid JSON after <|python_tag|>: {e}")


def validate_sample(
    sample: Dict[str, Any],
    sample_type: str,
    sample_idx: int,
) -> SampleValidationResult:
    """
    Validate a single sample from a batch.
    
    Args:
        sample: The sample dict
        sample_type: "harmful" or "benign"
        sample_idx: Index within the array
    
    Returns:
        SampleValidationResult
    """
    sample_id = sample.get("id", f"idx_{sample_idx}")
    result = SampleValidationResult(sample_id, sample_type)
    
    # Run validators
    validate_sample_text(sample, result)
    validate_sample_agentic_flags(sample, result)
    validate_sample_tool_call_format(sample, result)
    
    return result


# =============================================================================
# Batch Validators
# =============================================================================

def validate_batch_structure(batch: Dict[str, Any], result: BatchValidationResult):
    """Validate basic batch structure."""
    if "id" not in batch:
        result.add_warning("Missing 'id' field in batch")
    
    if "harmful" not in batch:
        result.add_error("Missing 'harmful' array in batch")
    elif not isinstance(batch["harmful"], list):
        result.add_error(f"'harmful' should be list, got {type(batch['harmful']).__name__}")
    
    if "benign" not in batch:
        result.add_error("Missing 'benign' array in batch")
    elif not isinstance(batch["benign"], list):
        result.add_error(f"'benign' should be list, got {type(batch['benign']).__name__}")


def validate_batch_non_empty(batch: Dict[str, Any], result: BatchValidationResult):
    """Check batch has at least one sample."""
    harmful = batch.get("harmful", [])
    benign = batch.get("benign", [])
    
    if len(harmful) == 0 and len(benign) == 0:
        result.add_error("Batch is empty (no harmful or benign samples)")
    
    if len(harmful) == 0:
        result.add_warning("Batch has no harmful samples")
    
    if len(benign) == 0:
        result.add_warning("Batch has no benign samples")


def validate_batch(batch: Dict[str, Any], batch_idx: int) -> BatchValidationResult:
    """
    Validate a single batch.
    
    Args:
        batch: The batch dict
        batch_idx: Line number / index
    
    Returns:
        BatchValidationResult
    """
    batch_id = batch.get("id", f"line_{batch_idx}")
    result = BatchValidationResult(batch_id)
    
    # Validate batch structure
    validate_batch_structure(batch, result)
    validate_batch_non_empty(batch, result)
    
    # If structure is invalid, don't validate samples
    if len(result.errors) > 0:
        return result
    
    # Validate harmful samples
    for idx, sample in enumerate(batch.get("harmful", [])):
        if not isinstance(sample, dict):
            result.add_error(f"harmful[{idx}] is not a dict")
            continue
        sample_result = validate_sample(sample, "harmful", idx)
        result.add_sample_result(sample_result)
    
    # Validate benign samples
    for idx, sample in enumerate(batch.get("benign", [])):
        if not isinstance(sample, dict):
            result.add_error(f"benign[{idx}] is not a dict")
            continue
        sample_result = validate_sample(sample, "benign", idx)
        result.add_sample_result(sample_result)
    
    return result
# =============================================================================
# File Validation
# =============================================================================

def validate_file(
    path: Path,
    verbose: bool = True,
    max_errors: int = 10,
) -> Dict[str, Any]:
    """
    Validate all batches in a JSONL file.
    
    Args:
        path: Path to JSONL file
        verbose: Print detailed errors
        max_errors: Maximum errors to print per type
    
    Returns:
        Dict with validation statistics
    """
    results: List[BatchValidationResult] = []
    error_counts: Dict[str, int] = {}
    warning_counts: Dict[str, int] = {}
    
    total_harmful_samples = 0
    total_benign_samples = 0
    total_tool_calls = 0
    
    logger.info(f"Validating {path}...")
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            
            try:
                batch = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON: {e}")
                error_counts["invalid_json"] = error_counts.get("invalid_json", 0) + 1
                continue
            
            result = validate_batch(batch, line_num)
            results.append(result)
            
            # Count samples
            total_harmful_samples += len(batch.get("harmful", []))
            total_benign_samples += len(batch.get("benign", []))
            
            # Count tool calls
            for s in batch.get("harmful", []) + batch.get("benign", []):
                if s.get("has_tool_call_in_text", False):
                    total_tool_calls += 1
            
            # Aggregate errors
            for error in result.all_errors:
                error_key = error.split(":")[0] if ":" in error else error[:60]
                error_counts[error_key] = error_counts.get(error_key, 0) + 1
            
            for warning in result.all_warnings:
                warning_key = warning.split(":")[0] if ":" in warning else warning[:60]
                warning_counts[warning_key] = warning_counts.get(warning_key, 0) + 1
    
    # Compute statistics
    total_batches = len(results)
    valid_batches = sum(1 for r in results if r.is_valid)
    invalid_batches = total_batches - valid_batches
    total_errors = sum(len(r.all_errors) for r in results)
    total_warnings = sum(len(r.all_warnings) for r in results)
    
    stats = {
        "file": str(path),
        "total_batches": total_batches,
        "valid_batches": valid_batches,
        "invalid_batches": invalid_batches,
        "validity_rate": valid_batches / total_batches if total_batches > 0 else 0,
        "total_harmful_samples": total_harmful_samples,
        "total_benign_samples": total_benign_samples,
        "total_samples": total_harmful_samples + total_benign_samples,
        "total_tool_calls": total_tool_calls,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "error_types": error_counts,
        "warning_types": warning_counts,
    }
    
    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("VALIDATION RESULTS (Batch Format)")
    logger.info("=" * 60)
    logger.info(f"File: {path}")
    logger.info(f"Total batches: {total_batches}")
    logger.info(f"Valid batches: {valid_batches} ({stats['validity_rate']:.1%})")
    logger.info(f"Invalid batches: {invalid_batches}")
    logger.info("")
    logger.info(f"Total harmful samples: {total_harmful_samples}")
    logger.info(f"Total benign samples: {total_benign_samples}")
    logger.info(f"Total samples with tool calls: {total_tool_calls}")
    logger.info("")
    logger.info(f"Total errors: {total_errors}")
    logger.info(f"Total warnings: {total_warnings}")
    
    if error_counts and verbose:
        logger.info("")
        logger.info("Error breakdown:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1])[:max_errors]:
            logger.info(f"  {count:4d} | {error_type}")
    
    if warning_counts and verbose:
        logger.info("")
        logger.info("Warning breakdown:")
        for warning_type, count in sorted(warning_counts.items(), key=lambda x: -x[1])[:max_errors]:
            logger.info(f"  {count:4d} | {warning_type}")
    
    # Show sample errors
    if verbose and invalid_batches > 0:
        logger.info("")
        logger.info("Sample errors (first 5 invalid batches):")
        shown = 0
        for result in results:
            if not result.is_valid and shown < 5:
                logger.info(f"  Batch {result.batch_id}:")
                for error in result.all_errors[:3]:
                    logger.info(f"    - {error}")
                shown += 1
    
    logger.info("=" * 60)
    
    # Overall pass/fail
    if invalid_batches == 0:
        logger.info("✅ ALL VALIDATION CHECKS PASSED")
    else:
        logger.info(f"❌ VALIDATION FAILED: {invalid_batches} invalid batches")
    
    return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate training data batch format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to JSONL data file to validate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save validation report JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any validation errors",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="Maximum errors to show per type",
    )
    
    args = parser.parse_args()
    
    # Check file exists
    if not args.data.exists():
        logger.error(f"File not found: {args.data}")
        return 1
    
    # Validate
    stats = validate_file(
        args.data,
        verbose=not args.quiet,
        max_errors=args.max_errors,
    )
    
    # Save report
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Validation report saved to {args.output}")
    
    # Exit code
    if args.strict and stats["invalid_batches"] > 0:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
