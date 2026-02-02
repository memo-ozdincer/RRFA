#!/usr/bin/env python3
"""
ETL_A: Tier A (raw formats) -> Tier B (trace_v1) conversion.

This module only handles raw source files and converts them to canonical traces.

Injection Detection:
    The module detects injection spans in tool responses using configurable patterns.
    Default patterns cover common injection markers like <INFORMATION>, <IMPORTANT>, etc.

CB Training Categories:
    - harmful: Attack present AND model complied (attack_succeeded=True)
    - resisted: Attack present BUT model refused (attack_succeeded=False)
    - benign: No attack present

    Only "harmful" traces go to CB loss; "benign" and "resisted" go to retain loss.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.schemas.trace import (
    Trace,
    Message,
    TraceSource,
    TraceTask,
    TraceLabels,
    TraceTraining,
    TraceToolAttack,
    TraceLinks,
    TraceMixture,
    SignalHints,
    InjectionCharSpan,
    ToolCall,
    ToolCallFunction,
    RawMetadata,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Injection Pattern Detection (Configurable)
# =============================================================================

# Default patterns for detecting injection spans in tool responses
# These can be extended or overridden via --injection-patterns flag
DEFAULT_INJECTION_PATTERNS = [
    # AgentDojo-style patterns
    r"<INFORMATION>.*?</INFORMATION>",
    r"<IMPORTANT>.*?</IMPORTANT>",
    r"<INSTRUCTION>.*?</INSTRUCTION>",
    r"<SECRET>.*?</SECRET>",
    r"<ADMIN>.*?</ADMIN>",
    r"<SYSTEM>.*?</SYSTEM>",
    # Common prompt injection markers
    r"\[INSTRUCTION\].*?\[/INSTRUCTION\]",
    r"\[SYSTEM\].*?\[/SYSTEM\]",
    r"(?:^|\n)(?:IMPORTANT|URGENT|NOTE|ADMIN|SYSTEM):\s*.*?(?=\n\n|\Z)",
    # Embedded commands pattern
    r"(?:ignore (?:previous|above|all) (?:instructions?|prompts?)|new instructions?:).*?(?=\n\n|\Z)",
]


def compile_injection_patterns(patterns: Optional[List[str]] = None) -> List[re.Pattern]:
    """Compile injection detection patterns into regex objects."""
    patterns = patterns or DEFAULT_INJECTION_PATTERNS
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE | re.DOTALL))
        except re.error as e:
            logger.warning("Invalid injection pattern '%s': %s", pattern, e)
    return compiled


def detect_injection_spans_in_content(
    content: str,
    message_index: int,
    patterns: Optional[List[re.Pattern]] = None,
) -> List[InjectionCharSpan]:
    """
    Detect injection spans in message content using regex patterns.

    Args:
        content: The message content to scan
        message_index: Index of the message in the trace
        patterns: Compiled regex patterns (uses defaults if None)

    Returns:
        List of InjectionCharSpan objects for detected injections
    """
    if not content:
        return []

    if patterns is None:
        patterns = compile_injection_patterns()

    spans = []
    for pattern in patterns:
        for match in pattern.finditer(content):
            spans.append(InjectionCharSpan(
                message_index=message_index,
                char_start=match.start(),
                char_end=match.end(),
            ))

    # Merge overlapping spans
    if len(spans) > 1:
        spans.sort(key=lambda s: (s.char_start, s.char_end))
        merged = [spans[0]]
        for span in spans[1:]:
            last = merged[-1]
            if span.char_start <= last.char_end:
                # Overlapping - extend the last span
                merged[-1] = InjectionCharSpan(
                    message_index=last.message_index,
                    char_start=last.char_start,
                    char_end=max(last.char_end, span.char_end),
                )
            else:
                merged.append(span)
        spans = merged

    return spans


def detect_injection_in_messages(
    messages: List[Dict[str, Any]],
    patterns: Optional[List[re.Pattern]] = None,
) -> Tuple[Optional[InjectionCharSpan], List[InjectionCharSpan]]:
    """
    Detect injection spans across all messages in a trace.

    Primarily looks in tool responses (role="tool") but also checks user messages.

    Args:
        messages: List of message dictionaries
        patterns: Compiled regex patterns

    Returns:
        Tuple of (first_injection_span, all_injection_spans)
        first_injection_span is for backward compatibility with signal_hints.injection_char_span
    """
    all_spans = []

    for idx, msg in enumerate(messages):
        content = msg.get("content") or ""
        role = msg.get("role", "")

        # Skip system messages - they're not injection targets
        if role == "system":
            continue

        # Check tool responses and user messages for injections
        if role in ("tool", "user"):
            spans = detect_injection_spans_in_content(content, idx, patterns)
            all_spans.extend(spans)

    # Return first span for backward compatibility, plus all spans
    first_span = all_spans[0] if all_spans else None
    return first_span, all_spans

# Only keeping B4 - removed B1, B2, B3
FUJITSU_FILE_MAP = {
    "b4": "orchestrator_attacks_combined_deduplicated.jsonl",
}


# =============================================================================
# Helpers
# =============================================================================

def _iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with open(path, "r") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            yield idx, json.loads(line)


def _record_locator(record: Dict[str, Any], line_number: int, keys: List[str]) -> Dict[str, str]:
    for key in keys:
        value = record.get(key)
        if value:
            if key == "index":
                kind = "index"
            else:
                kind = "uuid" if "id" in key else "path"
            return {"kind": kind, "value": str(value)}
    return {"kind": "jsonl_line", "value": str(line_number)}


def _find_injection_span(content: str, injection_text: Optional[str], message_index: int) -> Optional[InjectionCharSpan]:
    if not injection_text:
        return None
    idx = content.find(injection_text)
    if idx >= 0:
        return InjectionCharSpan(
            message_index=message_index,
            char_start=idx,
            char_end=idx + len(injection_text),
        )
    return None


def _has_assistant_message(messages: List[Message]) -> bool:
    """Check if trace has any assistant messages (used to determine completeness)."""
    return any(msg.role == "assistant" for msg in messages)


def _determine_completeness(messages: List[Message]) -> str:
    """Determine completeness based on assistant message presence."""
    return "complete" if _has_assistant_message(messages) else "skeleton"


def _determine_tier(messages: List[Message]) -> str:
    """Determine tier based on completeness: B1 = skeleton, B2 = complete."""
    return "B2" if _has_assistant_message(messages) else "B1"


def _build_user_with_context(
    user_query: str,
    context_label: str,
    context_text: Optional[str],
    message_index: int,
) -> Tuple[str, Optional[InjectionCharSpan]]:
    if not context_text:
        return user_query, None

    base = user_query.strip() if user_query else ""
    if base:
        base += "\n\n"

    label_open = f"[{context_label}]\n"
    label_close = f"\n[END_{context_label}]"
    start = len(base) + len(label_open)
    content = f"{base}{label_open}{context_text}{label_close}"

    span = InjectionCharSpan(
        message_index=message_index,
        char_start=start,
        char_end=start + len(context_text),
    )
    return content, span


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if any(token in lowered for token in ["success", "unsafe", "attack succeeded"]):
            return True
        if any(token in lowered for token in ["safe", "benign", "failed", "unsuccess"]):
            return False
    return default


# =============================================================================
# Fujitsu B4 (Orchestrator Tool-Flip)
# =============================================================================

def convert_fujitsu_b4_record(
    record: Dict[str, Any],
    split: str,
    line_number: int,
    system_prompt: Optional[str] = None,
) -> Optional[Trace]:
    """
    Convert Fujitsu B4 (orchestrator tool-flip) record to trace_v1.

    B4 records are SKELETON traces (no assistant response) that need
    generation via generate_completions.py before they can be used for training.

    Args:
        record: Raw Fujitsu B4 record
        split: Data split (train/eval/test)
        line_number: Line number in source file
        system_prompt: System prompt to use. If None, a placeholder is used.
                      The actual system prompt should be provided at generation time
                      via generate_completions.py --tool-schema.
    """
    # Default to a placeholder that indicates the system prompt should be overridden
    # at generation time. generate_completions.py will replace this with the
    # system prompt from the tool schema.
    if system_prompt is None:
        system_prompt = "[PLACEHOLDER: System prompt will be provided from tool schema at generation time]"

    try:
        user_content = record.get("combined_query") or record.get("benign_query") or ""
        messages = [
            Message(
                role="system",
                content=system_prompt,
            ),
            Message(
                role="user",
                content=user_content,
            ),
        ]

        success = _coerce_bool(record.get("success"), default=True)
        injection_span = _find_injection_span(user_content, record.get("malicious_injection"), 1)

        signal_hints = SignalHints(
            injection_char_span=injection_span,
            expected_tool_name=record.get("expected_tool"),
            observed_tool_name=record.get("simulated_tool"),
        )
        
        # Determine completeness based on messages
        completeness = _determine_completeness(messages)
        tier = _determine_tier(messages)

        trace = Trace(
            id=Trace.generate_id("fujitsu_b4", messages=messages),
            completeness=completeness,
            tier=tier,
            source=TraceSource(
                dataset="fujitsu_b4",
                tier="raw",
                subset="orchestrator",
                record_locator=_record_locator(record, line_number, ["record_id", "attack_id"]),
                ingest_version="etl_a_v1",
            ),
            messages=messages,
            split=split,
            task=TraceTask(
                family="tool_flip",
                name="orchestrator",
                variant=record.get("subtype"),
            ),
            labels=TraceLabels(
                category="harmful" if success else "benign",
                security_outcome="unsafe" if success else "safe",
                attack_type="tool_flip",
                attack_succeeded=success,
            ),
            tool_attack=TraceToolAttack(
                expected_tool=record.get("expected_tool"),
                observed_tool=record.get("simulated_tool"),
                attack_vector=record.get("subtype"),
                injection_text=record.get("malicious_injection"),
            ),
            training=TraceTraining(
                sample_weight=1.0,
                loss_mask_policy="assistant_only",
                mixture=TraceMixture(class_id="fujitsu_b4/tool_flip"),
            ),
            links=TraceLinks(raw_id=record.get("record_id")),
            signal_hints=signal_hints,
        )
        return trace
    except Exception as exc:
        logger.warning("Failed to convert Fujitsu B4 record: %s", exc)
        return None


# =============================================================================
# AgentDojo
# =============================================================================

def _detect_model_family(metadata: Dict[str, Any], messages: List[Dict[str, Any]]) -> Optional[str]:
    """Detect model family from AgentDojo metadata or message patterns."""
    # Check metadata for model info
    model_name = metadata.get("model") or metadata.get("model_name") or ""
    model_name_lower = model_name.lower()

    if "llama" in model_name_lower:
        return "llama"
    if "claude" in model_name_lower or "anthropic" in model_name_lower:
        return "claude"
    if "gpt" in model_name_lower or "openai" in model_name_lower:
        return "gpt"
    if "gemini" in model_name_lower:
        return "gemini"

    # Check message content for format patterns
    for msg in messages:
        content = msg.get("content", "")
        if "<|python_tag|>" in content:
            return "llama"
        if "<function_calls>" in content or "<function_calls>" in content:
            return "claude"

    return None


def _detect_format_family(model_family: Optional[str], messages: List[Dict[str, Any]]) -> Optional[str]:
    """Detect tool call format family from model family or message content."""
    if model_family == "llama":
        return "llama_python_tag"
    if model_family == "claude":
        return "anthropic_xml"
    if model_family == "gpt":
        return "openai_json"

    # Fall back to content inspection
    for msg in messages:
        content = msg.get("content", "")
        if "<|python_tag|>" in content:
            return "llama_python_tag"
        if "<function_calls>" in content:
            return "anthropic_xml"

    return "generic_json"


def _parse_tool_calls(msg: Dict[str, Any], preserve_raw_content: bool = True, strip_tool_syntax: bool = False) -> Tuple[Optional[List[ToolCall]], Optional[str]]:
    """
    Parse tool calls from a message, preserving both structured and raw formats.

    Args:
        msg: Message dictionary potentially containing tool_calls
        preserve_raw_content: If True, attempt to preserve the raw tool call string
        strip_tool_syntax: If True, also return cleaned content with tool syntax stripped

    Returns:
        Tuple of (List of ToolCall objects or None, cleaned_content or None)
        cleaned_content is only returned if strip_tool_syntax=True and tool calls found
    """
    tool_calls = []
    raw_calls = msg.get("tool_calls")
    if raw_calls is None and msg.get("tool_call"):
        raw_calls = [msg.get("tool_call")]

    if not raw_calls:
        return None, None

    # Extract clean content if strip_tool_syntax is enabled
    cleaned_content = None
    if strip_tool_syntax:
        content = msg.get("content", "")
        if content:
            # Strip tool call syntax based on detected format
            if "<|python_tag|>" in content:
                # Llama format: take everything before <|python_tag|>
                cleaned_content = content.split("<|python_tag|>", 1)[0].strip()
            elif "<function_calls>" in content or "<invoke>" in content:
                # Anthropic/Claude XML format
                import re
                cleaned_content = re.sub(r'<function_calls>.*?</function_calls>', '', content, flags=re.DOTALL)
                cleaned_content = re.sub(r'<invoke>.*?</invoke>', '', cleaned_content, flags=re.DOTALL)
                cleaned_content = cleaned_content.strip()
            elif content.strip().startswith("{") and ('"name"' in content or '"function"' in content):
                # OpenAI JSON format - if content is just JSON, clear it (tool call only)
                try:
                    json.loads(content)
                    cleaned_content = ""  # Pure tool call, no reasoning
                except json.JSONDecodeError:
                    cleaned_content = content  # Keep as-is if not valid JSON
            else:
                # No tool syntax detected, keep as-is
                cleaned_content = content

    for call in raw_calls:
        if not call:
            continue
        func = call.get("function") or {}
        if isinstance(func, str):
            func = {"name": func}
        name = func.get("name") or call.get("name") or ""

        raw_args = func.get("arguments") or call.get("args")
        args_obj = {}
        args_json = None
        raw_content = None

        if isinstance(raw_args, str):
            args_json = raw_args
            try:
                args_obj = json.loads(raw_args)
            except json.JSONDecodeError:
                args_obj = {"raw": raw_args}
        elif isinstance(raw_args, dict):
            args_obj = raw_args
            # ALWAYS serialize dict to JSON for cross-model compatibility
            # This ensures we can always re-render with any target model
            args_json = json.dumps(args_obj, ensure_ascii=False)

        # Preserve raw tool call content if available
        if preserve_raw_content:
            raw_content = call.get("raw_content") or call.get("raw") or func.get("raw_content")

        tool_calls.append(
            ToolCall(
                call_id=call.get("id"),
                function=ToolCallFunction(
                    name=name,
                    arguments=args_obj,
                    arguments_json=args_json,
                    raw_content=raw_content,
                ),
            )
        )

    return tool_calls or None, cleaned_content


def convert_agentdojo_record(
    record: Dict[str, Any],
    split: str,
    line_number: int,
    strip_tool_syntax: bool = False,
    injection_patterns: Optional[List[re.Pattern]] = None,
) -> Optional[Trace]:
    """
    Convert AgentDojo record to trace_v1 with full information preservation.

    Preserves:
    - Raw tool call arguments (arguments_json)
    - Source model information (if available)
    - Original assistant content (raw_assistant_content)
    - All metadata fields (in raw_metadata)

    CB Training Categories (NEW):
    - harmful: Attack present AND model complied (attack_succeeded=True)
    - resisted: Attack present BUT model refused (attack_succeeded=False)
    - benign: No attack present

    Args:
        strip_tool_syntax: If True, strip tool call syntax from assistant message content
        injection_patterns: Compiled regex patterns for injection detection
    """
    try:
        metadata = record.get("metadata", {})
        raw_messages = record.get("messages", [])

        if not raw_messages:
            return None

        # Detect model family and format from metadata/messages
        model_family = _detect_model_family(metadata, raw_messages)
        format_family = _detect_format_family(model_family, raw_messages)
        model_id = metadata.get("model") or metadata.get("model_name")

        # Capture raw assistant content for signal hints
        raw_assistant_contents = []

        messages: List[Message] = []
        for msg in raw_messages:
            role = msg.get("role", "user")
            tool_calls, cleaned_content = _parse_tool_calls(msg, preserve_raw_content=True, strip_tool_syntax=strip_tool_syntax)
            name = msg.get("name")
            if not name and role == "tool":
                tool_call = msg.get("tool_call") or {}
                if isinstance(tool_call, dict):
                    func = tool_call.get("function") or {}
                    if isinstance(func, str):
                        name = func
                    elif isinstance(func, dict):
                        name = func.get("name")
            if role == "tool":
                tool_calls = None

            content = msg.get("content") or ""

            # Use cleaned content if stripping was requested and we got a result
            if strip_tool_syntax and cleaned_content is not None:
                content = cleaned_content

            # Capture raw assistant content (before stripping)
            if role == "assistant":
                raw_assistant_contents.append(msg.get("content") or "")

            messages.append(
                Message(
                    role=role,
                    content=content,
                    name=name,
                    tool_calls=tool_calls,
                    tool_call_id=msg.get("tool_call_id"),
                    thinking=msg.get("thinking"),
                )
            )

        # =================================================================
        # Attack Detection and Category Assignment (UPDATED)
        # =================================================================
        # Check if injection was present in metadata
        is_attack = metadata.get("injection_task_id") is not None

        # AgentDojo security field: True = model was secure (resisted), False = compromised
        security_passed = bool(metadata.get("security", True))

        # Attack succeeded = injection present AND security check failed
        attack_succeeded = is_attack and not security_passed

        # Detect injection spans in tool responses (NEW)
        injection_span, all_injection_spans = detect_injection_in_messages(
            raw_messages, injection_patterns
        )

        # If we detected injections via patterns but metadata says no attack,
        # this might be a data quality issue - log it
        if all_injection_spans and not is_attack:
            logger.debug(
                "Detected injection patterns in trace but metadata says no attack "
                "(line %d). Patterns found: %d",
                line_number, len(all_injection_spans)
            )

        # Determine CB training category (NEW LOGIC)
        # - harmful: attack succeeded (model was compromised)
        # - resisted: attack present but model refused (correct behavior)
        # - benign: no attack present
        if is_attack:
            if attack_succeeded:
                category = "harmful"
            else:
                category = "resisted"
        else:
            category = "benign"

        # Determine completeness based on messages
        completeness = _determine_completeness(messages)
        tier = _determine_tier(messages)

        # Build signal hints with raw format preservation AND injection detection
        signal_hints = SignalHints(
            injection_char_span=injection_span,  # First detected injection
            raw_format=format_family,
            raw_assistant_content="\n".join(raw_assistant_contents) if raw_assistant_contents else None,
        )

        # Preserve all metadata fields in raw_metadata
        # Also store all injection spans if multiple were found
        extra_fields = {}
        if len(all_injection_spans) > 1:
            extra_fields["all_injection_spans"] = [
                {"message_index": s.message_index, "char_start": s.char_start, "char_end": s.char_end}
                for s in all_injection_spans
            ]

        raw_metadata = RawMetadata(
            source_fields={
                **{k: v for k, v in metadata.items()
                   if k not in {"suite_name", "name", "model", "model_name"}},
                **extra_fields,
            } if metadata or extra_fields else None,
        )

        trace = Trace(
            id=Trace.generate_id("agentdojo", messages=messages),
            completeness=completeness,
            tier=tier,
            source=TraceSource(
                dataset="agentdojo",
                tier="raw",
                subset=metadata.get("suite_name"),
                record_locator=_record_locator(
                    {"index": record.get("index"), **metadata},
                    line_number,
                    ["index", "name", "user_task_id", "injection_task_id"],
                ),
                ingest_version="etl_a_v3",  # Updated version for CB-aware categories
                model_id=model_id,
                model_family=model_family,
                format_family=format_family,
            ),
            messages=messages,
            split=split,
            task=TraceTask(
                family="prompt_injection",
                name=metadata.get("suite_name"),
                variant=metadata.get("attack_type") or metadata.get("injection_task_id"),
            ),
            labels=TraceLabels(
                category=category,  # Now properly: harmful/resisted/benign
                security_outcome="unsafe" if attack_succeeded else "safe",
                attack_type="prompt_injection" if is_attack else None,
                attack_succeeded=attack_succeeded,
                attack_present=is_attack,  # NEW: explicit attack presence flag
                capability_category=metadata.get("suite_name") if not is_attack else None,
            ),
            training=TraceTraining(
                sample_weight=1.0,
                loss_mask_policy="assistant_only",
                mixture=TraceMixture(
                    # Class ID reflects actual category for MWCS
                    class_id=f"agentdojo/{category}"
                ),
            ),
            links=TraceLinks(raw_id=metadata.get("name")),
            signal_hints=signal_hints,
            raw_metadata=raw_metadata if raw_metadata.source_fields else None,
        )
        return trace
    except Exception as exc:
        logger.warning("Failed to convert AgentDojo record: %s", exc)
        return None


# =============================================================================
# Writers
# =============================================================================

def _write_converted(
    input_path: Path,
    converter,
    output_handle,
    split: str,
    limit: Optional[int] = None,
) -> Tuple[int, int]:
    converted = 0
    failed = 0
    for line_number, record in _iter_jsonl(input_path):
        if limit is not None and converted >= limit:
            break
        trace = converter(record, split, line_number)
        if trace:
            output_handle.write(json.dumps(trace.to_dict()) + "\n")
            converted += 1
        else:
            failed += 1
    return converted, failed


# =============================================================================
# CLI
# =============================================================================

def _load_system_prompt_from_schema(schema_path: Path) -> Optional[str]:
    """Load system prompt from a tool schema JSON file."""
    try:
        with open(schema_path, "r") as f:
            schema = json.load(f)
        return schema.get("system_prompt")
    except Exception as e:
        logger.warning("Failed to load system prompt from %s: %s", schema_path, e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw Tier A data to trace_v1")

    parser.add_argument("--fujitsu-dir", type=Path, help="Directory containing Fujitsu JSONL files")
    parser.add_argument("--fujitsu-b4", type=Path, help="Fujitsu B4 JSONL file (orchestrator attacks)")

    parser.add_argument("--agentdojo", type=Path, help="AgentDojo JSONL file")
    parser.add_argument("--agentdojo-limit", type=int, help="Limit AgentDojo records (for testing)")

    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file for trace_v1")
    parser.add_argument("--split", default="train", help="Split assignment")

    # System prompt options (for datasets that don't include one, like Fujitsu B4)
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt to use for datasets that don't include one (e.g., Fujitsu B4). "
             "If not provided, a placeholder will be used that should be overridden at generation time.",
    )
    parser.add_argument(
        "--tool-schema",
        type=Path,
        default=None,
        help="Path to tool schema JSON file to extract system_prompt from. "
             "Takes precedence over --system-prompt.",
    )

    # Format-agnostic mode
    parser.add_argument(
        "--strip-tool-syntax",
        action="store_true",
        default=False,
        help="Strip tool call syntax from assistant message content, storing only reasoning text. "
             "Enables format-agnostic storage for cross-model training.",
    )

    # Injection detection configuration
    parser.add_argument(
        "--injection-patterns",
        type=str,
        nargs="*",
        default=None,
        help="Custom regex patterns for injection detection. If not provided, uses defaults. "
             "Example: --injection-patterns '<IMPORTANT>.*?</IMPORTANT>' '<SECRET>.*?</SECRET>'"
    )
    parser.add_argument(
        "--injection-patterns-file",
        type=Path,
        default=None,
        help="Path to JSON file containing list of injection patterns. "
             "Takes precedence over --injection-patterns."
    )

    args = parser.parse_args()

    # Load system prompt from tool schema if provided
    system_prompt = args.system_prompt
    if args.tool_schema:
        schema_prompt = _load_system_prompt_from_schema(args.tool_schema)
        if schema_prompt:
            system_prompt = schema_prompt
            logger.info("Loaded system prompt from %s", args.tool_schema)
        else:
            logger.warning("No system_prompt found in %s, using default", args.tool_schema)

    # Load injection patterns
    injection_pattern_strings = None
    if args.injection_patterns_file and args.injection_patterns_file.exists():
        try:
            with open(args.injection_patterns_file) as f:
                injection_pattern_strings = json.load(f)
            logger.info("Loaded %d injection patterns from %s",
                       len(injection_pattern_strings), args.injection_patterns_file)
        except Exception as e:
            logger.warning("Failed to load injection patterns from %s: %s",
                          args.injection_patterns_file, e)
    elif args.injection_patterns:
        injection_pattern_strings = args.injection_patterns
        logger.info("Using %d custom injection patterns", len(injection_pattern_strings))

    # Compile patterns (uses defaults if None)
    injection_patterns = compile_injection_patterns(injection_pattern_strings)
    logger.info("Compiled %d injection detection patterns", len(injection_patterns))

    inputs: List[Tuple[Path, Any, Optional[int]]] = []

    # Use functools.partial to bind system_prompt to Fujitsu B4 converter
    from functools import partial
    fujitsu_b4_converter = partial(convert_fujitsu_b4_record, system_prompt=system_prompt)

    # Bind strip_tool_syntax and injection_patterns to AgentDojo converter
    agentdojo_converter = partial(
        convert_agentdojo_record,
        strip_tool_syntax=args.strip_tool_syntax,
        injection_patterns=injection_patterns,
    )

    if args.fujitsu_dir:
        for key, filename in FUJITSU_FILE_MAP.items():
            file_path = args.fujitsu_dir / filename
            if file_path.exists():
                converter = {
                    "b4": fujitsu_b4_converter,
                }[key]
                inputs.append((file_path, converter, None))
            else:
                logger.warning("Missing Fujitsu file: %s", file_path)

    if args.fujitsu_b4:
        inputs.append((args.fujitsu_b4, fujitsu_b4_converter, None))

    if args.agentdojo:
        inputs.append((args.agentdojo, agentdojo_converter, args.agentdojo_limit))

    if not inputs:
        parser.error("No input files provided. Use --fujitsu-dir, --fujitsu-b4, or --agentdojo.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_converted = 0
    total_failed = 0

    with open(args.output, "w") as out_f:
        for path, converter, limit in inputs:
            logger.info("Converting %s", path)
            converted, failed = _write_converted(path, converter, out_f, args.split, limit=limit)
            total_converted += converted
            total_failed += failed

    logger.info("Converted %s traces, %s failed", total_converted, total_failed)


if __name__ == "__main__":
    main()
