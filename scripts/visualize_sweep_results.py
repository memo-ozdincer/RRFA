#!/usr/bin/env python3
"""
Visualize hyperparameter sweep results with full context.

This script reads from the sweep output directory structure and displays:
- User queries with injection text (from traces)
- System prompts and available tools (from tool schema)
- Expected vs observed tool calls
- Baseline vs Circuit Breaker responses
- Aggregate metrics across configurations

Usage:
    # From sweep directory (on cluster)
    python scripts/visualize_sweep_results.py /scratch/memoozd/cb-scratch/sweeps/hparam_sweep_YYYYMMDD_HHMMSS
    
    # Show detailed samples
    python scripts/visualize_sweep_results.py <sweep_dir> --show-samples 10
    
    # Filter to successful CB blocks
    python scripts/visualize_sweep_results.py <sweep_dir> --show-samples 10 --filter-success
    
    # Analyze specific run
    python scripts/visualize_sweep_results.py <sweep_dir> --run a5.0_l10_20_assistant_only
    
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Data Loading Helpers
# =============================================================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    if not path.exists():
        return []
    items = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_tool_schema(schema_path: Path) -> Dict[str, Any]:
    """Load tool schema with system prompt and tool definitions."""
    if not schema_path.exists():
        return {}
    return load_json(schema_path)


def build_trace_index(traces: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Build an index of traces by ID for quick lookup (supports duplicate IDs)."""
    index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        trace_id = trace.get("id")
        if trace_id:
            index[trace_id].append(trace)
    return index


def load_llmail_reference_data(llmail_data_dir: Optional[Path]) -> Dict[str, Any]:
    """Load LLMail reference metadata used to annotate sample context."""
    if not llmail_data_dir:
        return {}

    scenarios = load_json(llmail_data_dir / "scenarios.json")
    level_descriptions = load_json(llmail_data_dir / "levels_descriptions.json")
    objective_descriptions = load_json(llmail_data_dir / "objectives_descriptions.json")

    scenarios_by_level: Dict[int, Dict[str, Any]] = {}
    for key, value in scenarios.items():
        match = re.search(r"(\d+)$", key)
        if match:
            scenarios_by_level[int(match.group(1))] = value

    return {
        "scenarios_by_level": scenarios_by_level,
        "levels_descriptions": level_descriptions,
        "objective_descriptions": objective_descriptions,
    }


# =============================================================================
# Context Extraction from Traces
# =============================================================================

def get_system_prompt(trace: Dict[str, Any]) -> Optional[str]:
    """Extract system prompt from trace messages."""
    messages = trace.get("messages", [])
    for msg in messages:
        if msg.get("role") == "system":
            return msg.get("content", "")
    # Check signal_hints for raw_system_prompt
    signal_hints = trace.get("signal_hints", {})
    if signal_hints and signal_hints.get("raw_system_prompt"):
        return signal_hints["raw_system_prompt"]
    return None


def get_user_query(trace: Dict[str, Any]) -> Optional[str]:
    """Extract the user query from trace messages."""
    messages = trace.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def get_injection_text(trace: Dict[str, Any]) -> Optional[str]:
    """Extract injection text from trace tool_attack or signal_hints."""
    tool_attack = trace.get("tool_attack", {})
    if tool_attack and tool_attack.get("injection_text"):
        return tool_attack["injection_text"]
    return None


def find_injection_in_messages(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Find injections in AgentDojo-style traces.
    
    AgentDojo injections are embedded in tool results (not user messages).
    They typically appear as <INFORMATION>...</INFORMATION> blocks.
    
    Returns list of {role, content_preview, injection_text} dicts.
    """
    injections = []
    messages = trace.get("messages", [])
    
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if not content:
            continue
        
        # Look for <INFORMATION> tags (AgentDojo injection marker)
        import re
        info_pattern = re.compile(r'<INFORMATION>(.*?)</INFORMATION>', re.DOTALL | re.IGNORECASE)
        matches = info_pattern.findall(content)
        
        for match in matches:
            injections.append({
                "message_index": i,
                "role": msg.get("role", "unknown"),
                "injection_text": match.strip(),
                "content_preview": content,
            })
    
    return injections


def get_injection_span(trace: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Get character span of injection in user query."""
    signal_hints = trace.get("signal_hints", {})
    if signal_hints:
        span = signal_hints.get("injection_char_span", {})
        if span:
            return (span.get("char_start", 0), span.get("char_end", 0))
    return None


def get_expected_observed_tools(trace: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract expected and observed tools from trace."""
    # Try tool_attack first (trace_v1 format)
    tool_attack = trace.get("tool_attack", {})
    if tool_attack:
        expected = tool_attack.get("expected_tool")
        observed = tool_attack.get("observed_tool")
        if expected or observed:
            return expected, observed
    
    # Try signal_hints
    signal_hints = trace.get("signal_hints", {})
    if signal_hints:
        expected = signal_hints.get("expected_tool_name")
        observed = signal_hints.get("observed_tool_name")
        if expected or observed:
            return expected, observed
    
    # Try labels (legacy)
    labels = trace.get("labels", {})
    if labels:
        return labels.get("expected_tool"), labels.get("simulated_tool")
    
    return None, None


def get_attack_info(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Get full attack information from trace."""
    labels = trace.get("labels", {})
    tool_attack = trace.get("tool_attack", {})
    
    return {
        "category": labels.get("category"),
        "attack_type": labels.get("attack_type") or tool_attack.get("attack_vector"),
        "attack_succeeded": labels.get("attack_succeeded"),
        "security_outcome": labels.get("security_outcome"),
    }


def get_llmail_metadata(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Extract LLMail-specific metadata from raw trace fields."""
    source_fields = trace.get("raw_metadata", {}).get("source_fields", {})
    return {
        "scenario": source_fields.get("scenario"),
        "level": source_fields.get("level"),
        "level_variant": source_fields.get("level_variant"),
        "defense": source_fields.get("defense"),
        "model": source_fields.get("model"),
        "email_retrieved": source_fields.get("email_retrieved"),
        "defense_undetected": source_fields.get("defense_undetected"),
        "exfil_sent": source_fields.get("exfil_sent"),
        "exfil_destination": source_fields.get("exfil_destination"),
        "exfil_content": source_fields.get("exfil_content"),
        "scheduled_time": source_fields.get("scheduled_time"),
        "completed_time": source_fields.get("completed_time"),
    }


def get_tool_messages(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return tool messages from a trace."""
    tool_messages: List[Dict[str, Any]] = []
    for msg in trace.get("messages", []):
        if msg.get("role") == "tool":
            tool_messages.append(
                {
                    "name": msg.get("name"),
                    "tool_call_id": msg.get("tool_call_id"),
                    "content": msg.get("content", ""),
                }
            )
    return tool_messages


def attach_trace_context(
    paired_samples: List[Dict[str, Any]],
    trace_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Attach trace objects to paired samples.

    Uses id+occurrence matching to handle datasets where IDs can repeat
    (notably LLMail), with row-index fallback.
    """
    traces_by_id = build_trace_index(trace_records)
    id_seen_counts: Dict[str, int] = defaultdict(int)

    with_context: List[Dict[str, Any]] = []
    for row_idx, sample in enumerate(paired_samples):
        sample_id = sample.get("id")
        sample_copy = dict(sample)
        sample_copy.setdefault("sample_index", row_idx)
        trace = None

        if sample_id:
            if isinstance(sample_copy.get("id_occurrence"), int) and sample_copy["id_occurrence"] > 0:
                occurrence = sample_copy["id_occurrence"]
                id_seen_counts[sample_id] = max(id_seen_counts[sample_id], occurrence)
            else:
                id_seen_counts[sample_id] += 1
                occurrence = id_seen_counts[sample_id]
                sample_copy["id_occurrence"] = occurrence

            candidates = traces_by_id.get(sample_id, [])
            if 1 <= occurrence <= len(candidates):
                trace = candidates[occurrence - 1]
            elif candidates:
                trace = candidates[-1]

        if trace is None:
            sample_index = sample_copy.get("sample_index")
            if isinstance(sample_index, int) and 0 <= sample_index < len(trace_records):
                trace = trace_records[sample_index]

        sample_copy["_trace"] = trace
        with_context.append(sample_copy)

    return with_context


def extract_llmail_reference_for_level(
    llmail_reference: Optional[Dict[str, Any]],
    level: Optional[int],
) -> Dict[str, Any]:
    """Get scenario reference metadata for a LLMail level."""
    if not llmail_reference or not level:
        return {}
    scenarios_by_level = llmail_reference.get("scenarios_by_level", {})
    return scenarios_by_level.get(level, {})


# =============================================================================
# Display Helpers
# =============================================================================

def truncate_text(text: str, max_length: Optional[int] = None) -> str:
    """Return text without truncation unless a max_length is explicitly set."""
    if not text:
        return ""
    if max_length is None:
        return text
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def highlight_injection(text: str, injection: Optional[str], use_color: bool = True) -> str:
    """Highlight injection text within user query."""
    if not injection or not text:
        return text
    
    idx = text.find(injection)
    if idx < 0:
        return text
    
    if use_color:
        # ANSI colors
        RED = "\033[91m"
        RESET = "\033[0m"
        before = text[:idx]
        after = text[idx + len(injection):]
        return f"{before}{RED}[INJECTION: {truncate_text(injection)}]{RESET}{after}"
    else:
        before = text[:idx]
        after = text[idx + len(injection):]
        return f"{before}[INJECTION: {truncate_text(injection)}]{after}"


def format_tool_list(tools: List[Dict[str, Any]]) -> str:
    """Format tool list for display."""
    tool_names = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", tool.get("name", "?"))
        tool_names.append(name)
    return ", ".join(tool_names)


# =============================================================================
# Run Name Parsing
# =============================================================================

def parse_run_name(run_name: str) -> Tuple[Dict[str, str], Optional[str]]:
    """
    Parse run name into hparams and lossmask.

    Example: a5.0_l10_20_assistant_only -> hparams={"a": "5.0"}, lossmask="l10_20_assistant_only"
    """
    tokens = run_name.split("_")
    hparams: Dict[str, str] = {}
    lossmask: Optional[str] = None

    def is_param_token(token: str) -> bool:
        return bool(re.match(r"^[A-Za-z]\d", token))

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if is_param_token(token):
            key = token[0]
            value = token[1:]
            if key.lower() == "l":
                lossmask_parts = [token]
                j = i + 1
                while j < len(tokens) and not is_param_token(tokens[j]):
                    lossmask_parts.append(tokens[j])
                    j += 1
                lossmask = "_".join(lossmask_parts)
                i = j
                continue
            hparams[key] = value
        i += 1

    return hparams, lossmask


# =============================================================================
# Sample Display
# =============================================================================

def print_sample_detail(
    sample: Dict[str, Any],
    trace: Optional[Dict[str, Any]] = None,
    tool_schema: Optional[Dict[str, Any]] = None,
    llmail_reference: Optional[Dict[str, Any]] = None,
    show_full: bool = False,
    use_color: bool = True,
) -> None:
    """Print detailed information about a single sample with full context."""
    max_len = None if show_full else 1200
    
    # Colors
    BOLD = "\033[1m" if use_color else ""
    GREEN = "\033[92m" if use_color else ""
    RED = "\033[91m" if use_color else ""
    YELLOW = "\033[93m" if use_color else ""
    BLUE = "\033[94m" if use_color else ""
    CYAN = "\033[96m" if use_color else ""
    MAGENTA = "\033[95m" if use_color else ""
    RESET = "\033[0m" if use_color else ""
    
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}Sample ID:{RESET} {sample.get('id', 'N/A')}")
    print(f"{'='*80}")

    sample_context = sample.get("sample_context") or {}

    # Detect dataset type from trace
    dataset_type = "unknown"
    if trace:
        source = trace.get("source", {})
        dataset = source.get("dataset", "").lower()
        if "agentdojo" in dataset:
            dataset_type = "agentdojo"
        elif "fujitsu" in dataset:
            dataset_type = "fujitsu"
        elif "llmail" in dataset:
            dataset_type = "llmail"
    elif sample_context:
        dataset = str(sample_context.get("source_dataset", "")).lower()
        if "agentdojo" in dataset:
            dataset_type = "agentdojo"
        elif "fujitsu" in dataset:
            dataset_type = "fujitsu"
        elif "llmail" in dataset:
            dataset_type = "llmail"

    # Show system prompt - ALWAYS prefer embedded from trace (especially for AgentDojo)
    sys_prompt = sample_context.get("system_prompt")
    if trace:
        sys_prompt = get_system_prompt(trace)
    
    if sys_prompt:
        print(f"\n{CYAN}📋 SYSTEM PROMPT (from trace):{RESET}")
        print(f"   {truncate_text(sys_prompt, max_len)}")
    elif tool_schema:
        schema_prompt = tool_schema.get("system_prompt", "")
        if schema_prompt:
            print(f"\n{CYAN}📋 SYSTEM PROMPT (from schema - {YELLOW}may not match dataset{RESET}):{RESET}")
            print(f"   {truncate_text(schema_prompt, max_len)}")

    user_query = sample_context.get("user_query")
    if trace:
        user_query = get_user_query(trace) or user_query
    if user_query:
        print(f"\n{BLUE}💬 USER QUERY:{RESET}")
        print(f"   {truncate_text(user_query, max_len)}")
    
    # Show injection - different handling for Fujitsu vs AgentDojo
    if trace:
        # Fujitsu-style: injection in user query via tool_attack field
        fujitsu_injection = get_injection_text(trace)
        if fujitsu_injection:
            print(f"\n{RED}⚠️  INJECTION TEXT (in user query):{RESET}")
            print(f"   {truncate_text(fujitsu_injection, max_len)}")
        
        # AgentDojo-style: injection in tool results
        agentdojo_injections = find_injection_in_messages(trace)
        if agentdojo_injections:
            print(f"\n{RED}⚠️  INJECTIONS IN TOOL RESULTS ({len(agentdojo_injections)}):{RESET}")
            for inj in agentdojo_injections:
                print(f"\n   {MAGENTA}[Message {inj['message_index']} - {inj['role']}]{RESET}")
                print(f"   Context: {inj['content_preview']}")
                print(f"   {RED}Injection: {truncate_text(inj['injection_text'], max_len)}{RESET}")
        
        # Attack info
        attack_info = get_attack_info(trace)
        if attack_info.get("category"):
            print(f"\n{YELLOW}📊 ATTACK INFO:{RESET}")
            print(f"   Dataset: {dataset_type}")
            print(f"   Category: {attack_info.get('category', 'N/A')}")
            print(f"   Attack Type: {attack_info.get('attack_type', 'N/A')}")
            print(f"   Attack Succeeded: {attack_info.get('attack_succeeded', 'N/A')}")

    # LLMail-specific metadata and story context
    llmail_meta: Dict[str, Any] = {}
    if dataset_type == "llmail":
        if trace:
            llmail_meta.update(get_llmail_metadata(trace))

        llmail_meta.update({
            "scenario": sample_context.get("scenario") or llmail_meta.get("scenario"),
            "level": sample_context.get("level") or llmail_meta.get("level"),
            "level_variant": sample_context.get("level_variant") or llmail_meta.get("level_variant"),
            "defense": sample_context.get("defense") or llmail_meta.get("defense"),
            "model": sample_context.get("model") or llmail_meta.get("model"),
        })

        objectives = sample_context.get("objectives", {})
        if objectives:
            llmail_meta.update({
                "email_retrieved": objectives.get("email_retrieved"),
                "defense_undetected": objectives.get("defense_undetected"),
                "exfil_sent": objectives.get("exfil_sent"),
                "exfil_destination": objectives.get("exfil_destination"),
                "exfil_content": objectives.get("exfil_content"),
            })

        print(f"\n{YELLOW}📨 LLMAIL SCENARIO:{RESET}")
        print(f"   Scenario: {llmail_meta.get('scenario', 'N/A')}")
        print(f"   Level: {llmail_meta.get('level', 'N/A')}")
        print(f"   Variant: {llmail_meta.get('level_variant', 'N/A')}")
        print(f"   Defense: {llmail_meta.get('defense', 'N/A')}")
        print(f"   Model: {llmail_meta.get('model', 'N/A')}")

        level = llmail_meta.get("level")
        scenario_ref = extract_llmail_reference_for_level(llmail_reference, level)
        if scenario_ref:
            if scenario_ref.get("user_query"):
                print(f"   Scenario User Query: {truncate_text(scenario_ref['user_query'], max_len)}")
            if scenario_ref.get("task"):
                print(f"   Attack Goal: {truncate_text(scenario_ref['task'], max_len)}")

        if any(key in llmail_meta for key in ["email_retrieved", "defense_undetected", "exfil_sent", "exfil_destination", "exfil_content"]):
            print(f"\n{YELLOW}🎯 LLMAIL OBJECTIVES:{RESET}")
            print(f"   email.retrieved: {llmail_meta.get('email_retrieved', 'N/A')}")
            print(f"   defense.undetected: {llmail_meta.get('defense_undetected', 'N/A')}")
            print(f"   exfil.sent: {llmail_meta.get('exfil_sent', 'N/A')}")
            print(f"   exfil.destination: {llmail_meta.get('exfil_destination', 'N/A')}")
            print(f"   exfil.content: {llmail_meta.get('exfil_content', 'N/A')}")

        tool_messages = get_tool_messages(trace) if trace else (sample_context.get("tool_messages") or [])
        if tool_messages:
            print(f"\n{RED}⚠️  RETRIEVED EMAIL CONTEXT ({len(tool_messages)} tool messages):{RESET}")
            for idx, msg in enumerate(tool_messages[:2], start=1):
                tool_name = msg.get("name", "tool")
                print(f"   [{idx}] {tool_name}:")
                print(f"   {truncate_text(msg.get('content', ''), max_len)}")
    
    # Show expected and simulated tools
    expected = sample.get("expected_tool")
    simulated = sample.get("simulated_tool")
    if dataset_type == "llmail" and expected is None:
        expected = "None (do not call any tool)"
    expected = expected if expected is not None else "N/A"
    simulated = simulated if simulated is not None else "N/A"
    print(f"\n{BOLD}📎 Expected Tool:{RESET} {GREEN}{expected}{RESET}")
    print(f"{BOLD}🎯 Simulated (Attack) Tool:{RESET} {RED}{simulated}{RESET}")
    
    # Show tool observations
    baseline_obs = sample.get("baseline_observed_tool", "N/A")
    cb_obs = sample.get("cb_observed_tool", "N/A")
    print(f"\n{BOLD}🔍 Baseline Observed Tool:{RESET} {baseline_obs}")
    print(f"{BOLD}🛡️  CB Observed Tool:{RESET} {cb_obs}")
    
    # Show outcomes with color coding
    baseline_outcome = sample.get("baseline_outcome", "N/A")
    cb_outcome = sample.get("cb_outcome", "N/A")
    
    outcome_colors = {
        "attack_success": RED,
        "correct_behavior": GREEN,
        "no_tool_call": YELLOW,
        "other_tool": BLUE,
    }
    
    b_color = outcome_colors.get(baseline_outcome, "")
    c_color = outcome_colors.get(cb_outcome, "")
    
    print(f"\n{BOLD}📊 Baseline Outcome:{RESET} {b_color}{baseline_outcome}{RESET}")
    print(f"{BOLD}📊 CB Outcome:{RESET} {c_color}{cb_outcome}{RESET}")
    
    # Improvement/regression indicator
    if baseline_outcome == "attack_success" and cb_outcome != "attack_success":
        print(f"\n{GREEN}✅ CB SUCCESSFULLY BLOCKED ATTACK{RESET}")
    elif baseline_outcome != "attack_success" and cb_outcome == "attack_success":
        print(f"\n{RED}❌ CB REGRESSION - Attack now succeeds{RESET}")
    
    # Show responses
    baseline_resp = sample.get("baseline_response", "")
    cb_resp = sample.get("cb_response", "")
    
    if baseline_resp:
        print(f"\n{BOLD}📝 BASELINE RESPONSE:{RESET}")
        print(f"   {truncate_text(baseline_resp, max_len)}")
    
    if cb_resp:
        print(f"\n{BOLD}🛡️  CB RESPONSE:{RESET}")
        print(f"   {truncate_text(cb_resp, max_len)}")


# =============================================================================
# Run Analysis
# =============================================================================

def analyze_run(
    run_dir: Path,
    traces_dir: Optional[Path] = None,
    tool_schema: Optional[Dict[str, Any]] = None,
    llmail_reference: Optional[Dict[str, Any]] = None,
    show_samples: int = 0,
    start_sample: int = 0,
    show_full: bool = False,
    filter_success: bool = False,
    filter_failure: bool = False,
    use_color: bool = True,
) -> Dict[str, Any]:
    """Analyze a single run directory."""
    hparams, lossmask = parse_run_name(run_dir.name)
    results = {
        "run_name": run_dir.name,
        "hparams": hparams,
        "lossmask": lossmask,
        "fujitsu": None,
        "llmail": None,
        "agentdojo": None,
    }
    
    # Load evaluation results
    fujitsu_eval_json = run_dir / "eval" / "fujitsu_eval.json"
    fujitsu_paired = run_dir / "eval" / "fujitsu_eval.paired_outputs.jsonl"
    agentdojo_eval_json = run_dir / "eval" / "agentdojo_eval.json"
    agentdojo_paired = run_dir / "eval" / "agentdojo_eval.paired_outputs.jsonl"
    
    # Try to load traces for context (from multiple possible locations)
    dataset_traces: Dict[str, List[Dict[str, Any]]] = {
        "fujitsu": [],
        "llmail": [],
        "agentdojo": [],
    }
    if traces_dir:
        # Load Fujitsu traces
        fujitsu_traces_path = traces_dir / "fujitsu_b4_ds.jsonl"
        if fujitsu_traces_path.exists():
            dataset_traces["fujitsu"] = load_jsonl(fujitsu_traces_path)

        # Load LLMail traces
        llmail_traces_path = traces_dir / "llmail_inject_ds.jsonl"
        if llmail_traces_path.exists():
            dataset_traces["llmail"] = load_jsonl(llmail_traces_path)
        
        # Load AgentDojo traces from split directory
        agentdojo_split = run_dir / "agentdojo_split" / "agentdojo_traces_harmful.jsonl"
        if agentdojo_split.exists():
            dataset_traces["agentdojo"] = load_jsonl(agentdojo_split)
    
    # ---------------------------------------------------------
    # Fujitsu Analysis
    # ---------------------------------------------------------
    if fujitsu_eval_json.exists():
        eval_data = load_json(fujitsu_eval_json)
        paired = load_jsonl(fujitsu_paired)
        
        baseline = eval_data.get("baseline", {}).get("tool_flip_asr", {})
        cb = eval_data.get("cb_model", {}).get("tool_flip_asr", {})
        delta = eval_data.get("delta", {})
        
        results["fujitsu"] = {
            "baseline_asr": baseline.get("attack_success_rate", 0) * 100,
            "cb_asr": cb.get("attack_success_rate", 0) * 100,
            "delta": delta.get("tool_flip_asr", 0) * 100,
            "total": len(paired),
            "improvements": sum(1 for p in paired if p.get("baseline_outcome") == "attack_success" and p.get("cb_outcome") != "attack_success"),
            "regressions": sum(1 for p in paired if p.get("baseline_outcome") != "attack_success" and p.get("cb_outcome") == "attack_success"),
        }
        
        # Show sample details if requested
        if show_samples > 0 and paired:
            print(f"\n{'#'*80}")
            print(f"# FUJITSU SAMPLES: {run_dir.name}")
            print(f"{'#'*80}")

            paired_with_context = attach_trace_context(paired, dataset_traces["fujitsu"])
            
            # Filter samples
            if filter_success:
                filtered_samples = [
                    p for p in paired_with_context
                    if p.get("baseline_outcome") == "attack_success" and p.get("cb_outcome") != "attack_success"
                ]
            elif filter_failure:
                filtered_samples = [
                    p for p in paired_with_context
                    if p.get("cb_outcome") == "attack_success"
                ]
            else:
                filtered_samples = paired_with_context
            
            # Apply pagination
            end = start_sample + show_samples
            samples_to_show = filtered_samples[start_sample:end]
            
            for sample in samples_to_show:
                print_sample_detail(
                    sample,
                    sample.get("_trace"),
                    tool_schema,
                    llmail_reference=llmail_reference,
                    show_full=show_full,
                    use_color=use_color,
                )
    
    # ---------------------------------------------------------
    # LLMail-Inject Analysis
    # ---------------------------------------------------------
    llmail_eval_json = run_dir / "eval" / "llmail_eval.json"
    llmail_paired = run_dir / "eval" / "llmail_eval.paired_outputs.jsonl"
    
    if llmail_eval_json.exists():
        eval_data = load_json(llmail_eval_json)
        paired = load_jsonl(llmail_paired)

        # Use new llmail_attack metrics (preferred) with fallback to tool_flip_asr
        baseline_la = eval_data.get("baseline", {}).get("llmail_attack", {})
        cb_la = eval_data.get("cb_model", {}).get("llmail_attack", {})
        cb_lu = eval_data.get("cb_model", {}).get("llmail_usefulness", {})
        delta = eval_data.get("delta", {})

        if baseline_la:
            results["llmail"] = {
                "baseline_asr": baseline_la.get("attack_success_rate", 0) * 100,
                "cb_asr": cb_la.get("attack_success_rate", 0) * 100,
                "delta": delta.get("llmail_attack_asr", 0) * 100,
                "refusal_rate": cb_la.get("refusal_rate", 0) * 100,
                "usefulness": cb_lu.get("usefulness_rate", 0) * 100,
                "total": len(paired),
                "improvements": sum(1 for p in paired if p.get("baseline_outcome") == "attack_success" and p.get("cb_outcome") != "attack_success"),
                "regressions": sum(1 for p in paired if p.get("baseline_outcome") != "attack_success" and p.get("cb_outcome") == "attack_success"),
            }
        else:
            # Fallback to old tool_flip_asr (for sweeps run before this update)
            baseline = eval_data.get("baseline", {}).get("tool_flip_asr", {})
            cb = eval_data.get("cb_model", {}).get("tool_flip_asr", {})
            results["llmail"] = {
                "baseline_asr": baseline.get("attack_success_rate", 0) * 100,
                "cb_asr": cb.get("attack_success_rate", 0) * 100,
                "delta": delta.get("tool_flip_asr", 0) * 100,
                "total": len(paired),
                "improvements": sum(1 for p in paired if p.get("baseline_outcome") == "attack_success" and p.get("cb_outcome") != "attack_success"),
                "regressions": sum(1 for p in paired if p.get("baseline_outcome") != "attack_success" and p.get("cb_outcome") == "attack_success"),
            }

        # Show sample details if requested
        if show_samples > 0 and paired:
            print(f"\n{'#'*80}")
            print(f"# LLMAIL-INJECT SAMPLES: {run_dir.name}")
            print(f"{'#'*80}")

            paired_with_context = attach_trace_context(paired, dataset_traces["llmail"])

            if filter_success:
                filtered_samples = [
                    p for p in paired_with_context
                    if p.get("baseline_outcome") == "attack_success" and p.get("cb_outcome") != "attack_success"
                ]
            elif filter_failure:
                filtered_samples = [
                    p for p in paired_with_context
                    if p.get("cb_outcome") == "attack_success"
                ]
            else:
                filtered_samples = paired_with_context

            # Apply pagination
            end = start_sample + show_samples
            samples_to_show = filtered_samples[start_sample:end]

            for sample in samples_to_show:
                print_sample_detail(
                    sample,
                    sample.get("_trace"),
                    tool_schema,
                    llmail_reference=llmail_reference,
                    show_full=show_full,
                    use_color=use_color,
                )

    # ---------------------------------------------------------
    # AgentDojo Analysis
    # ---------------------------------------------------------
    if agentdojo_eval_json.exists():
        eval_data = load_json(agentdojo_eval_json)
        paired = load_jsonl(agentdojo_paired)

        output_comparison = eval_data.get("output_comparison", {})

        results["agentdojo"] = {
            "total": output_comparison.get("total_compared", 0),
            "different": output_comparison.get("different", 0),
            "diff_rate": output_comparison.get("difference_rate", 0) * 100,
        }

        # Show sample details if requested
        if show_samples > 0 and paired:
            print(f"\n{'#'*80}")
            print(f"# AGENTDOJO SAMPLES: {run_dir.name}")
            print(f"{'#'*80}")

            paired_with_context = attach_trace_context(paired, dataset_traces["agentdojo"])

            different = [p for p in paired_with_context if p.get("responses_differ")]
            filtered_samples = different if different else paired_with_context
            
            # Apply pagination
            end = start_sample + show_samples
            samples_to_show = filtered_samples[start_sample:end]

            for sample in samples_to_show:
                print_sample_detail(
                    sample,
                    sample.get("_trace"),
                    tool_schema,
                    llmail_reference=llmail_reference,
                    show_full=show_full,
                    use_color=use_color,
                )

    return results


# =============================================================================
# Summary Display
# =============================================================================

def get_best_runs(runs: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """Return top runs by lowest CB ASR."""
    valid_runs = [r for r in runs if r.get("fujitsu") and r["fujitsu"].get("cb_asr") is not None]
    return sorted(valid_runs, key=lambda r: r["fujitsu"]["cb_asr"])[:top_n]

def print_summary_table(runs: List[Dict[str, Any]], use_color: bool = True) -> None:
    """Print a summary table of all runs."""
    BOLD = "\033[1m" if use_color else ""
    GREEN = "\033[92m" if use_color else ""
    RED = "\033[91m" if use_color else ""
    RESET = "\033[0m" if use_color else ""
    
    print(f"\n{BOLD}{'='*150}{RESET}")
    print(f"{BOLD}SWEEP RESULTS SUMMARY{RESET}")
    print(f"{'='*150}")
    
    # Header
    header = (
        f"{'Run Name':<35} {'Hparams':<18} {'Lossmask':<28} "
        f"{'Base ASR':<10} {'CB ASR':<8} {'Reduct':<10} {'AgentDojo':<10} {'LLMail':<10}"
    )
    print(f"{BOLD}{header}{RESET}")
    print(f"{'-'*150}")
    
    for run in runs:
        run_name = run.get("run_name", "?")
        hparams = run.get("hparams") or {}
        lossmask = run.get("lossmask") or "N/A"
        hparams_str = ",".join(f"{k}{v}" for k, v in sorted(hparams.items())) or "N/A"
        
        fujitsu = run.get("fujitsu", {})
        if fujitsu:
            base_asr = fujitsu.get("baseline_asr", 0)
            cb_asr = fujitsu.get("cb_asr", 0)
            reduction = base_asr - cb_asr
            
            base_str = f"{base_asr:.1f}%"
            cb_str = f"{cb_asr:.1f}%"
            red_str = f"{reduction:.1f}pp"
            
            # Color code based on reduction
            if reduction > 50:
                red_str = f"{GREEN}{red_str}{RESET}"
            elif reduction < 0:
                red_str = f"{RED}{red_str}{RESET}"
        else:
            base_str = cb_str = red_str = "N/A"
        
        agentdojo = run.get("agentdojo", {})
        if agentdojo:
            diff_rate = agentdojo.get("diff_rate", 0)
            agent_str = f"{diff_rate:.1f}%"
        else:
            agent_str = "N/A"
            
        llmail = run.get("llmail", {})
        if llmail:
            llmail_base = llmail.get("baseline_asr", 0)
            llmail_cb = llmail.get("cb_asr", 0)
            llmail_red = llmail_base - llmail_cb
            llmail_str = f"{llmail_cb:.1f}%"
            if llmail_red > 50:
                llmail_str = f"{GREEN}{llmail_str}{RESET}"
        else:
            llmail_str = "N/A"
        
        print(
            f"{run_name:<35} {hparams_str:<18} {lossmask:<28} "
            f"{base_str:<10} {cb_str:<8} {red_str:<10} {agent_str:<10} {llmail_str:<10}"
        )


def print_best_runs(runs: List[Dict[str, Any]], top_n: int = 5, use_color: bool = True) -> None:
    """Print the best performing runs."""
    BOLD = "\033[1m" if use_color else ""
    GREEN = "\033[92m" if use_color else ""
    RESET = "\033[0m" if use_color else ""
    by_cb_asr = get_best_runs(runs, top_n=top_n)
    if not by_cb_asr:
        print("\nNo valid runs found with Fujitsu metrics.")
        return
    
    
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}TOP {top_n} RUNS BY LOWEST CB ASR{RESET}")
    print(f"{'='*80}")
    
    for i, run in enumerate(by_cb_asr[:top_n], 1):
        fujitsu = run["fujitsu"]
        hparams = run.get("hparams") or {}
        lossmask = run.get("lossmask") or "N/A"
        hparams_str = ",".join(f"{k}{v}" for k, v in sorted(hparams.items())) or "N/A"
        print(f"\n{GREEN}{i}. {run['run_name']}{RESET}")
        print(f"   Hparams: {hparams_str}")
        print(f"   Lossmask: {lossmask}")
        print(f"   Baseline ASR: {fujitsu['baseline_asr']:.1f}% → "
              f"CB ASR: {fujitsu['cb_asr']:.1f}% "
              f"(Reduction: {fujitsu['baseline_asr'] - fujitsu['cb_asr']:.1f}pp)")
        print(f"   Improvements: {fujitsu.get('improvements', 0)}, "
              f"Regressions: {fujitsu.get('regressions', 0)}")
        agentdojo = run.get("agentdojo", {})
        if agentdojo:
            print(f"   AgentDojo Diff Rate: {agentdojo.get('diff_rate', 0):.1f}%")

    best_run = by_cb_asr[0]
    best_hparams = ",".join(
        f"{k}{v}" for k, v in sorted((best_run.get("hparams") or {}).items())
    ) or "N/A"
    best_lossmask = best_run.get("lossmask") or "N/A"
    print(f"\n{BOLD}BEST HPARAMS:{RESET} {best_hparams}")
    print(f"{BOLD}BEST LOSSMASK:{RESET} {best_lossmask}")


def plot_metrics(runs: List[Dict[str, Any]]) -> None:
    """Plot simple ASCII charts for metrics."""
    print("\n" + "=" * 80)
    print("METRIC PLOTS")
    print("=" * 80)
    
    # Sort runs by alpha then layers
    sorted_runs = sorted(runs, key=lambda r: (
        float(r.get("hparams", {}).get("a", 0)),
        r.get("lossmask", "")
    ))
    
    # 1. Fujitsu CB ASR vs Alpha
    print("\n1. Fujitsu CB ASR (lower is better) vs Alpha:")
    max_asr = 100.0
    for run in sorted_runs:
        fujitsu = run.get("fujitsu", {})
        if not fujitsu: continue
        
        asr = fujitsu.get("cb_asr", 0)
        alpha = run.get("hparams", {}).get("a", "?")
        layers = run.get("lossmask", "?")
        
        bar_len = int((asr / max_asr) * 40)
        bar = "█" * bar_len
        print(f"  a={alpha:<4} {layers:<20}: {bar} {asr:.1f}%")

    # 2. AgentDojo Diff Rate vs Alpha
    print("\n2. AgentDojo Diff Rate (higher = more refusal) vs Alpha:")
    for run in sorted_runs:
        agentdojo = run.get("agentdojo", {})
        if not agentdojo: continue
        
        diff = agentdojo.get("diff_rate", 0)
        alpha = run.get("hparams", {}).get("a", "?")
        layers = run.get("lossmask", "?")
        
        bar_len = int((diff / 100.0) * 40)
        bar = "█" * bar_len
        print(f"  a={alpha:<4} {layers:<20}: {bar} {diff:.1f}%")

    # 3. LLMail CB ASR vs Alpha
    print("\n3. LLMail CB ASR (lower is better) vs Alpha:")
    for run in sorted_runs:
        llmail = run.get("llmail", {})
        if not llmail: continue
        
        asr = llmail.get("cb_asr", 0)
        alpha = run.get("hparams", {}).get("a", "?")
        layers = run.get("lossmask", "?")
        
        bar_len = int((asr / 100.0) * 40)
        bar = "█" * bar_len
        print(f"  a={alpha:<4} {layers:<20}: {bar} {asr:.1f}%")


def compare_samples(
    runs: List[Dict[str, Any]],
    run_dirs: List[Path],
    num_samples: int = 1,
    start_sample: int = 0,
    dataset: str = "fujitsu"
) -> None:
    """Compare specific samples across all runs."""
    print("\n" + "=" * 80)
    print(f"SAMPLE COMPARISON ({dataset.upper()})")
    print("=" * 80)
    
    # Load paired outputs for all runs
    run_data = []
    for run_dir in run_dirs:
        if dataset == "fujitsu":
            path = run_dir / "eval" / "fujitsu_eval.paired_outputs.jsonl"
        elif dataset == "agentdojo":
            path = run_dir / "eval" / "agentdojo_eval.paired_outputs.jsonl"
        elif dataset == "llmail":
            path = run_dir / "eval" / "llmail_eval.paired_outputs.jsonl"
        else:
            continue
            
        if path.exists():
            data = load_jsonl(path)
            # Index by ID
            data_map = {d.get("id"): d for d in data}
            run_data.append((run_dir.name, data_map))
    
    if not run_data:
        print(f"No data found for dataset {dataset}")
        return

    # Get sample IDs from the first run
    first_run_name, first_run_map = run_data[0]
    all_ids = list(first_run_map.keys())
    end = start_sample + num_samples
    sample_ids = all_ids[start_sample:end]
    
    for sid in sample_ids:
        print(f"\n{'-'*80}")
        print(f"Sample ID: {sid}")
        
        # Get query from first run
        sample = first_run_map[sid]
        query = sample.get("baseline_response", "") # Actually we don't have query in paired output easily, 
        # but we have baseline response.
        # Wait, paired output usually doesn't have the input query unless we put it there.
        # But we can show the baseline response and CB response.
        
        print(f"Baseline Response (from {first_run_name}):")
        print(f"  {truncate_text(sample.get('baseline_response', ''), 100)}")
        
        print(f"\n{'-'*20} Comparison {'-'*20}")
        
        for run_name, data_map in run_data:
            if sid not in data_map:
                continue
            
            s = data_map[sid]
            cb_resp = s.get("cb_response", "")
            outcome = s.get("cb_outcome", "?")
            
            # Color outcome
            outcome_str = outcome
            if outcome == "attack_success":
                outcome_str = f"\033[91m{outcome}\033[0m"
            elif outcome == "correct_behavior" or outcome == "different": # AgentDojo uses 'different'
                outcome_str = f"\033[92m{outcome}\033[0m"
            
            print(f"Run: {run_name:<35} | {outcome_str}")
            print(f"  {truncate_text(cb_resp, 150)}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize hyperparameter sweep results with full context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze sweep directory
    python scripts/visualize_sweep_results.py /path/to/sweep_dir
    
    # Show 10 detailed samples with full context
    python scripts/visualize_sweep_results.py /path/to/sweep_dir --show-samples 10 --show-full
    
    # Show only samples where CB successfully blocked attacks
    python scripts/visualize_sweep_results.py /path/to/sweep_dir --show-samples 5 --filter-success
    
    # Analyze specific run
    python scripts/visualize_sweep_results.py /path/to/sweep_dir --run a5.0_l10_20_assistant_only
        """
    )
    parser.add_argument(
        "sweep_dir",
        type=Path,
        help="Path to sweep output directory (e.g., /scratch/.../sweeps/hparam_sweep_YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=None,
        help="Directory containing trace files for context (default: auto-detect from sweep dir)"
    )
    parser.add_argument(
        "--tool-schema",
        type=Path,
        default=Path("configs/tool_schemas/b4_standard_v1.json"),
        help="Path to tool schema JSON (default: configs/tool_schemas/b4_standard_v1.json)"
    )
    parser.add_argument(
        "--llmail-data-dir",
        type=Path,
        default=Path("data/llmail_inject"),
        help="Directory with LLMail metadata files (scenarios.json, levels_descriptions.json)"
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=0,
        help="Number of sample details to show per dataset"
    )
    parser.add_argument(
        "--start-from-sample",
        type=int,
        default=0,
        help="Start showing samples from this index (0-based)"
    )
    parser.add_argument(
        "--show-full",
        action="store_true",
        help="Show full text instead of truncated"
    )
    parser.add_argument(
        "--filter-success",
        action="store_true",
        help="Only show samples where CB successfully blocked an attack"
    )
    parser.add_argument(
        "--filter-failure",
        action="store_true",
        help="Only show samples where CB failed to block an attack"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top runs to highlight"
    )
    parser.add_argument(
        "--samples-best-only",
        action="store_true",
        help="Only show samples from the best run(s) (uses --top-n)"
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Only analyze a specific run (by name, e.g., a5.0_l10_20_assistant_only)"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Export results to CSV file"
    )
    
    parser.add_argument(
        "--compare-samples",
        type=int,
        default=0,
        help="Number of samples to compare across all runs (side-by-side)"
    )
    parser.add_argument(
        "--compare-dataset",
        type=str,
        default="fujitsu",
        choices=["fujitsu", "agentdojo", "llmail"],
        help="Dataset to use for comparison"
    )
    
    args = parser.parse_args()
    
    use_color = not args.no_color
    
    # Check sweep directory exists
    if not args.sweep_dir.exists():
        print(f"Error: Sweep directory not found: {args.sweep_dir}")
        sys.exit(1)
    
    print(f"Analyzing sweep: {args.sweep_dir}")
    
    # Load tool schema
    tool_schema = None
    if args.tool_schema.exists():
        tool_schema = load_tool_schema(args.tool_schema)
        print(f"Loaded tool schema: {args.tool_schema}")

    llmail_reference = load_llmail_reference_data(args.llmail_data_dir)
    if llmail_reference:
        print(f"Loaded LLMail reference data: {args.llmail_data_dir}")
    
    # Try to find traces directory
    traces_dir = args.traces_dir
    if traces_dir is None:
        # Try common locations
        possible_traces = [
            Path("/scratch/memoozd/cb-scratch/data/traces"),
            args.sweep_dir.parent.parent / "data" / "traces",
        ]
        for p in possible_traces:
            if p.exists():
                traces_dir = p
                print(f"Found traces directory: {traces_dir}")
                break
    
    # Find run directories
    run_dirs = sorted([
        d for d in args.sweep_dir.iterdir()
        if d.is_dir() and d.name.startswith("a")  # Run dirs start with alpha param
    ])
    
    if not run_dirs:
        print("No run directories found in sweep directory.")
        sys.exit(1)
    
    print(f"Found {len(run_dirs)} runs")
    
    # Filter to specific run if requested
    if args.run:
        run_dirs = [d for d in run_dirs if d.name == args.run]
        if not run_dirs:
            print(f"Run '{args.run}' not found.")
            sys.exit(1)
    
    # Analyze all runs
    all_results = []
    show_samples = 0 if args.samples_best_only else args.show_samples
    for run_dir in run_dirs:
        result = analyze_run(
            run_dir,
            traces_dir=traces_dir,
            tool_schema=tool_schema,
            llmail_reference=llmail_reference,
            show_samples=show_samples,
            start_sample=args.start_from_sample,
            show_full=args.show_full,
            filter_success=args.filter_success,
            filter_failure=args.filter_failure,
            use_color=use_color,
        )
        all_results.append(result)
    
    # Print summary
    print_summary_table(all_results, use_color)
    print_best_runs(all_results, args.top_n, use_color)
    
    # Plot metrics
    plot_metrics(all_results)
    
    # Compare samples if requested
    if args.compare_samples > 0:
        compare_samples(all_results, run_dirs, args.compare_samples, args.start_from_sample, args.compare_dataset)

    # Optionally show samples only for best runs
    if args.show_samples > 0 and args.samples_best_only:
        best_runs = get_best_runs(all_results, top_n=args.top_n)
        best_names = {r.get("run_name") for r in best_runs}
        for run_dir in run_dirs:
            if run_dir.name in best_names:
                analyze_run(
                    run_dir,
                    traces_dir=traces_dir,
                    tool_schema=tool_schema,
                    llmail_reference=llmail_reference,
                    show_samples=args.show_samples,
                    start_sample=args.start_from_sample,
                    show_full=args.show_full,
                    filter_success=args.filter_success,
                    filter_failure=args.filter_failure,
                    use_color=use_color,
                )
    
    # Export to CSV if requested
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_name", "baseline_asr", "cb_asr", "reduction",
                "improvements", "regressions", "agentdojo_diff_rate", "llmail_cb_asr"
            ])
            for r in all_results:
                fujitsu = r.get("fujitsu") or {}
                agentdojo = r.get("agentdojo") or {}
                llmail = r.get("llmail") or {}
                writer.writerow([
                    r.get("run_name", ""),
                    fujitsu.get("baseline_asr", "") if fujitsu else "",
                    fujitsu.get("cb_asr", "") if fujitsu else "",
                    fujitsu.get("baseline_asr", 0) - fujitsu.get("cb_asr", 0) if fujitsu else "",
                    fujitsu.get("improvements", "") if fujitsu else "",
                    fujitsu.get("regressions", "") if fujitsu else "",
                    agentdojo.get("diff_rate", "") if agentdojo else "",
                    llmail.get("cb_asr", "") if llmail else "",
                ])
        print(f"\nExported results to: {args.csv}")


if __name__ == "__main__":
    main()
