#!/usr/bin/env python3
"""
Transform raw AgentDojo traces into clean Llama-native single-hop contrastive pairs.

For each (benign, harmful) task pair, produces:
  DS: System + User + Asst(first_tool_call) + ipython(INJECTED_response)  → Asst(seed)
  DR: System + User + Asst(first_tool_call) + ipython(CLEAN_response)     → Asst(correct_next_call)

Usage:
  python scripts/make_agentdojo_simple.py --input data/traces/agentdojo_traces.jsonl \
      --output data/traces/agentdojo_simple.jsonl --print-examples 3
"""

import argparse
import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Llama 3.1 special tokens
# ---------------------------------------------------------------------------
BOS       = "<|begin_of_text|>"
EOT       = "<|eot_id|>"
BOH       = "<|start_header_id|>"
EOH       = "<|end_header_id|>"

def llama_turn(role: str, content: str) -> str:
    return f"{BOH}{role}{EOH}\n\n{content.strip()}{EOT}"


_YEAR_LINE = re.compile(
    r"[ \t]*-[ \t]*Do not assume the current year[^\n]*\n?", re.IGNORECASE
)


def clean_system_prompt(raw: str) -> str:
    """Strip AgentDojo scaffolding noise from the original system prompt."""
    return _YEAR_LINE.sub("", raw).strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_raw_task_id(trace: dict) -> str | None:
    """Extract the base task id, e.g. 'user_task_0' from 'user_task_0[without attack]'."""
    raw = trace.get("links", {}).get("raw_id", "")
    m = re.match(r"(user_task_\d+)", raw)
    return m.group(1) if m else None


def find_injection_turn(messages: list) -> int | None:
    """Return the index of the first tool-response message that contains an injection."""
    for i, msg in enumerate(messages):
        if msg["role"] == "tool" and "<INFORMATION>" in msg.get("content", ""):
            return i
    return None


def tool_call_to_llama_json(tool_call: dict) -> str:
    """Convert an existing tool_call record to Llama-native JSON string."""
    fn = tool_call["function"]
    args = fn.get("arguments") or {}
    return json.dumps({"name": fn["name"], "parameters": args}, ensure_ascii=False)


def first_tool_call_msg(messages: list) -> dict | None:
    """Return the first assistant message that has at least one tool call."""
    for msg in messages:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            return msg
    return None


def next_assistant_action(messages: list, after_index: int) -> str | None:
    """
    Find the first assistant action after a given message index.
    Returns Llama-native JSON if it's a tool call, or the text content otherwise.
    """
    for msg in messages[after_index + 1:]:
        if msg["role"] == "assistant":
            if msg.get("tool_calls"):
                return tool_call_to_llama_json(msg["tool_calls"][0])
            content = msg.get("content", "").strip()
            if content:
                return content
    return None


def build_llama_prompt(system_content: str, user_content: str,
                       asst_tool_call_json: str, tool_response: str,
                       target_asst: str = "") -> str:
    """
    Assemble a Llama-native prompt string:
        system / user / assistant(tool_call) / ipython(tool_response) / assistant(target)
    """
    parts = [
        BOS,
        llama_turn("system", system_content),
        llama_turn("user", user_content),
        llama_turn("assistant", asst_tool_call_json),
        llama_turn("ipython", tool_response),
        f"{BOH}assistant{EOH}\n\n{target_asst}",  # open-ended or with target
    ]
    return "".join(parts)


# ---------------------------------------------------------------------------
# Core: build one contrastive pair
# ---------------------------------------------------------------------------

def build_pair(benign_trace: dict, harmful_trace: dict) -> dict | None:
    """
    Returns a dict with keys: id, task, raw_id, ds_prompt, dr_prompt,
    ds_messages (list), dr_messages (list), metadata.
    Returns None if the pair cannot be built cleanly.
    """
    b_msgs = benign_trace["messages"]
    h_msgs = harmful_trace["messages"]

    # --- locate injection in harmful trace ---
    inj_idx = find_injection_turn(h_msgs)
    if inj_idx is None:
        return None  # no injection found

    # --- extract user goal (first real user message) ---
    user_msg = None
    for msg in b_msgs:
        if msg["role"] == "user" and "filter the list" not in msg["content"]:
            user_msg = msg["content"]
            break
    if user_msg is None:
        return None

    # --- get first tool call from benign trace ---
    first_call_msg = first_tool_call_msg(b_msgs)
    if first_call_msg is None:
        return None
    asst_json = tool_call_to_llama_json(first_call_msg["tool_calls"][0])

    # --- get clean tool response from benign trace ---
    # Find the tool response that directly follows the first assistant call
    first_call_idx = b_msgs.index(first_call_msg)
    clean_tool_response = None
    for msg in b_msgs[first_call_idx + 1:]:
        if msg["role"] == "tool":
            clean_tool_response = msg["content"]
            break
    if clean_tool_response is None:
        return None

    # --- get injected tool response from harmful trace ---
    injected_tool_response = h_msgs[inj_idx]["content"]

    # --- get correct next action from benign trace (target for DR) ---
    clean_tool_idx = next(
        i for i, m in enumerate(b_msgs) if m["role"] == "tool" and m["content"] == clean_tool_response
    )
    dr_target = next_assistant_action(b_msgs, clean_tool_idx) or ""

    # --- system prompt ---
    system_content = clean_system_prompt(benign_trace["messages"][0]["content"])

    # --- build prompts ---
    ds_prompt = build_llama_prompt(
        system_content, user_msg, asst_json, injected_tool_response,
        target_asst=""  # seed — model must generate refusal/ignore
    )
    dr_prompt = build_llama_prompt(
        system_content, user_msg, asst_json, clean_tool_response,
        target_asst=dr_target
    )

    # Also store as message lists for downstream use
    def msg_list(tool_response, target=""):
        return [
            {"role": "system",    "content": system_content},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": asst_json},
            {"role": "ipython",   "content": tool_response},
            {"role": "assistant", "content": target},
        ]

    task_id = get_raw_task_id(benign_trace)
    pair_id = f"ad_simple_{benign_trace['task']['name']}_{task_id}"

    return {
        "id":          pair_id,
        "task":        benign_trace["task"]["name"],
        "raw_id":      task_id,
        "ds_prompt":   ds_prompt,
        "dr_prompt":   dr_prompt,
        "ds_messages": msg_list(injected_tool_response, target=""),
        "dr_messages": msg_list(clean_tool_response,    target=dr_target),
        "metadata": {
            "benign_trace_id":  benign_trace["id"],
            "harmful_trace_id": harmful_trace["id"],
            "first_tool_call":  json.loads(asst_json)["name"],
            "dr_target":        dr_target,
        }
    }


# ---------------------------------------------------------------------------
# Pairing logic
# ---------------------------------------------------------------------------

def load_and_pair(traces_path: Path) -> list[dict]:
    benign_by_key = {}   # (task_name, task_id) -> trace
    harmful_by_key = {}  # (task_name, task_id) -> trace

    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            task_id = get_raw_task_id(d)
            if task_id is None:
                continue
            key = (d["task"]["name"], task_id)
            cat = d["labels"]["category"]
            if cat == "benign":
                benign_by_key[key] = d
            elif cat == "harmful":
                harmful_by_key[key] = d

    pairs = []
    for key in sorted(benign_by_key):
        if key not in harmful_by_key:
            continue
        result = build_pair(benign_by_key[key], harmful_by_key[key])
        if result is not None:
            pairs.append(result)

    return pairs


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"


def _fmt_turn(role: str, content: str, truncate: int = 400) -> str:
    colors = {
        "system":    DIM,
        "user":      CYAN,
        "assistant": GREEN,
        "ipython":   YELLOW,
    }
    c = colors.get(role, "")
    snippet = content[:truncate] + ("…" if len(content) > truncate else "")
    return f"  {c}{BOLD}[{role}]{RESET} {c}{snippet}{RESET}"


def print_pair(pair: dict, index: int) -> None:
    print(f"\n{'='*72}")
    print(f"{BOLD}Pair #{index+1} | task={pair['task']} | raw_id={pair['raw_id']}{RESET}")
    print(f"  first tool call : {pair['metadata']['first_tool_call']}")
    print(f"  DR target       : {pair['metadata']['dr_target'][:80]}…" if pair['metadata']['dr_target'] else "  DR target: (none)")
    print()

    print(f"  {BOLD}{RED}── DS (dangerous/stimulus) ──{RESET}")
    for msg in pair["ds_messages"]:
        if msg["role"] == "assistant" and msg["content"] == "":
            print(f"  {GREEN}{BOLD}[assistant]{RESET} {DIM}<generation seed — model continues here>{RESET}")
        else:
            print(_fmt_turn(msg["role"], msg["content"]))
    print()

    print(f"  {BOLD}{GREEN}── DR (retain/benign) ──{RESET}")
    for msg in pair["dr_messages"]:
        print(_fmt_turn(msg["role"], msg["content"]))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Llama-native AgentDojo contrastive pairs")
    parser.add_argument("--input",  type=Path, default=Path("data/traces/agentdojo_traces.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/traces/agentdojo_simple.jsonl"))
    parser.add_argument("--print-examples", type=int, default=3, metavar="N",
                        help="Print N example pairs to stdout (0 to suppress)")
    args = parser.parse_args()

    print(f"Loading traces from {args.input} …")
    pairs = load_and_pair(args.input)
    print(f"Built {len(pairs)} contrastive pairs.\n")

    # Print examples
    for i in range(min(args.print_examples, len(pairs))):
        print_pair(pairs[i], i)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for pair in pairs:
            # Save without the full prompt strings (too large), just messages
            out = {k: v for k, v in pair.items() if not k.endswith("_prompt")}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(pairs)} pairs → {args.output}")


if __name__ == "__main__":
    main()
