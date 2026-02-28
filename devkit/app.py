#!/usr/bin/env python3
"""
RRFA Data Pipeline Dev Kit - Local visualization and validation tool.

Usage:
    streamlit run devkit/app.py

Requires: streamlit, pandas, transformers
"""

import json
import re
import sys
import html as html_lib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schemas.trace import Trace, TraceLabels
from src.schemas.lossmask import LossMask
from src.schemas.registry import load_lmp_registry, load_mwcs_registry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
TRACES_DIR = DATA_DIR / "traces"
RENDERS_DIR = DATA_DIR / "renders"
LOSSMASKS_DIR = DATA_DIR / "lossmasks"
LMP_REGISTRY_PATH = ROOT / "configs" / "lmp_registry_v1.json"
MWCS_REGISTRY_PATH = ROOT / "configs" / "mwcs_registry_v1.json"
DEFAULT_TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"

ROLE_COLORS = {
    "system": "#6c757d",
    "user": "#0d6efd",
    "assistant": "#198754",
    "tool": "#dc3545",
}

ROLE_BG = {
    "system": "#f8f9fa",
    "user": "#e7f1ff",
    "assistant": "#d1e7dd",
    "tool": "#f8d7da",
}

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading traces...")
def load_traces(path: str) -> List[Dict]:
    """Load trace JSONL file."""
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


@st.cache_data(show_spinner="Loading renders...")
def load_renders(path: str) -> List[Dict]:
    renders = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                renders.append(json.loads(line))
    return renders


@st.cache_data(show_spinner="Loading lossmasks...")
def load_lossmasks(path: str) -> List[Dict]:
    masks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                masks.append(json.loads(line))
    return masks


def discover_jsonl(directory: Path) -> List[Path]:
    """Find all JSONL files in a directory."""
    if not directory.exists():
        return []
    return sorted(directory.glob("*.jsonl"))


@st.cache_resource(show_spinner="Loading tokenizer (one-time)...")
def get_tokenizer(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


@st.cache_resource
def get_lmp_registry():
    if LMP_REGISTRY_PATH.exists():
        return load_lmp_registry(LMP_REGISTRY_PATH)
    return load_lmp_registry()


@st.cache_resource
def get_mwcs_registry():
    try:
        if MWCS_REGISTRY_PATH.exists():
            return load_mwcs_registry(MWCS_REGISTRY_PATH)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def trace_dataset(t: Dict) -> str:
    return t.get("source", {}).get("dataset", "unknown")


def trace_category(t: Dict) -> str:
    return t.get("labels", {}).get("category", "unknown") if t.get("labels") else "unknown"


def trace_completeness(t: Dict) -> str:
    return t.get("completeness", "unknown")


def trace_has_tool_calls(t: Dict) -> bool:
    for m in t.get("messages", []):
        if m.get("tool_calls"):
            return True
    return False


def highlight_injection(text: str, span: Optional[Dict]) -> str:
    """Highlight injection span in message content."""
    if not span:
        return html_lib.escape(text)
    start = span.get("char_start", 0)
    end = span.get("char_end", 0)
    if start >= end or start >= len(text):
        return html_lib.escape(text)
    before = html_lib.escape(text[:start])
    injection = html_lib.escape(text[start:end])
    after = html_lib.escape(text[end:])
    return f'{before}<span style="background-color:#ff6b6b;color:white;padding:1px 3px;border-radius:3px;font-weight:bold">{injection}</span>{after}'


def render_message_html(msg: Dict, msg_idx: int, injection_span: Optional[Dict] = None) -> str:
    """Render a single message as styled HTML."""
    role = msg.get("role", "unknown")
    color = ROLE_COLORS.get(role, "#333")
    bg = ROLE_BG.get(role, "#fff")
    content = msg.get("content", "")

    # Highlight injection if this is the injected message
    if injection_span and injection_span.get("message_index") == msg_idx:
        content_html = highlight_injection(content, injection_span)
    else:
        content_html = html_lib.escape(content)

    # Truncate for display (keep full in expandable)
    display_content = content_html
    if len(content) > 500:
        display_content = content_html[:500] + "..."

    tool_badge = ""
    if msg.get("tool_calls"):
        tc_count = len(msg["tool_calls"])
        tc_names = [tc.get("function", {}).get("name", "?") for tc in msg["tool_calls"]]
        tool_badge = f' <span style="background:#ffc107;color:#000;padding:1px 6px;border-radius:10px;font-size:0.8em">tool_calls: {", ".join(tc_names)}</span>'

    name_badge = ""
    if msg.get("name"):
        name_badge = f' <span style="background:#17a2b8;color:white;padding:1px 6px;border-radius:10px;font-size:0.8em">{html_lib.escape(msg["name"])}</span>'

    return f"""
    <div style="border-left:4px solid {color};background:{bg};padding:8px 12px;margin:4px 0;border-radius:0 6px 6px 0;">
        <div style="font-weight:bold;color:{color};margin-bottom:4px;">
            [{msg_idx}] {role.upper()}{tool_badge}{name_badge}
        </div>
        <div style="white-space:pre-wrap;font-family:monospace;font-size:0.85em;line-height:1.4;">
            {display_content}
        </div>
    </div>
    """


def render_lossmask_heatmap_html(
    tokens: List[str],
    mask: List[float],
    message_spans: Optional[List[Dict]] = None,
    max_tokens: int = 500,
) -> str:
    """Render token-level lossmask as colored HTML.

    Green = loss applied (mask > 0), gray = masked out (mask = 0).
    """
    n = min(len(tokens), len(mask), max_tokens)
    if n == 0:
        return "<p>No tokens to display.</p>"

    # Build span boundary set for role labels
    span_starts = {}
    if message_spans:
        for sp in message_spans:
            span_starts[sp.get("token_start", -1)] = sp.get("role", "?")

    parts = []
    parts.append('<div style="font-family:monospace;font-size:0.8em;line-height:2.0;word-break:break-all;">')
    for i in range(n):
        # Role boundary marker
        if i in span_starts:
            role = span_starts[i]
            parts.append(
                f'<br><span style="background:{ROLE_COLORS.get(role,"#333")};color:white;'
                f'padding:1px 6px;border-radius:3px;font-size:0.75em;margin:2px 0;">'
                f'{role}</span><br>'
            )

        m = mask[i]
        tok = html_lib.escape(tokens[i].replace("\n", "\\n").replace("\t", "\\t"))
        if m > 0:
            # Green gradient based on mask value
            alpha = min(m, 1.0)
            bg = f"rgba(25,135,84,{0.2 + 0.6 * alpha})"
            fg = "white" if alpha > 0.5 else "#000"
        else:
            bg = "#e9ecef"
            fg = "#999"
        parts.append(
            f'<span title="[{i}] mask={m:.2f}" style="background:{bg};color:{fg};'
            f'padding:0 2px;margin:0 1px;border-radius:2px;cursor:default;">{tok}</span>'
        )

    if n < len(tokens):
        parts.append(f'<span style="color:#999;">... ({len(tokens) - n} more tokens)</span>')
    parts.append("</div>")
    return "".join(parts)


def render_policy_comparison_html(
    tokens: List[str],
    policy_masks: Dict[str, List[float]],
    max_tokens: int = 300,
) -> str:
    """Render compact heatmap grid: one row per policy."""
    n = min(len(tokens), max_tokens)
    if n == 0:
        return "<p>No tokens.</p>"

    rows = []
    for policy_name, mask in policy_masks.items():
        cells = []
        for i in range(n):
            m = mask[i] if i < len(mask) else 0
            if m > 0:
                color = f"rgba(25,135,84,{0.3 + 0.5 * min(m, 1.0)})"
            else:
                color = "#e9ecef"
            tok = html_lib.escape(tokens[i].replace("\n", " "))
            cells.append(
                f'<span title="{tok} [{i}]" style="display:inline-block;width:4px;height:16px;'
                f'background:{color};margin:0;">&nbsp;</span>'
            )
        ratio = sum(1 for x in mask[:n] if x > 0) / n if n > 0 else 0
        rows.append(
            f'<div style="margin:2px 0;display:flex;align-items:center;">'
            f'<span style="width:180px;font-size:0.8em;font-family:monospace;text-align:right;padding-right:8px;">'
            f'{policy_name} ({ratio:.0%})</span>'
            f'<div style="display:inline-flex;">{"".join(cells)}</div></div>'
        )

    return f'<div style="overflow-x:auto;">{"".join(rows)}</div>'


# ===========================================================================
# PAGE: Dashboard
# ===========================================================================

def page_dashboard():
    st.header("Dashboard")

    # Discover files
    trace_files = discover_jsonl(TRACES_DIR)
    render_files = discover_jsonl(RENDERS_DIR)
    lossmask_files = discover_jsonl(LOSSMASKS_DIR)

    col1, col2, col3 = st.columns(3)
    col1.metric("Trace files", len(trace_files))
    col2.metric("Render files", len(render_files))
    col3.metric("Lossmask files", len(lossmask_files))

    if not trace_files:
        st.warning("No trace files found in data/traces/")
        return

    # Per-file stats
    st.subheader("Trace Files")
    for tf in trace_files:
        with st.expander(f"{tf.name}", expanded=len(trace_files) <= 4):
            traces = load_traces(str(tf))
            if not traces:
                st.write("Empty file")
                continue

            # Basic counts
            datasets = Counter(trace_dataset(t) for t in traces)
            categories = Counter(trace_category(t) for t in traces)
            completeness = Counter(trace_completeness(t) for t in traces)
            has_tools = sum(1 for t in traces if trace_has_tool_calls(t))

            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Records:** {len(traces)}")
                st.write("**Datasets:**")
                st.dataframe(
                    pd.DataFrame(datasets.items(), columns=["Dataset", "Count"]).sort_values("Count", ascending=False),
                    hide_index=True, width="stretch",
                )
            with c2:
                st.write(f"**With tool calls:** {has_tools}/{len(traces)}")
                st.write("**Categories:**")
                cat_df = pd.DataFrame(categories.items(), columns=["Category", "Count"]).sort_values("Count", ascending=False)
                st.dataframe(cat_df, hide_index=True, width="stretch")

            st.write("**Completeness:**", dict(completeness))

            # Category distribution chart
            if categories:
                chart_df = pd.DataFrame(categories.items(), columns=["Category", "Count"])
                st.bar_chart(chart_df.set_index("Category"))

    # Render/lossmask stats
    if render_files:
        st.subheader("Render Files")
        for rf in render_files:
            with st.expander(rf.name):
                renders = load_renders(str(rf))
                seq_lens = [r.get("sequence_length", len(r.get("input_ids", []))) for r in renders]
                st.write(f"**Records:** {len(renders)}")
                if seq_lens:
                    st.write(f"**Seq length:** min={min(seq_lens)}, max={max(seq_lens)}, mean={sum(seq_lens)/len(seq_lens):.0f}")
                    hist = pd.Series(seq_lens).value_counts().sort_index().head(50)
                    st.bar_chart(pd.DataFrame({"Count": hist.values}, index=hist.index))

    if lossmask_files:
        st.subheader("Lossmask Files")
        for lf in lossmask_files:
            with st.expander(lf.name):
                masks = load_lossmasks(str(lf))
                st.write(f"**Records:** {len(masks)}")
                policies = Counter(m.get("policy_id", "?") for m in masks)
                ratios = [m.get("stats", {}).get("mask_ratio", 0) for m in masks if m.get("stats")]
                st.write("**Policies:**", dict(policies))
                if ratios:
                    st.write(f"**Mask ratio:** min={min(ratios):.3f}, max={max(ratios):.3f}, mean={sum(ratios)/len(ratios):.3f}")


# ===========================================================================
# PAGE: Trace Explorer
# ===========================================================================

def page_trace_explorer():
    st.header("Trace Explorer")

    trace_files = discover_jsonl(TRACES_DIR)
    if not trace_files:
        st.warning("No trace files found.")
        return

    selected_file = st.selectbox("Trace file", trace_files, format_func=lambda p: p.name)
    traces = load_traces(str(selected_file))

    if not traces:
        st.warning("No traces in file.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    datasets = sorted(set(trace_dataset(t) for t in traces))
    categories = sorted(set(trace_category(t) for t in traces))

    with col1:
        filter_dataset = st.multiselect("Dataset", datasets, default=datasets)
    with col2:
        filter_category = st.multiselect("Category", categories, default=categories)
    with col3:
        filter_completeness = st.multiselect(
            "Completeness",
            ["skeleton", "complete"],
            default=["skeleton", "complete"],
        )

    filtered = [
        t for t in traces
        if trace_dataset(t) in filter_dataset
        and trace_category(t) in filter_category
        and trace_completeness(t) in filter_completeness
    ]

    st.write(f"**Showing {len(filtered)}/{len(traces)} traces**")

    if not filtered:
        return

    # Navigation via callbacks (must set state BEFORE widget renders)
    def _go_prev():
        cur = st.session_state.get("trace_idx", 0)
        if cur > 0:
            st.session_state.trace_idx = cur - 1

    def _go_next():
        cur = st.session_state.get("trace_idx", 0)
        if cur < len(filtered) - 1:
            st.session_state.trace_idx = cur + 1

    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 6])
    with nav_col1:
        st.button("< Prev", on_click=_go_prev)
    with nav_col2:
        st.button("Next >", on_click=_go_next)

    # Trace selector
    trace_idx = st.number_input("Trace index", 0, len(filtered) - 1, 0, key="trace_idx")
    t = filtered[trace_idx]

    # Trace metadata
    st.subheader("Metadata")
    meta_cols = st.columns(4)
    meta_cols[0].write(f"**ID:** `{t.get('id', '?')[:40]}...`")
    meta_cols[1].write(f"**Dataset:** {trace_dataset(t)}")
    meta_cols[2].write(f"**Category:** {trace_category(t)}")
    meta_cols[3].write(f"**Tier:** {t.get('tier', '?')} / {trace_completeness(t)}")

    # Labels
    if t.get("labels"):
        st.write("**Labels:**", t["labels"])
    if t.get("tool_attack"):
        st.write("**Tool Attack:**", t["tool_attack"])
    if t.get("training"):
        st.write("**Training Config:**", t["training"])
    if t.get("signal_hints"):
        st.write("**Signal Hints:**", t["signal_hints"])

    # Dataset-specific metadata (AgentHarm, etc.)
    raw_meta = t.get("raw_metadata", {}).get("source_fields", {})
    if raw_meta:
        extra = {}
        if raw_meta.get("target_functions"):
            extra["target_functions"] = raw_meta["target_functions"]
        if raw_meta.get("category"):
            extra["agentharm_category"] = raw_meta["category"]
        if raw_meta.get("hint_included") is not None:
            extra["hint_included"] = raw_meta["hint_included"]
        if raw_meta.get("grading_function"):
            extra["grading_function"] = raw_meta["grading_function"]
        if raw_meta.get("name"):
            extra["behavior_name"] = raw_meta["name"]
        if extra:
            st.write("**Source Metadata:**", extra)

    # Messages
    st.subheader("Messages")
    injection_span = None
    if t.get("signal_hints") and t["signal_hints"].get("injection_char_span"):
        injection_span = t["signal_hints"]["injection_char_span"]

    messages_html = []
    for i, msg in enumerate(t.get("messages", [])):
        messages_html.append(render_message_html(msg, i, injection_span))
    st.markdown("".join(messages_html), unsafe_allow_html=True)

    # Tool calls detail
    for i, msg in enumerate(t.get("messages", [])):
        if msg.get("tool_calls"):
            with st.expander(f"Tool calls in message [{i}]"):
                for tc in msg["tool_calls"]:
                    st.json(tc)

    # Raw JSON
    with st.expander("Raw JSON"):
        st.json(t)


# ===========================================================================
# PAGE: Render & Lossmask Viewer
# ===========================================================================

def page_render_lossmask():
    st.header("Render & Lossmask Viewer")

    trace_files = discover_jsonl(TRACES_DIR)
    if not trace_files:
        st.warning("No trace files found.")
        return

    selected_file = st.selectbox("Trace file", trace_files, format_func=lambda p: p.name, key="rlm_file")
    traces = load_traces(str(selected_file))
    if not traces:
        st.warning("No traces.")
        return

    # Filter by completeness
    include_skeleton = st.checkbox("Include skeleton (B1) traces", value=True, key="rlm_skeleton")
    if include_skeleton:
        filtered_traces = traces
    else:
        filtered_traces = [t for t in traces if t.get("completeness") == "complete" or t.get("tier") == "B2"]
    if not filtered_traces:
        st.info("No traces match the filter.")
        return

    trace_idx = st.number_input("Trace index", 0, len(filtered_traces) - 1, 0, key="rlm_idx")
    t = filtered_traces[trace_idx]

    is_skeleton = t.get("completeness") == "skeleton" or t.get("tier") == "B1"
    tier_badge = "B1 skeleton" if is_skeleton else "B2 complete"
    st.write(f"**Trace:** `{t.get('id', '?')[:50]}` | **Dataset:** {trace_dataset(t)} | **Category:** {trace_category(t)} | **Tier:** {tier_badge}")

    # Show AgentHarm-specific metadata
    raw_meta = t.get("raw_metadata", {}).get("source_fields", {})
    if raw_meta.get("target_functions"):
        st.write(f"**Target functions:** `{raw_meta['target_functions']}`")
    if raw_meta.get("category"):
        st.write(f"**AgentHarm category:** {raw_meta['category']}")
    if raw_meta.get("hint_included") is not None:
        st.write(f"**Hint included:** {raw_meta['hint_included']}")

    if is_skeleton:
        st.info("This is a **skeleton trace** (B1) - no assistant messages. Only `full_sequence` and `cb_full_sequence` policies produce meaningful masks. Other policies will show empty (all-gray) masks.")

    # Policy selection
    registry = get_lmp_registry()
    policies = registry.list_policies()
    # Default to cb_full_sequence for skeletons, assistant_only for complete
    if is_skeleton:
        default_policy = "cb_full_sequence" if "cb_full_sequence" in policies else "full_sequence"
    else:
        default_policy = "cb_full_sequence" if "cb_full_sequence" in policies else "assistant_only"
    default_idx = policies.index(default_policy) if default_policy in policies else 0
    selected_policy = st.selectbox("Loss mask policy", policies, index=default_idx)

    tokenizer_name = st.text_input("Tokenizer", DEFAULT_TOKENIZER)
    max_length = st.slider("Max sequence length", 256, 16384, 4096, 256)

    if st.button("Render", type="primary"):
        with st.spinner("Rendering trace..."):
            try:
                from src.schemas.tools import ETL_B as etl_b

                tokenizer = get_tokenizer(tokenizer_name)
                trace_obj = Trace.from_dict(t)

                render = etl_b.render_trace(
                    trace_obj, tokenizer,
                    max_length=max_length,
                    add_generation_prompt=False,
                    include_rendered_text=True,
                )

                policy_id, policy = etl_b._resolve_policy(trace_obj, registry, selected_policy)
                mask_values = etl_b._apply_lmp_policy(render, policy)
                lossmask = LossMask.from_render(
                    render, policy_id=policy_id,
                    mask_fn=lambda _: mask_values,
                    policy_version=registry.version,
                    policy_params=policy.params,
                    sample_weight=trace_obj.training.sample_weight if trace_obj.training else 1.0,
                )

                # Store in session state
                st.session_state.last_render = render
                st.session_state.last_lossmask = lossmask
                st.session_state.last_tokenizer_name = tokenizer_name

            except Exception as e:
                st.error(f"Render failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Display results
    if "last_render" in st.session_state:
        render = st.session_state.last_render
        lossmask = st.session_state.last_lossmask
        tokenizer = get_tokenizer(st.session_state.last_tokenizer_name)

        # Stats
        st.subheader("Stats")
        s = lossmask.stats
        if s:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total tokens", s.total_tokens)
            c2.metric("Unmasked (loss)", s.unmasked_tokens)
            c3.metric("Masked (no loss)", s.masked_tokens)
            c4.metric("Mask ratio", f"{s.mask_ratio:.1%}")

        # Rendered text
        if render.rendered_text:
            with st.expander("Rendered text (chat template output)", expanded=False):
                st.code(render.rendered_text[:5000], language=None)

        # Alignment
        if render.alignment:
            with st.expander("Alignment spans"):
                if render.alignment.message_spans:
                    st.write("**Message spans:**")
                    spans_data = [
                        {"idx": s.message_index, "role": s.role, "start": s.token_start, "end": s.token_end, "length": s.token_end - s.token_start}
                        for s in render.alignment.message_spans
                    ]
                    st.dataframe(pd.DataFrame(spans_data), hide_index=True, width="stretch")

                if render.alignment.assistant_spans:
                    st.write("**Assistant spans:**")
                    st.dataframe(pd.DataFrame([
                        {"idx": s.message_index, "start": s.token_start, "end": s.token_end, "length": s.token_end - s.token_start}
                        for s in render.alignment.assistant_spans
                    ]), hide_index=True, width="stretch")

                if render.alignment.tool_call_spans:
                    st.write("**Tool call spans:**")
                    st.dataframe(pd.DataFrame([
                        {"msg_idx": s.message_index, "call_idx": s.call_index, "tool": s.tool_name,
                         "start": s.token_start, "end": s.token_end, "name_end": s.name_token_end}
                        for s in render.alignment.tool_call_spans
                    ]), hide_index=True, width="stretch")

        # Token-level heatmap
        st.subheader("Token-Level Loss Mask")
        max_display = st.slider("Max tokens to display", 50, 2000, 500, 50, key="heatmap_max")
        tokens = [tokenizer.decode([tid]) for tid in render.input_ids[:max_display]]
        msg_spans = [
            {"token_start": s.token_start, "role": s.role}
            for s in (render.alignment.message_spans or [])
        ] if render.alignment else None

        heatmap_html = render_lossmask_heatmap_html(tokens, lossmask.loss_mask, msg_spans, max_display)
        st.markdown(heatmap_html, unsafe_allow_html=True)

        st.caption("Green = loss applied (model learns from these tokens), Gray = masked out (ignored during training). Hover for token index and mask value.")


# ===========================================================================
# PAGE: Policy Comparator
# ===========================================================================

def page_policy_comparator():
    st.header("Policy Comparator")
    st.write("Compare all loss mask policies on the same trace.")

    trace_files = discover_jsonl(TRACES_DIR)
    if not trace_files:
        st.warning("No trace files.")
        return

    selected_file = st.selectbox("Trace file", trace_files, format_func=lambda p: p.name, key="pc_file")
    traces = load_traces(str(selected_file))
    if not traces:
        st.warning("No traces.")
        return

    include_skeleton = st.checkbox("Include skeleton (B1) traces", value=True, key="pc_skeleton")
    if include_skeleton:
        filtered = traces
    else:
        filtered = [t for t in traces if t.get("completeness") == "complete"]
    if not filtered:
        st.warning("No traces match the filter.")
        return

    trace_idx = st.number_input("Trace index", 0, len(filtered) - 1, 0, key="pc_idx")
    t = filtered[trace_idx]
    is_skeleton = t.get("completeness") == "skeleton" or t.get("tier") == "B1"
    tier_badge = "B1 skeleton" if is_skeleton else "B2 complete"
    st.write(f"**Trace:** `{t.get('id','?')[:50]}` | {trace_dataset(t)} | {trace_category(t)} | {tier_badge}")

    if is_skeleton:
        st.info("Skeleton trace: only `full_sequence` and `cb_full_sequence` will produce non-empty masks.")

    tokenizer_name = st.text_input("Tokenizer", DEFAULT_TOKENIZER, key="pc_tok")
    max_length = st.slider("Max seq length", 256, 8192, 4096, 256, key="pc_maxlen")

    if st.button("Compare All Policies", type="primary"):
        with st.spinner("Rendering with all policies..."):
            try:
                from src.schemas.tools import ETL_B as etl_b

                tokenizer = get_tokenizer(tokenizer_name)
                trace_obj = Trace.from_dict(t)
                registry = get_lmp_registry()

                render = etl_b.render_trace(
                    trace_obj, tokenizer,
                    max_length=max_length,
                    add_generation_prompt=False,
                    include_rendered_text=False,
                )

                policy_masks = {}
                policy_stats = {}
                for pid in registry.list_policies():
                    try:
                        _, policy = etl_b._resolve_policy(trace_obj, registry, pid)
                        mask = etl_b._apply_lmp_policy(render, policy)
                        policy_masks[pid] = mask
                        n = len(mask)
                        unmasked = sum(1 for x in mask if x > 0)
                        policy_stats[pid] = {
                            "policy": pid,
                            "total": n,
                            "unmasked": unmasked,
                            "masked": n - unmasked,
                            "ratio": f"{unmasked/n:.1%}" if n > 0 else "N/A",
                        }
                    except Exception as e:
                        policy_stats[pid] = {"policy": pid, "error": str(e)}

                st.session_state.pc_render = render
                st.session_state.pc_masks = policy_masks
                st.session_state.pc_stats = policy_stats
                st.session_state.pc_tokenizer = tokenizer_name

            except Exception as e:
                st.error(f"Failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if "pc_render" in st.session_state:
        render = st.session_state.pc_render
        policy_masks = st.session_state.pc_masks
        policy_stats = st.session_state.pc_stats
        tokenizer = get_tokenizer(st.session_state.pc_tokenizer)

        # Stats table
        st.subheader("Policy Stats")
        stats_rows = [v for v in policy_stats.values() if "error" not in v]
        if stats_rows:
            st.dataframe(pd.DataFrame(stats_rows), hide_index=True, width="stretch")

        errors = {k: v["error"] for k, v in policy_stats.items() if "error" in v}
        if errors:
            st.write("**Policies with errors:**", errors)

        # Compact heatmap
        st.subheader("Visual Comparison")
        max_display = st.slider("Max tokens", 50, 1500, 300, 50, key="pc_heatmax")
        tokens = [tokenizer.decode([tid]) for tid in render.input_ids[:max_display]]
        html = render_policy_comparison_html(tokens, policy_masks, max_display)
        st.markdown(html, unsafe_allow_html=True)
        st.caption("Each row = one policy. Green = loss applied, gray = masked. Width = token position.")


# ===========================================================================
# PAGE: New Dataset Importer
# ===========================================================================

def page_dataset_importer():
    st.header("New Dataset Importer")
    st.write("Test ETL_A conversion on raw records. Paste JSON or point to a file.")

    import_mode = st.radio("Input mode", ["Paste JSON", "File path"], horizontal=True)

    raw_record = None
    if import_mode == "Paste JSON":
        json_input = st.text_area("Paste a single raw JSON record:", height=300)
        if json_input.strip():
            try:
                raw_record = json.loads(json_input)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
    else:
        file_path = st.text_input("Path to JSONL file (will read first record)")
        line_num = st.number_input("Line number (1-indexed)", 1, 10000, 1)
        if file_path and Path(file_path).exists():
            with open(file_path) as f:
                for i, line in enumerate(f, 1):
                    if i == line_num:
                        raw_record = json.loads(line.strip())
                        break
        elif file_path:
            st.warning("File not found.")

    if raw_record is None:
        st.info("Provide a raw record to test ETL_A conversion.")
        # Show converter templates
        with st.expander("ETL_A Converter Template"):
            st.code("""
def convert_my_dataset_record(
    record: Dict[str, Any],
    split: str = "train",
    line_number: int = 0,
    injection_patterns: Optional[List[re.Pattern]] = None,
) -> Optional[Trace]:
    \\"\\"\\"Convert a record from my_dataset to trace_v1.\\"\\"\\"

    messages = []
    # ... extract messages from your format ...

    return Trace(
        id=Trace.generate_id("my_dataset", messages=messages),
        source=TraceSource(dataset="my_dataset", tier="raw"),
        messages=messages,
        split=split,
        completeness="complete",  # or "skeleton"
        tier="B2",               # or "B1"
        labels=TraceLabels(
            category="harmful",   # or "benign", "resisted"
            attack_succeeded=True,
            attack_present=True,
        ),
    )
""", language="python")
        return

    st.subheader("Raw Record Preview")
    st.json(raw_record)

    # Try converting with known converters
    st.subheader("ETL_A Conversion")
    converter_choice = st.selectbox("Converter", [
        "auto-detect", "fujitsu_b4", "agentdojo", "agentharm", "llmail_inject",
    ])

    if st.button("Convert", type="primary"):
        try:
            from src.schemas.tools.ETL_A import (
                convert_fujitsu_b4_record,
                convert_agentdojo_record,
                convert_agentharm_record,
                convert_llmail_inject_record,
                DEFAULT_INJECTION_PATTERNS,
            )

            patterns = [re.compile(p, re.DOTALL | re.IGNORECASE) for p in DEFAULT_INJECTION_PATTERNS]

            trace = None
            errors = []

            if converter_choice == "auto-detect":
                # Try each converter
                for name, fn in [
                    ("agentdojo", convert_agentdojo_record),
                    ("agentharm", convert_agentharm_record),
                    ("fujitsu_b4", convert_fujitsu_b4_record),
                    ("llmail_inject", convert_llmail_inject_record),
                ]:
                    try:
                        trace = fn(raw_record, split="train", line_number=0, injection_patterns=patterns)
                        if trace:
                            st.success(f"Converted with **{name}** converter")
                            break
                    except Exception as e:
                        errors.append((name, str(e)))
                if not trace:
                    st.error("No converter succeeded.")
                    for name, err in errors:
                        st.write(f"- **{name}**: {err}")
            else:
                fn_map = {
                    "fujitsu_b4": convert_fujitsu_b4_record,
                    "agentdojo": convert_agentdojo_record,
                    "agentharm": convert_agentharm_record,
                    "llmail_inject": convert_llmail_inject_record,
                }
                fn = fn_map[converter_choice]
                trace = fn(raw_record, split="train", line_number=0, injection_patterns=patterns)

            if trace:
                # Display converted trace
                trace_dict = trace.to_dict()

                st.subheader("Converted Trace")
                cols = st.columns(4)
                cols[0].write(f"**ID:** `{trace.id[:40]}...`")
                cols[1].write(f"**Dataset:** {trace.source.dataset}")
                cols[2].write(f"**Category:** {trace_category(trace_dict)}")
                cols[3].write(f"**Tier:** {trace.tier}")

                st.write("**Labels:**", trace.labels)
                if trace.tool_attack:
                    st.write("**Tool Attack:**", trace.tool_attack)
                if trace.signal_hints:
                    st.write("**Signal Hints:**", trace.signal_hints)

                st.subheader("Messages")
                injection_span = None
                if trace.signal_hints and trace.signal_hints.injection_char_span:
                    injection_span = {
                        "message_index": trace.signal_hints.injection_char_span.message_index,
                        "char_start": trace.signal_hints.injection_char_span.char_start,
                        "char_end": trace.signal_hints.injection_char_span.char_end,
                    }
                msgs_html = []
                for i, msg in enumerate(trace.messages):
                    msg_dict = {"role": msg.role, "content": msg.content, "name": msg.name}
                    if msg.tool_calls:
                        msg_dict["tool_calls"] = [{"function": {"name": tc.function.name}} for tc in msg.tool_calls]
                    msgs_html.append(render_message_html(msg_dict, i, injection_span))
                st.markdown("".join(msgs_html), unsafe_allow_html=True)

                # Schema validation
                st.subheader("Schema Validation")
                try:
                    roundtrip = Trace.from_dict(trace_dict)
                    st.success("Roundtrip validation passed (to_dict -> from_dict)")
                except Exception as e:
                    st.error(f"Roundtrip failed: {e}")

                with st.expander("Full trace JSON"):
                    st.json(trace_dict)

        except Exception as e:
            st.error(f"Conversion failed: {e}")
            import traceback
            st.code(traceback.format_exc())


# ===========================================================================
# PAGE: Stats & Validation
# ===========================================================================

def page_stats_validation():
    st.header("Stats & Validation")

    trace_files = discover_jsonl(TRACES_DIR)
    if not trace_files:
        st.warning("No trace files.")
        return

    # Aggregate stats across all trace files
    all_traces = []
    for tf in trace_files:
        all_traces.extend(load_traces(str(tf)))

    st.subheader("Cross-Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total traces", len(all_traces))

    # Per-dataset breakdown
    ds_stats = defaultdict(lambda: {"total": 0, "harmful": 0, "benign": 0, "resisted": 0, "unknown": 0,
                                      "skeleton": 0, "complete": 0, "has_tools": 0, "has_injection": 0})
    for t in all_traces:
        ds = trace_dataset(t)
        cat = trace_category(t)
        ds_stats[ds]["total"] += 1
        ds_stats[ds][cat] += 1
        ds_stats[ds][trace_completeness(t)] += 1
        if trace_has_tool_calls(t):
            ds_stats[ds]["has_tools"] += 1
        if t.get("signal_hints", {}).get("injection_char_span"):
            ds_stats[ds]["has_injection"] += 1

    c2.metric("Datasets", len(ds_stats))
    c3.metric("With injection spans", sum(s["has_injection"] for s in ds_stats.values()))

    rows = []
    for ds, stats in sorted(ds_stats.items()):
        rows.append({
            "Dataset": ds,
            "Total": stats["total"],
            "Harmful": stats["harmful"],
            "Benign": stats["benign"],
            "Resisted": stats["resisted"],
            "Unknown": stats["unknown"],
            "Skeleton": stats["skeleton"],
            "Complete": stats["complete"],
            "Has Tools": stats["has_tools"],
            "Has Injection": stats["has_injection"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # Message count distribution
    st.subheader("Message Count Distribution")
    msg_counts = [len(t.get("messages", [])) for t in all_traces]
    if msg_counts:
        msg_df = pd.DataFrame({"Messages per trace": msg_counts})
        hist = msg_df["Messages per trace"].value_counts().sort_index().head(30)
        st.bar_chart(pd.DataFrame({"Count": hist.values}, index=hist.index))

    # Schema roundtrip validation
    st.subheader("Schema Validation")
    if st.button("Run roundtrip validation (sample)"):
        sample_size = min(100, len(all_traces))
        import random
        sample = random.sample(all_traces, sample_size)
        passed = 0
        failed = 0
        errors = []
        for t in sample:
            try:
                trace_obj = Trace.from_dict(t)
                _ = trace_obj.to_dict()
                passed += 1
            except Exception as e:
                failed += 1
                errors.append((t.get("id", "?")[:30], str(e)))

        if failed == 0:
            st.success(f"All {passed}/{sample_size} sampled traces passed roundtrip validation.")
        else:
            st.warning(f"{passed} passed, {failed} failed out of {sample_size} sampled.")
            for tid, err in errors[:10]:
                st.write(f"- `{tid}`: {err}")

    # Eval routing preview
    st.subheader("Eval Routing Preview")
    st.write("Shows which evaluation function would handle each dataset type.")
    for ds in sorted(ds_stats.keys()):
        if "fujitsu" in ds:
            route = "evaluate_tool_flip_asr (tool-flip ASR)"
        elif "agentdojo" in ds:
            route = "evaluate_tool_flip_asr + generation_comparison (with sample context)"
        elif "agentharm" in ds:
            route = "skeleton (B1) - needs DS/DR generation or judge-based eval"
        elif "llmail" in ds:
            route = "evaluate_llmail_attack + evaluate_llmail_usefulness"
        else:
            route = "evaluate_tool_flip_asr (default) or needs custom evaluator"
        st.write(f"- **{ds}** -> `{route}`")

    # MWCS Preview
    st.subheader("MWCS Preview")
    mwcs = get_mwcs_registry()
    if mwcs:
        schedule_id = st.selectbox("Schedule", mwcs.list_schedules())
        step = st.slider("Training step", 0, 200, 0)
        schedule = mwcs.get_schedule(schedule_id)
        weights = schedule.get_weights_at_step(step)
        st.write(f"**Schedule:** {schedule.name}")
        st.write(f"**Description:** {schedule.description}")
        st.write(f"**Weights at step {step}:**")
        st.json(weights)

        st.write("**Mixture classes:**")
        for cid in mwcs.list_mixture_classes():
            mc = mwcs.get_mixture_class(cid)
            st.write(f"- `{cid}`: {mc.name} ({mc.category}) - datasets: {mc.source_datasets}")
    else:
        st.info("No MWCS registry found.")


# ===========================================================================
# Main App
# ===========================================================================

st.set_page_config(
    page_title="RRFA Dev Kit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("RRFA Dev Kit")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Trace Explorer", "Render & Lossmask", "Policy Comparator", "Dataset Importer", "Stats & Validation"],
)

if page == "Dashboard":
    page_dashboard()
elif page == "Trace Explorer":
    page_trace_explorer()
elif page == "Render & Lossmask":
    page_render_lossmask()
elif page == "Policy Comparator":
    page_policy_comparator()
elif page == "Dataset Importer":
    page_dataset_importer()
elif page == "Stats & Validation":
    page_stats_validation()
