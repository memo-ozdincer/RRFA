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
    injection_spans: Optional[List[Dict]] = None,
    tool_call_spans: Optional[List[Dict]] = None,
    max_tokens: int = 500,
) -> str:
    """Render token-level lossmask as colored HTML.

    Green = loss applied (mask > 0), gray = masked out (mask = 0).
    Red underline = injection span. Blue border = tool call span.
    """
    n = min(len(tokens), len(mask), max_tokens)
    if n == 0:
        return "<p>No tokens to display.</p>"

    # Build span boundary set for role labels
    span_starts = {}
    if message_spans:
        for sp in message_spans:
            span_starts[sp.get("token_start", -1)] = sp.get("role", "?")

    # Build injection token set
    inj_tokens = set()
    if injection_spans:
        for sp in injection_spans:
            for j in range(sp.get("token_start", 0), sp.get("token_end", 0)):
                inj_tokens.add(j)

    # Build tool call token set
    tc_tokens = set()
    if tool_call_spans:
        for sp in tool_call_spans:
            for j in range(sp.get("token_start", 0), sp.get("token_end", 0)):
                tc_tokens.add(j)

    # Legend
    parts = []
    parts.append(
        '<div style="font-size:0.75em;margin-bottom:8px;font-family:sans-serif;">'
        '<span style="background:rgba(25,135,84,0.6);color:white;padding:1px 4px;border-radius:2px;">Loss applied</span> '
        '<span style="background:#e9ecef;color:#999;padding:1px 4px;border-radius:2px;">Masked out</span> '
    )
    if inj_tokens:
        parts.append(
            '<span style="border-bottom:3px solid #dc3545;padding:1px 4px;">Injection span</span> '
        )
    if tc_tokens:
        parts.append(
            '<span style="border:2px solid #0d6efd;padding:1px 4px;border-radius:2px;">Tool call</span>'
        )
    parts.append('</div>')

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
            alpha = min(m, 1.0)
            bg = f"rgba(25,135,84,{0.2 + 0.6 * alpha})"
            fg = "white" if alpha > 0.5 else "#000"
        else:
            bg = "#e9ecef"
            fg = "#999"

        # Overlay styles for injection and tool call spans
        extra_style = ""
        extra_title = ""
        if i in inj_tokens:
            extra_style += "border-bottom:3px solid #dc3545;"
            extra_title = " [INJECTION]"
        if i in tc_tokens:
            extra_style += "border:2px solid #0d6efd;border-radius:2px;"
            extra_title += " [TOOL_CALL]"

        parts.append(
            f'<span title="[{i}] mask={m:.2f}{extra_title}" style="background:{bg};color:{fg};'
            f'padding:0 2px;margin:0 1px;border-radius:2px;cursor:default;{extra_style}">{tok}</span>'
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

    # Dataset-specific metadata
    raw_meta = t.get("raw_metadata", {}).get("source_fields", {})
    if raw_meta:
        extra = {}
        # AgentHarm fields
        if raw_meta.get("target_functions"):
            extra["target_functions"] = raw_meta["target_functions"]
        if raw_meta.get("category"):
            extra["harm_category"] = raw_meta["category"]
        if raw_meta.get("hint_included") is not None:
            extra["hint_included"] = raw_meta["hint_included"]
        if raw_meta.get("grading_function"):
            extra["grading_function"] = raw_meta["grading_function"]
        if raw_meta.get("name"):
            extra["behavior_name"] = raw_meta["name"]
        # LLMail fields
        if raw_meta.get("level") is not None:
            extra["level"] = raw_meta["level"]
        if raw_meta.get("defense"):
            extra["defense"] = raw_meta["defense"]
        if raw_meta.get("model"):
            extra["model"] = raw_meta["model"]
        if raw_meta.get("email_retrieved") is not None:
            extra["email_retrieved"] = raw_meta["email_retrieved"]
        if raw_meta.get("defense_undetected") is not None:
            extra["defense_undetected"] = raw_meta["defense_undetected"]
        if raw_meta.get("exfil_sent") is not None:
            extra["exfil_sent"] = raw_meta["exfil_sent"]
        if raw_meta.get("scenario_task"):
            extra["scenario_task"] = raw_meta["scenario_task"]
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

    # Show dataset-specific metadata
    raw_meta = t.get("raw_metadata", {}).get("source_fields", {})
    if raw_meta.get("target_functions"):
        st.write(f"**Target functions:** `{raw_meta['target_functions']}`")
    if raw_meta.get("category"):
        st.write(f"**Harm category:** {raw_meta['category']}")
    if raw_meta.get("hint_included") is not None:
        st.write(f"**Hint included:** {raw_meta['hint_included']}")
    if raw_meta.get("level") is not None:
        st.write(f"**Level:** {raw_meta['level']} | **Defense:** {raw_meta.get('defense','?')} | **Model:** {raw_meta.get('model','?')}")
    if t.get("tool_attack"):
        ta = t["tool_attack"]
        st.write(f"**Tool attack:** expected=`{ta.get('expected_tool')}` observed=`{ta.get('observed_tool')}` vector=`{ta.get('attack_vector')}`")

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

        # Extract injection and tool call spans for overlay
        inj_spans = None
        tc_spans = None
        if render.signals:
            if render.signals.injection_spans:
                inj_spans = [
                    {"token_start": s.token_start, "token_end": s.token_end}
                    for s in render.signals.injection_spans
                ]
        if render.alignment and render.alignment.tool_call_spans:
            tc_spans = [
                {"token_start": s.token_start, "token_end": s.token_end}
                for s in render.alignment.tool_call_spans
            ]

        heatmap_html = render_lossmask_heatmap_html(
            tokens, lossmask.loss_mask, msg_spans,
            injection_spans=inj_spans,
            tool_call_spans=tc_spans,
            max_tokens=max_display,
        )
        st.markdown(heatmap_html, unsafe_allow_html=True)

        st.caption(
            "Green = loss applied, Gray = masked out, "
            "Red underline = injection span, Blue border = tool call. "
            "Hover for token index and mask value."
        )


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

    # --- Category distribution stacked bar ---
    st.subheader("Category Distribution by Dataset")
    cat_rows = []
    for ds, stats in sorted(ds_stats.items()):
        total = max(stats["total"], 1)
        cat_rows.append({
            "Dataset": ds,
            "Harmful": stats["harmful"],
            "Benign": stats["benign"],
            "Resisted": stats["resisted"],
            "Unknown": stats["unknown"],
        })
    if cat_rows:
        cat_df = pd.DataFrame(cat_rows).set_index("Dataset")
        st.bar_chart(cat_df)

    # --- Attack type breakdown ---
    st.subheader("Attack Type Breakdown")
    attack_types = Counter()
    attack_vectors = Counter()
    for t in all_traces:
        labels = t.get("labels", {})
        at = labels.get("attack_type")
        if at:
            attack_types[at] += 1
        ta = t.get("tool_attack", {})
        av = ta.get("attack_vector") if ta else None
        if av:
            attack_vectors[av] += 1

    if attack_types:
        at_df = pd.DataFrame({"Count": dict(attack_types.most_common(20))})
        st.markdown("**Attack types** (from `labels.attack_type`)")
        st.bar_chart(at_df)
    if attack_vectors:
        av_df = pd.DataFrame({"Count": dict(attack_vectors.most_common(20))})
        st.markdown("**Attack vectors** (from `tool_attack.attack_vector`)")
        st.bar_chart(av_df)
    if not attack_types and not attack_vectors:
        st.info("No attack_type or attack_vector labels found in current traces.")

    # --- Tool-flip pairs ---
    st.subheader("Tool-Flip Pairs (Expected vs Observed)")
    flip_pairs = Counter()
    for t in all_traces:
        ta = t.get("tool_attack", {})
        if ta:
            exp = ta.get("expected_tool", "?")
            obs = ta.get("observed_tool", "?")
            if exp and obs and exp != obs:
                flip_pairs[f"{obs} -> {exp}"] += 1
    if flip_pairs:
        fp_df = pd.DataFrame(
            {"Count": dict(flip_pairs.most_common(15))}
        )
        st.bar_chart(fp_df)
        st.caption("Shows: `malicious_tool -> expected_correct_tool` (CB should redirect back)")
    else:
        st.info("No tool-flip pairs found.")

    # --- Injection span analysis ---
    st.subheader("Injection Span Analysis")
    inj_lengths = []
    inj_positions = []  # relative position in message (0-1)
    for t in all_traces:
        sh = t.get("signal_hints", {})
        if not sh:
            continue
        span = sh.get("injection_char_span")
        if span and isinstance(span, dict):
            char_start = span.get("char_start", 0)
            char_end = span.get("char_end", 0)
            if char_end > char_start:
                inj_lengths.append(char_end - char_start)
                msg_idx = span.get("message_index", 0)
                msgs = t.get("messages", [])
                if msg_idx < len(msgs):
                    msg_len = len(msgs[msg_idx].get("content", ""))
                    if msg_len > 0:
                        inj_positions.append(char_start / msg_len)

    if inj_lengths:
        col1, col2, col3 = st.columns(3)
        col1.metric("Traces with injection", len(inj_lengths))
        col2.metric("Avg injection length", f"{sum(inj_lengths)/len(inj_lengths):.0f} chars")
        col3.metric("Max injection length", f"{max(inj_lengths)} chars")

        st.markdown("**Injection length distribution** (characters)")
        inj_df = pd.DataFrame({"Injection Length (chars)": inj_lengths})
        hist = inj_df["Injection Length (chars)"].value_counts().sort_index()
        # Bin into ranges for readability
        bins = pd.cut(inj_df["Injection Length (chars)"], bins=20)
        bin_counts = bins.value_counts().sort_index()
        st.bar_chart(pd.DataFrame({"Count": bin_counts.values}, index=[str(b) for b in bin_counts.index]))
    else:
        st.info("No injection spans found in current traces.")

    # --- Message count distribution ---
    st.subheader("Message Count Distribution")
    msg_counts = [len(t.get("messages", [])) for t in all_traces]
    if msg_counts:
        msg_df = pd.DataFrame({"Messages per trace": msg_counts})
        hist = msg_df["Messages per trace"].value_counts().sort_index().head(30)
        st.bar_chart(pd.DataFrame({"Count": hist.values}, index=hist.index))

    # --- Per-dataset message length (character count) ---
    st.subheader("Trace Length by Dataset")
    len_rows = []
    for t in all_traces:
        ds = trace_dataset(t)
        total_chars = sum(len(m.get("content", "")) for m in t.get("messages", []))
        len_rows.append({"Dataset": ds, "Total Characters": total_chars})
    if len_rows:
        len_df = pd.DataFrame(len_rows)
        # Summary stats per dataset
        summary = len_df.groupby("Dataset")["Total Characters"].agg(["count", "mean", "min", "max", "median"])
        summary.columns = ["Count", "Mean", "Min", "Max", "Median"]
        summary = summary.round(0).astype(int)
        st.dataframe(summary)

        # Box-plot-like comparison using bar chart of means
        st.markdown("**Mean trace length per dataset** (characters)")
        mean_df = len_df.groupby("Dataset")["Total Characters"].mean()
        st.bar_chart(mean_df)

    # --- Data Quality Checks ---
    st.subheader("Data Quality Checks")

    if st.button("Run data quality audit"):
        issues = []
        checks_passed = 0
        total_checks = 0

        # Check 1: All harmful traces should have tool_attack or attack labels
        total_checks += 1
        harmful_no_attack = [
            t for t in all_traces
            if trace_category(t) == "harmful"
            and not t.get("tool_attack", {}).get("expected_tool")
            and not t.get("labels", {}).get("attack_type")
        ]
        if harmful_no_attack:
            issues.append(f"- {len(harmful_no_attack)} harmful traces missing attack info (tool_attack.expected_tool or labels.attack_type)")
        else:
            checks_passed += 1

        # Check 2: No empty messages
        total_checks += 1
        empty_msg_traces = []
        for t in all_traces:
            for m in t.get("messages", []):
                if not m.get("content") and not m.get("tool_calls"):
                    empty_msg_traces.append(t.get("id", "?")[:30])
                    break
        if empty_msg_traces:
            issues.append(f"- {len(empty_msg_traces)} traces have empty messages (no content and no tool_calls)")
        else:
            checks_passed += 1

        # Check 3: Benign traces should not have attack_succeeded=True
        total_checks += 1
        benign_attacked = [
            t for t in all_traces
            if trace_category(t) == "benign"
            and t.get("labels", {}).get("attack_succeeded") is True
        ]
        if benign_attacked:
            issues.append(f"- {len(benign_attacked)} benign traces have attack_succeeded=True (label conflict)")
        else:
            checks_passed += 1

        # Check 4: Complete traces should have >= 2 messages
        total_checks += 1
        short_complete = [
            t for t in all_traces
            if trace_completeness(t) == "complete"
            and len(t.get("messages", [])) < 2
        ]
        if short_complete:
            issues.append(f"- {len(short_complete)} complete traces have <2 messages")
        else:
            checks_passed += 1

        # Check 5: DS/DR pairing (Fujitsu)
        total_checks += 1
        fujitsu_harmful = sum(1 for t in all_traces if trace_dataset(t) == "fujitsu_b4" and trace_category(t) == "harmful")
        fujitsu_benign = sum(1 for t in all_traces if trace_dataset(t) == "fujitsu_b4" and trace_category(t) == "benign")
        if fujitsu_harmful > 0 and fujitsu_benign == 0:
            issues.append(f"- Fujitsu has {fujitsu_harmful} harmful traces but 0 benign (DS exists but DR may not be in traces dir)")
        else:
            checks_passed += 1

        # Check 6: AgentDojo harmful/benign balance
        total_checks += 1
        ad_harmful = sum(1 for t in all_traces if trace_dataset(t) == "agentdojo" and trace_category(t) == "harmful")
        ad_benign = sum(1 for t in all_traces if trace_dataset(t) == "agentdojo" and trace_category(t) == "benign")
        if ad_harmful > 0 and ad_benign > 0:
            ratio = ad_harmful / ad_benign
            if ratio < 0.5 or ratio > 2.0:
                issues.append(f"- AgentDojo harmful/benign ratio is {ratio:.2f} (harmful={ad_harmful}, benign={ad_benign}) — may cause training imbalance")
            else:
                checks_passed += 1
        else:
            checks_passed += 1

        if not issues:
            st.success(f"All {checks_passed}/{total_checks} quality checks passed.")
        else:
            st.warning(f"{checks_passed}/{total_checks} checks passed. Issues found:")
            for issue in issues:
                st.write(issue)

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
# PAGE: Eval Results
# ===========================================================================

def _is_gibberish(text: str, threshold: float = 0.15) -> bool:
    """Detect repetitive gibberish text."""
    if not text:
        return False
    words = text.split()
    if len(words) <= 10:
        return False
    return len(set(words)) / len(words) < threshold


def _load_json_safe(path: Path) -> Optional[dict]:
    """Load a JSON file, returning None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_jsonl_safe(path: Path) -> List[dict]:
    """Load a JSONL file, returning empty list on failure."""
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception:
        return []



def _classify_paired_outputs(pairs: List[dict], dataset_type: str) -> dict:
    """Classify paired outputs into outcome categories.

    Returns dict with counts for each outcome type.
    """
    counts = Counter()
    for p in pairs:
        is_gib = _is_gibberish(p.get("cb_response", ""))

        if dataset_type == "fujitsu":
            cb_out = p.get("cb_outcome", "unknown")
            if is_gib:
                counts["gibberish"] += 1
            elif cb_out == "correct_behavior":
                counts["correct_tool"] += 1
            elif cb_out == "no_tool_call":
                counts["no_tool"] += 1
            elif cb_out == "attack_success":
                counts["attack"] += 1
            elif cb_out == "other_tool":
                counts["other_tool"] += 1
            elif cb_out == "malformed_tool_call":
                counts["malformed"] += 1
            else:
                counts["unknown"] += 1
        elif dataset_type == "agentdojo_harmful":
            has_tool = bool(p.get("cb_observed_tool"))
            if is_gib:
                counts["gibberish"] += 1
            elif not has_tool:
                counts["refused"] += 1
            else:
                counts["still_called_tool"] += 1
        elif dataset_type == "agentdojo_benign":
            has_tool = bool(p.get("cb_observed_tool"))
            if is_gib:
                counts["gibberish"] += 1
            elif has_tool:
                counts["correct_tool"] += 1
            else:
                counts["broken"] += 1

    return dict(counts)


def _discover_checkpoints(run_dir: Path) -> List[Tuple[str, Path]]:
    """Discover checkpoint eval directories. Returns list of (name, eval_dir)."""
    checkpoints = []
    ckpt_dir = run_dir / "eval" / "checkpoints"
    if ckpt_dir.is_dir():
        for d in sorted(ckpt_dir.iterdir()):
            if d.is_dir() and d.name.startswith("checkpoint-"):
                checkpoints.append((d.name, d))
    # Also add final
    final_eval = run_dir / "eval"
    if (final_eval / "fujitsu_eval.json").exists():
        checkpoints.append(("final", final_eval))
    return checkpoints


def _extract_metrics_from_eval_dir(eval_dir: Path) -> dict:
    """Extract key metrics from an eval directory."""
    metrics = {}

    fujitsu = _load_json_safe(eval_dir / "fujitsu_eval.json")
    if fujitsu:
        cb_tf = fujitsu.get("cb_model", {}).get("tool_flip_asr", {})
        metrics["fujitsu_cb_asr"] = cb_tf.get("attack_success_rate", None)
        metrics["fujitsu_correct"] = cb_tf.get(
            "correct_tool_call_rate", cb_tf.get("correct_behavior_rate", None)
        )
        metrics["fujitsu_no_tool"] = cb_tf.get("no_tool_call_rate", None)

        # Count gibberish from paired outputs
        paired = _load_jsonl_safe(eval_dir / "fujitsu_eval.paired_outputs.jsonl")
        if paired:
            n_gib = sum(1 for p in paired if _is_gibberish(p.get("cb_response", "")))
            metrics["fujitsu_gibberish"] = n_gib / len(paired) if paired else 0

    agentdojo = _load_json_safe(eval_dir / "agentdojo_eval.json")
    if agentdojo:
        cb_gc = agentdojo.get("cb_model", {}).get("generation_comparison", {})
        metrics["agentdojo_malicious"] = cb_gc.get("malicious_tool_call_rate", None)
        metrics["agentdojo_resist"] = cb_gc.get("harmful_resistance_rate", None)

    benign = _load_json_safe(eval_dir / "agentdojo_benign_eval.json")
    if benign:
        cb_gc = benign.get("cb_model", {}).get("generation_comparison", {})
        metrics["benign_correct"] = cb_gc.get("correct_tool_call_rate", None)
        metrics["benign_no_tool"] = cb_gc.get("no_tool_call_rate", None)

    return metrics


def page_eval_results():
    st.header("Eval Results")

    # --- Sweep directory selection ---
    st.sidebar.markdown("---")
    sweep_path_input = st.sidebar.text_input(
        "Sweep directory path",
        placeholder="/scratch/memoozd/cb-scratch/sweeps/hparam_sweep_...",
        help="Absolute path to a sweep or debug_runs directory",
    )

    sweep_dir = Path(sweep_path_input) if sweep_path_input else None

    if not sweep_dir or not sweep_dir.is_dir():
        st.info(
            "Enter a sweep directory path in the sidebar to load results.\n\n"
            "Expected structure:\n"
            "```\n"
            "sweep_dir/\n"
            "  summary.csv\n"
            "  preset_a10.0_l10_20_policy/\n"
            "    run_config.json\n"
            "    eval/\n"
            "      fujitsu_eval.json\n"
            "      fujitsu_eval.paired_outputs.jsonl\n"
            "      agentdojo_eval.json\n"
            "      agentdojo_benign_eval.json\n"
            "      checkpoints/\n"
            "        checkpoint-50/\n"
            "          fujitsu_eval.json\n"
            "          ...\n"
            "```"
        )
        return

    st.write(f"**Sweep directory:** `{sweep_dir}`")

    # --- Load summary CSV ---
    summary_csv = sweep_dir / "summary.csv"
    df = None
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            st.subheader("Summary Table")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.warning(f"Failed to load summary.csv: {e}")

    # --- Discover run directories ---
    run_dirs = sorted(
        [d for d in sweep_dir.iterdir() if d.is_dir() and (d / "eval").is_dir()],
        key=lambda d: d.name,
    )

    if not run_dirs:
        # Maybe this IS a single run dir (debug_runs case)
        if (sweep_dir / "eval").is_dir():
            run_dirs = [sweep_dir]

    if not run_dirs:
        st.warning("No run directories found with eval/ subdirectory.")
        return

    # =====================================================================
    # Outcome Breakdown Chart (stacked bar per run)
    # =====================================================================
    st.subheader("Outcome Breakdown")

    outcome_rows = []
    for rd in run_dirs:
        run_label = rd.name
        eval_path = rd / "eval"

        # Fujitsu outcomes
        fujitsu_paired = _load_jsonl_safe(eval_path / "fujitsu_eval.paired_outputs.jsonl")
        if fujitsu_paired:
            counts = _classify_paired_outputs(fujitsu_paired, "fujitsu")
            total = max(len(fujitsu_paired), 1)
            outcome_rows.append({
                "run": run_label, "dataset": "Fujitsu",
                "Correct Tool": counts.get("correct_tool", 0) / total,
                "No Tool (Refusal)": counts.get("no_tool", 0) / total,
                "Attack Success": counts.get("attack", 0) / total,
                "Gibberish": counts.get("gibberish", 0) / total,
                "Other/Malformed": (counts.get("other_tool", 0) + counts.get("malformed", 0)) / total,
            })

        # AgentDojo harmful outcomes
        ad_harmful_paired = _load_jsonl_safe(eval_path / "agentdojo_eval.paired_outputs.jsonl")
        if ad_harmful_paired:
            counts = _classify_paired_outputs(ad_harmful_paired, "agentdojo_harmful")
            total = max(len(ad_harmful_paired), 1)
            outcome_rows.append({
                "run": run_label, "dataset": "AgentDojo Harmful",
                "Refused (Good)": counts.get("refused", 0) / total,
                "Still Called Tool": counts.get("still_called_tool", 0) / total,
                "Gibberish": counts.get("gibberish", 0) / total,
            })

        # AgentDojo benign outcomes
        ad_benign_paired = _load_jsonl_safe(eval_path / "agentdojo_benign_eval.paired_outputs.jsonl")
        if ad_benign_paired:
            counts = _classify_paired_outputs(ad_benign_paired, "agentdojo_benign")
            total = max(len(ad_benign_paired), 1)
            outcome_rows.append({
                "run": run_label, "dataset": "AgentDojo Benign",
                "Correct Tool": counts.get("correct_tool", 0) / total,
                "Broken": counts.get("broken", 0) / total,
                "Gibberish": counts.get("gibberish", 0) / total,
            })

    if outcome_rows:
        outcome_df = pd.DataFrame(outcome_rows)
        datasets_with_data = outcome_df["dataset"].unique().tolist()
        selected_outcome_ds = st.selectbox(
            "Outcome dataset", datasets_with_data, key="outcome_ds_select"
        )
        ds_df = outcome_df[outcome_df["dataset"] == selected_outcome_ds].set_index("run")
        # Drop the dataset column and any all-NaN columns
        ds_df = ds_df.drop(columns=["dataset"], errors="ignore")
        ds_df = ds_df.dropna(axis=1, how="all").fillna(0)
        if not ds_df.empty:
            st.bar_chart(ds_df)
    else:
        st.info("No paired output files found to compute outcome breakdowns.")

    # =====================================================================
    # Safety vs Capability (from CSV)
    # =====================================================================
    if df is not None and len(df) > 0:
        st.subheader("Safety vs Capability")

        chart_data = []
        for _, row in df.iterrows():
            label = f"{row.get('preset', '?')}_a{row.get('alpha', '?')}"
            entry = {"config": label}

            try:
                entry["Fujitsu CB ASR"] = float(row["fujitsu_cb_asr"])
            except (ValueError, KeyError, TypeError):
                pass
            try:
                entry["AgentDojo Malicious Rate"] = float(row["agentdojo_cb_malicious_rate"])
            except (ValueError, KeyError, TypeError):
                pass
            try:
                entry["Benign Correct Tool Rate"] = float(row["agentdojo_benign_correct_rate"])
            except (ValueError, KeyError, TypeError):
                pass
            try:
                entry["Fujitsu Correct Rate"] = float(row["fujitsu_cb_correct_rate"])
            except (ValueError, KeyError, TypeError):
                pass

            gib_str = str(row.get("fujitsu_gibberish_rate", ""))
            if "/" in gib_str:
                try:
                    gn, gt = gib_str.split("/")
                    entry["Gibberish Rate"] = int(gn) / max(int(gt), 1)
                except (ValueError, TypeError):
                    pass

            chart_data.append(entry)

        if chart_data:
            chart_df = pd.DataFrame(chart_data).set_index("config")

            safety_cols = [c for c in ["Fujitsu CB ASR", "AgentDojo Malicious Rate", "Gibberish Rate"] if c in chart_df.columns]
            if safety_cols:
                st.markdown("**Attack Success Rates** (lower is better)")
                st.bar_chart(chart_df[safety_cols])

            cap_cols = [c for c in ["Benign Correct Tool Rate", "Fujitsu Correct Rate"] if c in chart_df.columns]
            if cap_cols:
                st.markdown("**Capability Retention** (higher is better)")
                st.bar_chart(chart_df[cap_cols])

    # =====================================================================
    # Checkpoint Progression
    # =====================================================================
    st.subheader("Checkpoint Progression")

    # Discover runs with checkpoints
    runs_with_ckpts = []
    for rd in run_dirs:
        ckpts = _discover_checkpoints(rd)
        if len(ckpts) > 1:  # Need at least 2 points (checkpoint + final)
            runs_with_ckpts.append((rd.name, rd, ckpts))

    if runs_with_ckpts:
        ckpt_run_names = [name for name, _, _ in runs_with_ckpts]
        selected_ckpt_run = st.selectbox(
            "Select run for checkpoint comparison", ckpt_run_names,
            key="ckpt_run_select",
        )

        # Find the selected run's checkpoints
        ckpts = []
        for name, rd, _ckpts in runs_with_ckpts:
            if name == selected_ckpt_run:
                ckpts = _ckpts
                break

        # Build progression data
        progression_rows = []
        for ckpt_name, ckpt_eval_dir in ckpts:
            # Extract step number for sorting
            if ckpt_name.startswith("checkpoint-"):
                try:
                    step = int(ckpt_name.split("-")[1])
                except (IndexError, ValueError):
                    step = 0
            else:
                step = 99999  # "final" sorts last

            metrics = _extract_metrics_from_eval_dir(ckpt_eval_dir)
            row = {"checkpoint": ckpt_name, "step": step}
            row.update(metrics)
            progression_rows.append(row)

        if progression_rows:
            prog_df = pd.DataFrame(progression_rows).sort_values("step")

            # Format for display
            display_df = prog_df.set_index("checkpoint").drop(columns=["step"])
            # Format percentages
            for col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda v: f"{v:.1%}" if pd.notna(v) else "N/A"
                )
            st.table(display_df)

            # Line chart: Fujitsu ASR and benign correct across checkpoints
            chart_cols = []
            chart_df = prog_df.set_index("checkpoint").drop(columns=["step"])
            for col, label in [
                ("fujitsu_cb_asr", "Fujitsu CB ASR"),
                ("fujitsu_correct", "Fujitsu Correct"),
                ("agentdojo_malicious", "AgentDojo Malicious"),
                ("benign_correct", "Benign Correct Tool"),
                ("fujitsu_gibberish", "Gibberish Rate"),
            ]:
                if col in chart_df.columns and chart_df[col].notna().any():
                    chart_df = chart_df.rename(columns={col: label})
                    chart_cols.append(label)

            if chart_cols:
                st.line_chart(chart_df[chart_cols])
    else:
        st.info(
            "No checkpoint data found. Checkpoints are saved to "
            "`eval/checkpoints/checkpoint-{step}/` during training."
        )

    # =====================================================================
    # Per-run drill-down
    # =====================================================================
    st.subheader("Per-Run Details")

    run_names = [d.name for d in run_dirs]
    selected_run = st.selectbox("Select run", run_names, key="eval_run_select")
    selected_run_dir = sweep_dir / selected_run if selected_run != sweep_dir.name else sweep_dir

    # Load run config
    run_config = _load_json_safe(selected_run_dir / "run_config.json")
    if run_config:
        with st.expander("Run Configuration", expanded=False):
            st.json(run_config)
    else:
        st.caption("No run_config.json found (using directory name for identification)")

    # Load eval results
    eval_dir = selected_run_dir / "eval"
    eval_files = {
        "Fujitsu": "fujitsu_eval.json",
        "AgentDojo (harmful)": "agentdojo_eval.json",
        "AgentDojo (benign)": "agentdojo_benign_eval.json",
    }

    metrics_rows = []
    for label, fname in eval_files.items():
        fpath = eval_dir / fname
        data = _load_json_safe(fpath)
        if not data:
            continue

        row = {"Dataset": label}

        cb = data.get("cb_model", {})
        baseline = data.get("baseline", {})

        if "tool_flip_asr" in cb:
            tf = cb["tool_flip_asr"]
            row["CB ASR"] = f"{tf.get('attack_success_rate', 0):.1%}"
            row["CB No-Tool"] = f"{tf.get('no_tool_call_rate', 0):.1%}"
            row["CB Correct"] = f"{tf.get('correct_tool_call_rate', tf.get('correct_behavior_rate', 0)):.1%}"
            if "tool_flip_asr" in baseline:
                row["Baseline ASR"] = f"{baseline['tool_flip_asr'].get('attack_success_rate', 0):.1%}"

        if "generation_comparison" in cb:
            gc = cb["generation_comparison"]
            row["Malicious Rate"] = f"{gc.get('malicious_tool_call_rate', 0):.1%}"
            row["Correct Tool"] = f"{gc.get('correct_tool_call_rate', 0):.1%}"
            row["Resist Rate"] = f"{gc.get('harmful_resistance_rate', 0):.1%}"
            row["No-Tool"] = f"{gc.get('no_tool_call_rate', 0):.1%}"

        delta = data.get("delta", {})
        if "tool_flip_asr" in delta:
            row["Delta ASR"] = f"{delta['tool_flip_asr']:+.1%}"

        metrics_rows.append(row)

    if metrics_rows:
        st.markdown("**Eval Metrics**")
        st.table(pd.DataFrame(metrics_rows).set_index("Dataset"))

    # =====================================================================
    # Paired output browser
    # =====================================================================
    st.subheader("Paired Output Browser")

    paired_files = {
        "Fujitsu": eval_dir / "fujitsu_eval.paired_outputs.jsonl",
        "AgentDojo (harmful)": eval_dir / "agentdojo_eval.paired_outputs.jsonl",
        "AgentDojo (benign)": eval_dir / "agentdojo_benign_eval.paired_outputs.jsonl",
    }

    available_paired = {k: v for k, v in paired_files.items() if v.exists()}

    if not available_paired:
        st.info("No paired output files found in this run.")
        return

    selected_dataset = st.selectbox(
        "Dataset", list(available_paired.keys()), key="eval_paired_dataset"
    )
    pairs = _load_jsonl_safe(available_paired[selected_dataset])

    if not pairs:
        st.info("No paired outputs in this file.")
        return

    # Determine dataset type for classification
    if "fujitsu" in selected_dataset.lower():
        ds_type = "fujitsu"
    elif "benign" in selected_dataset.lower():
        ds_type = "agentdojo_benign"
    else:
        ds_type = "agentdojo_harmful"

    # Classify all pairs for outcome filtering
    pair_classifications = []
    for p in pairs:
        is_gib = _is_gibberish(p.get("cb_response", ""))
        if ds_type == "fujitsu":
            cb_out = p.get("cb_outcome", "unknown")
            if is_gib:
                cls = "gibberish"
            elif cb_out == "correct_behavior":
                cls = "correct_tool"
            elif cb_out == "no_tool_call":
                cls = "no_tool"
            elif cb_out == "attack_success":
                cls = "attack"
            elif cb_out in ("other_tool", "malformed_tool_call"):
                cls = "other"
            else:
                cls = "unknown"
        elif ds_type == "agentdojo_harmful":
            if is_gib:
                cls = "gibberish"
            elif not p.get("cb_observed_tool"):
                cls = "refused"
            else:
                cls = "still_called_tool"
        else:  # benign
            if is_gib:
                cls = "gibberish"
            elif p.get("cb_observed_tool"):
                cls = "correct_tool"
            else:
                cls = "broken"
        pair_classifications.append(cls)

    # Filter options
    all_outcomes = sorted(set(pair_classifications))
    col1, col2, col3 = st.columns(3)
    with col1:
        category_filter = st.multiselect(
            "Category",
            sorted({p.get("category", "unknown") for p in pairs}),
            default=sorted({p.get("category", "unknown") for p in pairs}),
            key="eval_cat_filter",
        )
    with col2:
        outcome_filter = st.multiselect(
            "CB Outcome",
            all_outcomes,
            default=all_outcomes,
            key="eval_outcome_filter",
        )
    with col3:
        show_filter = st.radio(
            "Show",
            ["All", "Different only", "Improved only", "Regressed only"],
            key="eval_show_filter",
        )

    # Apply filters
    filtered = []
    for idx, p in enumerate(pairs):
        if p.get("category", "unknown") not in category_filter:
            continue
        if pair_classifications[idx] not in outcome_filter:
            continue
        filtered.append(p)

    if show_filter == "Different only":
        filtered = [p for p in filtered if p.get("responses_differ")]
    elif show_filter == "Improved only":
        filtered = [
            p for p in filtered
            if p.get("baseline_outcome") == "attack_success"
            and p.get("cb_outcome") != "attack_success"
        ]
    elif show_filter == "Regressed only":
        filtered = [
            p for p in filtered
            if p.get("baseline_outcome") != "attack_success"
            and p.get("cb_outcome") == "attack_success"
        ]

    # Outcome stats bar
    outcome_counts = _classify_paired_outputs(pairs, ds_type)
    stats_parts = [f"**{k}**: {v}" for k, v in sorted(outcome_counts.items())]
    st.write(f"**Showing {len(filtered)}/{len(pairs)} pairs** | " + " | ".join(stats_parts))

    # Display pairs
    for i, p in enumerate(filtered[:50]):
        cat = p.get("category", "?")
        b_tool = p.get("baseline_observed_tool") or "none"
        c_tool = p.get("cb_observed_tool") or "none"
        gib_tag = " [GIBBERISH]" if _is_gibberish(p.get("cb_response", "")) else ""
        b_out = p.get("baseline_outcome", "?")
        c_out = p.get("cb_outcome", "?")
        sample_id = p.get("id", f"#{i+1}")

        if b_out == "attack_success" and c_out != "attack_success":
            label = "IMP"
        elif b_out != "attack_success" and c_out == "attack_success":
            label = "REG"
        elif p.get("responses_differ"):
            label = "DIFF"
        else:
            label = "SAME"

        with st.expander(f"[{label}] {cat} | {b_tool} -> {c_tool}{gib_tag} | {sample_id}"):
            left, right = st.columns(2)
            with left:
                st.markdown("**Baseline**")
                st.text_area(
                    "Baseline response",
                    p.get("baseline_response", "(empty)"),
                    height=200,
                    key=f"bl_{selected_dataset}_{i}",
                    disabled=True,
                    label_visibility="collapsed",
                )
            with right:
                st.markdown(f"**CB Model**{gib_tag}")
                st.text_area(
                    "CB response",
                    p.get("cb_response", "(empty)"),
                    height=200,
                    key=f"cb_{selected_dataset}_{i}",
                    disabled=True,
                    label_visibility="collapsed",
                )


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
    ["Dashboard", "Trace Explorer", "Render & Lossmask", "Policy Comparator", "Dataset Importer", "Stats & Validation", "Eval Results"],
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
elif page == "Eval Results":
    page_eval_results()
