#!/usr/bin/env python3
"""
Streamlit UI for browsing sweep results.

Usage:
    streamlit run scripts/visualize_sweep_streamlit.py -- --sweep-dir /path/to/sweep_dir
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st


# =============================================================================
# Data loading helpers
# =============================================================================

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def build_trace_index(traces: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {t.get("id", ""): t for t in traces if t.get("id")}


# =============================================================================
# Trace helpers (subset of scripts/visualize_sweep_results.py)
# =============================================================================

def get_system_prompt(trace: Dict[str, Any]) -> Optional[str]:
    messages = trace.get("messages", [])
    for msg in messages:
        if msg.get("role") == "system":
            return msg.get("content", "")
    signal_hints = trace.get("signal_hints", {})
    if signal_hints and signal_hints.get("raw_system_prompt"):
        return signal_hints["raw_system_prompt"]
    return None


def get_user_query(trace: Dict[str, Any]) -> Optional[str]:
    messages = trace.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def get_injection_text(trace: Dict[str, Any]) -> Optional[str]:
    tool_attack = trace.get("tool_attack", {})
    if tool_attack and tool_attack.get("injection_text"):
        return tool_attack["injection_text"]
    return None


def find_injection_in_messages(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    injections = []
    messages = trace.get("messages", [])
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if not content:
            continue
        info_pattern = re_info_pattern()
        matches = info_pattern.findall(content)
        for match in matches:
            injections.append(
                {
                    "message_index": i,
                    "role": msg.get("role", "unknown"),
                    "injection_text": match.strip(),
                    "content_preview": content,
                }
            )
    return injections


def re_info_pattern():
    import re

    return re.compile(r"<INFORMATION>(.*?)</INFORMATION>", re.DOTALL | re.IGNORECASE)


def get_attack_info(trace: Dict[str, Any]) -> Dict[str, Any]:
    labels = trace.get("labels", {})
    tool_attack = trace.get("tool_attack", {})
    return {
        "category": labels.get("category"),
        "attack_type": labels.get("attack_type") or tool_attack.get("attack_vector"),
        "attack_succeeded": labels.get("attack_succeeded"),
        "security_outcome": labels.get("security_outcome"),
    }


# =============================================================================
# Sweep discovery and parsing
# =============================================================================

@dataclass
class RunRecord:
    run_name: str
    run_dir: Path
    baseline_asr: Optional[float]
    cb_asr: Optional[float]
    reduction_pp: Optional[float]
    improvements: Optional[int]
    regressions: Optional[int]
    fujitsu_total: Optional[int]
    agentdojo_diff_rate: Optional[float]
    agentdojo_total: Optional[int]


def discover_run_dirs(sweep_dir: Path) -> List[Path]:
    if not sweep_dir.exists():
        return []
    return sorted([d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith("a")])


def load_run_metrics(run_dir: Path) -> RunRecord:
    fujitsu_eval_json = run_dir / "eval" / "fujitsu_eval.json"
    fujitsu_paired = run_dir / "eval" / "fujitsu_eval.paired_outputs.jsonl"
    agentdojo_eval_json = run_dir / "eval" / "agentdojo_eval.json"

    baseline_asr = cb_asr = reduction_pp = None
    improvements = regressions = None
    fujitsu_total = None
    agentdojo_diff_rate = None
    agentdojo_total = None

    if fujitsu_eval_json.exists():
        eval_data = load_json(fujitsu_eval_json)
        paired = load_jsonl(fujitsu_paired)
        baseline = eval_data.get("baseline", {}).get("tool_flip_asr", {})
        cb = eval_data.get("cb_model", {}).get("tool_flip_asr", {})
        baseline_asr = baseline.get("attack_success_rate", 0) * 100
        cb_asr = cb.get("attack_success_rate", 0) * 100
        reduction_pp = baseline_asr - cb_asr
        improvements = sum(
            1
            for p in paired
            if p.get("baseline_outcome") == "attack_success" and p.get("cb_outcome") != "attack_success"
        )
        regressions = sum(
            1
            for p in paired
            if p.get("baseline_outcome") != "attack_success" and p.get("cb_outcome") == "attack_success"
        )
        fujitsu_total = len(paired)

    if agentdojo_eval_json.exists():
        eval_data = load_json(agentdojo_eval_json)
        output_comparison = eval_data.get("output_comparison", {})
        agentdojo_diff_rate = output_comparison.get("difference_rate", 0) * 100
        agentdojo_total = output_comparison.get("total_compared", 0)

    return RunRecord(
        run_name=run_dir.name,
        run_dir=run_dir,
        baseline_asr=baseline_asr,
        cb_asr=cb_asr,
        reduction_pp=reduction_pp,
        improvements=improvements,
        regressions=regressions,
        fujitsu_total=fujitsu_total,
        agentdojo_diff_rate=agentdojo_diff_rate,
        agentdojo_total=agentdojo_total,
    )


def parse_run_name(run_name: str) -> Dict[str, str]:
    parts = run_name.split("_")
    parsed: Dict[str, str] = {"run_name": run_name}
    if parts:
        parsed["alpha"] = parts[0].lstrip("a")
    if len(parts) > 1:
        parsed["lossmask"] = parts[1]
    if len(parts) > 2:
        parsed["hparams"] = "_".join(parts[2:])
    return parsed


def build_runs_table(run_dirs: List[Path]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for rd in run_dirs:
        metrics = load_run_metrics(rd)
        parsed = parse_run_name(metrics.run_name)
        records.append(
            {
                **parsed,
                "run_dir": str(metrics.run_dir),
                "baseline_asr": metrics.baseline_asr,
                "cb_asr": metrics.cb_asr,
                "reduction_pp": metrics.reduction_pp,
                "improvements": metrics.improvements,
                "regressions": metrics.regressions,
                "fujitsu_total": metrics.fujitsu_total,
                "agentdojo_diff_rate": metrics.agentdojo_diff_rate,
                "agentdojo_total": metrics.agentdojo_total,
            }
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_traces(traces_dir: Optional[Path], run_dir: Path) -> Dict[str, Dict[str, Any]]:
    traces_index: Dict[str, Dict[str, Any]] = {}
    if traces_dir:
        fujitsu_traces_path = traces_dir / "fujitsu_b4_ds.jsonl"
        if fujitsu_traces_path.exists():
            traces_index.update(build_trace_index(load_jsonl(fujitsu_traces_path)))
    agentdojo_split = run_dir / "agentdojo_split" / "agentdojo_traces_harmful.jsonl"
    if agentdojo_split.exists():
        traces_index.update(build_trace_index(load_jsonl(agentdojo_split)))
    return traces_index


def load_samples(run_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    fujitsu_paired = run_dir / "eval" / "fujitsu_eval.paired_outputs.jsonl"
    agentdojo_paired = run_dir / "eval" / "agentdojo_eval.paired_outputs.jsonl"
    return {
        "fujitsu": load_jsonl(fujitsu_paired),
        "agentdojo": load_jsonl(agentdojo_paired),
    }


# =============================================================================
# Streamlit UI
# =============================================================================

def render_metrics_table(df: pd.DataFrame) -> None:
    st.subheader("Sweep summary")
    st.dataframe(
        df.sort_values(["cb_asr", "reduction_pp"], ascending=[True, False]),
        use_container_width=True,
        hide_index=True,
    )


def render_sample(trace: Optional[Dict[str, Any]], sample: Dict[str, Any]) -> None:
    st.markdown("#### Sample detail")

    cols = st.columns(3)
    cols[0].metric("Expected tool", sample.get("expected_tool", "N/A"))
    cols[1].metric("Baseline observed", sample.get("baseline_observed_tool", "N/A"))
    cols[2].metric("CB observed", sample.get("cb_observed_tool", "N/A"))

    cols2 = st.columns(2)
    cols2[0].metric("Baseline outcome", sample.get("baseline_outcome", "N/A"))
    cols2[1].metric("CB outcome", sample.get("cb_outcome", "N/A"))

    if trace:
        user_query = get_user_query(trace)
        system_prompt = get_system_prompt(trace)
        injection = get_injection_text(trace)
        agentdojo_injections = find_injection_in_messages(trace)
        attack_info = get_attack_info(trace)

        if system_prompt:
            st.markdown("**System prompt**")
            st.code(system_prompt, language="text")
        if user_query:
            st.markdown("**User query**")
            st.code(user_query, language="text")
        if injection:
            st.markdown("**Injection text (user query)**")
            st.code(injection, language="text")
        if agentdojo_injections:
            st.markdown(f"**Injections in tool results ({len(agentdojo_injections)})**")
            for inj in agentdojo_injections:
                st.markdown(f"Message {inj['message_index']} ({inj['role']})")
                st.code(inj["content_preview"], language="text")
                st.code(inj["injection_text"], language="text")
        if any(attack_info.values()):
            st.markdown("**Attack info**")
            st.json({k: v for k, v in attack_info.items() if v is not None})

    baseline_resp = sample.get("baseline_response", "")
    cb_resp = sample.get("cb_response", "")
    if baseline_resp:
        st.markdown("**Baseline response**")
        st.code(baseline_resp, language="text")
    if cb_resp:
        st.markdown("**CB response**")
        st.code(cb_resp, language="text")


def render_samples_browser(run_dir: Path, traces_dir: Optional[Path]) -> None:
    st.subheader("Samples")
    samples = load_samples(run_dir)
    dataset = st.radio("Dataset", ["fujitsu", "agentdojo"], horizontal=True)
    records = samples.get(dataset, [])

    if not records:
        st.info("No samples found for this dataset.")
        return

    filter_mode = st.selectbox(
        "Filter",
        ["All", "CB blocked (improvement)", "CB failed (regression)", "Different outputs"],
    )

    if filter_mode == "CB blocked (improvement)":
        records = [
            r
            for r in records
            if r.get("baseline_outcome") == "attack_success" and r.get("cb_outcome") != "attack_success"
        ]
    elif filter_mode == "CB failed (regression)":
        records = [r for r in records if r.get("cb_outcome") == "attack_success"]
    elif filter_mode == "Different outputs":
        records = [r for r in records if r.get("responses_differ")]

    max_show = st.slider("Max samples", min_value=1, max_value=min(200, len(records)), value=min(25, len(records)))
    records = records[:max_show]

    traces_index = load_traces(traces_dir, run_dir)

    for i, sample in enumerate(records, start=1):
        st.markdown(f"### {i}. {sample.get('id', 'N/A')}")
        trace = traces_index.get(sample.get("id", ""))
        render_sample(trace, sample)


def render_trace_view(df: pd.DataFrame, traces_dir: Optional[Path]) -> None:
    st.subheader("Trace-first view")
    selected_run = st.selectbox("Run", df["run_name"].tolist())
    row = df[df["run_name"] == selected_run].iloc[0]
    run_dir = Path(row["run_dir"])
    samples = load_samples(run_dir)

    trace_ids = sorted({s.get("id", "") for ds in samples.values() for s in ds if s.get("id")})
    if not trace_ids:
        st.info("No trace IDs found.")
        return

    trace_id = st.selectbox("Trace ID", trace_ids)
    traces_index = load_traces(traces_dir, run_dir)
    trace = traces_index.get(trace_id)

    st.markdown("**Trace context**")
    if trace:
        st.json(trace)
    else:
        st.info("Trace not found in traces directory.")

    st.markdown("**Runs containing this trace**")
    matching = []
    for _, r in df.iterrows():
        run_dir = Path(r["run_dir"])
        run_samples = load_samples(run_dir)
        if any(s.get("id") == trace_id for ds in run_samples.values() for s in ds):
            matching.append(r)
    if matching:
        match_df = pd.DataFrame(matching)[
            ["run_name", "baseline_asr", "cb_asr", "reduction_pp", "agentdojo_diff_rate"]
        ]
        st.dataframe(match_df, use_container_width=True, hide_index=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--traces-dir", type=Path, default=None)
    return parser


def main() -> None:
    st.set_page_config(page_title="Sweep Viewer", layout="wide")

    parser = build_arg_parser()
    args, _ = parser.parse_known_args()

    sweep_dir = args.sweep_dir
    traces_dir = args.traces_dir

    st.title("Sweep Results Viewer")
    st.caption("Minimal Streamlit UI to browse sweep metrics and sample traces.")

    if not sweep_dir.exists():
        st.error(f"Sweep directory not found: {sweep_dir}")
        st.stop()

    run_dirs = discover_run_dirs(sweep_dir)
    if not run_dirs:
        st.warning("No run directories found in sweep directory.")
        st.stop()

    df = build_runs_table(run_dirs)
    if df.empty:
        st.warning("No metrics found in sweep directory.")
        st.stop()

    with st.sidebar:
        st.header("Filters")
        lmp_values = sorted({v for v in df["lossmask"].dropna().unique().tolist()})
        hparam_values = sorted({v for v in df["hparams"].dropna().unique().tolist()})

        selected_lmp = st.multiselect("Lossmask policy", lmp_values, default=lmp_values)
        selected_hparams = st.multiselect("Hparams", hparam_values, default=hparam_values)

        max_cb_asr = st.slider(
            "Max CB ASR",
            min_value=0.0,
            max_value=100.0,
            value=100.0,
            step=1.0,
        )

        sort_key = st.selectbox("Sort by", ["cb_asr", "reduction_pp", "baseline_asr"])

    filtered = df.copy()
    if selected_lmp:
        filtered = filtered[filtered["lossmask"].isin(selected_lmp)]
    if selected_hparams:
        filtered = filtered[filtered["hparams"].isin(selected_hparams)]
    filtered = filtered[filtered["cb_asr"].fillna(0) <= max_cb_asr]

    render_metrics_table(filtered.sort_values(sort_key, ascending=True))

    st.divider()
    st.subheader("Run details")

    run_name = st.selectbox("Run", filtered["run_name"].tolist())
    run_row = filtered[filtered["run_name"] == run_name].iloc[0]
    run_dir = Path(run_row["run_dir"])

    cols = st.columns(5)
    cols[0].metric("Baseline ASR", f"{run_row['baseline_asr']:.2f}%" if pd.notna(run_row['baseline_asr']) else "N/A")
    cols[1].metric("CB ASR", f"{run_row['cb_asr']:.2f}%" if pd.notna(run_row['cb_asr']) else "N/A")
    cols[2].metric("Reduction", f"{run_row['reduction_pp']:.2f} pp" if pd.notna(run_row['reduction_pp']) else "N/A")
    cols[3].metric("Improvements", int(run_row["improvements"]) if pd.notna(run_row["improvements"]) else 0)
    cols[4].metric("Regressions", int(run_row["regressions"]) if pd.notna(run_row["regressions"]) else 0)

    render_samples_browser(run_dir, traces_dir)

    st.divider()
    render_trace_view(filtered, traces_dir)


if __name__ == "__main__":
    main()
