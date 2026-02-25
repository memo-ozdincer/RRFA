#!/usr/bin/env python3
"""
Pipeline artifact verifier for CB runs.

Checks concrete invariants only:
1. Tool schema validity
2. Trace/context integrity
3. Lossmask coverage
4. Training-log/loss sanity
5. Evaluation-metric sanity and context correctness

Usage examples:
  python scripts/verify_pipeline_run.py \
    --tool-schema configs/tool_schemas/b4_standard_v1.json \
    --train-log /path/to/train.log \
    --eval-json /path/to/eval/fujitsu_eval.json \
    --eval-json /path/to/eval/agentdojo_eval.json \
    --trace-jsonl /path/to/fujitsu_b4_ds.jsonl \
    --trace-jsonl /path/to/agentdojo_traces_harmful.jsonl \
    --lossmask-jsonl /path/to/fujitsu_b4_ds.jsonl \
    --report-json /path/to/verification_report.json \
    --fail-on-error
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


VALID_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}

# Keys that should be bounded to [0, 1] when present.
RATE_KEYS = {
    "attack_success_rate",
    "correct_behavior_rate",
    "no_tool_call_rate",
    "valid_json_rate",
    "forced_call_asr",
    "capability_retention",
    "difference_rate",
    "refusal_rate",
    "other_tool_rate",
    "json_parse_failure_rate",
    "target_accuracy_rate",
    "partial_neutralization_rate",
    "usefulness_rate",
    "no_tool_rate",
    "wrong_tool_rate",
    "tool_call_rate",
    "benign_tool_match_rate",
    "harmful_resistance_rate",
}


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_jsonl(path: Path, max_lines: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if max_lines > 0 and line_no > max_lines:
                break
            yield line_no, json.loads(line)


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isfinite(f):
            return f
        return None
    return None


def _tool_names_from_schema(schema: Dict[str, Any]) -> Set[str]:
    names: Set[str] = set()
    for tool in schema.get("tools", []) or []:
        fn = tool.get("function") or {}
        name = fn.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _collect_trace_tool_names(trace: Dict[str, Any]) -> Set[str]:
    names: Set[str] = set()
    for message in trace.get("messages", []) or []:
        role = message.get("role")
        if role == "assistant":
            for tool_call in message.get("tool_calls") or []:
                fn = tool_call.get("function") or {}
                name = fn.get("name")
                if isinstance(name, str) and name:
                    names.add(name)
        elif role == "tool":
            name = message.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return names


def _collect_embedded_tool_names(trace: Dict[str, Any]) -> Set[str]:
    names: Set[str] = set()

    def _consume_tools(payload: Any) -> None:
        if not isinstance(payload, list):
            return
        for tool in payload:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") or {}
            name = fn.get("name")
            if isinstance(name, str) and name:
                names.add(name)

    _consume_tools(trace.get("tools"))
    source_fields = (trace.get("raw_metadata") or {}).get("source_fields") or {}
    _consume_tools(source_fields.get("tools"))
    _consume_tools(source_fields.get("available_tools"))
    return names


def _add_finding(findings: List[Dict[str, Any]], level: str, check: str, detail: str) -> None:
    findings.append({"level": level, "check": check, "detail": detail})


def _check_rate(findings: List[Dict[str, Any]], check: str, key: str, value: Any) -> None:
    f = _safe_float(value)
    if f is None:
        return
    if f < 0.0 or f > 1.0:
        _add_finding(
            findings,
            "error",
            check,
            f"Rate `{key}` is outside [0, 1]: {f}",
        )


def verify_tool_schema(path: Path, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "path": str(path),
        "tool_count": 0,
        "tool_names": [],
    }
    if not path.exists():
        _add_finding(findings, "error", "tool_schema", f"Tool schema not found: {path}")
        return summary

    schema = _read_json(path)
    tools = schema.get("tools")
    if not isinstance(tools, list) or not tools:
        _add_finding(findings, "error", "tool_schema", "Schema has no `tools` list")
        return summary

    names: List[str] = []
    for idx, tool in enumerate(tools):
        fn = (tool or {}).get("function") or {}
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            _add_finding(findings, "error", "tool_schema", f"Tool[{idx}] missing function.name")
            continue
        names.append(name)

        params = fn.get("parameters")
        if not isinstance(params, dict) or params.get("type") != "object":
            _add_finding(
                findings,
                "error",
                "tool_schema",
                f"Tool `{name}` has invalid parameters schema (expected object).",
            )
            continue

        props = params.get("properties", {})
        required = params.get("required", [])
        if not isinstance(props, dict):
            _add_finding(findings, "error", "tool_schema", f"Tool `{name}` properties must be a dict.")
            continue
        if not isinstance(required, list):
            _add_finding(findings, "error", "tool_schema", f"Tool `{name}` required must be a list.")
            continue

        missing_required = [k for k in required if k not in props]
        if missing_required:
            _add_finding(
                findings,
                "error",
                "tool_schema",
                f"Tool `{name}` required keys missing in properties: {missing_required}",
            )

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        _add_finding(findings, "error", "tool_schema", f"Duplicate tool names in schema: {duplicates}")

    summary["tool_count"] = len(names)
    summary["tool_names"] = sorted(set(names))
    return summary


def verify_traces(
    paths: Sequence[Path],
    schema_tool_names: Set[str],
    findings: List[Dict[str, Any]],
    max_lines: int,
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for path in paths:
        report: Dict[str, Any] = {
            "path": str(path),
            "rows": 0,
            "datasets": {},
            "rows_without_messages": 0,
            "rows_without_system_first": 0,
            "rows_without_user": 0,
            "rows_with_embedded_tools": 0,
            "observed_tool_names": [],
            "embedded_tool_names": [],
            "unknown_vs_schema": [],
        }
        if not path.exists():
            _add_finding(findings, "error", "traces", f"Trace file not found: {path}")
            reports.append(report)
            continue

        observed: Set[str] = set()
        embedded: Set[str] = set()
        datasets: Dict[str, int] = {}

        try:
            for _, row in _iter_jsonl(path, max_lines=max_lines):
                report["rows"] += 1

                dataset = ((row.get("source") or {}).get("dataset")) or "unknown"
                datasets[dataset] = datasets.get(dataset, 0) + 1

                messages = row.get("messages") or []
                if not isinstance(messages, list) or not messages:
                    report["rows_without_messages"] += 1
                    continue

                first_role = (messages[0] or {}).get("role")
                if first_role != "system":
                    report["rows_without_system_first"] += 1

                has_user = any((m or {}).get("role") == "user" for m in messages)
                if not has_user:
                    report["rows_without_user"] += 1

                for message in messages:
                    role = (message or {}).get("role")
                    if role not in VALID_MESSAGE_ROLES:
                        _add_finding(
                            findings,
                            "warning",
                            "traces",
                            f"{path}: invalid role `{role}` in messages.",
                        )

                observed |= _collect_trace_tool_names(row)
                row_embedded = _collect_embedded_tool_names(row)
                if row_embedded:
                    report["rows_with_embedded_tools"] += 1
                embedded |= row_embedded
        except json.JSONDecodeError as exc:
            _add_finding(findings, "error", "traces", f"{path}: invalid JSONL ({exc})")

        report["datasets"] = datasets
        report["observed_tool_names"] = sorted(observed)
        report["embedded_tool_names"] = sorted(embedded)

        if report["rows"] == 0:
            _add_finding(findings, "error", "traces", f"{path}: no records found")

        if report["rows_without_system_first"] > 0:
            _add_finding(
                findings,
                "warning",
                "traces",
                f"{path}: {report['rows_without_system_first']} rows do not start with system message.",
            )

        if report["rows_without_user"] > 0:
            _add_finding(
                findings,
                "warning",
                "traces",
                f"{path}: {report['rows_without_user']} rows have no user message.",
            )

        unknown_vs_schema = sorted(name for name in observed if name not in schema_tool_names)
        report["unknown_vs_schema"] = unknown_vs_schema

        if unknown_vs_schema:
            # AgentDojo may legitimately use per-sample toolsets; mark warning by default.
            datasets_lower = {k.lower() for k in datasets.keys()}
            level = "warning"
            if datasets_lower and datasets_lower <= {"fujitsu_b4"}:
                level = "error"
            _add_finding(
                findings,
                level,
                "traces",
                f"{path}: tool names not in schema: {unknown_vs_schema}",
            )

        reports.append(report)
    return reports


def verify_lossmasks(
    paths: Sequence[Path],
    findings: List[Dict[str, Any]],
    max_lines: int,
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for path in paths:
        report: Dict[str, Any] = {
            "path": str(path),
            "rows": 0,
            "rows_with_nonzero_loss": 0,
            "nonzero_token_ratio": 0.0,
        }
        if not path.exists():
            _add_finding(findings, "error", "lossmasks", f"Lossmask file not found: {path}")
            reports.append(report)
            continue

        nonzero_tokens = 0
        total_tokens = 0
        try:
            for _, row in _iter_jsonl(path, max_lines=max_lines):
                report["rows"] += 1
                mask = row.get("loss_mask")
                if not isinstance(mask, list):
                    _add_finding(findings, "warning", "lossmasks", f"{path}: row missing list loss_mask.")
                    continue
                row_nonzero = 0
                for value in mask:
                    f = _safe_float(value)
                    if f is None:
                        continue
                    total_tokens += 1
                    if f > 0:
                        row_nonzero += 1
                        nonzero_tokens += 1
                if row_nonzero > 0:
                    report["rows_with_nonzero_loss"] += 1
        except json.JSONDecodeError as exc:
            _add_finding(findings, "error", "lossmasks", f"{path}: invalid JSONL ({exc})")

        if report["rows"] == 0:
            _add_finding(findings, "error", "lossmasks", f"{path}: no rows found.")
        elif report["rows_with_nonzero_loss"] == 0:
            _add_finding(findings, "error", "lossmasks", f"{path}: all rows have zero loss_mask.")

        if total_tokens > 0:
            report["nonzero_token_ratio"] = nonzero_tokens / total_tokens
            if report["nonzero_token_ratio"] < 0.001:
                _add_finding(
                    findings,
                    "warning",
                    "lossmasks",
                    f"{path}: very low non-zero mask ratio ({report['nonzero_token_ratio']:.4%}).",
                )

        reports.append(report)
    return reports


STEP_RE = re.compile(r"Step\s+(\d+):\s*mode=([^,]+),\s*loss=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
LOSS_COMPONENT_RE = re.compile(r"(triplet_b|triplet_h|triplet_kl|reroute|retain|kl)=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def verify_training_log(
    path: Optional[Path],
    findings: List[Dict[str, Any]],
    expected_steps: Optional[int],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "path": str(path) if path else None,
        "step_logs": 0,
        "max_step_logged": 0,
        "modes": [],
        "small_grad_warnings": 0,
        "has_training_complete": False,
        "zero_triplet_tail_runs": 0,
    }
    if path is None:
        return report
    if not path.exists():
        _add_finding(findings, "error", "training_log", f"Training log not found: {path}")
        return report

    text = path.read_text(encoding="utf-8", errors="replace")
    report["has_training_complete"] = "Training complete" in text
    if not report["has_training_complete"]:
        _add_finding(findings, "error", "training_log", "Training log missing completion marker.")

    if re.search(r"\b(traceback|runtimeerror|valueerror)\b", text, re.IGNORECASE):
        _add_finding(findings, "warning", "training_log", "Training log contains exception-like tokens.")

    if re.search(r"\b(?:nan|inf)\b", text, re.IGNORECASE):
        _add_finding(findings, "error", "training_log", "Training log contains NaN/Inf token(s).")

    steps: List[int] = []
    losses: List[float] = []
    modes: Set[str] = set()
    zero_tail_run = 0
    max_zero_tail_run = 0
    for line in text.splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        mode = match.group(2).strip()
        loss = float(match.group(3))
        if not math.isfinite(loss):
            _add_finding(findings, "error", "training_log", f"Non-finite loss at step {step}: {loss}")
        steps.append(step)
        losses.append(loss)
        modes.add(mode)

        components = dict((name, float(value)) for name, value in LOSS_COMPONENT_RE.findall(line))
        if "triplet_b" in components and "triplet_h" in components:
            if abs(components["triplet_b"]) < 1e-12 and abs(components["triplet_h"]) < 1e-12:
                zero_tail_run += 1
                max_zero_tail_run = max(max_zero_tail_run, zero_tail_run)
            else:
                zero_tail_run = 0

    report["step_logs"] = len(steps)
    report["max_step_logged"] = max(steps) if steps else 0
    report["modes"] = sorted(modes)
    report["small_grad_warnings"] = text.count("Very small gradients detected")
    report["zero_triplet_tail_runs"] = max_zero_tail_run

    if not steps:
        _add_finding(findings, "error", "training_log", "No `Step ... loss=...` entries found in training log.")
        return report

    for prev, curr in zip(steps, steps[1:]):
        if curr <= prev:
            _add_finding(findings, "warning", "training_log", f"Non-increasing step sequence: {prev} -> {curr}")
            break

    if expected_steps is not None and report["max_step_logged"] < expected_steps:
        _add_finding(
            findings,
            "error",
            "training_log",
            f"Max logged step {report['max_step_logged']} < expected {expected_steps}.",
        )

    if report["small_grad_warnings"] > max(3, len(steps) // 2):
        _add_finding(
            findings,
            "warning",
            "training_log",
            f"Frequent small-gradient warnings: {report['small_grad_warnings']}",
        )

    if report["zero_triplet_tail_runs"] >= 5:
        _add_finding(
            findings,
            "warning",
            "training_log",
            f"Triplet benign/harmful losses hit zero for {report['zero_triplet_tail_runs']} consecutive log points.",
        )

    return report


def verify_training_manifest(
    path: Optional[Path],
    findings: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[int]]:
    report: Dict[str, Any] = {
        "path": str(path) if path else None,
        "found": False,
        "harmful_samples": None,
        "benign_samples": None,
        "total_steps": None,
        "loss_mode": None,
    }
    if path is None:
        return report, None
    if not path.exists():
        _add_finding(findings, "warning", "training_manifest", f"Training manifest not found: {path}")
        return report, None

    try:
        payload = _read_json(path)
    except Exception as exc:  # pragma: no cover
        _add_finding(findings, "error", "training_manifest", f"Failed to parse training manifest: {exc}")
        return report, None

    config = payload.get("config") or {}
    data = payload.get("data") or {}
    report["found"] = True
    report["harmful_samples"] = data.get("harmful_samples")
    report["benign_samples"] = data.get("benign_samples")
    report["loss_mode"] = config.get("loss_mode")
    report["total_steps"] = config.get("total_steps")

    if _safe_float(data.get("harmful_samples")) in (None, 0.0):
        _add_finding(findings, "error", "training_manifest", "harmful_samples missing or zero.")
    if _safe_float(data.get("benign_samples")) in (None, 0.0):
        _add_finding(findings, "error", "training_manifest", "benign_samples missing or zero.")

    loss_mode = str(config.get("loss_mode", ""))
    if not loss_mode:
        _add_finding(findings, "error", "training_manifest", "config.loss_mode missing.")

    if loss_mode == "triplet_full":
        for key in ("triplet_beta_harmful", "triplet_gamma_kl", "triplet_margin_benign", "triplet_margin_harmful"):
            value = _safe_float(config.get(key))
            if value is None or value <= 0:
                _add_finding(
                    findings,
                    "error",
                    "training_manifest",
                    f"Triplet config `{key}` missing or non-positive ({config.get(key)}).",
                )

    total_steps = config.get("total_steps")
    total_steps_i = int(total_steps) if isinstance(total_steps, int) else None
    return report, total_steps_i


def _check_rates_in_dict(prefix: str, payload: Dict[str, Any], findings: List[Dict[str, Any]]) -> None:
    for key, value in payload.items():
        if key in RATE_KEYS:
            _check_rate(findings, prefix, key, value)


def _validate_model_eval_block(
    model_label: str,
    model_block: Dict[str, Any],
    dataset_type: str,
    use_sample_context: bool,
    findings: List[Dict[str, Any]],
    strict_agentdojo_context: bool,
) -> None:
    check_name = f"eval:{model_label}"

    for key in ("tool_flip_asr", "forced_function_call", "capability_retention"):
        if key not in model_block or not isinstance(model_block[key], dict):
            _add_finding(findings, "error", check_name, f"Missing `{key}` block.")

    tool_flip_samples = int((model_block.get("tool_flip_asr") or {}).get("total_samples", 0))
    forced_samples = int((model_block.get("forced_function_call") or {}).get("total_samples", 0))
    cap_samples = int((model_block.get("capability_retention") or {}).get("total_samples", 0))
    gen_samples = int((model_block.get("generation_comparison") or {}).get("total_samples", 0))
    llmail_samples = int((model_block.get("llmail_attack") or {}).get("total_samples", 0))

    if tool_flip_samples == 0 and forced_samples == 0 and cap_samples == 0 and gen_samples == 0 and llmail_samples == 0:
        _add_finding(
            findings,
            "error",
            check_name,
            "No evaluable samples in any metric block.",
        )

    for key, value in model_block.items():
        if isinstance(value, dict):
            _check_rates_in_dict(f"{check_name}:{key}", value, findings)

    context = model_block.get("evaluation_context") or {}
    if dataset_type == "agentdojo":
        tool_source = context.get("tool_source")
        if strict_agentdojo_context and use_sample_context and tool_source in {None, "schema", "schema_fallback"}:
            _add_finding(
                findings,
                "error",
                check_name,
                f"AgentDojo eval used weak context/tool source `{tool_source}` under sample-context mode.",
            )
        tool_count = int(context.get("tool_count", 0))
        if tool_count <= 0:
            _add_finding(findings, "error", check_name, "evaluation_context.tool_count is zero.")


def verify_eval_outputs(
    eval_paths: Sequence[Path],
    findings: List[Dict[str, Any]],
    strict_agentdojo_context: bool,
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for path in eval_paths:
        report: Dict[str, Any] = {"path": str(path)}
        if not path.exists():
            _add_finding(findings, "error", "evaluation", f"Eval JSON not found: {path}")
            reports.append(report)
            continue

        payload = _read_json(path)
        dataset_type = str(payload.get("dataset_type", "unknown")).lower()
        use_sample_context = bool(payload.get("use_sample_context", False))
        report["dataset_type"] = dataset_type
        report["num_samples"] = int(payload.get("num_samples", 0))
        report["use_sample_context"] = use_sample_context

        if report["num_samples"] <= 0:
            _add_finding(findings, "error", "evaluation", f"{path}: num_samples <= 0.")

        baseline = payload.get("baseline")
        cb_model = payload.get("cb_model")
        if not isinstance(baseline, dict) or not isinstance(cb_model, dict):
            _add_finding(findings, "error", "evaluation", f"{path}: missing baseline/cb_model blocks.")
            reports.append(report)
            continue

        _validate_model_eval_block(
            model_label=f"{path.name}:baseline",
            model_block=baseline,
            dataset_type=dataset_type,
            use_sample_context=use_sample_context,
            findings=findings,
            strict_agentdojo_context=strict_agentdojo_context,
        )
        _validate_model_eval_block(
            model_label=f"{path.name}:cb_model",
            model_block=cb_model,
            dataset_type=dataset_type,
            use_sample_context=use_sample_context,
            findings=findings,
            strict_agentdojo_context=strict_agentdojo_context,
        )

        diagnostics = payload.get("diagnostics") or {}
        if diagnostics.get("tools_broken") is True:
            _add_finding(findings, "error", "evaluation", f"{path}: diagnostics.tools_broken = true")

        if dataset_type != "llmail":
            if "llmail_attack" in baseline or "llmail_attack" in cb_model:
                _add_finding(
                    findings,
                    "warning",
                    "evaluation",
                    f"{path}: LLMail metrics present for non-LLMail dataset (`{dataset_type}`).",
                )

        output_cmp = payload.get("output_comparison") or {}
        _check_rates_in_dict(f"evaluation:{path.name}:output_comparison", output_cmp, findings)

        baseline_gen = (baseline.get("generation_comparison") or {})
        cb_gen = (cb_model.get("generation_comparison") or {})
        b_tcr = _safe_float(baseline_gen.get("tool_call_rate"))
        c_tcr = _safe_float(cb_gen.get("tool_call_rate"))
        if dataset_type == "agentdojo" and b_tcr is not None and c_tcr is not None:
            if b_tcr > 0.30 and c_tcr < 0.05:
                _add_finding(
                    findings,
                    "error",
                    "evaluation",
                    f"{path}: CB tool-call rate collapsed ({c_tcr:.3f}) vs baseline ({b_tcr:.3f}).",
                )

        if dataset_type == "agentdojo":
            source_counts = cb_gen.get("tools_source_counts") or {}
            if isinstance(source_counts, dict):
                inferred_or_embedded = (_safe_float(source_counts.get("per_sample_inferred")) or 0.0) + (
                    _safe_float(source_counts.get("sample_embedded")) or 0.0
                )
                total_count = 0.0
                for value in source_counts.values():
                    fv = _safe_float(value)
                    if fv is not None:
                        total_count += fv
                if strict_agentdojo_context and use_sample_context and total_count > 0 and inferred_or_embedded <= 0:
                    _add_finding(
                        findings,
                        "error",
                        "evaluation",
                        f"{path}: generation_comparison.tools_source_counts lacks per-sample tools under sample-context.",
                    )
                fallback = _safe_float(source_counts.get("global_fallback")) or 0.0
                if total_count > 0 and (fallback / total_count) > 0.25:
                    _add_finding(
                        findings,
                        "warning",
                        "evaluation",
                        f"{path}: global_fallback tool source used heavily ({fallback}/{total_count}).",
                    )

            details = cb_gen.get("details") or []
            if isinstance(details, list) and details:
                untrimmed_tail_assistant = 0
                for row in details:
                    prep_meta = row.get("prep_meta") or {}
                    if not isinstance(prep_meta, dict):
                        continue
                    raw_last_role = prep_meta.get("raw_last_role")
                    trimmed = int(prep_meta.get("trimmed_trailing_assistant_count", 0) or 0)
                    prepared_last_role = prep_meta.get("prepared_last_role")
                    if raw_last_role == "assistant" and trimmed == 0:
                        untrimmed_tail_assistant += 1
                    if prepared_last_role == "assistant":
                        untrimmed_tail_assistant += 1
                if untrimmed_tail_assistant > 0:
                    _add_finding(
                        findings,
                        "error",
                        "evaluation",
                        f"{path}: found {untrimmed_tail_assistant} AgentDojo rows with trailing assistant in generation context.",
                    )

        reports.append(report)
    return reports


def _discover_eval_jsons_from_run_dir(run_dir: Path) -> List[Path]:
    eval_dir = run_dir / "eval"
    if not eval_dir.exists():
        return []
    return sorted(eval_dir.glob("*_eval.json"))


def _print_findings(findings: Sequence[Dict[str, Any]]) -> None:
    if not findings:
        print("No findings.")
        return
    for item in findings:
        print(f"[{item['level'].upper()}] {item['check']}: {item['detail']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify CB pipeline outputs and logs.")
    parser.add_argument("--run-dir", type=Path, help="Run directory containing eval/*.json")
    parser.add_argument("--tool-schema", type=Path, help="Tool schema JSON path")
    parser.add_argument("--train-log", type=Path, help="Training log path")
    parser.add_argument("--training-manifest", type=Path, help="Resolved training manifest JSON path")
    parser.add_argument("--eval-json", action="append", default=[], type=Path, help="Eval JSON path (repeatable)")
    parser.add_argument("--trace-jsonl", action="append", default=[], type=Path, help="Trace JSONL path (repeatable)")
    parser.add_argument("--lossmask-jsonl", action="append", default=[], type=Path, help="Lossmask JSONL path (repeatable)")
    parser.add_argument("--report-json", type=Path, help="Write report JSON to this path")
    parser.add_argument("--max-lines", type=int, default=0, help="Max lines to scan from each JSONL (0 = all)")
    parser.add_argument("--expected-steps", type=int, help="Expected training steps for max-step check")
    parser.add_argument(
        "--strict-agentdojo-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require AgentDojo eval to use non-schema tool context when sample-context is enabled.",
    )
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero if any errors are found")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    findings: List[Dict[str, Any]] = []

    eval_paths: List[Path] = list(args.eval_json)
    if args.run_dir:
        eval_paths.extend(_discover_eval_jsons_from_run_dir(args.run_dir))
    # Deduplicate while preserving order
    seen: Set[str] = set()
    dedup_eval_paths: List[Path] = []
    for path in eval_paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        dedup_eval_paths.append(path)

    schema_report: Dict[str, Any] = {}
    schema_tool_names: Set[str] = set()
    if args.tool_schema:
        schema_report = verify_tool_schema(args.tool_schema, findings)
        schema_tool_names = set(schema_report.get("tool_names", []))
    else:
        _add_finding(findings, "warning", "tool_schema", "No tool schema provided; schema checks skipped.")

    trace_reports = verify_traces(args.trace_jsonl, schema_tool_names, findings, max_lines=args.max_lines)
    lossmask_reports = verify_lossmasks(args.lossmask_jsonl, findings, max_lines=args.max_lines)

    manifest_path = args.training_manifest
    if manifest_path is None and args.train_log is not None:
        inferred_manifest = args.train_log.parent / "training_manifest.json"
        manifest_path = inferred_manifest

    training_manifest_report, manifest_steps = verify_training_manifest(manifest_path, findings)
    effective_expected_steps = args.expected_steps if args.expected_steps is not None else manifest_steps
    training_report = verify_training_log(args.train_log, findings, expected_steps=effective_expected_steps)
    eval_reports = verify_eval_outputs(
        dedup_eval_paths,
        findings,
        strict_agentdojo_context=args.strict_agentdojo_context,
    )

    error_count = sum(1 for item in findings if item["level"] == "error")
    warning_count = sum(1 for item in findings if item["level"] == "warning")

    report = {
        "summary": {
            "errors": error_count,
            "warnings": warning_count,
            "passed": error_count == 0,
        },
        "schema": schema_report,
        "traces": trace_reports,
        "lossmasks": lossmask_reports,
        "training_manifest": training_manifest_report,
        "training": training_report,
        "evaluation": eval_reports,
        "findings": findings,
    }

    print("============================================================")
    print("PIPELINE VERIFICATION REPORT")
    print("============================================================")
    print(f"Errors:   {error_count}")
    print(f"Warnings: {warning_count}")
    print(f"Status:   {'PASS' if error_count == 0 else 'FAIL'}")
    print("")
    _print_findings(findings)

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print("")
        print(f"Report JSON: {args.report_json}")

    if args.fail_on_error and error_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
