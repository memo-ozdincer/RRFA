#!/usr/bin/env python3
"""
Export a self-contained render audit JSONL.

Each row links:
- canonical trace context (messages/system/tools/labels/source)
- rendered sequence text/options/alignment/signals
- lossmask metadata (policy/sample_weight/stats, optional full mask values)

This is intended for run-level debugging and reproducibility checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _trace_id_of(row: Dict[str, Any]) -> Optional[str]:
    trace_id = row.get("id") or row.get("trace_id")
    if isinstance(trace_id, str) and trace_id:
        return trace_id
    return None


def _render_id_of(row: Dict[str, Any]) -> Optional[str]:
    render_id = row.get("render_id")
    if isinstance(render_id, str) and render_id:
        return render_id
    return None


def _extract_system_prompt(messages: List[Dict[str, Any]]) -> Optional[str]:
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            return str(content) if content is not None else None
    return None


def _extract_embedded_tools(trace_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def _consume(candidate: Any) -> None:
        if not isinstance(candidate, list):
            return
        for item in candidate:
            if isinstance(item, dict):
                out.append(item)

    _consume(trace_row.get("tools"))
    source_fields = (trace_row.get("raw_metadata") or {}).get("source_fields") or {}
    _consume(source_fields.get("tools"))
    _consume(source_fields.get("available_tools"))
    return out


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full render audit JSONL.")
    parser.add_argument("--trace-jsonl", action="append", default=[], type=Path, help="Trace JSONL (repeatable)")
    parser.add_argument("--render-jsonl", action="append", default=[], type=Path, help="Render JSONL (repeatable)")
    parser.add_argument("--lossmask-jsonl", action="append", default=[], type=Path, help="Lossmask JSONL (repeatable)")
    parser.add_argument("--output", required=True, type=Path, help="Output audit JSONL")
    parser.add_argument("--summary", type=Path, default=None, help="Optional summary text file")
    parser.add_argument(
        "--include-loss-mask-values",
        action="store_true",
        help="Include full per-token loss_mask array in output rows",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    missing_inputs: List[str] = []
    for path in args.trace_jsonl + args.render_jsonl + args.lossmask_jsonl:
        if not path.exists():
            missing_inputs.append(str(path))
    if missing_inputs:
        print("ERROR: Missing input file(s):")
        for item in missing_inputs:
            print(f"  - {item}")
        return 1

    if not args.render_jsonl:
        print("ERROR: At least one --render-jsonl is required.")
        return 1

    trace_by_id: Dict[str, Dict[str, Any]] = {}
    for trace_path in args.trace_jsonl:
        for row in _iter_jsonl(trace_path):
            trace_id = _trace_id_of(row)
            if trace_id is not None:
                trace_by_id[trace_id] = row

    lossmask_by_render_id: Dict[str, Dict[str, Any]] = {}
    lossmask_by_trace_id: Dict[str, Dict[str, Any]] = {}
    for lossmask_path in args.lossmask_jsonl:
        for row in _iter_jsonl(lossmask_path):
            render_id = _render_id_of(row)
            trace_id = _trace_id_of(row)
            if render_id is not None:
                lossmask_by_render_id[render_id] = row
            if trace_id is not None and trace_id not in lossmask_by_trace_id:
                lossmask_by_trace_id[trace_id] = row

    output_rows: List[Dict[str, Any]] = []
    dataset_counts: Dict[str, int] = {}
    missing_trace_count = 0
    missing_render_text_count = 0
    missing_lossmask_count = 0

    for render_path in args.render_jsonl:
        for render_row in _iter_jsonl(render_path):
            render_id = _render_id_of(render_row)
            trace_id = _trace_id_of(render_row)
            trace_row = trace_by_id.get(trace_id or "", {})
            if not trace_row:
                missing_trace_count += 1

            source = trace_row.get("source") or {}
            dataset = source.get("dataset") or "unknown"
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

            messages = trace_row.get("messages") or []
            if not isinstance(messages, list):
                messages = []

            embedded_tools = _extract_embedded_tools(trace_row)
            lossmask_row = (
                lossmask_by_render_id.get(render_id or "")
                or lossmask_by_trace_id.get(trace_id or "")
                or {}
            )
            if not lossmask_row:
                missing_lossmask_count += 1

            rendered_text = render_row.get("rendered_text")
            if not isinstance(rendered_text, str):
                missing_render_text_count += 1

            lossmask_payload: Dict[str, Any] = {
                "policy_id": lossmask_row.get("policy_id"),
                "policy_version": lossmask_row.get("policy_version"),
                "sample_weight": lossmask_row.get("sample_weight"),
                "stats": lossmask_row.get("stats"),
            }
            if args.include_loss_mask_values:
                lossmask_payload["loss_mask"] = lossmask_row.get("loss_mask")

            output_rows.append(
                {
                    "trace_id": trace_id,
                    "render_id": render_id,
                    "source": source,
                    "labels": trace_row.get("labels"),
                    "messages": messages,
                    "system_prompt": _extract_system_prompt(messages),
                    "embedded_tools": embedded_tools,
                    "rendered_text": rendered_text,
                    "tokenizer_id": render_row.get("tokenizer_id"),
                    "sequence_length": render_row.get("sequence_length"),
                    "render_options": render_row.get("render_options"),
                    "alignment": render_row.get("alignment"),
                    "signals": render_row.get("signals"),
                    "lossmask": lossmask_payload,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = _write_jsonl(args.output, output_rows)

    summary_lines = [
        "FULL RENDER AUDIT SUMMARY",
        f"output: {args.output}",
        f"rows_written: {written}",
        f"datasets: {json.dumps(dataset_counts, sort_keys=True)}",
        f"missing_trace_rows: {missing_trace_count}",
        f"missing_rendered_text_rows: {missing_render_text_count}",
        f"missing_lossmask_rows: {missing_lossmask_count}",
        f"include_loss_mask_values: {args.include_loss_mask_values}",
    ]
    summary_text = "\n".join(summary_lines) + "\n"

    print(summary_text, end="")
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(summary_text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

