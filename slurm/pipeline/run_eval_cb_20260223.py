#!/usr/bin/env python3
"""Standalone eval script for cb_20260223_113325 adapter.

Run on the cluster with:
    python slurm/pipeline/run_eval_cb_20260223.py

Writes results to VIS_DIR/eval/ and also prints everything to stdout.
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# ─── Configuration ───────────────────────────────────────────────────────────
ADAPTER_PATH = "/scratch/memoozd/cb-scratch/models/cb_20260223_113325"
CB_SCRATCH = "/scratch/memoozd/cb-scratch"
HF_HUB_CACHE = f"{CB_SCRATCH}/cache/hf/hub"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LIMIT_EVAL = 50
DTYPE = "bfloat16"

VIS_DIR = f"{CB_SCRATCH}/eval_runs/eval_cb_20260223_113325_20260223_165521"
VIS_EVAL_DIR = f"{VIS_DIR}/eval"

TOOL_SCHEMA_FUJITSU = "configs/tool_schemas/b4_standard_v1.json"
TOOL_SCHEMA_LLMAIL = "configs/tool_schemas/llmail_inject_challenge_v2.json"

# Trace / split candidates (picks first that exists)
TRACES_CANDIDATES = [
    f"{CB_SCRATCH}/data/traces",
    f"{CB_SCRATCH}/data_all/traces",
    f"{CB_SCRATCH}/data_llmail_only/traces",
]
SPLIT_CANDIDATES = [
    f"{CB_SCRATCH}/data/split/agentdojo",
    f"{CB_SCRATCH}/data_all/split/agentdojo",
]

EVAL_DATASETS = ["llmail", "agentdojo"]  # add "fujitsu" if desired
# ─────────────────────────────────────────────────────────────────────────────


def find_first_dir(candidates):
    for c in candidates:
        if Path(c).is_dir():
            return c
    return None


def resolve_adapter(adapter_path):
    p = Path(adapter_path)
    if (p / "adapter_config.json").is_file():
        return str(p)
    if (p / "final" / "adapter_config.json").is_file():
        return str(p / "final")
    sys.exit(f"ERROR: No adapter_config.json at {p} or {p / 'final'}")


def resolve_baseline():
    cache_name = MODEL_ID.replace("/", "--")
    snapshot_root = Path(HF_HUB_CACHE) / f"models--{cache_name}" / "snapshots"
    if not snapshot_root.is_dir():
        sys.exit(f"ERROR: Model not cached at {snapshot_root.parent}")
    snapshots = sorted(snapshot_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    if not snapshots:
        sys.exit(f"ERROR: Empty snapshots dir: {snapshot_root}")
    return str(snapshots[0])


def run_eval(cmd, label, log_path):
    """Run an eval command, tee to log file and stdout."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log:     {log_path}\n")

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        proc.wait()

    if proc.returncode == 0:
        print(f"\n  OK: {label}")
    else:
        print(f"\n  FAIL: {label} (exit code {proc.returncode})")
    return proc.returncode


def main():
    adapter = resolve_adapter(ADAPTER_PATH)
    baseline = resolve_baseline()
    traces_dir = find_first_dir(TRACES_CANDIDATES)
    split_dir = find_first_dir(SPLIT_CANDIDATES)

    if not traces_dir:
        sys.exit("ERROR: Could not find traces directory")

    os.makedirs(VIS_EVAL_DIR, exist_ok=True)

    print(f"Adapter:       {adapter}")
    print(f"Baseline:      {baseline}")
    print(f"Traces:        {traces_dir}")
    print(f"Split dir:     {split_dir or '<not found>'}")
    print(f"Eval datasets: {EVAL_DATASETS}")
    print(f"Output:        {VIS_DIR}")

    results = {}

    # ── Fujitsu ──────────────────────────────────────────────────────────
    if "fujitsu" in EVAL_DATASETS:
        ds = f"{traces_dir}/fujitsu_b4_ds.jsonl"
        if Path(ds).is_file():
            rc = run_eval(
                [
                    sys.executable, "src/evaluation/eval.py",
                    "--baseline", baseline,
                    "--cb-adapter", adapter,
                    "--eval-data", ds,
                    "--tool-schema", TOOL_SCHEMA_FUJITSU,
                    "--output", f"{VIS_EVAL_DIR}/fujitsu_eval.json",
                    "--limit", str(LIMIT_EVAL),
                    "--merge-adapter",
                    "--dtype", DTYPE,
                ],
                label="Fujitsu B4",
                log_path=f"{VIS_EVAL_DIR}/fujitsu_eval.log",
            )
            results["fujitsu"] = "OK" if rc == 0 else "FAIL"
        else:
            print(f"Skipping Fujitsu: {ds} not found")

    # ── LLMail ───────────────────────────────────────────────────────────
    if "llmail" in EVAL_DATASETS:
        ds = f"{traces_dir}/llmail_inject_ds.jsonl"
        if Path(ds).is_file():
            rc = run_eval(
                [
                    sys.executable, "src/evaluation/eval.py",
                    "--baseline", baseline,
                    "--cb-adapter", adapter,
                    "--eval-data", ds,
                    "--tool-schema", TOOL_SCHEMA_LLMAIL,
                    "--output", f"{VIS_EVAL_DIR}/llmail_eval.json",
                    "--limit", str(LIMIT_EVAL),
                    "--llmail-retrieved-only", "true",
                    "--merge-adapter",
                    "--dtype", DTYPE,
                ],
                label="LLMail Inject",
                log_path=f"{VIS_EVAL_DIR}/llmail_eval.log",
            )
            results["llmail"] = "OK" if rc == 0 else "FAIL"
        else:
            print(f"Skipping LLMail: {ds} not found")

    # ── AgentDojo ────────────────────────────────────────────────────────
    if "agentdojo" in EVAL_DATASETS:
        agentdojo_ds = None
        for candidate in [
            f"{split_dir}/agentdojo_traces_cb.jsonl" if split_dir else None,
            f"{traces_dir}/../split/agentdojo/agentdojo_traces_cb.jsonl",
        ]:
            if candidate and Path(candidate).is_file():
                agentdojo_ds = candidate
                break

        if agentdojo_ds:
            rc = run_eval(
                [
                    sys.executable, "src/evaluation/eval.py",
                    "--baseline", baseline,
                    "--cb-adapter", adapter,
                    "--eval-data", agentdojo_ds,
                    "--tool-schema", TOOL_SCHEMA_FUJITSU,
                    "--output", f"{VIS_EVAL_DIR}/agentdojo_eval.json",
                    "--use-sample-context",
                    "--limit", str(LIMIT_EVAL),
                    "--merge-adapter",
                    "--dtype", DTYPE,
                ],
                label="AgentDojo",
                log_path=f"{VIS_EVAL_DIR}/agentdojo_eval.log",
            )
            results["agentdojo"] = "OK" if rc == 0 else "FAIL"
        else:
            print("Skipping AgentDojo: harmful traces not found")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  EVAL SUMMARY")
    print(f"{'='*60}")
    for ds_name, status in results.items():
        print(f"  {ds_name:<15} {status}")
    print(f"  Output dir: {VIS_DIR}")

    # Write run_info.json
    import json
    run_info = {
        "run_label": "cb_20260223_113325",
        "adapter_path": adapter,
        "baseline_model": baseline,
        "model_id": MODEL_ID,
        "eval_datasets": ",".join(EVAL_DATASETS),
        "traces_dir": traces_dir,
        "limit_eval": LIMIT_EVAL,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    info_path = f"{VIS_DIR}/run_info.json"
    with open(info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"\n  run_info.json -> {info_path}")

    # ── Visualization ────────────────────────────────────────────────────
    ok_count = sum(1 for v in results.values() if v == "OK")
    if ok_count > 0:
        print("\nRunning visualization...")
        vis_cmd = [
            sys.executable, "scripts/visualize_sweep_results.py",
            VIS_DIR,
            "--no-color",
            "--csv", f"{VIS_DIR}/eval_summary.csv",
        ]
        if traces_dir:
            vis_cmd += ["--traces-dir", traces_dir]

        vis_log = f"{VIS_DIR}/visualization.txt"
        run_eval(vis_cmd, label="Visualization", log_path=vis_log)
        print(f"  CSV:    {VIS_DIR}/eval_summary.csv")
        print(f"  Log:    {vis_log}")

    print(f"\nDone at {datetime.now()}")


if __name__ == "__main__":
    main()
