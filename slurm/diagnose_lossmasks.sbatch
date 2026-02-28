#!/bin/bash
#SBATCH --job-name=diagnose_lossmasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# LOSSMASK DIAGNOSTIC JOB
# =============================================================================
#
# CPU-only job: runs scripts/diagnose_lossmasks.py + view_lossmasks.py
# on cluster data, writing a report to $REPORT_FILE.
#
# Two modes:
#
#   Mode A (default): Run ETL_B from scratch on existing cluster traces,
#   then diagnose. Covers all policies in POLICIES.
#
#   Mode B: Diagnose renders/lossmasks from an existing sweep run dir.
#   Set SWEEP_RUN_DIR to the run directory.
#
# Examples:
#
#   # Mode A — run ETL_B + diagnose all configured policies
#   sbatch slurm/diagnose_lossmasks.sbatch
#
#   # Mode A — custom policy set
#   POLICIES="assistant_only,cb_full_sequence,tool_calls_only" \
#     sbatch slurm/diagnose_lossmasks.sbatch
#
#   # Mode B — diagnose an existing sweep run
#   SWEEP_RUN_DIR=/scratch/memoozd/cb-scratch/sweeps/hparam_sweep_20260228_120000/a10.0_l4_8_14_assistant_only \
#     sbatch slurm/diagnose_lossmasks.sbatch
#
# =============================================================================

set -euo pipefail

# =============================================================================
# Environment Setup
# =============================================================================
PROJECT_DIR="/project/def-zhijing/memoozd"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"
CB_SCRATCH="/scratch/memoozd/cb-scratch"

cd "$REPO_DIR"

module --force purge || true
module load StdEnv/2023
module load python/3.11.5

source "$VENV_DIR/bin/activate"

# =============================================================================
# Config (override with env vars)
# =============================================================================
# Mode A: policies to run ETL_B for (comma-separated)
POLICIES="${POLICIES:-assistant_only,cb_full_sequence}"

# Mode B: if set, skip ETL_B and diagnose this run dir directly
SWEEP_RUN_DIR="${SWEEP_RUN_DIR:-}"

# Cluster trace files (Mode A)
TRACES_DIR="$CB_SCRATCH/data/traces"
FUJITSU_DS_TRACE="$TRACES_DIR/fujitsu_b4_ds.jsonl"
FUJITSU_DR_TRACE="$TRACES_DIR/fujitsu_b4_dr.jsonl"
AGENTDOJO_TRACE="$TRACES_DIR/agentdojo_complete.jsonl"
TOKENIZER_PATH="${TOKENIZER_PATH:-meta-llama/Llama-3.1-8B-Instruct}"

# Scratch space for ETL_B output (Mode A)
WORK_DIR="$CB_SCRATCH/lossmask_diag/${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
REPORT_FILE="$WORK_DIR/lossmask_report.txt"

# View sample count
VIEW_LIMIT="${VIEW_LIMIT:-8}"

# =============================================================================
mkdir -p "$WORK_DIR/logs"
echo "========================================" | tee "$REPORT_FILE"
echo "LOSSMASK DIAGNOSTIC REPORT"              | tee -a "$REPORT_FILE"
echo "Date: $(date)"                           | tee -a "$REPORT_FILE"
echo "Job: ${SLURM_JOB_ID:-manual}"           | tee -a "$REPORT_FILE"
echo "Work dir: $WORK_DIR"                     | tee -a "$REPORT_FILE"
echo "========================================" | tee -a "$REPORT_FILE"

# =============================================================================
# Mode B: diagnose existing sweep run
# =============================================================================
if [[ -n "$SWEEP_RUN_DIR" ]]; then
    echo "" | tee -a "$REPORT_FILE"
    echo "MODE B: Diagnosing existing sweep run" | tee -a "$REPORT_FILE"
    echo "  Run dir: $SWEEP_RUN_DIR"            | tee -a "$REPORT_FILE"

    RENDERS_DIR="$SWEEP_RUN_DIR/renders"
    MASKS_DIR="$SWEEP_RUN_DIR/lossmasks"

    if [[ ! -d "$RENDERS_DIR" || ! -d "$MASKS_DIR" ]]; then
        echo "ERROR: renders/ or lossmasks/ not found in $SWEEP_RUN_DIR" | tee -a "$REPORT_FILE"
        exit 1
    fi

    echo "" | tee -a "$REPORT_FILE"
    echo "=== DIAGNOSTIC ===" | tee -a "$REPORT_FILE"
    python scripts/diagnose_lossmasks.py \
        --renders-dir "$RENDERS_DIR" \
        --lossmasks-dir "$MASKS_DIR" \
        --verbose 2>&1 | tee -a "$REPORT_FILE"

    echo "" | tee -a "$REPORT_FILE"
    echo "=== SAMPLE VIEW (first $VIEW_LIMIT non-zero samples) ===" | tee -a "$REPORT_FILE"
    # Find the first matching render/lossmask pair
    first_mask=$(ls "$MASKS_DIR"/*.jsonl 2>/dev/null | head -n 1 || true)
    if [[ -n "$first_mask" ]]; then
        stem=$(basename "$first_mask" .jsonl)
        first_render="$RENDERS_DIR/${stem}.jsonl"
        if [[ -f "$first_render" ]]; then
            python scripts/view_lossmasks.py \
                --renders "$first_render" \
                --lossmasks "$first_mask" \
                --nonzero-only \
                --limit "$VIEW_LIMIT" 2>&1 | tee -a "$REPORT_FILE"
        fi
    fi

    echo "" | tee -a "$REPORT_FILE"
    echo "Report written to: $REPORT_FILE"
    exit 0
fi

# =============================================================================
# Mode A: Run ETL_B then diagnose
# =============================================================================
echo "" | tee -a "$REPORT_FILE"
echo "MODE A: ETL_B + diagnose for policies: $POLICIES" | tee -a "$REPORT_FILE"

# Validate trace files
TRACE_FILES=()
for tf in "$FUJITSU_DS_TRACE" "$FUJITSU_DR_TRACE" "$AGENTDOJO_TRACE"; do
    if [[ -f "$tf" ]]; then
        TRACE_FILES+=("$tf")
        echo "  Found: $tf" | tee -a "$REPORT_FILE"
    else
        echo "  MISSING: $tf — skipping" | tee -a "$REPORT_FILE"
    fi
done

if [[ ${#TRACE_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No trace files found. Check TRACES_DIR=$TRACES_DIR" | tee -a "$REPORT_FILE"
    exit 1
fi

# Resolve tokenizer
HF_HUB_CACHE="$CB_SCRATCH/cache/hf/hub"
SNAPSHOT_ROOT="$HF_HUB_CACHE/models--meta-llama--Llama-3.1-8B-Instruct/snapshots"
if [[ -d "$SNAPSHOT_ROOT" ]]; then
    RESOLVED_TOK=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1 || true)
    [[ -n "$RESOLVED_TOK" ]] && TOKENIZER_PATH="$RESOLVED_TOK"
fi
echo "  Tokenizer: $TOKENIZER_PATH" | tee -a "$REPORT_FILE"

# Split policy string
IFS=',' read -ra POLICY_ARRAY <<< "$POLICIES"

for policy in "${POLICY_ARRAY[@]}"; do
    echo "" | tee -a "$REPORT_FILE"
    echo "========================================" | tee -a "$REPORT_FILE"
    echo "POLICY: $policy"                          | tee -a "$REPORT_FILE"
    echo "========================================" | tee -a "$REPORT_FILE"

    POLICY_DIR="$WORK_DIR/$policy"
    mkdir -p "$POLICY_DIR/renders" "$POLICY_DIR/lossmasks"

    # --- ETL_B ---
    echo "Running ETL_B..." | tee -a "$REPORT_FILE"
    etl_start=$(date +%s)

    for trace_file in "${TRACE_FILES[@]}"; do
        stem=$(basename "$trace_file" .jsonl)
        render_out="$POLICY_DIR/renders/${stem}.jsonl"
        lossmask_out="$POLICY_DIR/lossmasks/${stem}.jsonl"

        EXTRA_ARGS=()
        # Allow skeleton traces (prompt-only) — they'll produce zero-mask samples
        # which the diagnostic will flag; keeping them makes the report accurate
        [[ "$stem" == *skeleton* || "$stem" == fujitsu_b4_dr* ]] && EXTRA_ARGS+=(--allow-skeleton)

        echo "  ETL_B: $stem → policy=$policy" | tee -a "$REPORT_FILE"
        python src/schemas/tools/ETL_B.py \
            --traces "$trace_file" \
            --render-out "$render_out" \
            --lossmask-out "$lossmask_out" \
            --tokenizer "$TOKENIZER_PATH" \
            --max-length 4096 \
            --policy-override "$policy" \
            "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$WORK_DIR/logs/etl_b_${policy}_${stem}.log"
    done

    etl_end=$(date +%s)
    echo "  ETL_B done in $((etl_end - etl_start))s" | tee -a "$REPORT_FILE"

    # --- Diagnostic ---
    echo "" | tee -a "$REPORT_FILE"
    echo "--- DIAGNOSTIC: $policy ---" | tee -a "$REPORT_FILE"
    python scripts/diagnose_lossmasks.py \
        --renders-dir "$POLICY_DIR/renders" \
        --lossmasks-dir "$POLICY_DIR/lossmasks" \
        --verbose 2>&1 | tee -a "$REPORT_FILE"

    # --- Sample view ---
    echo "" | tee -a "$REPORT_FILE"
    echo "--- SAMPLE VIEW: $policy (first $VIEW_LIMIT non-zero) ---" | tee -a "$REPORT_FILE"
    first_mask=$(ls "$POLICY_DIR/lossmasks"/*.jsonl 2>/dev/null | head -n 1 || true)
    if [[ -n "$first_mask" ]]; then
        stem=$(basename "$first_mask" .jsonl)
        render_file="$POLICY_DIR/renders/${stem}.jsonl"
        if [[ -f "$render_file" ]]; then
            python scripts/view_lossmasks.py \
                --renders "$render_file" \
                --lossmasks "$first_mask" \
                --nonzero-only \
                --limit "$VIEW_LIMIT" 2>&1 | tee -a "$REPORT_FILE"
        fi
    fi
done

echo "" | tee -a "$REPORT_FILE"
echo "========================================" | tee -a "$REPORT_FILE"
echo "All done. Report: $REPORT_FILE"         | tee -a "$REPORT_FILE"
echo "========================================" | tee -a "$REPORT_FILE"
