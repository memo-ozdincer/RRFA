#!/bin/bash
#SBATCH --job-name=eval_cb_0223
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# EVAL for cb_20260223_113325 adapter
# =============================================================================
#
# Usage:
#   sbatch slurm/pipeline/run_eval_cb_20260223.sbatch
#
# Output goes to:
#   /scratch/memoozd/cb-scratch/eval_runs/eval_cb_20260223_113325_20260223_165521/
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"

# Load modules
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv not found at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# =============================================================================
# Cache setup
# =============================================================================
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,wandb,triton}
mkdir -p "$CB_SCRATCH/logs"

export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export WANDB_DIR="$CACHE_DIR/wandb"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export XDG_CACHE_HOME="$CACHE_DIR"
export DS_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
mkdir -p "$TRITON_CACHE_DIR" "$TRITON_CACHE_DIR/autotune"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# =============================================================================
# Configuration
# =============================================================================
ADAPTER_PATH="/scratch/memoozd/cb-scratch/models/cb_20260223_113325"
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
LIMIT_EVAL=50
DTYPE="bfloat16"
EVAL_DATASETS="llmail,agentdojo"

VIS_DIR="$CB_SCRATCH/eval_runs/eval_cb_20260223_113325_20260223_165521"
VIS_EVAL_DIR="$VIS_DIR/eval"
mkdir -p "$VIS_EVAL_DIR"

# Resolve adapter
if [[ -f "$ADAPTER_PATH/adapter_config.json" ]]; then
    RESOLVED_ADAPTER="$ADAPTER_PATH"
elif [[ -d "$ADAPTER_PATH/final" && -f "$ADAPTER_PATH/final/adapter_config.json" ]]; then
    RESOLVED_ADAPTER="$ADAPTER_PATH/final"
else
    echo "ERROR: No adapter_config.json found at $ADAPTER_PATH or $ADAPTER_PATH/final"
    exit 1
fi

# Resolve baseline model from cache
MODEL_CACHE_NAME=$(echo "$MODEL_ID" | sed 's/\//--/g')
MODEL_CACHE_DIR="$HF_HUB_CACHE/models--${MODEL_CACHE_NAME}"
SNAPSHOT_ROOT="$MODEL_CACHE_DIR/snapshots"

if [[ -d "$SNAPSHOT_ROOT" ]]; then
    BASELINE_MODEL=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1)
    if [[ -z "$BASELINE_MODEL" ]]; then
        echo "ERROR: Snapshot directory exists but is empty: $SNAPSHOT_ROOT"
        exit 1
    fi
else
    echo "ERROR: Model not cached at: $MODEL_CACHE_DIR"
    exit 1
fi

# Traces directory
TRACES_DIR=""
for candidate in \
    "$CB_SCRATCH/data/traces" \
    "$CB_SCRATCH/data_all/traces" \
    "$CB_SCRATCH/data_llmail_only/traces"; do
    if [[ -d "$candidate" ]]; then
        TRACES_DIR="$candidate"
        break
    fi
done

if [[ -z "$TRACES_DIR" || ! -d "$TRACES_DIR" ]]; then
    echo "ERROR: Could not find traces directory. Set TRACES_DIR explicitly."
    exit 1
fi

# Tool schemas
TOOL_SCHEMA_FUJITSU="configs/tool_schemas/b4_standard_v1.json"
TOOL_SCHEMA_LLMAIL="configs/tool_schemas/llmail_inject_challenge_v2.json"

# Split dir for AgentDojo
SPLIT_DIR=""
for candidate in \
    "$CB_SCRATCH/data/split/agentdojo" \
    "$CB_SCRATCH/data_all/split/agentdojo"; do
    if [[ -d "$candidate" ]]; then
        SPLIT_DIR="$candidate"
        break
    fi
done

RUN_LABEL="cb_20260223_113325"

IFS=',' read -ra EVAL_DATASET_ARRAY <<< "$EVAL_DATASETS"

echo ""
echo "========================================"
echo "EVAL RUN: $RUN_LABEL"
echo "========================================"
echo "Adapter:          $RESOLVED_ADAPTER"
echo "Baseline:         $BASELINE_MODEL"
echo "Eval Datasets:    $EVAL_DATASETS"
echo "Traces Dir:       $TRACES_DIR"
echo "Split Dir:        ${SPLIT_DIR:-<not found>}"
echo "Limit Eval:       $LIMIT_EVAL"
echo "Output Dir:       $VIS_DIR"
echo ""

# =============================================================================
# Helper
# =============================================================================
should_eval() {
    local needle="$1"
    for d in "${EVAL_DATASET_ARRAY[@]}"; do
        d="$(echo "$d" | xargs)"
        if [[ "$d" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

# =============================================================================
# Run evaluations
# =============================================================================
EVAL_COUNT=0
EVAL_OK=0
EVAL_FAIL=0

# --- LLMail ---
if should_eval "llmail"; then
    LLMAIL_DS="$TRACES_DIR/llmail_inject_ds.jsonl"
    if [[ -f "$LLMAIL_DS" ]]; then
        echo "Evaluating LLMail..."
        EVAL_COUNT=$((EVAL_COUNT + 1))
        if python src/evaluation/eval.py \
            --baseline "$BASELINE_MODEL" \
            --cb-adapter "$RESOLVED_ADAPTER" \
            --eval-data "$LLMAIL_DS" \
            --tool-schema "$TOOL_SCHEMA_LLMAIL" \
            --output "$VIS_EVAL_DIR/llmail_eval.json" \
            --limit "$LIMIT_EVAL" \
            --llmail-retrieved-only true \
            --merge-adapter \
            --dtype "$DTYPE" 2>&1 | tee "$VIS_EVAL_DIR/llmail_eval.log"; then
            EVAL_OK=$((EVAL_OK + 1))
            echo "  OK: $VIS_EVAL_DIR/llmail_eval.json"
        else
            EVAL_FAIL=$((EVAL_FAIL + 1))
            echo "  FAIL: LLMail eval returned non-zero"
        fi
        echo ""
    else
        echo "Skipping LLMail: traces not found at $LLMAIL_DS"
    fi
fi

# --- AgentDojo ---
if should_eval "agentdojo"; then
    AGENTDOJO_HARMFUL=""
    for candidate in \
        "${SPLIT_DIR:-__none__}/agentdojo_traces_cb.jsonl" \
        "$TRACES_DIR/../split/agentdojo/agentdojo_traces_cb.jsonl"; do
        if [[ -f "$candidate" ]]; then
            AGENTDOJO_HARMFUL="$candidate"
            break
        fi
    done

    if [[ -n "$AGENTDOJO_HARMFUL" ]]; then
        echo "Evaluating AgentDojo..."
        EVAL_COUNT=$((EVAL_COUNT + 1))
        if python src/evaluation/eval.py \
            --baseline "$BASELINE_MODEL" \
            --cb-adapter "$RESOLVED_ADAPTER" \
            --eval-data "$AGENTDOJO_HARMFUL" \
            --tool-schema "$TOOL_SCHEMA_FUJITSU" \
            --output "$VIS_EVAL_DIR/agentdojo_eval.json" \
            --use-sample-context \
            --limit "$LIMIT_EVAL" \
            --merge-adapter \
            --dtype "$DTYPE" 2>&1 | tee "$VIS_EVAL_DIR/agentdojo_eval.log"; then
            EVAL_OK=$((EVAL_OK + 1))
            echo "  OK: $VIS_EVAL_DIR/agentdojo_eval.json"
        else
            EVAL_FAIL=$((EVAL_FAIL + 1))
            echo "  FAIL: AgentDojo eval returned non-zero"
        fi
        echo ""
    else
        echo "Skipping AgentDojo: harmful traces not found"
    fi
fi

# --- Fujitsu ---
if should_eval "fujitsu"; then
    FUJITSU_DS="$TRACES_DIR/fujitsu_b4_ds.jsonl"
    if [[ -f "$FUJITSU_DS" ]]; then
        echo "Evaluating Fujitsu..."
        EVAL_COUNT=$((EVAL_COUNT + 1))
        if python src/evaluation/eval.py \
            --baseline "$BASELINE_MODEL" \
            --cb-adapter "$RESOLVED_ADAPTER" \
            --eval-data "$FUJITSU_DS" \
            --tool-schema "$TOOL_SCHEMA_FUJITSU" \
            --output "$VIS_EVAL_DIR/fujitsu_eval.json" \
            --limit "$LIMIT_EVAL" \
            --merge-adapter \
            --dtype "$DTYPE" 2>&1 | tee "$VIS_EVAL_DIR/fujitsu_eval.log"; then
            EVAL_OK=$((EVAL_OK + 1))
            echo "  OK: $VIS_EVAL_DIR/fujitsu_eval.json"
        else
            EVAL_FAIL=$((EVAL_FAIL + 1))
            echo "  FAIL: Fujitsu eval returned non-zero"
        fi
        echo ""
    else
        echo "Skipping Fujitsu: traces not found at $FUJITSU_DS"
    fi
fi

# =============================================================================
# Write run_info.json
# =============================================================================
cat > "$VIS_DIR/run_info.json" <<RUNEOF
{
    "run_label": "$RUN_LABEL",
    "adapter_path": "$RESOLVED_ADAPTER",
    "baseline_model": "$BASELINE_MODEL",
    "model_id": "$MODEL_ID",
    "eval_datasets": "$EVAL_DATASETS",
    "traces_dir": "$TRACES_DIR",
    "limit_eval": $LIMIT_EVAL,
    "timestamp": "$(date -Iseconds)",
    "slurm_job_id": "${SLURM_JOB_ID:-local}",
    "eval_ok": $EVAL_OK,
    "eval_fail": $EVAL_FAIL
}
RUNEOF

echo "========================================"
echo "EVAL SUMMARY"
echo "========================================"
echo "Total evals:  $EVAL_COUNT"
echo "Succeeded:    $EVAL_OK"
echo "Failed:       $EVAL_FAIL"
echo "Output dir:   $VIS_DIR"
echo ""

echo "Output files:"
for f in "$VIS_EVAL_DIR"/*.json; do
    if [[ -f "$f" ]]; then
        echo "  $(basename "$f")"
    fi
done
echo ""

# =============================================================================
# Run visualization
# =============================================================================
if [[ "$EVAL_OK" -gt 0 ]]; then
    echo "Running visualization..."
    echo ""

    python scripts/visualize_sweep_results.py "$VIS_DIR" \
        --traces-dir "$TRACES_DIR" \
        --no-color \
        --show-samples 999 \
        --show-full \
        --csv "$VIS_DIR/eval_summary.csv" \
        2>&1 | tee "$VIS_DIR/visualization.txt"

    echo ""
    echo "Visualization:  $VIS_DIR/visualization.txt"
    echo "Summary CSV:    $VIS_DIR/eval_summary.csv"
fi

echo ""
echo "========================================"
echo "DONE"
echo "========================================"
echo "To re-visualize later:"
echo "  python scripts/visualize_sweep_results.py $VIS_DIR"
echo ""
echo "Finished at: $(date)"
