#!/bin/bash
#SBATCH --job-name=regen_evals_llmail
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# REGENERATE EVALS ONLY + LLMAIL VISUALIZATION
# =============================================================================
#
# This script:
# 1) Re-runs eval for existing sweep run directories (no ETL_B, no training)
# 2) Rewrites *.paired_outputs.jsonl using the current eval.py logic
# 3) Runs LLMail-focused sweep visualization
#
# Usage examples:
#   sbatch slurm/pipeline/regenerate_evals_and_visualize_llmail.sbatch
#
#   SWEEP_DIR=/scratch/memoozd/cb-scratch/sweeps/hparam_sweep_YYYYMMDD_HHMMSS \
#   sbatch slurm/pipeline/regenerate_evals_and_visualize_llmail.sbatch
#
#   # Re-eval all datasets (default is llmail only)
#   EVAL_DATASETS=llmail,fujitsu,agentdojo \
#   sbatch slurm/pipeline/regenerate_evals_and_visualize_llmail.sbatch
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"
CB_SCRATCH="/scratch/memoozd/cb-scratch"

if [[ -d "$REPO_DIR" ]]; then
    cd "$REPO_DIR"
else
    echo "ERROR: Repo not found at $REPO_DIR"
    exit 1
fi

module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
else
    echo "WARNING: Venv not found at $VENV_DIR"
fi

# -----------------------------------------------------------------------------
# Runtime config
# -----------------------------------------------------------------------------
EVAL_DATASETS="${EVAL_DATASETS:-llmail}"     # comma-separated: llmail,fujitsu,agentdojo
LIMIT_EVAL="${LIMIT_EVAL:-100}"
DTYPE="${DTYPE:-bfloat16}"
RUN_NAME="${RUN_NAME:-}"                     # optional: single run name

VIS_SHOW_SAMPLES="${VIS_SHOW_SAMPLES:-8}"
VIS_TOP_N="${VIS_TOP_N:-3}"
VIS_COMPARE_SAMPLES="${VIS_COMPARE_SAMPLES:-5}"
VIS_FILTER="${VIS_FILTER:-success}"          # success|failure|none
VIS_NO_COLOR="${VIS_NO_COLOR:-true}"

TRACES_DIR="${TRACES_DIR:-$CB_SCRATCH/data/traces}"
FUJITSU_DS_TRACES="$TRACES_DIR/fujitsu_b4_ds.jsonl"
LLMAIL_DS_TRACES="$TRACES_DIR/llmail_inject_ds.jsonl"
LLMAIL_TOOL_SCHEMA="${LLMAIL_TOOL_SCHEMA:-$REPO_DIR/configs/tool_schemas/llmail_inject_v1.json}"

# Select sweep dir (explicit or latest)
if [[ -z "${SWEEP_DIR:-}" ]]; then
    SWEEP_DIR="$(ls -1td "$CB_SCRATCH"/sweeps/hparam_sweep_* 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$SWEEP_DIR" || ! -d "$SWEEP_DIR" ]]; then
    echo "ERROR: Sweep directory not found. Set SWEEP_DIR explicitly."
    exit 1
fi

echo "========================================"
echo "REGENERATE EVALS + LLMAIL VISUALIZATION"
echo "========================================"
echo "Sweep Dir:      $SWEEP_DIR"
echo "Eval Datasets:  $EVAL_DATASETS"
echo "Limit Eval:     $LIMIT_EVAL"
echo "Run Name:       ${RUN_NAME:-<all runs>}"
echo "Traces Dir:     $TRACES_DIR"
echo ""

# -----------------------------------------------------------------------------
# Cache/offline setup (same pattern as sweep scripts)
# -----------------------------------------------------------------------------
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,wandb,triton}

export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export WANDB_DIR="$CACHE_DIR/wandb"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
TOKENIZER="$MODEL_ID"

echo "Resolving model '$MODEL_ID' to local cache path..."
MODEL_CACHE_NAME=$(echo "$MODEL_ID" | sed 's/\//--/g')
MODEL_CACHE_DIR="$HF_HUB_CACHE/models--${MODEL_CACHE_NAME}"
SNAPSHOT_ROOT="$MODEL_CACHE_DIR/snapshots"

if [[ -d "$SNAPSHOT_ROOT" ]]; then
    SNAPSHOT_PATH=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1)
    if [[ -n "$SNAPSHOT_PATH" ]]; then
        TOKENIZER="$SNAPSHOT_PATH"
        echo "  Using cached snapshot: $TOKENIZER"
    else
        echo "ERROR: Snapshot directory exists but is empty: $SNAPSHOT_ROOT"
        exit 1
    fi
else
    echo "ERROR: Model not cached at: $MODEL_CACHE_DIR"
    echo "Run slurm/cache_models.sh first."
    exit 1
fi

echo ""

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
should_eval() {
    local dataset="$1"
    [[ ",${EVAL_DATASETS}," == *",${dataset},"* ]]
}

run_eval() {
    local run_dir="$1"
    local adapter_path="$2"
    local dataset="$3"

    case "$dataset" in
        fujitsu)
            if [[ ! -f "$FUJITSU_DS_TRACES" ]]; then
                echo "  [fujitsu] missing traces: $FUJITSU_DS_TRACES"
                return 0
            fi
            local out="$run_dir/eval/fujitsu_eval.json"
            python src/evaluation/eval.py \
                --baseline "$TOKENIZER" \
                --cb-adapter "$adapter_path" \
                --eval-data "$FUJITSU_DS_TRACES" \
                --tool-schema "$REPO_DIR/configs/tool_schemas/b4_standard_v1.json" \
                --output "$out" \
                --limit "$LIMIT_EVAL" \
                --merge-adapter \
                --dtype "$DTYPE" 2>&1 | tee -a "$run_dir/eval/fujitsu_regen.log"
            ;;

        llmail)
            if [[ ! -f "$LLMAIL_DS_TRACES" ]]; then
                echo "  [llmail] missing traces: $LLMAIL_DS_TRACES"
                return 0
            fi
            local out="$run_dir/eval/llmail_eval.json"
            python src/evaluation/eval.py \
                --baseline "$TOKENIZER" \
                --cb-adapter "$adapter_path" \
                --eval-data "$LLMAIL_DS_TRACES" \
                --tool-schema "$LLMAIL_TOOL_SCHEMA" \
                --output "$out" \
                --limit "$LIMIT_EVAL" \
                --merge-adapter \
                --dtype "$DTYPE" 2>&1 | tee -a "$run_dir/eval/llmail_regen.log"
            ;;

        agentdojo)
            local agentdojo_eval_data="$run_dir/agentdojo_split/agentdojo_traces_harmful.jsonl"
            if [[ ! -f "$agentdojo_eval_data" ]]; then
                echo "  [agentdojo] missing eval data: $agentdojo_eval_data"
                return 0
            fi
            local out="$run_dir/eval/agentdojo_eval.json"
            python src/evaluation/eval.py \
                --baseline "$TOKENIZER" \
                --cb-adapter "$adapter_path" \
                --eval-data "$agentdojo_eval_data" \
                --tool-schema "$REPO_DIR/configs/tool_schemas/b4_standard_v1.json" \
                --output "$out" \
                --use-sample-context \
                --limit "$LIMIT_EVAL" \
                --merge-adapter \
                --dtype "$DTYPE" 2>&1 | tee -a "$run_dir/eval/agentdojo_regen.log"
            ;;
    esac
}

# -----------------------------------------------------------------------------
# Discover runs
# -----------------------------------------------------------------------------
mapfile -t RUN_DIRS < <(find "$SWEEP_DIR" -maxdepth 1 -mindepth 1 -type d -name 'a*' | sort)
if [[ -n "$RUN_NAME" ]]; then
    mapfile -t RUN_DIRS < <(printf "%s\n" "${RUN_DIRS[@]}" | grep "/${RUN_NAME}$" || true)
fi

if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
    echo "ERROR: No run directories found under $SWEEP_DIR"
    exit 1
fi

echo "Found ${#RUN_DIRS[@]} run(s) to re-evaluate."
echo ""

ok_runs=0
skip_runs=0
fail_runs=0
run_idx=0

for run_dir in "${RUN_DIRS[@]}"; do
    run_idx=$((run_idx + 1))
    run_name="$(basename "$run_dir")"

    echo "----------------------------------------"
    echo "[$run_idx/${#RUN_DIRS[@]}] $run_name"
    echo "----------------------------------------"

    if [[ -d "$run_dir/model/final" ]]; then
        ADAPTER_PATH="$run_dir/model/final"
    elif [[ -f "$run_dir/model/adapter_config.json" ]]; then
        ADAPTER_PATH="$run_dir/model"
    else
        echo "  Skipping: no adapter found in $run_dir/model"
        skip_runs=$((skip_runs + 1))
        echo ""
        continue
    fi

    mkdir -p "$run_dir/eval"

    if should_eval "fujitsu"; then
        echo "  Re-evaluating Fujitsu..."
        if ! run_eval "$run_dir" "$ADAPTER_PATH" "fujitsu"; then
            echo "  Fujitsu eval failed."
            fail_runs=$((fail_runs + 1))
            echo ""
            continue
        fi
    fi

    if should_eval "llmail"; then
        echo "  Re-evaluating LLMail..."
        if ! run_eval "$run_dir" "$ADAPTER_PATH" "llmail"; then
            echo "  LLMail eval failed."
            fail_runs=$((fail_runs + 1))
            echo ""
            continue
        fi
    fi

    if should_eval "agentdojo"; then
        echo "  Re-evaluating AgentDojo..."
        if ! run_eval "$run_dir" "$ADAPTER_PATH" "agentdojo"; then
            echo "  AgentDojo eval failed."
            fail_runs=$((fail_runs + 1))
            echo ""
            continue
        fi
    fi

    ok_runs=$((ok_runs + 1))
    echo ""
done

echo "========================================"
echo "EVAL REGEN SUMMARY"
echo "========================================"
echo "Succeeded: $ok_runs"
echo "Skipped:   $skip_runs"
echo "Failed:    $fail_runs"
echo ""

# -----------------------------------------------------------------------------
# LLMail-focused visualization
# -----------------------------------------------------------------------------
echo "Running LLMail-focused visualization..."

VIS_ARGS=(
    "$SWEEP_DIR"
    --traces-dir "$TRACES_DIR"
    --llmail-data-dir "$REPO_DIR/data/llmail_inject"
    --show-samples "$VIS_SHOW_SAMPLES"
    --samples-best-only
    --top-n "$VIS_TOP_N"
    --compare-samples "$VIS_COMPARE_SAMPLES"
    --compare-dataset llmail
    --csv "$SWEEP_DIR/llmail_eval_regen_summary.csv"
)

if [[ "$VIS_FILTER" == "success" ]]; then
    VIS_ARGS+=(--filter-success)
elif [[ "$VIS_FILTER" == "failure" ]]; then
    VIS_ARGS+=(--filter-failure)
fi

if [[ "$VIS_NO_COLOR" == "true" ]]; then
    VIS_ARGS+=(--no-color)
fi

python scripts/visualize_sweep_results.py "${VIS_ARGS[@]}" \
    | tee "$SWEEP_DIR/llmail_visualization_after_regen.txt"

echo ""
echo "Done."
echo "Visualization log: $SWEEP_DIR/llmail_visualization_after_regen.txt"
echo "Summary CSV:       $SWEEP_DIR/llmail_eval_regen_summary.csv"
