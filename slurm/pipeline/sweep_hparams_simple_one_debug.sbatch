#!/bin/bash
#SBATCH --job-name=sweep_debug_one
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# DEBUG: Single-config stability sweep for inspecting traces and eval output
# =============================================================================
# Purpose: Run one known-good config with small limits to verify:
#   1. Traces look correct
#   2. Eval JSON contains unified_tool_metrics
#   3. Malicious/correct tool call rates are computed
#   4. Summary CSV has all columns populated
#
# Usage:
#   sbatch slurm/pipeline/sweep_hparams_simple_one_debug.sbatch
# =============================================================================

set -euo pipefail

# =============================================================================
# Environment Setup
# =============================================================================
PROJECT_DIR="/project/def-zhijing/memoozd"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

if [[ -d "$REPO_DIR" ]]; then
    cd "$REPO_DIR"
    echo "Changed directory to: $(pwd)"
else
    echo "WARNING: Repo directory not found at $REPO_DIR. Assuming current directory is correct."
fi

# Load modules and activate venv (Required for data generation)
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated venv: $VENV_DIR"
else
    echo "WARNING: Venv not found at $VENV_DIR"
fi

# Locate the sweep script
SWEEP_SCRIPT="slurm/pipeline/sweep_hparams.sbatch"

if [[ ! -f "$SWEEP_SCRIPT" ]]; then
    SWEEP_SCRIPT="$REPO_DIR/slurm/pipeline/sweep_hparams.sbatch"
fi

if [[ ! -f "$SWEEP_SCRIPT" ]]; then
    echo "ERROR: Sweep script not found at: $SWEEP_SCRIPT"
    echo "Current directory: $(pwd)"
    echo "Contents of slurm/pipeline:"
    ls -l slurm/pipeline || echo "(dir not found)"
    exit 1
fi

# =============================================================================
# DEBUG OVERRIDES: minimal config for inspection
# =============================================================================
export QUICK_TEST="true"
export TOTAL_STEPS="${TOTAL_STEPS:-50}"
export ALPHAS="${ALPHAS:-10.0}"
export LAYER_CONFIGS="${LAYER_CONFIGS:-10,20}"
export POLICIES="${POLICIES:-cb_full_sequence}"
export MWCS_SCHEDULE="${MWCS_SCHEDULE:-capability_focus}"
export MWCS_REGISTRY="${MWCS_REGISTRY:-$REPO_DIR/configs/mwcs_registry_v1.json}"
export MWCS_STEP="${MWCS_STEP:-0}"
export AGENTDOJO_EVAL_SPLIT="${AGENTDOJO_EVAL_SPLIT:-complete}"
export LIMIT_EVAL="${LIMIT_EVAL:-20}"
export LIMIT_GEN="${LIMIT_GEN:-500}"
export NUM_EXAMPLES="${NUM_EXAMPLES:-5}"
export SHOW_EXAMPLES="${SHOW_EXAMPLES:-true}"
export FULL_RENDER_AUDIT="${FULL_RENDER_AUDIT:-false}"
export EVAL_INTERMEDIATE_CHECKPOINTS="${EVAL_INTERMEDIATE_CHECKPOINTS:-false}"
export CHECKPOINT_EVAL_STEPS="${CHECKPOINT_EVAL_STEPS:-}"
export CHECKPOINT_EVAL_LIMIT="${CHECKPOINT_EVAL_LIMIT:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export VERIFY_PIPELINE="${VERIFY_PIPELINE:-true}"
export VERIFY_FAIL_ON_ERROR="${VERIFY_FAIL_ON_ERROR:-false}"
export TRIPLET_BETA_HARMFUL="${TRIPLET_BETA_HARMFUL:-0.25}"
export TRIPLET_GAMMA_KL="${TRIPLET_GAMMA_KL:-1.2}"
export TRIPLET_MARGIN_BENIGN="${TRIPLET_MARGIN_BENIGN:-50.0}"
export TRIPLET_MARGIN_HARMFUL="${TRIPLET_MARGIN_HARMFUL:-150.0}"
export SAVE_STEPS="${SAVE_STEPS:-10}"

echo "=========================================="
echo "DEBUG: Single-config stability sweep"
echo "=========================================="
echo "  ALPHAS: ${ALPHAS}"
echo "  LAYER_CONFIGS: ${LAYER_CONFIGS}"
echo "  POLICIES: ${POLICIES}"
echo "  LIMIT_EVAL: ${LIMIT_EVAL}"
echo "  LIMIT_GEN: ${LIMIT_GEN}"
echo "  TOTAL_STEPS: ${TOTAL_STEPS}"
echo "  FULL_RENDER_AUDIT: ${FULL_RENDER_AUDIT}"
echo "  EVAL_INTERMEDIATE_CHECKPOINTS: ${EVAL_INTERMEDIATE_CHECKPOINTS}"
echo "  Datasets: Fujitsu, AgentDojo"
echo ""

# =============================================================================
# Cache Setup (Offline Mode)
# =============================================================================
CB_SCRATCH="/scratch/memoozd/cb-scratch"
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,wandb,triton}

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

# Resolve Tokenizer to Local Cache (for generate_completions.py)
MODEL_NAME="models--meta-llama--Llama-3.1-8B-Instruct"
MODEL_CACHE_DIR="$HF_HUB_CACHE/models--meta-llama--Llama-3.1-8B-Instruct"
SNAPSHOT_ROOT="$MODEL_CACHE_DIR/snapshots"

if [[ -d "$SNAPSHOT_ROOT" ]]; then
    SNAPSHOT_PATH=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1)
    if [[ -n "$SNAPSHOT_PATH" ]]; then
        echo "Found local snapshot: $SNAPSHOT_PATH"
        MODEL="$SNAPSHOT_PATH"
    else
        echo "WARNING: Snapshot directory exists but is empty: $SNAPSHOT_ROOT"
        MODEL="$MODEL_NAME"
    fi
else
    echo "WARNING: Model not cached at: $MODEL_CACHE_DIR"
    MODEL="$MODEL_NAME"
fi

# =============================================================================
# Auto-Generation of Missing Data
# =============================================================================
TRACES_DIR="/scratch/memoozd/cb-scratch/data/traces"
FUJITSU_SKELETON="$TRACES_DIR/fujitsu_b4_skeletons.jsonl"
FUJITSU_DS="$TRACES_DIR/fujitsu_b4_ds.jsonl"
FUJITSU_DR="$TRACES_DIR/fujitsu_b4_dr.jsonl"
TOOL_SCHEMA="configs/tool_schemas/b4_standard_v1.json"

generate_data() {
    local name="$1"
    local skeleton="$2"
    local ds="$3"
    local dr="$4"
    local schema="$5"
    if [[ ! -f "$ds" || ! -f "$dr" ]]; then
        if [[ -f "$skeleton" ]]; then
            echo "Generating missing $name data (DS/DR)..."
            if [[ ! -f "$ds" ]]; then
                echo "  Generating DS (Harmful)..."
                python src/data_generation/generate_completions.py \
                    --traces "$skeleton" \
                    --output "$ds" \
                    --model "$MODEL" \
                    --tool-schema "$schema" \
                    --mode ds \
                    --temperature-ds 0.7 \
                    --strict-simulated-match true \
                    --use-vllm \
                    --limit "$LIMIT_GEN"
            fi
            if [[ ! -f "$dr" ]]; then
                echo "  Generating DR (Benign)..."
                python src/data_generation/generate_completions.py \
                    --ds-data "$ds" \
                    --output "$dr" \
                    --model "$MODEL" \
                    --tool-schema "$schema" \
                    --mode dr \
                    --temperature-dr 0.3 \
                    --use-vllm \
                    --limit "$LIMIT_GEN"
            fi
        else
            echo "WARNING: Skeleton file for $name not found at $skeleton. Cannot generate data."
        fi
    else
        echo "$name data already exists. Skipping generation."
    fi
}

generate_data "Fujitsu B4" "$FUJITSU_SKELETON" "$FUJITSU_DS" "$FUJITSU_DR" "$TOOL_SCHEMA"

echo ""
echo "Data preparation complete. Launching debug sweep..."
echo ""

exec bash "$SWEEP_SCRIPT"
