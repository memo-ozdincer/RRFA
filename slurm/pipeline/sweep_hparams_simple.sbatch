#!/bin/bash
#SBATCH --job-name=sweep_hparams_simple
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=09:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

set -euo pipefail

# =============================================================================
# Environment Setup
# =============================================================================
# Ensure we are running from the repo root
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
    # Try absolute path
    SWEEP_SCRIPT="$REPO_DIR/slurm/pipeline/sweep_hparams.sbatch"
fi

if [[ ! -f "$SWEEP_SCRIPT" ]]; then
    echo "ERROR: Sweep script not found at: $SWEEP_SCRIPT"
    echo "Current directory: $(pwd)"
    echo "Contents of slurm/pipeline:"
    ls -l slurm/pipeline || echo "(dir not found)"
    exit 1
fi

export TOTAL_STEPS="${TOTAL_STEPS:-100}"
export ALPHAS="${ALPHAS:-7.5,10.0}"
export LAYER_CONFIGS="${LAYER_CONFIGS:-10,20}"
export POLICIES="${POLICIES:-assistant_only}"
export LIMIT_EVAL="${LIMIT_EVAL:-50}"
export LIMIT_GEN="${LIMIT_GEN:-5000}"
export NUM_EXAMPLES="${NUM_EXAMPLES:-10}"
export SHOW_EXAMPLES="${SHOW_EXAMPLES:-true}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export VIS_CHECKPOINT_STEPS="${VIS_CHECKPOINT_STEPS:-50,100}"

# Pooling presets: "name:pooling_mode:distance:margin_b:margin_h"
# Legacy grid around working config (margins 500/1500, alpha 10)
# 2 presets × 2 alphas = 4 runs, each ~12 min (train + checkpoint eval + benign eval) ≈ 48 min
export POOLING_PRESETS="${POOLING_PRESETS:-m500:legacy:dmix:500:1500 m750:legacy:dmix:750:2250}"

echo "Starting sweep_hparams_simple.sbatch..."
echo "  ALPHAS: ${ALPHAS}"
echo "  LAYER_CONFIGS: ${LAYER_CONFIGS}"
echo "  POLICIES: ${POLICIES}"
echo "  POOLING_PRESETS: ${POOLING_PRESETS}"
echo "  TOTAL_STEPS: ${TOTAL_STEPS}"
echo "  Datasets: Fujitsu, AgentDojo"
echo "  Logging: Compatible with scripts/visualize_sweep_results.py"

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
        # Fallback to name, will fail if offline
        MODEL="$MODEL_NAME"
    fi
else
    echo "WARNING: Model not cached at: $MODEL_CACHE_DIR"
    MODEL="$MODEL_NAME"
fi

# =============================================================================
# Auto-Generation of Missing Data
# =============================================================================
# Check if Fujitsu DS/DR traces exist. If not, generate them.
# This ensures the sweep can run even if data prep wasn't done manually.

TRACES_DIR="/scratch/memoozd/cb-scratch/data/traces"
FUJITSU_SKELETON="$TRACES_DIR/fujitsu_b4_skeletons.jsonl"
FUJITSU_DS="$TRACES_DIR/fujitsu_b4_ds.jsonl"
FUJITSU_DR="$TRACES_DIR/fujitsu_b4_dr.jsonl"
TOOL_SCHEMA="configs/tool_schemas/b4_standard_v1.json"
# MODEL is now set above via cache resolution

# Helper to generate DS/DR
generate_data() {
    local name="$1"
    local skeleton="$2"
    local ds="$3"
    local dr="$4"
    local schema="$5"
    local raw_source="${6:-}"  # Optional raw source for ETL_A

    # 1. Check for Skeleton (ETL_A)
    if [[ ! -f "$skeleton" ]]; then
        if [[ -n "$raw_source" && -f "$raw_source" ]]; then
            echo "Generating missing skeleton for $name (ETL_A)..."
            echo "WARNING: No ETL_A logic defined for $name in simple script."
        fi
    fi

    # 2. Check for DS/DR (Generation)
    if [[ ! -f "$ds" || ! -f "$dr" ]]; then
        if [[ -f "$skeleton" ]]; then
            echo "Generating missing $name data (DS/DR)..."
            
            # Generate DS (Harmful)
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
            
            # Generate DR (Benign)
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

# Run generation
generate_data "Fujitsu B4" "$FUJITSU_SKELETON" "$FUJITSU_DS" "$FUJITSU_DR" "$TOOL_SCHEMA"

# AgentDojo: regenerate traces via ETL_A if missing or suspiciously small
AGENTDOJO_TRACES="$TRACES_DIR/agentdojo_complete.jsonl"
AGENTDOJO_RAW="$REPO_DIR/data/agent_dojo/agentdojo_spotlight_extract.jsonl"
AGENTDOJO_MIN_LINES=100  # expect ~3000+; if fewer, regenerate

if [[ -f "$AGENTDOJO_RAW" ]]; then
    regen=false
    if [[ ! -f "$AGENTDOJO_TRACES" ]]; then
        regen=true
        echo "AgentDojo traces not found. Regenerating via ETL_A..."
    else
        n_lines=$(wc -l < "$AGENTDOJO_TRACES")
        if [[ "$n_lines" -lt "$AGENTDOJO_MIN_LINES" ]]; then
            regen=true
            echo "AgentDojo traces too small ($n_lines lines < $AGENTDOJO_MIN_LINES). Regenerating via ETL_A..."
        fi
    fi
    if [[ "$regen" == "true" ]]; then
        python src/schemas/tools/ETL_A.py \
            --agentdojo "$AGENTDOJO_RAW" \
            --output "$AGENTDOJO_TRACES" \
            --split train
        echo "AgentDojo ETL_A complete: $(wc -l < "$AGENTDOJO_TRACES") traces"
    else
        echo "AgentDojo traces OK: $(wc -l < "$AGENTDOJO_TRACES") lines"
    fi
else
    echo "WARNING: AgentDojo raw file not found at $AGENTDOJO_RAW"
fi

echo ""
echo "Data preparation complete. Launching sweep..."
echo ""

bash "$SWEEP_SCRIPT"

LATEST_SWEEP_DIR="$(ls -1dt /scratch/memoozd/cb-scratch/sweeps/hparam_sweep_* 2>/dev/null | head -n 1 || true)"
if [[ -n "$LATEST_SWEEP_DIR" && -d "$LATEST_SWEEP_DIR" ]]; then
    echo ""
    echo "Simple sweep complete."
    echo "Checkpoint-aware visualization command:"
    echo "  python scripts/visualize_sweep_results.py \"$LATEST_SWEEP_DIR\" \\"
    echo "    --compare-checkpoints --checkpoint-steps \"$VIS_CHECKPOINT_STEPS\" --no-color \\"
    echo "    --csv \"$LATEST_SWEEP_DIR/eval_summary.csv\""
fi
