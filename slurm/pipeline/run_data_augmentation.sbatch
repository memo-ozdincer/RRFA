#!/bin/bash
#SBATCH --job-name=data_augment
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

set -euo pipefail

# =============================================================================
# DATA AUGMENTATION PIPELINE (no GPU needed)
# =============================================================================
# 1. Run augment_agentdojo.py (multi-model ingest + removal + transplant + taxonomy)
# 2. Run ETL_B to create renders + lossmasks
# 3. Run split to create CB/retain sets
# 4. Print summary
#
# Usage:
#   sbatch slurm/pipeline/run_data_augmentation.sbatch
#
# Output files (SCP these back):
#   $CB_SCRATCH/data/traces/agentdojo_augmented.jsonl
#   $CB_SCRATCH/data/renders/agentdojo_augmented.jsonl
#   $CB_SCRATCH/data/lossmasks/agentdojo_augmented.jsonl
# =============================================================================

PROJECT_DIR="/project/def-zhijing/memoozd"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

if [[ -d "$REPO_DIR" ]]; then
    cd "$REPO_DIR"
    echo "Working directory: $(pwd)"
else
    echo "ERROR: Repo not found at $REPO_DIR"
    exit 1
fi

module --force purge || true
module load StdEnv/2023
module load python/3.11.5

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated venv: $VENV_DIR"
else
    echo "WARNING: Venv not found at $VENV_DIR"
fi

# Cache setup
CB_SCRATCH="/scratch/memoozd/cb-scratch"
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch}
export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Output dirs
TRACES_OUT="$CB_SCRATCH/data/traces"
RENDERS_OUT="$CB_SCRATCH/data/renders"
LOSSMASKS_OUT="$CB_SCRATCH/data/lossmasks"
SPLIT_OUT="$CB_SCRATCH/data/split"
mkdir -p "$TRACES_OUT" "$RENDERS_OUT" "$LOSSMASKS_OUT" "$SPLIT_OUT"

# Configurable
OPERATIONS="${OPERATIONS:-ingest,backfill,removal,transplant,taxonomy}"
SEED="${SEED:-42}"
TOKENIZER="${TOKENIZER:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
PREFIX="agentdojo_augmented"

echo "============================================================"
echo "STEP 1: Data Augmentation"
echo "============================================================"

python scripts/augment_agentdojo.py \
    --raw-input data/agent_dojo/agentdojo_spotlight_extract.jsonl \
    --output "$TRACES_OUT/${PREFIX}.jsonl" \
    --operations "$OPERATIONS" \
    --seed "$SEED" \
    --stats

echo ""
echo "============================================================"
echo "STEP 2: ETL_B (Render + Lossmask)"
echo "============================================================"

python -m src.schemas.tools.ETL_B \
    --traces "$TRACES_OUT/${PREFIX}.jsonl" \
    --render-out "$RENDERS_OUT/${PREFIX}.jsonl" \
    --lossmask-out "$LOSSMASKS_OUT/${PREFIX}.jsonl" \
    --tokenizer "$TOKENIZER" \
    --max-length "$MAX_SEQ_LENGTH" \
    --allow-skeleton

RENDER_COUNT=$(wc -l < "$RENDERS_OUT/${PREFIX}.jsonl")
LOSSMASK_COUNT=$(wc -l < "$LOSSMASKS_OUT/${PREFIX}.jsonl")
echo "  Renders: $RENDER_COUNT"
echo "  Lossmasks: $LOSSMASK_COUNT"

echo ""
echo "============================================================"
echo "STEP 3: Split (CB / Retain)"
echo "============================================================"

python scripts/split_agentdojo.py \
    --traces "$TRACES_OUT/${PREFIX}.jsonl" \
    --renders "$RENDERS_OUT/${PREFIX}.jsonl" \
    --lossmasks "$LOSSMASKS_OUT/${PREFIX}.jsonl" \
    --output-dir "$SPLIT_OUT" \
    --prefix "$PREFIX"

echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"
echo ""
echo "Output files:"
echo "  Traces:    $TRACES_OUT/${PREFIX}.jsonl"
echo "  Renders:   $RENDERS_OUT/${PREFIX}.jsonl"
echo "  Lossmasks: $LOSSMASKS_OUT/${PREFIX}.jsonl"
echo "  Split CB:  $SPLIT_OUT/${PREFIX}_traces_cb.jsonl"
echo "  Split Ret: $SPLIT_OUT/${PREFIX}_traces_retain.jsonl"
echo ""
echo "SCP to local machine:"
echo "  scp cluster:$TRACES_OUT/${PREFIX}.jsonl data/traces/"
echo "  scp cluster:$RENDERS_OUT/${PREFIX}.jsonl data/renders/"
echo "  scp cluster:$LOSSMASKS_OUT/${PREFIX}.jsonl data/lossmasks/"
