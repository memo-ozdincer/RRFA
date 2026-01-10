#!/bin/bash
#SBATCH --job-name=mvp_sanity
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/project/def-zhijing/memoozd/harmful-agents-meta-dataset/logs/%x_%j.out
#SBATCH --error=/project/def-zhijing/memoozd/harmful-agents-meta-dataset/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Adapter Sanity Check (KL Divergence)
# Trillium 1×H100
# =============================================================================
#
# Verifies that the trained adapter actually changes model behavior:
# - Computes KL divergence between base and adapter outputs
# - Fails if mean KL < epsilon (adapter has no effect)
# - CRITICAL: Run this AFTER training, BEFORE evaluation
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_mvp_sanity_check.sbatch
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="$PROJECT_DIR/harmful-agents-meta-dataset"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"
mkdir -p "$SCRATCH_DIR/logs"

# Load modules
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

source "$VENV_DIR/bin/activate"

echo "Python: $(python -V)"

# =============================================================================
# Cache Setup
# =============================================================================
CACHE_ROOT="$SCRATCH_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,torch}

# CRITICAL FIX: Override HOME
export HOME="$CACHE_ROOT"

export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$CACHE_ROOT/hf/hub"  # Explicit hub cache location
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
# export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers" # Deprecated
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg_cache"
mkdir -p "$XDG_CACHE_HOME"

export HF_HUB_OFFLINE=1  # Fail fast if model not cached (no internet on compute nodes)
export TRANSFORMERS_OFFLINE=1  # Disable all network calls in transformers

# =============================================================================
# Job Info
# =============================================================================
echo "========================================"
echo "Stage 1 MVP: Adapter Sanity Check"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: 1 x H100 SXM 80GB"
echo "========================================"

# =============================================================================
# Configuration
# =============================================================================
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Find the most recent run directory (by modification time)
LATEST_RUN=$(ls -td $SCRATCH_DIR/cb_runs/*/ 2>/dev/null | head -1)
if [[ -z "$LATEST_RUN" ]]; then
    echo "ERROR: No run directory found in $SCRATCH_DIR/cb_runs/"
    exit 1
fi

ADAPTER_PATH="${LATEST_RUN}outputs/cb_mvp_adapter"

echo "Latest run: $LATEST_RUN"
echo "Adapter path: $ADAPTER_PATH"

# Check if adapter exists
if [[ ! -d "$ADAPTER_PATH" ]]; then
    echo "WARNING: Adapter path not found: $ADAPTER_PATH"
    echo "Running pre-training sanity check (base model only)..."
    ADAPTER_ARG=""
else
    ADAPTER_ARG="--adapter-path $ADAPTER_PATH"
fi

# =============================================================================
# Run Sanity Check
# =============================================================================
python scripts/circuit_breakers/sanity_check.py \
    --base-model "$BASE_MODEL" \
    $ADAPTER_ARG \
    --device auto \
    --dtype bfloat16 \
    --epsilon 1e-4 \
    --fail-on-error

echo "========================================"
echo "Sanity check complete!"
echo "========================================"
