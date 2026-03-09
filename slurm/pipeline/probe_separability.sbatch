#!/bin/bash
#SBATCH --job-name=probe_sep
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Linear Probe Diagnostic
# =============================================================================
# Answers: can the base model distinguish AD harmful vs benign at layers 5-30?
# If not, CB training at those layers cannot work.
#
# Usage:
#   sbatch slurm/pipeline/probe_separability.sbatch
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"

module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

source "$VENV_DIR/bin/activate"

# Cache
CACHE_DIR="$CB_SCRATCH/cache"
export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export TORCH_HOME="$CACHE_DIR/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Resolve model
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_CACHE_NAME=$(echo "$MODEL_ID" | sed 's/\//--/g')
SNAPSHOT_ROOT="$CACHE_DIR/hf/hub/models--${MODEL_CACHE_NAME}/snapshots"
MODEL_PATH=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1)
if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: Model not cached."; exit 1
fi

# Use truncated traces (matches training format)
TRACES="$CB_SCRATCH/data/traces/agentdojo_truncated.jsonl"
if [[ ! -f "$TRACES" ]]; then
    # Fall back to augmented
    TRACES="$CB_SCRATCH/data/traces/agentdojo_augmented.jsonl"
fi

echo "========================================================"
echo "Linear Probe Diagnostic"
echo "========================================================"
echo "Model: $MODEL_PATH"
echo "Traces: $TRACES"
echo "Date: $(date)"
echo ""

python scripts/probe_separability.py \
    --traces "$TRACES" \
    --model "$MODEL_PATH" \
    --layers 5 10 15 20 25 30 \
    --max-samples 200 \
    --max-length 2048

echo ""
echo "Done: $(date)"
