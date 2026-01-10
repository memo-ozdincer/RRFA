#!/bin/bash
#SBATCH --job-name=debug_toolcall
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Debug Tool Calling Format
# =============================================================================
# Inspects what the model actually generates to diagnose parsing failures
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="$PROJECT_DIR/harmful-agents-meta-dataset"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"

module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

source "$VENV_DIR/bin/activate"

# Cache setup
CACHE_ROOT="$SCRATCH_DIR/cb_cache"
export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$CACHE_ROOT/hf/hub"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$CACHE_ROOT/xdg_cache"
export XDG_CONFIG_HOME="$CACHE_ROOT/xdg_config"
export FLASHINFER_WORKSPACE_DIR="$CACHE_ROOT/flashinfer"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

echo "========================================"
echo "Debug Tool Calling"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "========================================"

# Test with both models
echo ""
echo "========================================" 
echo "Testing ABLITERATED model"
echo "========================================"
python scripts/cb_data_generation/debug_tool_calling.py \
    --model mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated \
    --backend vllm \
    --limit 3

echo ""
echo "========================================"
echo "Testing STANDARD model"  
echo "========================================"
python scripts/cb_data_generation/debug_tool_calling.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --backend vllm \
    --limit 3

echo ""
echo "========================================"
echo "Debug complete"
echo "========================================"
