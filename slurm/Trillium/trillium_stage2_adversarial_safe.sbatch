#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=stage2_advsafe
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err

# =============================================================================
# STAGE 2: GENERATE ADVERSARIAL-SAFE SAMPLES (Critical Dr expansion)
# =============================================================================
#
# Generates samples where the model RESISTS injection attacks:
# - Input: B4 adversarial prompts (with malicious injections)
# - Generate: Model responses at low temperature (0.1)
# - Filter: Keep only samples where observed_tool == expected_tool
# - Output: High-value Dr samples for capability retention
#
# Expected yield: ~95% (baseline ASR was only 5.2%)
# Target: 500 samples minimum
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_stage2_adversarial_safe.sbatch
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="/project/def-zhijing/memoozd/harmful-agents-meta-dataset"
VENV_DIR="/project/def-zhijing/memoozd/.venvs/cb_env"

cd "$REPO_DIR"

echo "=== Module setup ==="
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
module list

echo "=== Venv activation ==="
if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR"
  exit 1
fi
source "$VENV_DIR/bin/activate"
echo "Python: $(python -V)"

echo "=== Environment setup ==="
export OMP_NUM_THREADS=8

RUN_DIR="$SCRATCH_DIR/stage2_adversarial_safe/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

# =============================================================================
# Cache Setup (MUST match prefetch_models.sh)
# =============================================================================
CACHE_ROOT="$SCRATCH_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf/hub,hf/datasets,torch}

export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$CACHE_ROOT/hf/hub"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"

# Enable offline mode - fail fast if model not cached
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Redirect XDG directories to writable scratch space
export XDG_CACHE_HOME="$CACHE_ROOT/xdg_cache"
export XDG_CONFIG_HOME="$CACHE_ROOT/xdg_config"
export FLASHINFER_WORKSPACE_DIR="$CACHE_ROOT/flashinfer"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$FLASHINFER_WORKSPACE_DIR"

# Disable vLLM usage statistics (avoids writing to ~/.config/vllm)
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

# Force vLLM to use flash_attn backend instead of flashinfer
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# DO NOT SET TRANSFORMERS_CACHE - deprecated and causes cache fragmentation

export SCRATCH="$SCRATCH_DIR"

echo ""
echo "Cache Configuration:"
echo "  HF_HOME: $HF_HOME"
echo "  HF_HUB_CACHE: $HF_HUB_CACHE"
echo "  HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
echo ""

# Quick cache check
echo "Hub cache contents:"
ls -1d "$HF_HUB_CACHE"/models--* 2>/dev/null | head -5 || echo "  (no models cached yet - run prefetch_models.sh first!)"
echo ""

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
MODEL_CACHE_DIR="$HF_HUB_CACHE/models--${MODEL_ID//\//--}"
if [[ ! -d "$MODEL_CACHE_DIR" ]]; then
  echo "ERROR: Model cache missing: $MODEL_CACHE_DIR"
  echo "       Run: bash slurm/Trillium/prefetch_models.sh"
  exit 1
fi

MODEL_PATH="$(python - << 'PY'
from huggingface_hub import snapshot_download
model_id = "meta-llama/Llama-3.1-8B-Instruct"
local_path = snapshot_download(repo_id=model_id, local_files_only=True)
print(local_path)
PY
)"

if [[ -z "$MODEL_PATH" || ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: Failed to resolve local model path for $MODEL_ID"
  exit 1
fi

echo ""
echo "GENERATING ADVERSARIAL-SAFE SAMPLES"
echo ""

OUTPUT_DIR="$SCRATCH_DIR/cb_stage2_data/retain"
mkdir -p "$OUTPUT_DIR"

python scripts/cb_data_generation/generate_adversarial_safe.py \
    --b4-data data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl \
    --tool-schema configs/tool_schemas/b4_standard_v1.json \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_DIR/adversarial_safe.jsonl" \
    --target-n 500 \
    --temperature 0.1 \
    --batch-size 32
