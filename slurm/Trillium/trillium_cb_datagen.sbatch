#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=cb_datagen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err

# =============================================================================
# CB Data Generation - Trillium 4×H100 (Tensor Parallel)
# =============================================================================
#
# Generates Circuit Breaker training data using abliterated model
# Uses vLLM with tensor_parallel=4 for high-throughput batched inference
#
# Hardware: 4x NVIDIA H100 SXM (80GB VRAM each = 320GB total)
#           96 CPU cores, 755 GiB RAM (full node)
# Estimated time: ~1-2 hours for full dataset with TP=4
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   mkdir -p logs
#   sbatch slurm/Trillium/trillium_cb_datagen.sbatch
#
# =============================================================================

set -euo pipefail

mkdir -p logs

PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="/project/def-zhijing/memoozd/harmful-agents-meta-dataset"
VENV_DIR="/project/def-zhijing/memoozd/.venvs/cb_env"

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

echo "Python: $(python -V)"
echo "Which:  $(which python)"

# =============================================================================
# Preflight
# =============================================================================
python - << 'PY'
import sys

def check(mod: str) -> None:
    try:
        __import__(mod)
    except Exception as e:
        print(f"ERROR: failed to import '{mod}': {e}")
        sys.exit(1)

for m in ("torch", "vllm"):
    check(m)

import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
PY

# =============================================================================
# Cache Setup (SCRATCH - Trillium requirement)
# =============================================================================
CACHE_ROOT="$SCRATCH_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,torch}
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"

# =============================================================================
# Job Info
# =============================================================================
echo "========================================"
echo "CB Data Generation - Trillium"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPU: 4 x H100 SXM 80GB (tensor_parallel=4)"
echo "Model: mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
echo "========================================"

# =============================================================================
# Run Data Generation Pipeline
# =============================================================================

# IMPORTANT: Use abliterated model so it actually generates harmful completions
# Standard Llama-3.1-Instruct will refuse!

# Using tensor_parallel=4 to shard model across 4 GPUs for faster inference
# --resume: continue from where we left off
# --batch-size 64: reasonable batch size to avoid memory issues
python scripts/cb_data_generation/run_pipeline.py \
    --all \
    --model mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated \
    --backend vllm \
    --tensor-parallel 4 \
    --temperature 0.7 \
    --num-samples 3 \
    --batch-size 64 \
    --resume \
    --output-dir data/circuit_breakers/

echo "========================================"
echo "Data generation complete!"
echo "========================================"
echo "Outputs:"
echo "  Ds: data/circuit_breakers/ds/circuit_breaker_set.jsonl"
echo "  Dr: data/circuit_breakers/dr/retain_set.jsonl"
echo "  Eval: data/circuit_breakers/eval/eval_set.jsonl"
echo "  Report: data/circuit_breakers/quality_report.json"
echo "========================================"
