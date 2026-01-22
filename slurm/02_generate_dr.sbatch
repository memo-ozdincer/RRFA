#!/bin/bash
#SBATCH --job-name=mvp_dr_gen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Generate Dr (Retain Set) as Paired Benign Twins
# Trillium 4×H100 (tensor parallel)
# =============================================================================
#
# DEPENDENCY: Run 01_generate_ds.sbatch first!
#
# Submit:
#   sbatch slurm/02_generate_dr.sbatch
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

# Cache Setup
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch}
export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# Check Dependencies
DATA_DIR="$CB_SCRATCH/data"
DS_FILE="$DATA_DIR/ds_stage1.jsonl"

if [[ ! -f "$DS_FILE" ]]; then
    echo "ERROR: Ds file not found at $DS_FILE"
    echo "Please run 01_generate_ds.sbatch first!"
    exit 1
fi

echo "Found Ds file: $DS_FILE ($(wc -l < "$DS_FILE") samples)"

# Run Dr Generation
python src/data_generation/generate_dr.py \
    --ds-data "$DS_FILE" \
    --tool-schema configs/tool_schemas/b4_standard_v1.json \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --backend vllm \
    --tensor-parallel 1 \
    --batch-size 32 \
    --dtype bfloat16 \
    --temperature 0.3 \
    --output "$DATA_DIR/dr_stage1.jsonl" \
    --print-examples \
    --n-correct 10 \
    --n-wrong-tool 5 \
    --n-no-tool 5 \
    --n-format-errors 5

echo "Dr generation complete!"
echo "  Output: $DATA_DIR/dr_stage1.jsonl"
echo "  Stats: $DATA_DIR/dr_stage1.stats.json"
echo "  Examples: $DATA_DIR/dr_stage1.examples.json"
