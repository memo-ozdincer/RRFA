#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=stage2_merge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err

# =============================================================================
# STAGE 2: MERGE DATASETS (CPU)
# =============================================================================
#
# Merges harmful + retain sources into a single Stage 2 train/eval split.
#
# Output (scratch):
#   /scratch/memoozd/cb_stage2_data/stage2/train.jsonl
#   /scratch/memoozd/cb_stage2_data/stage2/eval.jsonl
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_stage2_merge.sbatch
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

RUN_DIR="$SCRATCH_DIR/stage2_merge/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

STAGE2_DIR="$SCRATCH_DIR/cb_stage2_data/stage2"
mkdir -p "$STAGE2_DIR"

echo ""
echo "MERGING STAGE 2 DATA"
echo ""

python scripts/cb_data_generation/merge_stage2_data.py \
    --harmful-dir "$SCRATCH_DIR/cb_stage2_data/harmful/" \
    --retain-dir "$SCRATCH_DIR/cb_stage2_data/retain/" \
    --output "$STAGE2_DIR/train.jsonl" \
    --eval-output "$STAGE2_DIR/eval.jsonl" \
    --eval-fraction 0.1 \
    --validate

echo ""
echo "========================================"
echo "Stage 2 merge complete!"
echo "========================================"
echo "Train: $STAGE2_DIR/train.jsonl"
echo "Eval:  $STAGE2_DIR/eval.jsonl"
echo "Counts:"
echo "  train: $(wc -l < "$STAGE2_DIR/train.jsonl" 2>/dev/null || echo 0)"
echo "  eval:  $(wc -l < "$STAGE2_DIR/eval.jsonl" 2>/dev/null || echo 0)"
