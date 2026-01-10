#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=prepare_cb_data
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err

# =============================================================================
# Circuit Breakers Data Preparation Pipeline (Trillium)
# =============================================================================
#
# This script runs the complete data preparation pipeline:
# 1. Ingest raw data from all sources (Fujitsu, AgentDojo, etc.)
# 2. Extract completions from raw data (CRITICAL for CB training!)
# 3. Create balanced training batches
#
# MUST BE RUN BEFORE TRAINING!
#
# Usage:
#   sbatch slurm/Trillium/trillium_prepare_data.sbatch
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
module load python/3.11.5

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR"
  exit 1
fi

source "$VENV_DIR/bin/activate"

echo "Python: $(python -V)"
echo "Which:  $(which python)"

# =============================================================================
# Environment
# =============================================================================

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "=========================================="
echo "CB Data Preparation Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=========================================="

# =============================================================================
# Run Full Data Preparation Pipeline
# =============================================================================

python scripts/prepare_cb_training.py \
  --fujitsu-success-only \
  --batch-size 16

# =============================================================================
# Verify Output
# =============================================================================

echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="

BATCHES_FILE="data/circuit_breakers/cb_training_batches.jsonl"

if [[ -f "$BATCHES_FILE" ]]; then
  N_BATCHES=$(wc -l < "$BATCHES_FILE")
  echo "Training batches created: $N_BATCHES"
  echo "File: $BATCHES_FILE"

  # Check for completions
  echo ""
  echo "Checking for harmful completions..."
  python -c "
import json
with open('$BATCHES_FILE') as f:
    batches = [json.loads(l) for l in f]

total_harmful = sum(len(b['harmful']) for b in batches)
with_completion = sum(
    1 for b in batches
    for s in b['harmful']
    if s.get('text') or s.get('harmful_completion')
)
pct = 100 * with_completion / total_harmful if total_harmful > 0 else 0
print(f'  Harmful samples with completions: {with_completion}/{total_harmful} ({pct:.1f}%)')
if pct < 50:
    print('  WARNING: Less than 50% have completions!')
else:
    print('  OK: Majority have completions')
"
else
  echo "ERROR: Training batches not created!"
  exit 1
fi

echo ""
echo "=========================================="
echo "Data preparation completed!"
echo "Date: $(date)"
echo ""
echo "Next: Run training with:"
echo "  sbatch slurm/Trillium/trillium_cb_llama31_8b_h100s_v2.sbatch"
echo "=========================================="
