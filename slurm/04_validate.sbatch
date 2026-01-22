#!/bin/bash
#SBATCH --job-name=mvp_validate
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1 MVP: Validate Data Format
# CPU-only (no GPU needed)
# =============================================================================
#
# DEPENDENCY: Run 01_generate_ds.sbatch and 02_generate_dr.sbatch first!
#
# Submit:
#   sbatch slurm/04_validate.sbatch
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"

module --force purge || true
module load StdEnv/2023
module load python/3.11.5

source "$VENV_DIR/bin/activate"

CACHE_DIR="$CB_SCRATCH/cache"
export HF_HOME="$CACHE_DIR/hf"

echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# Validate Files
DATA_DIR="$CB_SCRATCH/data"
DS_FILE="$DATA_DIR/ds_stage1.jsonl"
DR_FILE="$DATA_DIR/dr_stage1.jsonl"

VALIDATION_PASSED=true

# Validate Ds
if [[ -f "$DS_FILE" ]]; then
    echo "--- Validating Ds: $DS_FILE ---"
    if python src/data_generation/validate_format.py \
        --data "$DS_FILE" \
        --strict \
        --print-examples \
        --n-per-error-type 3; then
        echo "Ds validation PASSED"
    else
        echo "Ds validation FAILED"
        VALIDATION_PASSED=false
    fi
else
    echo "Ds file not found: $DS_FILE"
fi

# Validate Dr
if [[ -f "$DR_FILE" ]]; then
    echo "--- Validating Dr: $DR_FILE ---"
    if python src/data_generation/validate_format.py \
        --data "$DR_FILE" \
        --strict \
        --print-examples \
        --n-per-error-type 3; then
        echo "Dr validation PASSED"
    else
        echo "Dr validation FAILED"
        VALIDATION_PASSED=false
    fi
else
    echo "Dr file not found: $DR_FILE"
fi

# Summary
if $VALIDATION_PASSED; then
    echo "All validations PASSED"
    echo "Data ready for training: Ds=$DS_FILE, Dr=$DR_FILE"
    exit 0
else
    echo "Some validations FAILED"
    exit 1
fi
