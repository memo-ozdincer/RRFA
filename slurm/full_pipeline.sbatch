#!/bin/bash
#SBATCH --job-name=cb_full_1gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Full Pipeline - 1 GPU Variant
# Runs each stage script sequentially within this allocation.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "FULL PIPELINE (single allocation) - Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "========================================"

# NOTE: Running via bash ignores the #SBATCH headers in the stage scripts, so
# everything uses the resources allocated to this job.
bash "$SCRIPT_DIR/01_generate_ds.sbatch"
bash "$SCRIPT_DIR/02_generate_dr.sbatch"
bash "$SCRIPT_DIR/03_create_eval.sbatch"
bash "$SCRIPT_DIR/04_rebuild_data.sbatch"
bash "$SCRIPT_DIR/05_train.sbatch"
bash "$SCRIPT_DIR/06_eval.sbatch"

echo "========================================"
echo "✓✓✓ FULL PIPELINE COMPLETE ✓✓✓"
echo "========================================"
