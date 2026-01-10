#!/bin/bash
# =============================================================================
# Circuit Breaker MVP Pipeline - Full Dependency Chain
# =============================================================================
# Submits all jobs with proper dependencies
# Run from /scratch/memoozd/harmful-agents-meta-dataset
# =============================================================================

set -euo pipefail

echo "========================================"
echo "Circuit Breaker MVP Pipeline"
echo "========================================"
echo "Submitting jobs with dependencies..."
echo ""

# 1. Generate Ds (Harmful Set)
echo "Submitting Ds generation..."
DS_JOB=$(sbatch --parsable slurm/Trillium/trillium_mvp_generate_ds_1h100.sbatch)
echo "  Job ID: $DS_JOB"

# 2. Generate Dr (Benign Twins) - depends on Ds
echo "Submitting Dr generation (depends on $DS_JOB)..."
DR_JOB=$(sbatch --parsable --dependency=afterok:$DS_JOB slurm/Trillium/trillium_mvp_generate_dr_1h100.sbatch)
echo "  Job ID: $DR_JOB"

# 3. Validate - depends on Dr
echo "Submitting validation (depends on $DR_JOB)..."
VALIDATE_JOB=$(sbatch --parsable --dependency=afterok:$DR_JOB slurm/Trillium/trillium_mvp_validate.sbatch)
echo "  Job ID: $VALIDATE_JOB"

# 4. Create Eval Set - depends on Dr (can run parallel with validate)
echo "Submitting eval set creation (depends on $DR_JOB)..."
EVAL_SET_JOB=$(sbatch --parsable --dependency=afterok:$DR_JOB slurm/Trillium/trillium_mvp_create_eval.sbatch)
echo "  Job ID: $EVAL_SET_JOB"

# 5. Train - depends on validate and eval set creation
echo "Submitting training (depends on $VALIDATE_JOB and $EVAL_SET_JOB)..."
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$VALIDATE_JOB:$EVAL_SET_JOB slurm/Trillium/trillium_mvp_train.sbatch)
echo "  Job ID: $TRAIN_JOB"

# 6. Sanity Check - depends on training
echo "Submitting sanity check (depends on $TRAIN_JOB)..."
SANITY_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB slurm/Trillium/trillium_mvp_sanity_check.sbatch)
echo "  Job ID: $SANITY_JOB"

# 7. Full Evaluation - depends on training (can run parallel with sanity)
echo "Submitting evaluation (depends on $TRAIN_JOB)..."
EVAL_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB slurm/Trillium/trillium_mvp_eval.sbatch)
echo "  Job ID: $EVAL_JOB"

echo ""
echo "========================================"
echo "Pipeline submitted!"
echo "========================================"
echo "Job dependency chain:"
echo "  1. Ds generation: $DS_JOB"
echo "  2. Dr generation: $DR_JOB (after $DS_JOB)"
echo "  3. Validation:    $VALIDATE_JOB (after $DR_JOB)"
echo "  4. Eval set:      $EVAL_SET_JOB (after $DR_JOB)"
echo "  5. Training:      $TRAIN_JOB (after $VALIDATE_JOB, $EVAL_SET_JOB)"
echo "  6. Sanity check:  $SANITY_JOB (after $TRAIN_JOB)"
echo "  7. Evaluation:    $EVAL_JOB (after $TRAIN_JOB)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all with: scancel $DS_JOB $DR_JOB $VALIDATE_JOB $EVAL_SET_JOB $TRAIN_JOB $SANITY_JOB $EVAL_JOB"
echo "========================================"
