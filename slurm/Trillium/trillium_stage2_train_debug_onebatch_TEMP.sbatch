#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=cb_stage2_debug_onebatch
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8

# ============================================================================
# TEMP DEBUG: Verify non-zero gradients + masking on ONE tiny Stage2 batch.
#
# What it does:
# 1) Creates tiny batches (max 5 harmful, Dr:Ds=2:1, require tool calls)
# 2) Runs training for 3 steps
# 3) Trainer prints masking validation at start + grad norms at step 1
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_stage2_train_debug_onebatch_TEMP.sbatch
# ============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="$PROJECT_DIR/harmful-agents-meta-dataset"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"

echo "=== Module setup ==="
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
module list

echo "=== Venv activation ==="
source "$VENV_DIR/bin/activate"
echo "Python: $(python -V)"

export WANDB_MODE=disabled

# Offline mode (use local cache)
export HF_HOME="$SCRATCH_DIR/cb_cache/hf"
export HF_HUB_CACHE="$SCRATCH_DIR/cb_cache/hf/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

STAGE2_DATA_DIR="$SCRATCH_DIR/cb_stage2_data"
MERGED_FILE="$STAGE2_DATA_DIR/stage2/train.jsonl"

if [[ ! -f "$MERGED_FILE" ]]; then
  echo "ERROR: Missing merged stage2 file: $MERGED_FILE"
  echo "Run: sbatch slurm/Trillium/trillium_stage2_merge.sbatch"
  exit 1
fi

RUN_DIR="$SCRATCH_DIR/cb_runs/$SLURM_JOB_ID"
OUTPUT_DIR="$RUN_DIR/outputs/cb_stage2_debug_adapter"
mkdir -p "$RUN_DIR"/outputs

TINY_BATCHES="$STAGE2_DATA_DIR/stage2/train_batches_debug_tiny.jsonl"

echo ""
echo "Creating tiny Stage2 batches -> $TINY_BATCHES"
python scripts/cb_data_generation/create_stage2_batches.py \
  --input "$MERGED_FILE" \
  --output "$TINY_BATCHES" \
  --benign-per-harmful 2 \
  --max-harmful 5 \
  --require-tool-calls-harmful

echo "Tiny batches lines: $(wc -l < "$TINY_BATCHES")"
echo "Head of tiny batches:"
head -n 1 "$TINY_BATCHES" | head -c 600; echo

echo ""
echo "Running 3 training steps (should show NON-ZERO grad_norm_total at step 1)"

accelerate launch --num_processes ${SLURM_GPUS_ON_NODE:-1} \
  scripts/train_circuit_breaker.py \
  --preset llama-3.1-8b-stage2 \
  --base-model "meta-llama/Llama-3.1-8B-Instruct" \
  --data-path "$TINY_BATCHES" \
  --output-dir "$OUTPUT_DIR" \
  --loss-weighting dual \
  --alpha-max 0.5 \
  --alpha-decay-multiplier 2.0 \
  --total-steps 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 3e-5 \
  --lora-r 16 \
  --lora-alpha 32 \
  --cb-target-layers 15 \
  --no-wandb

echo ""
echo "Done. Output dir: $OUTPUT_DIR"
