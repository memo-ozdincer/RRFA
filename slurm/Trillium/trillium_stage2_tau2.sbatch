#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=stage2_tau2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err

# =============================================================================
# STAGE 2: INGEST TAU2 TRACES (CPU)
# =============================================================================
#
# Extracts TAU2 traces for retain set (Dr).
#
# Output (scratch):
#   /scratch/memoozd/cb_stage2_data/retain/tau2_traces.jsonl
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_stage2_tau2.sbatch
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

RUN_DIR="$SCRATCH_DIR/stage2_tau2/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

OUTPUT_DIR="$SCRATCH_DIR/cb_stage2_data/retain"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "INGESTING TAU2 TRACES"
echo ""

python scripts/cb_data_generation/ingest_tau2_traces.py \
    --tau2-path data/tau2_repo/ \
    --output "$OUTPUT_DIR/tau2_traces.jsonl" \
    --target-n 500

echo ""
echo "========================================"
echo "TAU2 ingestion complete!"
echo "========================================"
echo "Output: $OUTPUT_DIR/tau2_traces.jsonl"
echo "Count:  $(wc -l < "$OUTPUT_DIR/tau2_traces.jsonl" 2>/dev/null || echo 0)"
