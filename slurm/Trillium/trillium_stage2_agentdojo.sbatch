#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=stage2_agentdojo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err

# =============================================================================
# STAGE 2: INGEST AGENTDOJO SPLITS (CPU)
# =============================================================================
#
# Splits AgentDojo traces by security outcome:
# - security == False -> Ds (harmful)
# - security == True  -> Dr (retain)
#
# Output (scratch):
#   /scratch/memoozd/cb_stage2_data/harmful/agentdojo_failures.jsonl
#   /scratch/memoozd/cb_stage2_data/retain/agentdojo_resisted.jsonl
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_stage2_agentdojo.sbatch
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

RUN_DIR="$SCRATCH_DIR/stage2_agentdojo/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

OUTPUT_HARMFUL="$SCRATCH_DIR/cb_stage2_data/harmful"
OUTPUT_RETAIN="$SCRATCH_DIR/cb_stage2_data/retain"
mkdir -p "$OUTPUT_HARMFUL" "$OUTPUT_RETAIN"

echo ""
echo "INGESTING AGENTDOJO SPLITS"
echo ""

python scripts/cb_data_generation/ingest_agentdojo_splits.py \
    --input-dir data/agent_dojo/ \
    --output-harmful "$OUTPUT_HARMFUL/agentdojo_failures.jsonl" \
    --output-retain "$OUTPUT_RETAIN/agentdojo_resisted.jsonl" \
    --max-each 500

echo ""
echo "========================================"
echo "AgentDojo ingestion complete!"
echo "========================================"
echo "Harmful: $OUTPUT_HARMFUL/agentdojo_failures.jsonl"
echo "Retain:  $OUTPUT_RETAIN/agentdojo_resisted.jsonl"
echo "Counts:"
echo "  harmful: $(wc -l < "$OUTPUT_HARMFUL/agentdojo_failures.jsonl" 2>/dev/null || echo 0)"
echo "  retain:  $(wc -l < "$OUTPUT_RETAIN/agentdojo_resisted.jsonl" 2>/dev/null || echo 0)"
