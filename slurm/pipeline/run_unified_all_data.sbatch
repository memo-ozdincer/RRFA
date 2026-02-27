#!/bin/bash
#SBATCH --job-name=unified_all_data
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

set -euo pipefail

# Use the same hardcoded project path as unified_pipeline.sbatch.
# NOTE: BASH_SOURCE resolves to SLURM's spool dir, not the repo.
PROJECT_DIR="/project/def-zhijing/memoozd"
REPO_DIR="$PROJECT_DIR/rrfa"
UNIFIED_PIPELINE="$REPO_DIR/slurm/pipeline/unified_pipeline.sbatch"

if [[ ! -f "$UNIFIED_PIPELINE" ]]; then
    echo "ERROR: unified pipeline not found at $UNIFIED_PIPELINE"
    exit 1
fi

# Dataset profile
export DATASETS="${DATASETS:-agentdojo,fujitsu_b4}"

# Keep outputs isolated from other profiles unless overridden.
export OUTPUT_BASE="${OUTPUT_BASE:-/scratch/memoozd/cb-scratch/data_all}"

# Generation/eval controls synced with sweep_hparams_simple behavior
export STRICT_SIMULATED_MATCH="${STRICT_SIMULATED_MATCH:-true}"
export LIMIT_GEN="${LIMIT_GEN:-5000}"
export EVAL_MERGE_ADAPTER="${EVAL_MERGE_ADAPTER:-true}"
export EVAL_LIMIT="${EVAL_LIMIT:-100}"

# Run unified pipeline and then print a checkpoint-aware visualize command.
bash "$UNIFIED_PIPELINE"

LATEST_RUN_DIR="$(ls -1dt "${OUTPUT_BASE}"/run_* 2>/dev/null | head -n 1 || true)"
if [[ -n "$LATEST_RUN_DIR" && -d "$LATEST_RUN_DIR" ]]; then
    echo ""
    echo "All-data profile complete."
    echo "Visualize with primary metrics + checkpoint comparison:"
    echo "  python scripts/visualize_sweep_results.py \"$LATEST_RUN_DIR\" \\"
    echo "    --compare-checkpoints --no-color \\"
    echo "    --csv \"$LATEST_RUN_DIR/eval_summary.csv\""
fi
