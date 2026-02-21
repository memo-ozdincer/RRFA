#!/bin/bash
#SBATCH --job-name=unified_all_data
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

set -euo pipefail

# Resolve script directory robustly (works regardless of submit cwd).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
UNIFIED_PIPELINE="$SCRIPT_DIR/unified_pipeline.sbatch"

if [[ ! -f "$UNIFIED_PIPELINE" ]]; then
    echo "ERROR: unified pipeline not found at $UNIFIED_PIPELINE"
    exit 1
fi

# Dataset profile
export DATASETS="${DATASETS:-agentdojo,fujitsu_b4,llmail_inject}"

# Keep outputs isolated from other profiles unless overridden.
export OUTPUT_BASE="${OUTPUT_BASE:-/scratch/memoozd/cb-scratch/data_all}"

# LLMail retain mode: how LLMail benign data is sourced for training.
#   "generated"  - use model-generated DR completions (default)
#   "synthetic"  - use deterministic synthetic benign send-email traces
#   "none"       - no LLMail retain data (LLMail used as CB set only)
# Variant recipes:
#   LLMAIL_RETAIN_MODE=none      → all data + LLMail as CB set only
#   LLMAIL_RETAIN_MODE=synthetic → all data + LLMail CB + synthetic retain
#   LLMAIL_RETAIN_MODE=generated → all data + LLMail CB + model-generated retain (default)
export LLMAIL_RETAIN_MODE="${LLMAIL_RETAIN_MODE:-generated}"

# LLMail challenge-faithful defaults
export LLMAIL_TOOL_SCHEMA="${LLMAIL_TOOL_SCHEMA:-configs/tool_schemas/llmail_inject_challenge_v2.json}"
export LLMAIL_CONTEXT_MODE="${LLMAIL_CONTEXT_MODE:-tool_message_retrieval}"
export LLMAIL_EXCLUDE_NOT_RETRIEVED="${LLMAIL_EXCLUDE_NOT_RETRIEVED:-true}"
export LLMAIL_SCENARIOS_PATH="${LLMAIL_SCENARIOS_PATH:-data/llmail_inject/scenarios.json}"

# Generation/eval controls synced with sweep_hparams_simple behavior
export STRICT_SIMULATED_MATCH="${STRICT_SIMULATED_MATCH:-true}"
export SKIP_LLMAIL_NOT_RETRIEVED="${SKIP_LLMAIL_NOT_RETRIEVED:-true}"
export LIMIT_GEN="${LIMIT_GEN:-5000}"
export LLMAIL_LIMIT="${LLMAIL_LIMIT:-}"
export LLMAIL_RETRIEVED_ONLY="${LLMAIL_RETRIEVED_ONLY:-true}"
export EVAL_MERGE_ADAPTER="${EVAL_MERGE_ADAPTER:-true}"

exec bash "$UNIFIED_PIPELINE"
