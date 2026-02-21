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

# Resolve repo root from this script location and run via relative path.
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Dataset profile
export DATASETS="${DATASETS:-agentdojo,fujitsu_b4,llmail_inject}"

# Keep outputs isolated from other profiles unless overridden.
export OUTPUT_BASE="${OUTPUT_BASE:-/scratch/memoozd/cb-scratch/data_all}"

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

exec bash slurm/pipeline/unified_pipeline.sbatch
