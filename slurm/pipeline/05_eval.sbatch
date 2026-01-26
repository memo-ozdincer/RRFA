#!/bin/bash
#SBATCH --job-name=cb_eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 5: Evaluation (eval.py)
# Evaluates trained CB model against baseline
# Multi-GPU parallel evaluation supported
# =============================================================================
#
# This script:
# 1. Loads trained CB model and optional baseline
# 2. Runs tool-flip ASR evaluation (primary metric)
# 3. Runs forced function call evaluation
# 4. Runs capability retention evaluation
# 5. Outputs JSON results with Stage 1 gate pass/fail
#
# Evaluation Metrics:
#   Tool-flip ASR      - Attack success rate (lower is better for CB)
#   Forced Call ASR    - Prefill attack success rate (lower is better)
#   Capability Retention - Benign query performance (higher is better)
#
# Configuration via environment variables:
#   CB_MODEL_DIR      - Path to trained CB model (default: latest in models/)
#   BASELINE_MODEL    - Baseline model for comparison (optional)
#   EVAL_DATA         - Evaluation data JSONL (default: traces/fujitsu_b4_ds.jsonl)
#   TOOL_SCHEMA       - Tool schema JSON (default: configs/tool_schemas/b4_standard_v1.json)
#   OUTPUT_DIR        - Output directory for results (default: $CB_SCRATCH/eval)
#   OUTPUT_NAME       - Result file name (default: eval_results.json)
#   NUM_WORKERS       - Number of parallel workers (default: 4)
#   GPU_IDS           - Comma-separated GPU IDs (default: 0,1,2,3)
#   LIMIT             - Limit eval samples (optional, for quick tests)
#   FAIL_ON_GATE      - Exit with error if Stage 1 gates fail (default: false)
#   NO_DETAILS        - Don't save per-sample details (default: false)
#
# Submit:
#   sbatch slurm/pipeline/05_eval.sbatch
#
# Or with custom settings:
#   CB_MODEL_DIR=/path/to/model LIMIT=100 sbatch slurm/pipeline/05_eval.sbatch
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"

# Load modules
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv not found at $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"

# =============================================================================
# Cache Setup
# =============================================================================
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch}
mkdir -p "$CB_SCRATCH/logs"

export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================"
echo "Stage 5: Evaluation"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPUs: ${SLURM_GPUS_ON_NODE:-1}"
echo ""

# =============================================================================
# Configuration
# =============================================================================
# Find latest model if not specified
if [[ -z "${CB_MODEL_DIR:-}" ]]; then
    # Find the most recent model directory
    CB_MODEL_DIR=$(ls -dt "$CB_SCRATCH"/models/cb_train_*/final 2>/dev/null | head -1 || echo "")
    if [[ -z "$CB_MODEL_DIR" || ! -d "$CB_MODEL_DIR" ]]; then
        echo "ERROR: No trained model found. Set CB_MODEL_DIR or run 04_train.sbatch first."
        exit 1
    fi
fi

BASELINE_MODEL="${BASELINE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
EVAL_DATA="${EVAL_DATA:-$CB_SCRATCH/data/traces/fujitsu_b4_ds.jsonl}"
TOOL_SCHEMA="${TOOL_SCHEMA:-$REPO_DIR/configs/tool_schemas/b4_standard_v1.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$CB_SCRATCH/eval}"
OUTPUT_NAME="${OUTPUT_NAME:-eval_$(date +%Y%m%d_%H%M%S).json}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
LIMIT="${LIMIT:-}"
FAIL_ON_GATE="${FAIL_ON_GATE:-false}"
NO_DETAILS="${NO_DETAILS:-false}"
DTYPE="${DTYPE:-bfloat16}"

mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  CB_MODEL_DIR: $CB_MODEL_DIR"
echo "  BASELINE_MODEL: $BASELINE_MODEL"
echo "  EVAL_DATA: $EVAL_DATA"
echo "  TOOL_SCHEMA: $TOOL_SCHEMA"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  OUTPUT_NAME: $OUTPUT_NAME"
echo "  NUM_WORKERS: $NUM_WORKERS"
echo "  GPU_IDS: $GPU_IDS"
echo "  LIMIT: ${LIMIT:-unlimited}"
echo "  FAIL_ON_GATE: $FAIL_ON_GATE"
echo "  NO_DETAILS: $NO_DETAILS"
echo "  DTYPE: $DTYPE"
echo ""

# =============================================================================
# Validate inputs
# =============================================================================
if [[ ! -d "$CB_MODEL_DIR" ]]; then
    echo "ERROR: CB model not found: $CB_MODEL_DIR"
    exit 1
fi

if [[ ! -f "$EVAL_DATA" ]]; then
    echo "ERROR: Evaluation data not found: $EVAL_DATA"
    echo "Please run stages 1-2 first or set EVAL_DATA path."
    exit 1
fi

if [[ ! -f "$TOOL_SCHEMA" ]]; then
    echo "ERROR: Tool schema not found: $TOOL_SCHEMA"
    exit 1
fi

echo "Validation passed."
echo ""

# =============================================================================
# Determine model/adapter structure
# =============================================================================
# Check if CB_MODEL_DIR is a full model or just adapter
CB_ADAPTER=""
CB_MODEL=""

if [[ -f "$CB_MODEL_DIR/adapter_config.json" ]]; then
    # This is a LoRA adapter - need base model
    CB_ADAPTER="$CB_MODEL_DIR"
    CB_MODEL="$BASELINE_MODEL"
    echo "Detected LoRA adapter at: $CB_ADAPTER"
    echo "Using base model: $CB_MODEL"
elif [[ -f "$CB_MODEL_DIR/config.json" ]]; then
    # This is a full/merged model
    CB_MODEL="$CB_MODEL_DIR"
    echo "Detected full model at: $CB_MODEL"
else
    echo "ERROR: Invalid model directory structure: $CB_MODEL_DIR"
    echo "Expected either adapter_config.json (LoRA) or config.json (full model)"
    exit 1
fi

echo ""

# =============================================================================
# Build command arguments
# =============================================================================
EVAL_ARGS=(
    --baseline "$BASELINE_MODEL"
    --eval-data "$EVAL_DATA"
    --tool-schema "$TOOL_SCHEMA"
    --output "$OUTPUT_DIR/$OUTPUT_NAME"
    --num-workers "$NUM_WORKERS"
    --gpu-ids "$GPU_IDS"
    --dtype "$DTYPE"
)

if [[ -n "$CB_MODEL" ]]; then
    EVAL_ARGS+=(--cb-model "$CB_MODEL")
fi

if [[ -n "$CB_ADAPTER" ]]; then
    EVAL_ARGS+=(--cb-adapter "$CB_ADAPTER")
fi

if [[ -n "$LIMIT" ]]; then
    EVAL_ARGS+=(--limit "$LIMIT")
fi

if [[ "$FAIL_ON_GATE" == "true" ]]; then
    EVAL_ARGS+=(--fail-on-gate)
fi

if [[ "$NO_DETAILS" == "true" ]]; then
    EVAL_ARGS+=(--no-details)
fi

# =============================================================================
# Run Evaluation
# =============================================================================
echo "========================================"
echo "Running Evaluation"
echo "========================================"
echo "Command: python src/evaluation/eval.py ${EVAL_ARGS[*]}"
echo ""

python src/evaluation/eval.py "${EVAL_ARGS[@]}"
EVAL_EXIT_CODE=$?

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "Stage 5 Complete: Evaluation"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/$OUTPUT_NAME"
echo ""

# Print summary if output exists
if [[ -f "$OUTPUT_DIR/$OUTPUT_NAME" ]]; then
    echo "Quick Summary:"
    python3 -c "
import json
with open('$OUTPUT_DIR/$OUTPUT_NAME', 'r') as f:
    results = json.load(f)

if 'baseline' in results:
    print('  Baseline Tool-flip ASR: {:.1%}'.format(
        results['baseline']['tool_flip_asr']['attack_success_rate']))

if 'cb_model' in results:
    print('  CB Model Tool-flip ASR: {:.1%}'.format(
        results['cb_model']['tool_flip_asr']['attack_success_rate']))

if 'delta' in results:
    print('  ASR Reduction: {:.1%}'.format(
        -results['delta']['tool_flip_asr']))

if 'stage1_passed' in results:
    passed = results['stage1_passed']
    print('  Stage 1 Gates: {} PASSED'.format('✅' if passed else '❌ NOT'))
" 2>/dev/null || echo "  (summary extraction failed)"
fi

echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "  $OUTPUT_DIR/$OUTPUT_NAME"
echo ""

# Check if we need to exit with error
if [[ $EVAL_EXIT_CODE -ne 0 ]]; then
    echo "Evaluation exited with code $EVAL_EXIT_CODE"
    exit $EVAL_EXIT_CODE
fi

echo "Pipeline evaluation complete!"
