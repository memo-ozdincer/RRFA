#!/bin/bash
#SBATCH --job-name=pretrain_diag
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# PRE-TRAINING REPRESENTATION DIAGNOSTICS
# =============================================================================
# Runs on frozen base model only — no adapter, no training needed.
# Fast: ~50 samples × 6 layers × 3 diagnostics ≈ 30-60 minutes on 1 GPU.
#
# Diagnostics:
#   1. Base cosine similarity between paired harmful/benign at each layer
#   2. Four-way cosine matrix (tricked/refusal/correct/no-injection)
#   3. Alarm signal analysis (injection separation, propagation, attention)
#
# Also runs post-training drift analysis if ADAPTER_PATH is set.
#
# Usage:
#   sbatch slurm/diagnostics/run_pretrain_diagnostics.sbatch
#   ADAPTER_PATH=/path/to/adapter sbatch slurm/diagnostics/run_pretrain_diagnostics.sbatch
#   MAX_SAMPLES=100 sbatch slurm/diagnostics/run_pretrain_diagnostics.sbatch
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"

module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

source "$VENV_DIR/bin/activate"

# Cache
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch}
export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export TORCH_HOME="$CACHE_DIR/torch"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export XDG_CACHE_HOME="$CACHE_DIR"
mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TQDM_DISABLE=1

# =============================================================================
# Configuration
# =============================================================================
MAX_SAMPLES="${MAX_SAMPLES:-50}"
LAYERS="${LAYERS:-5 10 15 20 25 30}"
ADAPTER_PATH="${ADAPTER_PATH:-}"  # Set to run post-training drift analysis

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_CACHE_NAME=$(echo "$MODEL_ID" | sed 's/\//--/g')
SNAPSHOT_ROOT="$CACHE_DIR/hf/hub/models--${MODEL_CACHE_NAME}/snapshots"
TOKENIZER=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1)
if [[ -z "$TOKENIZER" ]]; then
    echo "ERROR: Model not cached. Run slurm/cache_models.sh first."; exit 1
fi

TRACES_DIR="$CB_SCRATCH/data/traces"
HARMFUL_TRACES="$TRACES_DIR/agentdojo_augmented.jsonl"
BENIGN_TRACES="$TRACES_DIR/agentdojo_augmented.jsonl"  # Same file, split by category
CONTRASTIVE_PAIRS="$TRACES_DIR/contrastive_pairs.jsonl"
AD_TOOL_SCHEMA="$REPO_DIR/configs/tool_schemas/agentdojo_v1.json"

# Refusal traces: generate if missing
REFUSAL_TRACES="$TRACES_DIR/agentdojo_refusal_synthetic.jsonl"
if [[ ! -f "$REFUSAL_TRACES" ]]; then
    TRUNCATED="$CB_SCRATCH/sweeps/shared_data_cb_full_sequence_ts/agentdojo_truncated.jsonl"
    if [[ -f "$TRUNCATED" ]]; then
        echo "Generating refusal traces..."
        python scripts/generate_refusal_traces.py \
            --traces "$TRUNCATED" \
            --output "$REFUSAL_TRACES"
    else
        echo "WARNING: No truncated traces for refusal generation"
        REFUSAL_TRACES=""
    fi
fi

# Generate contrastive pairs if missing
if [[ ! -f "$CONTRASTIVE_PAIRS" ]]; then
    echo "Generating contrastive pairs..."
    python scripts/augment_agentdojo.py \
        --traces-input "$HARMFUL_TRACES" \
        --operations removal,contrastive \
        --contrastive-output "$CONTRASTIVE_PAIRS" \
        --output "$TRACES_DIR/_augment_temp.jsonl" 2>&1 | tail -5
    rm -f "$TRACES_DIR/_augment_temp.jsonl"
fi

# Verify data
for f in "$HARMFUL_TRACES" "$CONTRASTIVE_PAIRS"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing $f"; exit 1
    fi
done

OUTPUT_DIR="$CB_SCRATCH/diagnostics/pretrain_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo "PRE-TRAINING REPRESENTATION DIAGNOSTICS"
echo "========================================================"
echo "Job: ${SLURM_JOB_ID:-local} | $(date)"
echo "Model: $TOKENIZER"
echo "Max samples: $MAX_SAMPLES"
echo "Layers: $LAYERS"
echo "Output: $OUTPUT_DIR"
echo ""

# =============================================================================
# Diagnostic 1-3: Pre-training analysis (frozen model)
# =============================================================================
echo "Running pre-training diagnostics..."

REFUSAL_ARGS=""
if [[ -n "$REFUSAL_TRACES" && -f "$REFUSAL_TRACES" ]]; then
    REFUSAL_ARGS="--refusal-traces $REFUSAL_TRACES"
fi

python scripts/diagnostics/pretrain_representation_analysis.py \
    --model "$TOKENIZER" \
    --harmful-traces "$HARMFUL_TRACES" \
    --benign-traces "$BENIGN_TRACES" \
    --contrastive-pairs "$CONTRASTIVE_PAIRS" \
    $REFUSAL_ARGS \
    --tool-schema "$AD_TOOL_SCHEMA" \
    --layers $LAYERS \
    --max-samples "$MAX_SAMPLES" \
    --output "$OUTPUT_DIR/pretrain_diagnostics.json" \
    --dtype bfloat16

echo ""
echo "Pre-training diagnostics complete."
echo ""

# =============================================================================
# Post-training drift analysis (if adapter provided)
# =============================================================================
if [[ -n "$ADAPTER_PATH" && -d "$ADAPTER_PATH" ]]; then
    echo "========================================================"
    echo "POST-TRAINING DRIFT ANALYSIS"
    echo "========================================================"
    echo "Adapter: $ADAPTER_PATH"
    echo ""

    python scripts/diagnostics/posttrain_drift_analysis.py \
        --model "$TOKENIZER" \
        --adapter "$ADAPTER_PATH" \
        --benign-traces "$BENIGN_TRACES" \
        --harmful-traces "$HARMFUL_TRACES" \
        --tool-schema "$AD_TOOL_SCHEMA" \
        --layers $LAYERS \
        --max-samples "$MAX_SAMPLES" \
        --output "$OUTPUT_DIR/posttrain_drift.json" \
        --dtype bfloat16

    echo ""
    echo "Post-training drift analysis complete."
fi

# =============================================================================
# Also run lossmask coverage diagnostics (fast, no GPU needed really)
# =============================================================================
SHARED_DATA="$CB_SCRATCH/sweeps/shared_data_cb_full_sequence_ts"
if [[ -d "$SHARED_DATA/lossmasks" ]]; then
    echo ""
    echo "========================================================"
    echo "LOSSMASK COVERAGE ANALYSIS"
    echo "========================================================"

    python3 -c "
import json, os, sys
from pathlib import Path

lm_dir = Path('$SHARED_DATA/lossmasks')
for lm_file in sorted(lm_dir.glob('*.jsonl')):
    total = 0; zero = 0; ratios = []
    with open(lm_file) as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            mask = d.get('loss_mask', [])
            total += 1
            r = sum(1 for x in mask if x > 0) / len(mask) if mask else 0
            ratios.append(r)
            if r == 0: zero += 1
    avg = sum(ratios) / len(ratios) if ratios else 0
    inj_count = sum(1 for r in ratios if 0 < r < 0.5)
    print(f'  {lm_file.name}: {total} samples, avg_coverage={avg:.1%}, zero_mask={zero}, partial_mask={inj_count}')
"
fi

echo ""
echo "========================================================"
echo "ALL DIAGNOSTICS COMPLETE"
echo "========================================================"
echo "Output: $OUTPUT_DIR"
echo "Date: $(date)"
