#!/bin/bash
#SBATCH --job-name=probe_sep
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Representation Probe Diagnostic — Extended
# =============================================================================
# Three comparisons × three poolings × two probe types:
#   1. Harmful vs Benign (different context) — original test
#   2. Harmful vs Refusal (SAME injection context, different response)
#   3. All 32 layers with last_token + MLP (find where signal lives)
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

CACHE_DIR="$CB_SCRATCH/cache"
export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export TORCH_HOME="$CACHE_DIR/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_CACHE_NAME=$(echo "$MODEL_ID" | sed 's/\//--/g')
SNAPSHOT_ROOT="$CACHE_DIR/hf/hub/models--${MODEL_CACHE_NAME}/snapshots"
MODEL_PATH=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1)
if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: Model not cached."; exit 1
fi

TRACES_DIR="$CB_SCRATCH/data/traces"
TRACES_TRUNC="$TRACES_DIR/agentdojo_truncated.jsonl"
TRACES_FULL="$TRACES_DIR/agentdojo_augmented.jsonl"

# Generate refusal traces if not present
REFUSAL_TRACES="$TRACES_DIR/agentdojo_refusal_synthetic.jsonl"
if [[ ! -f "$REFUSAL_TRACES" ]] || [[ ! -s "$REFUSAL_TRACES" ]]; then
    echo "Generating refusal traces..."
    # Use truncated traces as source
    SRC="$TRACES_TRUNC"
    [[ ! -f "$SRC" ]] && SRC="$TRACES_FULL"
    python scripts/generate_refusal_traces.py \
        --traces "$SRC" --output "$REFUSAL_TRACES" --seed 42
fi

echo "========================================================"
echo "Probe Diagnostic — Extended"
echo "========================================================"
echo "Model: $MODEL_PATH"
echo "Date: $(date)"
echo ""

# Use truncated if available, else full
TRACES="$TRACES_TRUNC"
[[ ! -f "$TRACES" ]] && TRACES="$TRACES_FULL"

# ── Test 1: Harmful vs Benign (original comparison) ──────────────────────
echo ""
echo "################################################################"
echo "TEST 1: Harmful vs Benign (different context)"
echo "################################################################"
python scripts/probe_separability.py \
    --traces "$TRACES" \
    --model "$MODEL_PATH" \
    --layers 5 10 15 20 25 30 \
    --pooling mean last_token last_quarter \
    --probe linear mlp \
    --max-samples 200

# ── Test 2: Harmful vs Refusal (same context, different response) ────────
echo ""
echo "################################################################"
echo "TEST 2: Harmful vs Refusal (same injection context)"
echo "################################################################"
python scripts/probe_separability.py \
    --class1 "$TRACES" --class1-label harmful \
    --class0 "$REFUSAL_TRACES" \
    --model "$MODEL_PATH" \
    --layers 5 10 15 20 25 30 \
    --pooling mean last_token last_quarter \
    --probe linear mlp \
    --max-samples 200

# ── Test 3: All 32 layers, last_token + MLP (find the signal) ───────────
echo ""
echo "################################################################"
echo "TEST 3: All 32 layers — last_token + MLP (find where signal is)"
echo "################################################################"
python scripts/probe_separability.py \
    --traces "$TRACES" \
    --model "$MODEL_PATH" \
    --layers $(seq 0 31) \
    --pooling last_token \
    --probe mlp \
    --max-samples 200

echo ""
echo "All probes done: $(date)"
