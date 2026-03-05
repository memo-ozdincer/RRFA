#!/bin/bash
#SBATCH --job-name=llm_augment
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

set -euo pipefail

# =============================================================================
# LLM-BASED DATA AUGMENTATION PIPELINE
# =============================================================================
# Uses vLLM with the same Llama-3.1-8B-Instruct model we train on.
# Produces completed traces optimized for injection_aware lossmasking.
#
# Operations:
#   1. Start vLLM server
#   2. Complete Fujitsu skeletons (harmful + benign pairs)
#   3. Canonicalize AgentDojo traces
#   4. Upsample minority variant (Comment-based, 5.6% -> 20%+)
#   5. Cross-pollinate injection patterns
#   6. Build contrastive pairs
#   7. Complete upsampled minority skeletons
#   8. Run ETL_B on all augmented data
#
# Usage:
#   sbatch slurm/pipeline/run_llm_augmentation.sbatch
#
# Outputs:
#   $CB_SCRATCH/data/traces/fujitsu_b4_completed.jsonl     (completed skeletons)
#   $CB_SCRATCH/data/traces/agentdojo_canonicalized.jsonl   (cleaned AD traces)
#   $CB_SCRATCH/data/traces/minority_upsampled.jsonl        (new minority traces)
#   $CB_SCRATCH/data/traces/cross_pollinated.jsonl          (cross-dataset)
#   $CB_SCRATCH/data/traces/minority_completed.jsonl        (completed minority)
#   $CB_SCRATCH/data/traces/contrastive_pairs_fujitsu.jsonl (pair mapping)
#   $CB_SCRATCH/data/traces/augmented_all.jsonl             (merged final)
#   $CB_SCRATCH/data/renders/augmented_all.jsonl            (tokenized)
#   $CB_SCRATCH/data/lossmasks/augmented_all.jsonl          (loss masks)
# =============================================================================

PROJECT_DIR="/project/def-zhijing/memoozd"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

if [[ -d "$REPO_DIR" ]]; then
    cd "$REPO_DIR"
    echo "Working directory: $(pwd)"
else
    echo "ERROR: Repo not found at $REPO_DIR"
    exit 1
fi

module --force purge || true
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.2

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated venv: $VENV_DIR"
fi

# Cache
CB_SCRATCH="/scratch/memoozd/cb-scratch"
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch}
export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Config
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TOKENIZER="${TOKENIZER:-$MODEL}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-42}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
MINORITY_TARGET="${MINORITY_TARGET:-3000}"
CROSS_MAX="${CROSS_MAX:-500}"
VLLM_PORT=8899

# Output dirs
OUT="$CB_SCRATCH/data/traces"
RENDERS_OUT="$CB_SCRATCH/data/renders"
LOSSMASKS_OUT="$CB_SCRATCH/data/lossmasks"
mkdir -p "$OUT" "$RENDERS_OUT" "$LOSSMASKS_OUT"

echo "============================================================"
echo "STEP 0: Start vLLM Server"
echo "============================================================"

# Start vLLM server in background
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port $VLLM_PORT \
    --max-model-len $MAX_SEQ_LENGTH \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for vLLM server..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

VLLM_URL="http://localhost:$VLLM_PORT"

echo ""
echo "============================================================"
echo "STEP 1: Complete Fujitsu Skeletons"
echo "============================================================"

python scripts/augment_llm.py \
    --operation complete_fujitsu \
    --input data/traces/fujitsu_b4_traces.jsonl \
    --output "$OUT/fujitsu_b4_completed.jsonl" \
    --vllm-url "$VLLM_URL" \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED"

echo ""
echo "============================================================"
echo "STEP 2: Canonicalize AgentDojo"
echo "============================================================"

python scripts/augment_llm.py \
    --operation canonicalize_ad \
    --input data/traces/agentdojo_augmented.jsonl \
    --output "$OUT/agentdojo_canonicalized.jsonl"

echo ""
echo "============================================================"
echo "STEP 3: Upsample Minority Variant"
echo "============================================================"

python scripts/augment_llm.py \
    --operation upsample_minority \
    --input data/traces/fujitsu_b4_traces.jsonl \
    --output "$OUT/minority_upsampled.jsonl" \
    --minority-target "$MINORITY_TARGET" \
    --seed "$SEED"

echo ""
echo "============================================================"
echo "STEP 4: Cross-Pollinate"
echo "============================================================"

python scripts/augment_llm.py \
    --operation cross_pollinate \
    --input data/traces/fujitsu_b4_traces.jsonl \
    --input2 data/traces/agentdojo_augmented.jsonl \
    --output "$OUT/cross_pollinated.jsonl" \
    --cross-max "$CROSS_MAX" \
    --seed "$SEED"

echo ""
echo "============================================================"
echo "STEP 5: Complete Minority Skeletons"
echo "============================================================"

python scripts/augment_llm.py \
    --operation complete_fujitsu \
    --input "$OUT/minority_upsampled.jsonl" \
    --output "$OUT/minority_completed.jsonl" \
    --vllm-url "$VLLM_URL" \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED"

echo ""
echo "============================================================"
echo "STEP 6: Build Contrastive Pairs"
echo "============================================================"

python scripts/augment_llm.py \
    --operation contrastive_fujitsu \
    --input "$OUT/fujitsu_b4_completed.jsonl" \
    --output /dev/null \
    --pairs-output "$OUT/contrastive_pairs_fujitsu.jsonl"

echo ""
echo "============================================================"
echo "STEP 7: Merge All Augmented Data"
echo "============================================================"

cat "$OUT/fujitsu_b4_completed.jsonl" \
    "$OUT/agentdojo_canonicalized.jsonl" \
    "$OUT/minority_completed.jsonl" \
    "$OUT/cross_pollinated.jsonl" \
    > "$OUT/augmented_all.jsonl"

TOTAL=$(wc -l < "$OUT/augmented_all.jsonl")
echo "Merged $TOTAL traces into augmented_all.jsonl"

echo ""
echo "============================================================"
echo "STEP 8: Verify Augmented Data"
echo "============================================================"

python scripts/augment_llm.py \
    --operation verify \
    --input "$OUT/augmented_all.jsonl"

echo ""
echo "============================================================"
echo "STEP 9: ETL_B (Render + Lossmask)"
echo "============================================================"

python -m src.schemas.tools.ETL_B \
    --traces "$OUT/augmented_all.jsonl" \
    --render-out "$RENDERS_OUT/augmented_all.jsonl" \
    --lossmask-out "$LOSSMASKS_OUT/augmented_all.jsonl" \
    --tokenizer "$TOKENIZER" \
    --max-length "$MAX_SEQ_LENGTH" \
    --policy-override injection_aware \
    --pooling-policy-override cb_full_sequence \
    --include-rendered-text

RENDER_COUNT=$(wc -l < "$RENDERS_OUT/augmented_all.jsonl")
LOSSMASK_COUNT=$(wc -l < "$LOSSMASKS_OUT/augmented_all.jsonl")
echo "  Renders: $RENDER_COUNT"
echo "  Lossmasks: $LOSSMASK_COUNT"

echo ""
echo "============================================================"
echo "STEP 10: Shutdown vLLM"
echo "============================================================"

kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "vLLM server stopped"

echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"
echo ""
echo "Output files:"
echo "  Completed Fujitsu:  $OUT/fujitsu_b4_completed.jsonl"
echo "  Canonicalized AD:   $OUT/agentdojo_canonicalized.jsonl"
echo "  Minority completed: $OUT/minority_completed.jsonl"
echo "  Cross-pollinated:   $OUT/cross_pollinated.jsonl"
echo "  Contrastive pairs:  $OUT/contrastive_pairs_fujitsu.jsonl"
echo "  MERGED ALL:         $OUT/augmented_all.jsonl"
echo "  Renders:            $RENDERS_OUT/augmented_all.jsonl"
echo "  Lossmasks:          $LOSSMASKS_OUT/augmented_all.jsonl"
echo ""
echo "Expected trace counts:"
echo "  Fujitsu completed:  ~26,492 (13,246 harmful + 13,246 benign)"
echo "  AgentDojo canon:    ~4,369"
echo "  Minority completed: ~6,000 (3,000 harmful + 3,000 benign)"
echo "  Cross-pollinated:   ~1,000"
echo "  TOTAL:              ~37,861"
