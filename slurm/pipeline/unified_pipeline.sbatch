#!/bin/bash
#SBATCH --job-name=cb_pipeline
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# UNIFIED CB TRAINING PIPELINE
# =============================================================================
#
# A configurable pipeline that handles any dataset type:
# - Complete traces with labels (AgentDojo)
# - Skeleton traces needing generation (Fujitsu B4)
# - Complete traces needing judge evaluation (new datasets)
#
# Pipeline Stages:
#   1. ETL_A: Convert raw data to trace_v1 format
#   2. (Optional) Generate: Fill skeleton traces with completions
#   3. (Optional) Judge: Evaluate traces to add attack_succeeded labels
#   4. ETL_B: Create renders and lossmasks
#   5. Split: Separate into CB and retain sets
#   6. Train: Run circuit breaker training
#   7. Eval: Evaluate trained model
#
# =============================================================================
#
# Configuration via environment variables:
#
# Dataset Selection:
#   DATASETS         - Comma-separated list: "agentdojo,fujitsu_b4" (default: both)
#   DATASET_CONFIG   - Path to dataset_config.yaml (default: configs/dataset_config.yaml)
#
# Stage Control:
#   SKIP_ETL_A       - Skip raw data conversion (default: false)
#   SKIP_GENERATE    - Skip skeleton completion (default: false)
#   SKIP_JUDGE       - Skip judge evaluation (default: true, only for unlabeled data)
#   SKIP_ETL_B       - Skip render/lossmask creation (default: false)
#   SKIP_SPLIT       - Skip CB/retain splitting (default: false)
#   SKIP_TRAIN       - Skip training (default: false)
#   SKIP_EVAL        - Skip evaluation (default: false)
#
# Data Paths:
#   FUJITSU_B4_PATH  - Path to Fujitsu B4 JSONL (default: auto-detected)
#   AGENTDOJO_PATH   - Path to AgentDojo JSONL (default: auto-detected)
#   OUTPUT_BASE      - Base output directory (default: $CB_SCRATCH/data)
#
# Training Configuration:
#   TOTAL_STEPS      - Training steps (default: 300)
#   BATCH_SIZE       - Per-GPU batch size (default: 8)
#   LEARNING_RATE    - Learning rate (default: 5e-5)
#   ALPHA_MAX        - Initial alpha for RR loss (default: 10.0)
#   CB_LAYERS        - CB target layers, comma-separated (default: 10,20)
#   LMP_POLICY       - Loss mask policy (default: assistant_only)
#
# Judge Configuration (for datasets needing evaluation):
#   JUDGE_PROVIDER   - openai, anthropic, or local (default: openai)
#   JUDGE_MODEL      - Model name (default: gpt-4o-mini)
#
# =============================================================================
#
# Usage Examples:
#
#   # Full pipeline with both datasets
#   sbatch slurm/pipeline/unified_pipeline.sbatch
#
#   # AgentDojo only, skip generation
#   DATASETS=agentdojo sbatch slurm/pipeline/unified_pipeline.sbatch
#
#   # Fujitsu B4 only
#   DATASETS=fujitsu_b4 sbatch slurm/pipeline/unified_pipeline.sbatch
#
#   # Skip training to just prepare data
#   SKIP_TRAIN=true SKIP_EVAL=true sbatch slurm/pipeline/unified_pipeline.sbatch
#
#   # Use judge for a new dataset
#   DATASETS=new_dataset SKIP_JUDGE=false JUDGE_PROVIDER=openai sbatch ...
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
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,wandb}
mkdir -p "$CB_SCRATCH/logs"

export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export WANDB_DIR="$CACHE_DIR/wandb"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================"
echo "UNIFIED CB TRAINING PIPELINE"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo ""

# =============================================================================
# Configuration
# =============================================================================

# Dataset selection
DATASETS="${DATASETS:-agentdojo,fujitsu_b4}"
DATASET_CONFIG="${DATASET_CONFIG:-$REPO_DIR/configs/dataset_config.yaml}"

# Stage control
SKIP_ETL_A="${SKIP_ETL_A:-false}"
SKIP_GENERATE="${SKIP_GENERATE:-false}"
SKIP_JUDGE="${SKIP_JUDGE:-true}"
SKIP_ETL_B="${SKIP_ETL_B:-false}"
SKIP_SPLIT="${SKIP_SPLIT:-false}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"
SKIP_EVAL="${SKIP_EVAL:-false}"

# Paths
OUTPUT_BASE="${OUTPUT_BASE:-$CB_SCRATCH/data}"
TRACES_DIR="$OUTPUT_BASE/traces"
RENDERS_DIR="$OUTPUT_BASE/renders"
LOSSMASKS_DIR="$OUTPUT_BASE/lossmasks"
MODELS_DIR="$CB_SCRATCH/models"
EVAL_DIR="$CB_SCRATCH/eval"

mkdir -p "$TRACES_DIR" "$RENDERS_DIR" "$LOSSMASKS_DIR" "$MODELS_DIR" "$EVAL_DIR"

# Default data paths
FUJITSU_B4_PATH="${FUJITSU_B4_PATH:-$REPO_DIR/data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl}"
AGENTDOJO_PATH="${AGENTDOJO_PATH:-$REPO_DIR/data/agent_dojo/agentdojo_spotlight_extract.jsonl}"

# Model/tokenizer
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TOKENIZER="${TOKENIZER:-$MODEL}"
TOOL_SCHEMA="${TOOL_SCHEMA:-$REPO_DIR/configs/tool_schemas/b4_standard_v1.json}"

# Generation settings
TEMPERATURE_DS="${TEMPERATURE_DS:-0.7}"
TEMPERATURE_DR="${TEMPERATURE_DR:-0.3}"
USE_VLLM="${USE_VLLM:-true}"

# Training settings
TOTAL_STEPS="${TOTAL_STEPS:-300}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
ALPHA_MAX="${ALPHA_MAX:-10.0}"
CB_LAYERS="${CB_LAYERS:-10,20}"
LMP_POLICY="${LMP_POLICY:-assistant_only}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
DTYPE="${DTYPE:-bfloat16}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
NO_WANDB="${NO_WANDB:-true}"

# Judge settings
JUDGE_PROVIDER="${JUDGE_PROVIDER:-openai}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"

# Eval settings
EVAL_LIMIT="${EVAL_LIMIT:-100}"

echo "Configuration:"
echo "  DATASETS: $DATASETS"
echo "  OUTPUT_BASE: $OUTPUT_BASE"
echo "  LMP_POLICY: $LMP_POLICY"
echo "  TOTAL_STEPS: $TOTAL_STEPS"
echo "  ALPHA_MAX: $ALPHA_MAX"
echo "  CB_LAYERS: $CB_LAYERS"
echo ""
echo "Stage Control:"
echo "  SKIP_ETL_A: $SKIP_ETL_A"
echo "  SKIP_GENERATE: $SKIP_GENERATE"
echo "  SKIP_JUDGE: $SKIP_JUDGE"
echo "  SKIP_ETL_B: $SKIP_ETL_B"
echo "  SKIP_SPLIT: $SKIP_SPLIT"
echo "  SKIP_TRAIN: $SKIP_TRAIN"
echo "  SKIP_EVAL: $SKIP_EVAL"
echo ""

# =============================================================================
# Resolve Tokenizer to Local Cache
# =============================================================================
if [[ "$TOKENIZER" != /* ]]; then
    echo "Resolving tokenizer '$TOKENIZER' to local cache path..."

    MODEL_CACHE_DIR="$HF_HUB_CACHE/models--meta-llama--Llama-3.1-8B-Instruct"
    SNAPSHOT_ROOT="$MODEL_CACHE_DIR/snapshots"

    if [[ -d "$SNAPSHOT_ROOT" ]]; then
        SNAPSHOT_PATH=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1)
        if [[ -n "$SNAPSHOT_PATH" ]]; then
            echo "  Found local snapshot: $SNAPSHOT_PATH"
            TOKENIZER="$SNAPSHOT_PATH"
            MODEL="$SNAPSHOT_PATH"
        else
            echo "  ERROR: Snapshot directory is empty: $SNAPSHOT_ROOT"
            exit 1
        fi
    else
        echo "  ERROR: Model not cached at: $MODEL_CACHE_DIR"
        exit 1
    fi
fi
echo ""

# =============================================================================
# Stage 1: ETL_A (Raw -> Traces)
# =============================================================================
if [[ "$SKIP_ETL_A" != "true" ]]; then
    echo "========================================"
    echo "Stage 1: ETL_A (Raw -> Traces)"
    echo "========================================"

    IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"

    for dataset in "${DATASET_ARRAY[@]}"; do
        dataset=$(echo "$dataset" | xargs)  # Trim whitespace

        case "$dataset" in
            agentdojo)
                if [[ -f "$AGENTDOJO_PATH" ]]; then
                    echo "Processing AgentDojo..."
                    OUTPUT_FILE="$TRACES_DIR/agentdojo_complete.jsonl"

                    python src/schemas/tools/ETL_A.py \
                        --agentdojo "$AGENTDOJO_PATH" \
                        --output "$OUTPUT_FILE" \
                        --split train \
                        --injection-patterns-file "$REPO_DIR/configs/injection_patterns.json"

                    echo "  Output: $OUTPUT_FILE ($(wc -l < "$OUTPUT_FILE") lines)"
                else
                    echo "  WARNING: AgentDojo file not found: $AGENTDOJO_PATH"
                fi
                ;;

            fujitsu_b4)
                if [[ -f "$FUJITSU_B4_PATH" ]]; then
                    echo "Processing Fujitsu B4..."
                    OUTPUT_FILE="$TRACES_DIR/fujitsu_b4_skeletons.jsonl"

                    python src/schemas/tools/ETL_A.py \
                        --fujitsu-b4 "$FUJITSU_B4_PATH" \
                        --output "$OUTPUT_FILE" \
                        --split train \
                        --tool-schema "$TOOL_SCHEMA"

                    echo "  Output: $OUTPUT_FILE ($(wc -l < "$OUTPUT_FILE") lines)"
                else
                    echo "  WARNING: Fujitsu B4 file not found: $FUJITSU_B4_PATH"
                fi
                ;;

            *)
                echo "  WARNING: Unknown dataset: $dataset"
                ;;
        esac
    done
    echo ""
fi

# =============================================================================
# Stage 2: Generate Completions (for skeleton traces)
# =============================================================================
if [[ "$SKIP_GENERATE" != "true" ]]; then
    SKELETON_FILE="$TRACES_DIR/fujitsu_b4_skeletons.jsonl"

    if [[ -f "$SKELETON_FILE" ]]; then
        echo "========================================"
        echo "Stage 2: Generate Completions"
        echo "========================================"

        DS_OUTPUT="$TRACES_DIR/fujitsu_b4_ds.jsonl"
        DR_OUTPUT="$TRACES_DIR/fujitsu_b4_dr.jsonl"

        # Check if we need to generate
        NEED_DS=true
        NEED_DR=true
        [[ -f "$DS_OUTPUT" ]] && NEED_DS=false
        [[ -f "$DR_OUTPUT" ]] && NEED_DR=false

        if [[ "$NEED_DS" == "true" || "$NEED_DR" == "true" ]]; then
            # Generate DS (attack succeeds)
            if [[ "$NEED_DS" == "true" ]]; then
                echo "Generating DS (attack succeeds) traces..."
                python src/generation/generate_completions.py \
                    --input "$SKELETON_FILE" \
                    --output "$DS_OUTPUT" \
                    --model "$MODEL" \
                    --tool-schema "$TOOL_SCHEMA" \
                    --mode ds \
                    --temperature "$TEMPERATURE_DS" \
                    ${USE_VLLM:+--use-vllm}

                echo "  DS output: $DS_OUTPUT ($(wc -l < "$DS_OUTPUT") lines)"
            fi

            # Generate DR (attack resisted)
            if [[ "$NEED_DR" == "true" ]]; then
                echo "Generating DR (resisted) traces..."
                python src/generation/generate_completions.py \
                    --input "$SKELETON_FILE" \
                    --output "$DR_OUTPUT" \
                    --model "$MODEL" \
                    --tool-schema "$TOOL_SCHEMA" \
                    --mode dr \
                    --temperature "$TEMPERATURE_DR" \
                    ${USE_VLLM:+--use-vllm}

                echo "  DR output: $DR_OUTPUT ($(wc -l < "$DR_OUTPUT") lines)"
            fi
        else
            echo "Skipping: DS/DR files already exist"
            echo "  DS: $DS_OUTPUT"
            echo "  DR: $DR_OUTPUT"
        fi
        echo ""
    fi
fi

# =============================================================================
# Stage 3: Judge Evaluation (for unlabeled datasets)
# =============================================================================
if [[ "$SKIP_JUDGE" != "true" ]]; then
    echo "========================================"
    echo "Stage 3: Judge Evaluation"
    echo "========================================"

    # Find traces that need evaluation (no attack_succeeded labels)
    for trace_file in "$TRACES_DIR"/*_complete.jsonl; do
        if [[ -f "$trace_file" ]]; then
            basename=$(basename "$trace_file" .jsonl)

            # Check if labels exist
            HAS_LABELS=$(head -1 "$trace_file" | python3 -c "
import json, sys
d = json.load(sys.stdin)
labels = d.get('labels', {})
print('yes' if 'attack_succeeded' in labels else 'no')
" 2>/dev/null || echo "no")

            if [[ "$HAS_LABELS" == "no" ]]; then
                echo "Evaluating: $basename (no attack_succeeded labels)"

                LABELED_OUTPUT="${trace_file%.jsonl}_labeled.jsonl"

                python src/evaluation/judge.py \
                    --traces "$trace_file" \
                    --output "$LABELED_OUTPUT" \
                    --provider "$JUDGE_PROVIDER" \
                    --model "$JUDGE_MODEL" \
                    --add-labels

                # Replace original with labeled version
                mv "$LABELED_OUTPUT" "$trace_file"
                echo "  Updated: $trace_file"
            else
                echo "Skipping: $basename (already has labels)"
            fi
        fi
    done
    echo ""
fi

# =============================================================================
# Stage 4: ETL_B (Traces -> Renders + Lossmasks)
# =============================================================================
if [[ "$SKIP_ETL_B" != "true" ]]; then
    echo "========================================"
    echo "Stage 4: ETL_B (Renders + Lossmasks)"
    echo "========================================"

    # Process all trace files
    for trace_file in "$TRACES_DIR"/*.jsonl; do
        if [[ -f "$trace_file" ]]; then
            basename=$(basename "$trace_file" .jsonl)

            # Skip skeleton files
            if [[ "$basename" == *"skeleton"* ]]; then
                echo "Skipping skeleton file: $basename"
                continue
            fi

            render_out="$RENDERS_DIR/${basename}.jsonl"
            lossmask_out="$LOSSMASKS_DIR/${basename}.jsonl"

            echo "Processing: $basename"
            python src/schemas/tools/ETL_B.py \
                --traces "$trace_file" \
                --render-out "$render_out" \
                --lossmask-out "$lossmask_out" \
                --tokenizer "$TOKENIZER" \
                --max-length 4096 \
                --policy-override "$LMP_POLICY"

            echo "  Render: $render_out ($(wc -l < "$render_out") lines)"
            echo "  Lossmask: $lossmask_out ($(wc -l < "$lossmask_out") lines)"
        fi
    done
    echo ""
fi

# =============================================================================
# Stage 5: Split into CB/Retain Sets
# =============================================================================
if [[ "$SKIP_SPLIT" != "true" ]]; then
    echo "========================================"
    echo "Stage 5: Split into CB/Retain Sets"
    echo "========================================"

    IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"

    for dataset in "${DATASET_ARRAY[@]}"; do
        dataset=$(echo "$dataset" | xargs)

        case "$dataset" in
            agentdojo)
                TRACE_FILE="$TRACES_DIR/agentdojo_complete.jsonl"
                RENDER_FILE="$RENDERS_DIR/agentdojo_complete.jsonl"
                LOSSMASK_FILE="$LOSSMASKS_DIR/agentdojo_complete.jsonl"
                SPLIT_DIR="$OUTPUT_BASE/split/agentdojo"

                if [[ -f "$TRACE_FILE" ]]; then
                    echo "Splitting AgentDojo..."
                    python scripts/split_dataset.py \
                        --traces "$TRACE_FILE" \
                        --renders "$RENDER_FILE" \
                        --lossmasks "$LOSSMASK_FILE" \
                        --output-dir "$SPLIT_DIR" \
                        --prefix "agentdojo" \
                        --resisted-handling retain
                fi
                ;;

            fujitsu_b4)
                # Fujitsu B4 already has DS (harmful) and DR (benign) split
                echo "Fujitsu B4: Using existing DS/DR split"
                echo "  DS (CB): $TRACES_DIR/fujitsu_b4_ds.jsonl"
                echo "  DR (Retain): $TRACES_DIR/fujitsu_b4_dr.jsonl"
                ;;
        esac
    done
    echo ""
fi

# =============================================================================
# Stage 6: Training
# =============================================================================
if [[ "$SKIP_TRAIN" != "true" ]]; then
    echo "========================================"
    echo "Stage 6: Circuit Breaker Training"
    echo "========================================"

    RUN_ID="cb_$(date +%Y%m%d_%H%M%S)"
    MODEL_OUTPUT_DIR="$MODELS_DIR/$RUN_ID"
    mkdir -p "$MODEL_OUTPUT_DIR"

    # Collect harmful and benign data
    HARMFUL_RENDERS=""
    HARMFUL_LOSSMASKS=""
    BENIGN_RENDERS=""
    BENIGN_LOSSMASKS=""

    # Fujitsu DS (harmful)
    if [[ -f "$RENDERS_DIR/fujitsu_b4_ds.jsonl" ]]; then
        HARMFUL_RENDERS="$RENDERS_DIR/fujitsu_b4_ds.jsonl"
        HARMFUL_LOSSMASKS="$LOSSMASKS_DIR/fujitsu_b4_ds.jsonl"
    fi

    # Fujitsu DR (benign)
    if [[ -f "$RENDERS_DIR/fujitsu_b4_dr.jsonl" ]]; then
        BENIGN_RENDERS="$RENDERS_DIR/fujitsu_b4_dr.jsonl"
        BENIGN_LOSSMASKS="$LOSSMASKS_DIR/fujitsu_b4_dr.jsonl"
    fi

    # AgentDojo split
    AGENTDOJO_SPLIT="$OUTPUT_BASE/split/agentdojo"
    if [[ -f "$AGENTDOJO_SPLIT/agentdojo_renders_cb.jsonl" ]]; then
        if [[ -n "$HARMFUL_RENDERS" ]]; then
            HARMFUL_RENDERS="$HARMFUL_RENDERS,$AGENTDOJO_SPLIT/agentdojo_renders_cb.jsonl"
            HARMFUL_LOSSMASKS="$HARMFUL_LOSSMASKS,$AGENTDOJO_SPLIT/agentdojo_lossmasks_cb.jsonl"
        else
            HARMFUL_RENDERS="$AGENTDOJO_SPLIT/agentdojo_renders_cb.jsonl"
            HARMFUL_LOSSMASKS="$AGENTDOJO_SPLIT/agentdojo_lossmasks_cb.jsonl"
        fi
    fi

    if [[ -f "$AGENTDOJO_SPLIT/agentdojo_renders_retain.jsonl" ]]; then
        if [[ -n "$BENIGN_RENDERS" ]]; then
            BENIGN_RENDERS="$BENIGN_RENDERS,$AGENTDOJO_SPLIT/agentdojo_renders_retain.jsonl"
            BENIGN_LOSSMASKS="$BENIGN_LOSSMASKS,$AGENTDOJO_SPLIT/agentdojo_lossmasks_retain.jsonl"
        else
            BENIGN_RENDERS="$AGENTDOJO_SPLIT/agentdojo_renders_retain.jsonl"
            BENIGN_LOSSMASKS="$AGENTDOJO_SPLIT/agentdojo_lossmasks_retain.jsonl"
        fi
    fi

    echo "Training data:"
    echo "  Harmful renders: $HARMFUL_RENDERS"
    echo "  Benign renders: $BENIGN_RENDERS"
    echo ""

    # Parse CB layers
    IFS=',' read -ra CB_LAYER_ARRAY <<< "$CB_LAYERS"

    # Build training command
    TRAIN_ARGS=(
        --preset "llama-3.1-8b-instruct"
        --model "$MODEL"
        --output-dir "$MODEL_OUTPUT_DIR"
        --total-steps "$TOTAL_STEPS"
        --batch-size "$BATCH_SIZE"
        --learning-rate "$LEARNING_RATE"
        --warmup-steps "$WARMUP_STEPS"
        --alpha-max "$ALPHA_MAX"
        --max-seq-length "$MAX_SEQ_LENGTH"
        --gradient-accumulation-steps 1
        --loss-weighting "dual"
        --lora-r "$LORA_R"
        --lora-alpha "$LORA_ALPHA"
        --lora-dropout 0.05
        --logging-steps 10
        --save-steps 50
        --cb-target-layers "${CB_LAYER_ARRAY[@]}"
        --mode mixed
    )

    if [[ "$NO_WANDB" == "true" ]]; then
        TRAIN_ARGS+=(--no-wandb)
    fi

    # Add data files
    IFS=',' read -ra HARMFUL_R_ARRAY <<< "$HARMFUL_RENDERS"
    IFS=',' read -ra HARMFUL_L_ARRAY <<< "$HARMFUL_LOSSMASKS"
    IFS=',' read -ra BENIGN_R_ARRAY <<< "$BENIGN_RENDERS"
    IFS=',' read -ra BENIGN_L_ARRAY <<< "$BENIGN_LOSSMASKS"

    TRAIN_ARGS+=(--harmful-renders "${HARMFUL_R_ARRAY[@]}")
    TRAIN_ARGS+=(--harmful-lossmasks "${HARMFUL_L_ARRAY[@]}")
    TRAIN_ARGS+=(--benign-renders "${BENIGN_R_ARRAY[@]}")
    TRAIN_ARGS+=(--benign-lossmasks "${BENIGN_L_ARRAY[@]}")

    echo "Running training..."
    python src/training/train_schema.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$MODEL_OUTPUT_DIR/train.log"

    echo ""
    echo "Model saved to: $MODEL_OUTPUT_DIR"
    echo ""
fi

# =============================================================================
# Stage 7: Evaluation
# =============================================================================
if [[ "$SKIP_EVAL" != "true" ]]; then
    echo "========================================"
    echo "Stage 7: Evaluation"
    echo "========================================"

    # Find latest model
    if [[ -z "${MODEL_OUTPUT_DIR:-}" ]]; then
        MODEL_OUTPUT_DIR=$(ls -dt "$MODELS_DIR"/cb_*/final 2>/dev/null | head -1 | xargs dirname || echo "")
    fi

    if [[ -z "$MODEL_OUTPUT_DIR" || ! -d "$MODEL_OUTPUT_DIR" ]]; then
        echo "WARNING: No trained model found, skipping evaluation"
    else
        # Find adapter path
        if [[ -d "$MODEL_OUTPUT_DIR/final" ]]; then
            ADAPTER_PATH="$MODEL_OUTPUT_DIR/final"
        elif [[ -f "$MODEL_OUTPUT_DIR/adapter_config.json" ]]; then
            ADAPTER_PATH="$MODEL_OUTPUT_DIR"
        else
            echo "WARNING: No adapter found in $MODEL_OUTPUT_DIR"
            ADAPTER_PATH=""
        fi

        if [[ -n "$ADAPTER_PATH" ]]; then
            EVAL_OUTPUT="$EVAL_DIR/eval_$(date +%Y%m%d_%H%M%S).json"

            # Evaluate on Fujitsu DS
            if [[ -f "$TRACES_DIR/fujitsu_b4_ds.jsonl" ]]; then
                echo "Evaluating on Fujitsu DS..."
                python src/evaluation/eval.py \
                    --baseline "$MODEL" \
                    --cb-adapter "$ADAPTER_PATH" \
                    --eval-data "$TRACES_DIR/fujitsu_b4_ds.jsonl" \
                    --tool-schema "$TOOL_SCHEMA" \
                    --output "$EVAL_OUTPUT" \
                    --limit "$EVAL_LIMIT" \
                    --dtype "$DTYPE"
            fi

            # Evaluate on AgentDojo harmful
            AGENTDOJO_HARMFUL="$OUTPUT_BASE/split/agentdojo/agentdojo_traces_cb.jsonl"
            if [[ -f "$AGENTDOJO_HARMFUL" ]]; then
                AGENTDOJO_EVAL="${EVAL_OUTPUT%.json}_agentdojo.json"
                echo "Evaluating on AgentDojo (harmful)..."
                python src/evaluation/eval.py \
                    --baseline "$MODEL" \
                    --cb-adapter "$ADAPTER_PATH" \
                    --eval-data "$AGENTDOJO_HARMFUL" \
                    --tool-schema "$TOOL_SCHEMA" \
                    --output "$AGENTDOJO_EVAL" \
                    --use-sample-context \
                    --limit "$EVAL_LIMIT" \
                    --dtype "$DTYPE"
            fi

            echo ""
            echo "Evaluation results: $EVAL_OUTPUT"
        fi
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "PIPELINE COMPLETE"
echo "========================================"
echo ""
echo "Output directories:"
echo "  Traces: $TRACES_DIR"
echo "  Renders: $RENDERS_DIR"
echo "  Lossmasks: $LOSSMASKS_DIR"
echo "  Models: $MODELS_DIR"
echo "  Eval: $EVAL_DIR"
echo ""
echo "Finished at: $(date)"
