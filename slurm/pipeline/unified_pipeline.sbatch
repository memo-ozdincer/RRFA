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
# CANONICAL PIPELINE ENTRYPOINT (1/2): REGULAR RUNS
# =============================================================================
#
# This is the only non-sweep pipeline script. It can run full end-to-end or any
# subset of stages via env vars.
#
# Stages:
#   1) ETL_A -> 2) Generate/Judge (optional) -> 3) ETL_B -> 4) Split
#   5) Train -> 6) Eval
#
# -----------------------------------------------------------------------------
# Common run recipes
# -----------------------------------------------------------------------------
# 1) Full default run (all datasets, triplet_full, full logging):
#    sbatch slurm/pipeline/unified_pipeline.sbatch
#
# 2) Train+eval only (reuse existing traces/renders/splits):
#    SKIP_ETL_A=true SKIP_GENERATE=true SKIP_ETL_B=true SKIP_SPLIT=true \
#    sbatch slurm/pipeline/unified_pipeline.sbatch
#
# 3) Data prep only (no train/eval):
#    SKIP_TRAIN=true SKIP_EVAL=true sbatch slurm/pipeline/unified_pipeline.sbatch
#
# 4) Single-dataset run:
#    DATASETS=agentdojo sbatch slurm/pipeline/unified_pipeline.sbatch
#
# 5) Quick smoke run:
#    TOTAL_STEPS=30 EVAL_LIMIT=20 DATASETS=agentdojo \
#    sbatch slurm/pipeline/unified_pipeline.sbatch
#
# 6) Legacy loss mode:
#    LOSS_MODE=legacy_cb LOSS_WEIGHTING=dual \
#    sbatch slurm/pipeline/unified_pipeline.sbatch
#
# -----------------------------------------------------------------------------
# Config knobs (pass with `VAR=value sbatch ...`)
# -----------------------------------------------------------------------------
# Dataset selection:
#   DATASETS, DATASET_CONFIG, FUJITSU_B4_PATH, AGENTDOJO_PATH, LLMAIL_INJECT_PATH
#
# Stage toggles:
#   SKIP_ETL_A, SKIP_GENERATE, SKIP_JUDGE, SKIP_ETL_B, SKIP_SPLIT, SKIP_TRAIN, SKIP_EVAL
#
# Core training:
#   TOTAL_STEPS, BATCH_SIZE, LEARNING_RATE, WARMUP_STEPS, ALPHA_MAX,
#   LOSS_MODE, LOSS_WEIGHTING, CB_LAYERS, LMP_POLICY, MAX_SEQ_LENGTH
#
# Triplet parameters:
#   TRIPLET_ALPHA_BENIGN, TRIPLET_BETA_HARMFUL, TRIPLET_GAMMA_KL,
#   TRIPLET_MARGIN_BENIGN, TRIPLET_MARGIN_HARMFUL,
#   TRIPLET_BENIGN_POS_DISTANCE, TRIPLET_BENIGN_NEG_DISTANCE,
#   TRIPLET_HARMFUL_POS_DISTANCE, TRIPLET_HARMFUL_NEG_DISTANCE,
#   TRIPLET_MIX_L2_WEIGHT, TRIPLET_MIX_COS_WEIGHT
#
# Runtime/logging:
#   OUTPUT_BASE, MODEL, TOKENIZER, DTYPE, LORA_R, LORA_ALPHA,
#   NO_WANDB, JUDGE_PROVIDER, JUDGE_MODEL, EVAL_LIMIT
#   LLMAIL_TOOL_SCHEMA (tool schema used for LLMail generation/eval)
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
DATASETS="${DATASETS:-agentdojo,fujitsu_b4,llmail_inject}"
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
LLMAIL_INJECT_PATH="${LLMAIL_INJECT_PATH:-$REPO_DIR/data/llmail_inject/raw_submissions_phase1.jsonl}"

# Model/tokenizer
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TOKENIZER="${TOKENIZER:-$MODEL}"
TOOL_SCHEMA="${TOOL_SCHEMA:-$REPO_DIR/configs/tool_schemas/b4_standard_v1.json}"
LLMAIL_TOOL_SCHEMA="${LLMAIL_TOOL_SCHEMA:-$REPO_DIR/configs/tool_schemas/llmail_inject_v1.json}"

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
LOSS_MODE="${LOSS_MODE:-triplet_full}"
LOSS_WEIGHTING="${LOSS_WEIGHTING:-dual}"
TRIPLET_ALPHA_BENIGN="${TRIPLET_ALPHA_BENIGN:-0.5}"
TRIPLET_BETA_HARMFUL="${TRIPLET_BETA_HARMFUL:-0.4}"
TRIPLET_GAMMA_KL="${TRIPLET_GAMMA_KL:-0.9}"
TRIPLET_MARGIN_BENIGN="${TRIPLET_MARGIN_BENIGN:-500.0}"
TRIPLET_MARGIN_HARMFUL="${TRIPLET_MARGIN_HARMFUL:-1500.0}"
TRIPLET_BENIGN_POS_DISTANCE="${TRIPLET_BENIGN_POS_DISTANCE:-dmix}"
TRIPLET_BENIGN_NEG_DISTANCE="${TRIPLET_BENIGN_NEG_DISTANCE:-dmix}"
TRIPLET_HARMFUL_POS_DISTANCE="${TRIPLET_HARMFUL_POS_DISTANCE:-dmix}"
TRIPLET_HARMFUL_NEG_DISTANCE="${TRIPLET_HARMFUL_NEG_DISTANCE:-dmix}"
TRIPLET_MIX_L2_WEIGHT="${TRIPLET_MIX_L2_WEIGHT:-0.5}"
TRIPLET_MIX_COS_WEIGHT="${TRIPLET_MIX_COS_WEIGHT:-0.5}"
CB_LAYERS="${CB_LAYERS:-10,20}"
LMP_POLICY="${LMP_POLICY:-assistant_only}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
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
echo "  LLMAIL_INJECT_PATH: $LLMAIL_INJECT_PATH"
echo "  LMP_POLICY: $LMP_POLICY"
echo "  TOTAL_STEPS: $TOTAL_STEPS"
echo "  ALPHA_MAX: $ALPHA_MAX"
echo "  LOSS_MODE: $LOSS_MODE"
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

            llmail_inject)
                if [[ -f "$LLMAIL_INJECT_PATH" ]]; then
                    echo "Processing LLMail-Inject..."
                    OUTPUT_FILE="$TRACES_DIR/llmail_inject_skeleton.jsonl"

                    python src/schemas/tools/ETL_A.py \
                        --llmail-inject "$LLMAIL_INJECT_PATH" \
                        --output "$OUTPUT_FILE" \
                        --split train \
                        --tool-schema "$LLMAIL_TOOL_SCHEMA"

                    echo "  Output: $OUTPUT_FILE ($(wc -l < "$OUTPUT_FILE") lines)"
                else
                    echo "  WARNING: LLMail-Inject file not found: $LLMAIL_INJECT_PATH"
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
    echo "========================================"
    echo "Stage 2: Generate Completions"
    echo "========================================"

    VLLM_ARGS=()
    if [[ "$USE_VLLM" == "true" ]]; then
        VLLM_ARGS+=(--use-vllm)
    fi

    generate_ds_dr_dataset() {
        local dataset_name="$1"
        local skeleton_file="$2"
        local ds_output="$3"
        local dr_output="$4"
        local dataset_tool_schema="$5"

        if [[ ! -f "$skeleton_file" ]]; then
            return
        fi

        local need_ds=true
        local need_dr=true
        [[ -f "$ds_output" ]] && need_ds=false
        [[ -f "$dr_output" ]] && need_dr=false

        if [[ "$need_ds" == "false" && "$need_dr" == "false" ]]; then
            echo "Skipping $dataset_name generation: DS/DR files already exist"
            echo "  DS: $ds_output"
            echo "  DR: $dr_output"
            return
        fi

        if [[ "$need_ds" == "true" ]]; then
            echo "Generating $dataset_name DS traces..."
            python src/data_generation/generate_completions.py \
                --traces "$skeleton_file" \
                --output "$ds_output" \
                --model "$MODEL" \
                --tool-schema "$dataset_tool_schema" \
                --mode ds \
                --temperature-ds "$TEMPERATURE_DS" \
                --max-model-len "$MAX_MODEL_LEN" \
                "${VLLM_ARGS[@]}"
            echo "  DS output: $ds_output ($(wc -l < "$ds_output") lines)"
        fi

        if [[ "$need_dr" == "true" ]]; then
            echo "Generating $dataset_name DR traces..."
            python src/data_generation/generate_completions.py \
                --ds-data "$ds_output" \
                --output "$dr_output" \
                --model "$MODEL" \
                --tool-schema "$dataset_tool_schema" \
                --mode dr \
                --temperature-dr "$TEMPERATURE_DR" \
                --max-model-len "$MAX_MODEL_LEN" \
                "${VLLM_ARGS[@]}"
            echo "  DR output: $dr_output ($(wc -l < "$dr_output") lines)"
        fi
    }

    generate_ds_dr_dataset \
        "Fujitsu B4" \
        "$TRACES_DIR/fujitsu_b4_skeletons.jsonl" \
        "$TRACES_DIR/fujitsu_b4_ds.jsonl" \
        "$TRACES_DIR/fujitsu_b4_dr.jsonl" \
        "$TOOL_SCHEMA"

    generate_ds_dr_dataset \
        "LLMail-Inject" \
        "$TRACES_DIR/llmail_inject_skeleton.jsonl" \
        "$TRACES_DIR/llmail_inject_ds.jsonl" \
        "$TRACES_DIR/llmail_inject_dr.jsonl" \
        "$LLMAIL_TOOL_SCHEMA"

    echo ""
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

            llmail_inject)
                echo "LLMail-Inject: Using existing DS/DR split"
                echo "  DS (CB): $TRACES_DIR/llmail_inject_ds.jsonl"
                echo "  DR (Retain): $TRACES_DIR/llmail_inject_dr.jsonl"
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

    # LLMail DS (harmful)
    if [[ -f "$RENDERS_DIR/llmail_inject_ds.jsonl" ]]; then
        if [[ -n "$HARMFUL_RENDERS" ]]; then
            HARMFUL_RENDERS="$HARMFUL_RENDERS,$RENDERS_DIR/llmail_inject_ds.jsonl"
            HARMFUL_LOSSMASKS="$HARMFUL_LOSSMASKS,$LOSSMASKS_DIR/llmail_inject_ds.jsonl"
        else
            HARMFUL_RENDERS="$RENDERS_DIR/llmail_inject_ds.jsonl"
            HARMFUL_LOSSMASKS="$LOSSMASKS_DIR/llmail_inject_ds.jsonl"
        fi
    fi

    # LLMail DR (benign)
    if [[ -f "$RENDERS_DIR/llmail_inject_dr.jsonl" ]]; then
        if [[ -n "$BENIGN_RENDERS" ]]; then
            BENIGN_RENDERS="$BENIGN_RENDERS,$RENDERS_DIR/llmail_inject_dr.jsonl"
            BENIGN_LOSSMASKS="$BENIGN_LOSSMASKS,$LOSSMASKS_DIR/llmail_inject_dr.jsonl"
        else
            BENIGN_RENDERS="$RENDERS_DIR/llmail_inject_dr.jsonl"
            BENIGN_LOSSMASKS="$LOSSMASKS_DIR/llmail_inject_dr.jsonl"
        fi
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
        --loss-mode "$LOSS_MODE"
        --loss-weighting "$LOSS_WEIGHTING"
        --triplet-alpha-benign "$TRIPLET_ALPHA_BENIGN"
        --triplet-beta-harmful "$TRIPLET_BETA_HARMFUL"
        --triplet-gamma-kl "$TRIPLET_GAMMA_KL"
        --triplet-margin-benign "$TRIPLET_MARGIN_BENIGN"
        --triplet-margin-harmful "$TRIPLET_MARGIN_HARMFUL"
        --triplet-benign-positive-distance "$TRIPLET_BENIGN_POS_DISTANCE"
        --triplet-benign-negative-distance "$TRIPLET_BENIGN_NEG_DISTANCE"
        --triplet-harmful-positive-distance "$TRIPLET_HARMFUL_POS_DISTANCE"
        --triplet-harmful-negative-distance "$TRIPLET_HARMFUL_NEG_DISTANCE"
        --triplet-mix-l2-weight "$TRIPLET_MIX_L2_WEIGHT"
        --triplet-mix-cos-weight "$TRIPLET_MIX_COS_WEIGHT"
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

            # Evaluate on LLMail DS
            if [[ -f "$TRACES_DIR/llmail_inject_ds.jsonl" ]]; then
                LLMAIL_EVAL="${EVAL_OUTPUT%.json}_llmail.json"
                echo "Evaluating on LLMail-Inject DS..."
                python src/evaluation/eval.py \
                    --baseline "$MODEL" \
                    --cb-adapter "$ADAPTER_PATH" \
                    --eval-data "$TRACES_DIR/llmail_inject_ds.jsonl" \
                    --tool-schema "$LLMAIL_TOOL_SCHEMA" \
                    --output "$LLMAIL_EVAL" \
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
