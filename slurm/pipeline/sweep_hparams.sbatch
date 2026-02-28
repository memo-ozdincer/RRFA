#!/bin/bash
#SBATCH --job-name=sweep_hparams
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=09:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# CANONICAL PIPELINE ENTRYPOINT (2/2): HPARAM SWEEP
# =============================================================================
#
# This is the only sweep script. It sweeps:
#   - `POLICIES` (lossmask policy)
#   - `LAYER_CONFIGS` (CB target layers)
#   - `ALPHAS` (alpha_max for legacy schedules)
#
# For each configuration: ETL_B (policy) -> train -> eval.
# Datasets: Fujitsu B4 + AgentDojo (+ optional AgentHarm).
#
# -----------------------------------------------------------------------------
# Common run recipes
# -----------------------------------------------------------------------------
# 1) Full sweep:
#    sbatch slurm/pipeline/sweep_hparams.sbatch
#
# 2) Quick sanity sweep:
#    QUICK_TEST=true sbatch slurm/pipeline/sweep_hparams.sbatch
#
# 3) One configuration only:
#    ALPHAS=10.0 LAYER_CONFIGS=10,20 POLICIES=assistant_only \
#    sbatch slurm/pipeline/sweep_hparams.sbatch
#
# 4) Policy-only sweep (fixed layers/alpha):
#    ALPHAS=10.0 LAYER_CONFIGS=10,20 \
#    POLICIES=assistant_only,assistant_and_tool,cb_full_sequence,tool_calls_only,completion_only \
#    sbatch slurm/pipeline/sweep_hparams.sbatch
#
# 5) Legacy loss sweep:
#    LOSS_MODE=legacy_cb LOSS_WEIGHTING=dual \
#    sbatch slurm/pipeline/sweep_hparams.sbatch
#
# -----------------------------------------------------------------------------
# Config knobs (pass with `VAR=value sbatch ...`)
# -----------------------------------------------------------------------------
# Sweep controls:
#   QUICK_TEST, TOTAL_STEPS, ALPHAS, LAYER_CONFIGS, POLICIES
#
# Eval/logging:
#   LIMIT_EVAL, NUM_EXAMPLES, SHOW_EXAMPLES
#
# Core training:
#   LOSS_MODE, LOSS_WEIGHTING, BATCH_SIZE, LEARNING_RATE, WARMUP_STEPS,
#   MAX_SEQ_LENGTH, MAX_MODEL_LEN, DTYPE, LORA_R, LORA_ALPHA
#
# Triplet parameters:
#   TRIPLET_ALPHA_BENIGN, TRIPLET_BETA_HARMFUL, TRIPLET_GAMMA_KL,
#   TRIPLET_MARGIN_BENIGN, TRIPLET_MARGIN_HARMFUL,
#   TRIPLET_BENIGN_POS_DISTANCE, TRIPLET_BENIGN_NEG_DISTANCE,
#   TRIPLET_HARMFUL_POS_DISTANCE, TRIPLET_HARMFUL_NEG_DISTANCE,
#   TRIPLET_MIX_L2_WEIGHT, TRIPLET_MIX_COS_WEIGHT
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

# =============================================================================
# Cache Setup
# =============================================================================
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,wandb}

export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export WANDB_DIR="$CACHE_DIR/wandb"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export XDG_CACHE_HOME="$CACHE_DIR"
export DS_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
mkdir -p "$TRITON_CACHE_DIR" "$TRITON_CACHE_DIR/autotune"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================"
echo "COMPREHENSIVE HYPERPARAMETER SWEEP"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo ""

# =============================================================================
# Configuration
# =============================================================================
QUICK_TEST="${QUICK_TEST:-false}"

if [[ "$QUICK_TEST" == "true" ]]; then
    echo "🔧 QUICK TEST MODE - Reduced configurations"
    DEFAULT_STEPS=50
    DEFAULT_ALPHAS="5.0,10.0"
    DEFAULT_LAYER_CONFIGS="10,20"
    DEFAULT_POLICIES="assistant_only"
    DEFAULT_LIMIT_EVAL=20
    DEFAULT_NUM_EXAMPLES=5
else
    DEFAULT_STEPS=200
    DEFAULT_ALPHAS="5.0,10.0,15.0"
    DEFAULT_LAYER_CONFIGS="10,20|8,16,24|12,18|6,12,18,24"
    DEFAULT_POLICIES="assistant_only,assistant_and_tool,cb_full_sequence,tool_calls_only,completion_only"
    DEFAULT_LIMIT_EVAL=50
    DEFAULT_NUM_EXAMPLES=50
fi

TOTAL_STEPS="${TOTAL_STEPS:-$DEFAULT_STEPS}"
ALPHAS="${ALPHAS:-$DEFAULT_ALPHAS}"
LAYER_CONFIGS="${LAYER_CONFIGS:-$DEFAULT_LAYER_CONFIGS}"
POLICIES="${POLICIES:-$DEFAULT_POLICIES}"
LIMIT_EVAL="${LIMIT_EVAL:-$DEFAULT_LIMIT_EVAL}"
NUM_EXAMPLES="${NUM_EXAMPLES:-$DEFAULT_NUM_EXAMPLES}"
SHOW_EXAMPLES="${SHOW_EXAMPLES:-true}"

# Training params (configurable via env vars, safe defaults for single GPU)
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DTYPE="${DTYPE:-bfloat16}"
LOSS_MODE="${LOSS_MODE:-triplet_full}"
LOSS_WEIGHTING="${LOSS_WEIGHTING:-dual}"
TRIPLET_ALPHA_BENIGN="${TRIPLET_ALPHA_BENIGN:-0.5}"
TRIPLET_BETA_HARMFUL="${TRIPLET_BETA_HARMFUL:-0.4}"
TRIPLET_GAMMA_KL="${TRIPLET_GAMMA_KL:-0.9}"
TRIPLET_MIX_L2_WEIGHT="${TRIPLET_MIX_L2_WEIGHT:-0.5}"
TRIPLET_MIX_COS_WEIGHT="${TRIPLET_MIX_COS_WEIGHT:-0.5}"

# Pooling presets: each is "name:pooling_mode:distance:margin_b:margin_h"
# Multiple presets separated by spaces. Swept as innermost dimension.
# Use POOLING_PRESETS env var to customize, or these defaults:
POOLING_PRESETS="${POOLING_PRESETS:-legacy:legacy:dmix:500:1500}"
STOP_AFTER_ETL_B="${STOP_AFTER_ETL_B:-false}"
STOP_AFTER_SPLIT="${STOP_AFTER_SPLIT:-false}"

echo "Configuration:"
echo "  TOTAL_STEPS: $TOTAL_STEPS"
echo "  ALPHAS: $ALPHAS"
echo "  LAYER_CONFIGS: $LAYER_CONFIGS"
echo "  POLICIES: $POLICIES"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  GRAD_ACCUM: $GRAD_ACCUM"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  LOSS_MODE: $LOSS_MODE"
echo "  LIMIT_EVAL: $LIMIT_EVAL"
echo "  NUM_EXAMPLES: $NUM_EXAMPLES"
echo "  SHOW_EXAMPLES: $SHOW_EXAMPLES"
echo "  QUICK_TEST: $QUICK_TEST"
echo "  POOLING_PRESETS: $POOLING_PRESETS"
echo "  STOP_AFTER_ETL_B: $STOP_AFTER_ETL_B"
echo "  STOP_AFTER_SPLIT: $STOP_AFTER_SPLIT"
echo ""

# =============================================================================
# Input Traces (ALL available data)
# =============================================================================
TRACES_DIR="$CB_SCRATCH/data/traces"
DS_TRACES="$TRACES_DIR/fujitsu_b4_ds.jsonl"
DR_TRACES="$TRACES_DIR/fujitsu_b4_dr.jsonl"
AGENTDOJO_TRACES="$TRACES_DIR/agentdojo_complete.jsonl"
AGENTHARM_TRACES="$TRACES_DIR/agentharm_mixed.jsonl"
echo "Input traces (ALL used for training):"
for trace in "$DS_TRACES" "$DR_TRACES"; do
    if [[ -f "$trace" ]]; then
        echo "  ✓ $(basename "$trace"): $(wc -l < "$trace") lines"
    else
        echo "  ✗ $(basename "$trace"): NOT FOUND"
        exit 1
    fi
done

# AgentDojo: regenerate via ETL_A if missing or suspiciously small
AGENTDOJO_RAW="$REPO_DIR/data/agent_dojo/agentdojo_spotlight_extract.jsonl"
AGENTDOJO_MIN_LINES=100
if [[ -f "$AGENTDOJO_RAW" ]]; then
    regen=false
    if [[ ! -f "$AGENTDOJO_TRACES" ]]; then
        regen=true
        echo "  AgentDojo traces not found. Regenerating via ETL_A..."
    else
        n_lines=$(wc -l < "$AGENTDOJO_TRACES")
        if [[ "$n_lines" -lt "$AGENTDOJO_MIN_LINES" ]]; then
            regen=true
            echo "  AgentDojo traces too small ($n_lines < $AGENTDOJO_MIN_LINES). Regenerating via ETL_A..."
        fi
    fi
    if [[ "$regen" == "true" ]]; then
        python src/schemas/tools/ETL_A.py \
            --agentdojo "$AGENTDOJO_RAW" \
            --output "$AGENTDOJO_TRACES" \
            --split train
        echo "  ✓ AgentDojo regenerated: $(wc -l < "$AGENTDOJO_TRACES") traces"
    else
        echo "  ✓ $(basename "$AGENTDOJO_TRACES"): $(wc -l < "$AGENTDOJO_TRACES") lines"
    fi
else
    echo "  ✗ AgentDojo raw file not found at $AGENTDOJO_RAW"
    exit 1
fi

HAS_AGENTHARM=false
if [[ -f "$AGENTHARM_TRACES" ]]; then
    HAS_AGENTHARM=true
    echo "  ✓ $(basename "$AGENTHARM_TRACES"): $(wc -l < "$AGENTHARM_TRACES") lines"
    echo "  AgentHarm: enabled"
else
    echo "  AgentHarm: not found (optional, continuing without it)"
fi

echo ""

# =============================================================================
# Resolve Tokenizer / Model
# =============================================================================
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
PRESET="${PRESET:-llama-3.1-8b-instruct}"
TOKENIZER="$MODEL_ID"
echo "Resolving model '$MODEL_ID' to local cache path..."

# Convert model ID to HF cache directory format (org--model)
MODEL_CACHE_NAME=$(echo "$MODEL_ID" | sed 's/\//--/g')
MODEL_CACHE_DIR="$HF_HUB_CACHE/models--${MODEL_CACHE_NAME}"
SNAPSHOT_ROOT="$MODEL_CACHE_DIR/snapshots"

if [[ -d "$SNAPSHOT_ROOT" ]]; then
    SNAPSHOT_PATH=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1)
    if [[ -n "$SNAPSHOT_PATH" ]]; then
        echo "  Found local snapshot: $SNAPSHOT_PATH"
        TOKENIZER="$SNAPSHOT_PATH"
    else
        echo "  ERROR: Snapshot directory exists but is empty: $SNAPSHOT_ROOT"
        echo "  Run slurm/cache_models.sh on login node first!"
        exit 1
    fi
else
    echo "  ERROR: Model not cached at: $MODEL_CACHE_DIR"
    echo "  Run slurm/cache_models.sh on login node first!"
    exit 1
fi
echo ""

# =============================================================================
# Create Sweep Directory
# =============================================================================
SWEEP_ID="hparam_sweep_$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="$CB_SCRATCH/sweeps/$SWEEP_ID"
mkdir -p "$SWEEP_DIR"

echo "Sweep output: $SWEEP_DIR"
echo ""

# =============================================================================
# Count Total Runs
# =============================================================================
count_configs() {
    local count=0
    IFS=',' read -ra ALPHA_ARRAY <<< "$ALPHAS"
    IFS='|' read -ra LAYER_ARRAY <<< "$LAYER_CONFIGS"
    IFS=',' read -ra POLICY_ARRAY <<< "$POLICIES"
    read -ra PRESET_ARRAY <<< "$POOLING_PRESETS"

    for alpha in "${ALPHA_ARRAY[@]}"; do
        for layers in "${LAYER_ARRAY[@]}"; do
            for policy in "${POLICY_ARRAY[@]}"; do
                for _p in "${PRESET_ARRAY[@]}"; do
                    count=$((count + 1))
                done
            done
        done
    done
    echo "$count"
}

TOTAL_RUNS=$(count_configs)
echo "========================================"
echo "TOTAL CONFIGURATIONS TO TEST: $TOTAL_RUNS"
echo "========================================"
echo ""

# =============================================================================
# Main Sweep Loop
# =============================================================================
run_idx=0
success_count=0
fail_count=0

IFS=',' read -ra ALPHA_ARRAY <<< "$ALPHAS"
IFS='|' read -ra LAYER_ARRAY <<< "$LAYER_CONFIGS"
IFS=',' read -ra POLICY_ARRAY <<< "$POLICIES"
read -ra PRESET_ARRAY <<< "$POOLING_PRESETS"

for policy in "${POLICY_ARRAY[@]}"; do
    for layers in "${LAYER_ARRAY[@]}"; do
        for alpha in "${ALPHA_ARRAY[@]}"; do
            # Create base name (without preset) for ETL_B sharing
            layers_short="${layers//,/_}"
            BASE_NAME="a${alpha}_l${layers_short}_${policy}"
            BASE_DIR="$SWEEP_DIR/$BASE_NAME"

            # For single-preset runs, use flat naming; multi-preset sets RUN_NAME in preset loop
            if [[ ${#PRESET_ARRAY[@]} -eq 1 ]]; then
                RUN_NAME="$BASE_NAME"
            else
                RUN_NAME="$BASE_NAME"  # placeholder, overridden per preset below
            fi

            run_idx=$((run_idx + 1))
            RUN_DIR="$BASE_DIR"

            echo "========================================"
            echo "[$run_idx/$TOTAL_RUNS] $RUN_NAME"
            echo "========================================"
            echo "  Alpha: $alpha"
            echo "  Layers: $layers"
            echo "  Policy: $policy"
            echo "  Output: $RUN_DIR"
            echo ""
            
            mkdir -p "$RUN_DIR"/{renders,lossmasks,model,eval}
            
            # =====================================================
            # Step 1: ETL_B with this policy
            # =====================================================
            echo "Step 1: Creating lossmasks with policy=$policy"
            
            etl_start=$(date +%s)
            TRACE_FILES=("$DS_TRACES" "$DR_TRACES" "$AGENTDOJO_TRACES")
            if [[ "$HAS_AGENTHARM" == "true" ]]; then
                TRACE_FILES+=("$AGENTHARM_TRACES")
            fi

            for trace_file in "${TRACE_FILES[@]}"; do
                basename=$(basename "$trace_file" .jsonl)
                render_out="$RUN_DIR/renders/${basename}.jsonl"
                lossmask_out="$RUN_DIR/lossmasks/${basename}.jsonl"

                ETL_B_LIMIT_ARGS=()
                ETL_B_EXTRA_ARGS=()
                if [[ "$basename" == agentharm_* ]]; then
                    # AgentHarm traces are prompt-only skeletons.
                    ETL_B_EXTRA_ARGS+=(--allow-skeleton)
                fi

                echo "  Processing: $basename"
                python src/schemas/tools/ETL_B.py \
                    --traces "$trace_file" \
                    --render-out "$render_out" \
                    --lossmask-out "$lossmask_out" \
                    --tokenizer "$TOKENIZER" \
                    --max-length 4096 \
                    --policy-override "$policy" \
                    "${ETL_B_LIMIT_ARGS[@]}" \
                    "${ETL_B_EXTRA_ARGS[@]}" 2>&1 | tee -a "$RUN_DIR/etl_b.log"
            done
            etl_end=$(date +%s)
            echo "  ETL_B completed in $((etl_end - etl_start))s"
            echo ""

            if [[ "$STOP_AFTER_ETL_B" == "true" ]]; then
                echo "Checkpoint reached: STOP_AFTER_ETL_B=true. Exiting after ETL_B for run $RUN_NAME."
                exit 0
            fi
            
            # Verify lossmasks are non-zero
            echo "  Verifying lossmasks..."
            for lm in "$RUN_DIR/lossmasks"/*.jsonl; do
                nonzero=$(head -5 "$lm" | python3 -c "
import json, sys
total = 0
for line in sys.stdin:
    if line.strip():
        d = json.loads(line)
        mask = d.get('loss_mask', [])
        total += sum(1 for x in mask if x > 0)
print(total)
" 2>/dev/null || echo "0")
                echo "    $(basename "$lm"): $nonzero non-zero mask tokens (first 5 samples)"
            done
            echo ""
            
            # =====================================================
            # Step 2: Split AgentDojo (same as sweep_policies.sbatch)
            # =====================================================
            echo "Step 2: Splitting AgentDojo data"
            
            SPLIT_DIR="$RUN_DIR/agentdojo_split"
            mkdir -p "$SPLIT_DIR"
            
            # Use legacy naming for backward compatibility with existing training args
            python "$REPO_DIR/scripts/split_agentdojo.py" \
                --traces "$AGENTDOJO_TRACES" \
                --renders "$RUN_DIR/renders/agentdojo_complete.jsonl" \
                --lossmasks "$RUN_DIR/lossmasks/agentdojo_complete.jsonl" \
                --output-dir "$SPLIT_DIR" \
                --legacy-harmful-naming 2>&1 | tee -a "$RUN_DIR/split.log"

            AGENTHARM_SPLIT_DIR="$RUN_DIR/agentharm_split"
            if [[ "$HAS_AGENTHARM" == "true" ]]; then
                mkdir -p "$AGENTHARM_SPLIT_DIR"
                echo "Step 2b: Splitting AgentHarm data"
                python "$REPO_DIR/scripts/split_dataset.py" \
                    --traces "$AGENTHARM_TRACES" \
                    --renders "$RUN_DIR/renders/agentharm_mixed.jsonl" \
                    --lossmasks "$RUN_DIR/lossmasks/agentharm_mixed.jsonl" \
                    --output-dir "$AGENTHARM_SPLIT_DIR" \
                    --prefix "agentharm" \
                    --legacy-naming \
                    --resisted-handling retain 2>&1 | tee -a "$RUN_DIR/split.log"
            fi
            
            echo ""

            if [[ "$STOP_AFTER_SPLIT" == "true" ]]; then
                echo "Checkpoint reached: STOP_AFTER_SPLIT=true. Exiting after split for run $RUN_NAME."
                exit 0
            fi
            
            # Parse layers to array (shared by all presets)
            IFS=',' read -ra CB_LAYER_ARRAY <<< "$layers"

            for preset_str in "${PRESET_ARRAY[@]}"; do
            # Parse preset: name:pooling_mode:distance:margin_b:margin_h
            IFS=':' read -r _PNAME _PMODE _PDIST _PMB _PMH <<< "$preset_str"

            # Adjust run dir for multi-preset sweeps
            if [[ ${#PRESET_ARRAY[@]} -gt 1 ]]; then
                RUN_NAME="${_PNAME}_a${alpha}_l${layers_short}_${policy}"
                RUN_DIR="$SWEEP_DIR/$RUN_NAME"
                mkdir -p "$RUN_DIR"/{model,eval}
            fi

            # =====================================================
            # Save run configuration
            # =====================================================
            cat > "$RUN_DIR/run_config.json" <<RUNCFG
{
  "preset": "$_PNAME",
  "pooling_mode": "$_PMODE",
  "distance": "$_PDIST",
  "margin_benign": $_PMB,
  "margin_harmful": $_PMH,
  "alpha": $alpha,
  "layers": "$layers",
  "policy": "$policy",
  "total_steps": $TOTAL_STEPS,
  "batch_size": $BATCH_SIZE,
  "grad_accum": $GRAD_ACCUM,
  "learning_rate": "$LEARNING_RATE",
  "loss_mode": "$LOSS_MODE",
  "loss_weighting": "$LOSS_WEIGHTING",
  "triplet_alpha_benign": $TRIPLET_ALPHA_BENIGN,
  "triplet_beta_harmful": $TRIPLET_BETA_HARMFUL,
  "triplet_gamma_kl": $TRIPLET_GAMMA_KL,
  "lora_r": $LORA_R,
  "lora_alpha": $LORA_ALPHA,
  "model": "$TOKENIZER",
  "run_name": "$RUN_NAME",
  "run_dir": "$RUN_DIR",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "job_id": "${SLURM_JOB_ID:-local}"
}
RUNCFG

            # =====================================================
            # Step 3: Training
            # =====================================================
            echo "Step 3: Training with alpha=$alpha, layers=$layers, policy=$policy, preset=$_PNAME (${_PMODE}/${_PDIST}/${_PMB}/${_PMH})"

            HARMFUL_RENDER_ARGS=(
                "$RUN_DIR/renders/fujitsu_b4_ds.jsonl"
                "$SPLIT_DIR/agentdojo_renders_harmful.jsonl"
            )
            HARMFUL_LOSSMASK_ARGS=(
                "$RUN_DIR/lossmasks/fujitsu_b4_ds.jsonl"
                "$SPLIT_DIR/agentdojo_lossmasks_harmful.jsonl"
            )
            BENIGN_RENDER_ARGS=(
                "$RUN_DIR/renders/fujitsu_b4_dr.jsonl"
                "$SPLIT_DIR/agentdojo_renders_benign.jsonl"
            )
            BENIGN_LOSSMASK_ARGS=(
                "$RUN_DIR/lossmasks/fujitsu_b4_dr.jsonl"
                "$SPLIT_DIR/agentdojo_lossmasks_benign.jsonl"
            )

            if [[ "$HAS_AGENTHARM" == "true" ]]; then
                if [[ -f "$AGENTHARM_SPLIT_DIR/agentharm_renders_harmful.jsonl" ]]; then
                    HARMFUL_RENDER_ARGS+=("$AGENTHARM_SPLIT_DIR/agentharm_renders_harmful.jsonl")
                    HARMFUL_LOSSMASK_ARGS+=("$AGENTHARM_SPLIT_DIR/agentharm_lossmasks_harmful.jsonl")
                fi
                if [[ -f "$AGENTHARM_SPLIT_DIR/agentharm_renders_benign.jsonl" ]]; then
                    BENIGN_RENDER_ARGS+=("$AGENTHARM_SPLIT_DIR/agentharm_renders_benign.jsonl")
                    BENIGN_LOSSMASK_ARGS+=("$AGENTHARM_SPLIT_DIR/agentharm_lossmasks_benign.jsonl")
                fi
            fi
            
            # Build training args (matching sweep_policies.sbatch exactly)
            TRAIN_ARGS=(
                --preset "$PRESET"
                --model "$TOKENIZER"
                --output-dir "$RUN_DIR/model"
                --total-steps "$TOTAL_STEPS"
                --batch-size "$BATCH_SIZE"
                --learning-rate "$LEARNING_RATE"
                --warmup-steps "$WARMUP_STEPS"
                --alpha-max "$alpha"
                --max-seq-length "$MAX_SEQ_LENGTH"
                --gradient-accumulation-steps "$GRAD_ACCUM"
                --loss-mode "$LOSS_MODE"
                --loss-weighting "$LOSS_WEIGHTING"
                --triplet-alpha-benign "$TRIPLET_ALPHA_BENIGN"
                --triplet-beta-harmful "$TRIPLET_BETA_HARMFUL"
                --triplet-gamma-kl "$TRIPLET_GAMMA_KL"
                --triplet-margin-benign "$_PMB"
                --triplet-margin-harmful "$_PMH"
                --triplet-benign-positive-distance "$_PDIST"
                --triplet-benign-negative-distance "$_PDIST"
                --triplet-harmful-positive-distance "$_PDIST"
                --triplet-harmful-negative-distance "$_PDIST"
                --triplet-mix-l2-weight "$TRIPLET_MIX_L2_WEIGHT"
                --triplet-mix-cos-weight "$TRIPLET_MIX_COS_WEIGHT"
                --pooling-mode "$_PMODE"
                --lora-r "$LORA_R"
                --lora-alpha "$LORA_ALPHA"
                --lora-dropout 0.05
                --logging-steps 10
                --save-steps 50
                --no-wandb
                --cb-target-layers "${CB_LAYER_ARRAY[@]}"
                --mode mixed
                --harmful-renders "${HARMFUL_RENDER_ARGS[@]}"
                --harmful-lossmasks "${HARMFUL_LOSSMASK_ARGS[@]}"
                --benign-renders "${BENIGN_RENDER_ARGS[@]}"
                --benign-lossmasks "${BENIGN_LOSSMASK_ARGS[@]}"
            )
            
            train_start=$(date +%s)
            if python src/training/train_schema.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$RUN_DIR/train.log"; then
                train_status="SUCCESS"
            else
                train_status="FAILED"
            fi
            train_end=$(date +%s)
            echo "  Training $train_status in $((train_end - train_start))s"
            echo ""
            
            # =====================================================
            # Step 4a: Checkpoint Evaluation (intermediate checkpoints)
            # =====================================================
            if [[ "$train_status" == "SUCCESS" ]]; then
                for ckpt_path in "$RUN_DIR/model/checkpoint-"*; do
                    [[ -d "$ckpt_path" ]] || continue
                    ckpt_name=$(basename "$ckpt_path")
                    CKPT_EVAL_DIR="$RUN_DIR/eval/checkpoints/$ckpt_name"
                    mkdir -p "$CKPT_EVAL_DIR"
                    echo "  Evaluating $ckpt_name..."

                    # Fujitsu checkpoint eval
                    python src/evaluation/eval.py \
                        --baseline "$TOKENIZER" \
                        --cb-adapter "$ckpt_path" \
                        --eval-data "$DS_TRACES" \
                        --tool-schema "$REPO_DIR/configs/tool_schemas/b4_standard_v1.json" \
                        --output "$CKPT_EVAL_DIR/fujitsu_eval.json" \
                        --limit "$LIMIT_EVAL" \
                        --merge-adapter \
                        --dtype "$DTYPE" 2>&1 | tee -a "$CKPT_EVAL_DIR/fujitsu.log"

                    # AgentDojo harmful checkpoint eval
                    if [[ -f "$SPLIT_DIR/agentdojo_traces_harmful.jsonl" ]]; then
                        python src/evaluation/eval.py \
                            --baseline "$TOKENIZER" \
                            --cb-adapter "$ckpt_path" \
                            --eval-data "$SPLIT_DIR/agentdojo_traces_harmful.jsonl" \
                            --tool-schema "$REPO_DIR/configs/tool_schemas/b4_standard_v1.json" \
                            --output "$CKPT_EVAL_DIR/agentdojo_eval.json" \
                            --use-sample-context \
                            --limit "$LIMIT_EVAL" \
                            --merge-adapter \
                            --dtype "$DTYPE" 2>&1 | tee -a "$CKPT_EVAL_DIR/agentdojo.log"
                    fi

                    # AgentDojo benign checkpoint eval
                    if [[ -f "$SPLIT_DIR/agentdojo_traces_benign.jsonl" ]]; then
                        python src/evaluation/eval.py \
                            --baseline "$TOKENIZER" \
                            --cb-adapter "$ckpt_path" \
                            --eval-data "$SPLIT_DIR/agentdojo_traces_benign.jsonl" \
                            --tool-schema "$REPO_DIR/configs/tool_schemas/b4_standard_v1.json" \
                            --output "$CKPT_EVAL_DIR/agentdojo_benign_eval.json" \
                            --use-sample-context \
                            --limit "$LIMIT_EVAL" \
                            --merge-adapter \
                            --dtype "$DTYPE" 2>&1 | tee -a "$CKPT_EVAL_DIR/agentdojo_benign.log"
                    fi

                    echo "  $ckpt_name eval complete."
                done
            fi

            # =====================================================
            # Step 4b: Final Evaluation with Examples
            # =====================================================
            if [[ "$train_status" == "SUCCESS" ]]; then
                echo "Step 4b: Evaluating final model"

                # Find trained model
                if [[ -d "$RUN_DIR/model/final" ]]; then
                    ADAPTER_PATH="$RUN_DIR/model/final"
                elif [[ -f "$RUN_DIR/model/adapter_config.json" ]]; then
                    ADAPTER_PATH="$RUN_DIR/model"
                else
                    echo "  WARNING: No adapter found, skipping evaluation"
                    ADAPTER_PATH=""
                fi
                
                if [[ -n "$ADAPTER_PATH" ]]; then
                    # ---------------------------------------------------------
                    # Evaluate on Fujitsu
                    # ---------------------------------------------------------
                    echo "  Evaluating on Fujitsu ($LIMIT_EVAL samples)..."
                    FUJITSU_OUTPUT="$RUN_DIR/eval/fujitsu_eval.json"
                    
                    python src/evaluation/eval.py \
                        --baseline "$TOKENIZER" \
                        --cb-adapter "$ADAPTER_PATH" \
                        --eval-data "$DS_TRACES" \
                        --tool-schema "$REPO_DIR/configs/tool_schemas/b4_standard_v1.json" \
                        --output "$FUJITSU_OUTPUT" \
                        --limit "$LIMIT_EVAL" \
                        --merge-adapter \
                        --dtype "$DTYPE" 2>&1 | tee -a "$RUN_DIR/eval/fujitsu.log"
                    
                    # Print Fujitsu examples
                    if [[ "$SHOW_EXAMPLES" == "true" ]]; then
                        PAIRED_FILE="${FUJITSU_OUTPUT%.json}.paired_outputs.jsonl"
                        if [[ -f "$PAIRED_FILE" ]]; then
                            echo ""
                            echo "    ========================================"
                            echo "    FUJITSU EXAMPLES: $RUN_NAME"
                            echo "    ========================================"
                            python3 << PYEOF
import json

with open('$PAIRED_FILE', 'r') as f:
    pairs = [json.loads(line) for line in f if line.strip()]

if not pairs:
    print('    No paired outputs')
    exit(0)

def is_gibberish(resp):
    if not resp: return False
    words = resp.split()
    return len(words) > 10 and len(set(words)) / len(words) < 0.15

improvements = [p for p in pairs if p.get('baseline_outcome') == 'attack_success' and p.get('cb_outcome') != 'attack_success']
regressions = [p for p in pairs if p.get('baseline_outcome') != 'attack_success' and p.get('cb_outcome') == 'attack_success']
n_gib = sum(1 for p in pairs if is_gibberish(p.get('cb_response', '')))
n_correct = sum(1 for p in pairs if p.get('cb_outcome') == 'correct_behavior')
n_no_tool = sum(1 for p in pairs if p.get('cb_outcome') == 'no_tool_call')
n_attack = sum(1 for p in pairs if p.get('cb_outcome') == 'attack_success')

print(f'    Pairs: {len(pairs)} | Improved: {len(improvements)} | Regressed: {len(regressions)}')
print(f'    CB outcomes: correct={n_correct} attack={n_attack} no_tool={n_no_tool} gibberish={n_gib}')
print()

for i, p in enumerate((improvements + regressions)[:$NUM_EXAMPLES], 1):
    b_out = p.get('baseline_outcome', '?')
    c_out = p.get('cb_outcome', '?')
    label = 'IMP' if b_out == 'attack_success' else 'REG'
    gib = ' [GIB]' if is_gibberish(p.get('cb_response', '')) else ''
    print(f'    [{i}] {label}: {b_out} -> {c_out}{gib}')
    print(f'        expected={p.get("expected_tool","?")} simulated={p.get("simulated_tool","?")}')
    print(f'    --- BASELINE ---')
    print((p.get('baseline_response') or '(empty)')[:300])
    print(f'    --- CB MODEL ---')
    print((p.get('cb_response') or '(empty)')[:300])
    print()
PYEOF
                        fi
                    fi

                    # ---------------------------------------------------------
                    # Evaluate on AgentDojo (harmful subset)
                    # ---------------------------------------------------------
                    echo "  Evaluating on AgentDojo ($LIMIT_EVAL samples)..."
                    AGENTDOJO_OUTPUT="$RUN_DIR/eval/agentdojo_eval.json"
                    
                    if [[ -f "$SPLIT_DIR/agentdojo_traces_harmful.jsonl" ]]; then
                        python src/evaluation/eval.py \
                            --baseline "$TOKENIZER" \
                            --cb-adapter "$ADAPTER_PATH" \
                            --eval-data "$SPLIT_DIR/agentdojo_traces_harmful.jsonl" \
                            --tool-schema "$REPO_DIR/configs/tool_schemas/b4_standard_v1.json" \
                            --output "$AGENTDOJO_OUTPUT" \
                            --use-sample-context \
                            --limit "$LIMIT_EVAL" \
                            --merge-adapter \
                            --dtype "$DTYPE" 2>&1 | tee -a "$RUN_DIR/eval/agentdojo.log"
                        
                        # Print AgentDojo examples
                        if [[ "$SHOW_EXAMPLES" == "true" ]]; then
                            PAIRED_FILE="${AGENTDOJO_OUTPUT%.json}.paired_outputs.jsonl"
                            if [[ -f "$PAIRED_FILE" ]]; then
                                echo ""
                                echo "    ========================================"
                                echo "    AGENTDOJO EXAMPLES: $RUN_NAME"
                                echo "    ========================================"
                                python3 << PYEOF
import json

with open('$PAIRED_FILE', 'r') as f:
    pairs = [json.loads(line) for line in f if line.strip()]

if not pairs:
    print('    No paired outputs')
    exit(0)

def is_gibberish(resp):
    if not resp: return False
    words = resp.split()
    return len(words) > 10 and len(set(words)) / len(words) < 0.15

differ = [p for p in pairs if p.get('responses_differ')]
same = [p for p in pairs if not p.get('responses_differ')]
n_gib = sum(1 for p in pairs if is_gibberish(p.get('cb_response', '')))
harmful = [p for p in pairs if p.get('category') == 'harmful']
n_refused = sum(1 for p in harmful if not p.get('cb_observed_tool'))
n_still_tool = sum(1 for p in harmful if p.get('cb_observed_tool'))

print(f'    Pairs: {len(pairs)} | Different: {len(differ)} ({100*len(differ)/len(pairs):.0f}%) | Same: {len(same)}')
print(f'    Harmful ({len(harmful)}): refused={n_refused} still_tool={n_still_tool} gibberish={n_gib}')
print()

for i, p in enumerate(differ[:$NUM_EXAMPLES], 1):
    cat = p.get('category', '?')
    b_tool = p.get('baseline_observed_tool') or 'none'
    c_tool = p.get('cb_observed_tool') or 'none'
    gib = ' [GIB]' if is_gibberish(p.get('cb_response', '')) else ''
    print(f'    [{i}] {cat}: baseline={b_tool} -> cb={c_tool}{gib}')
    print(f'    --- BASELINE ---')
    print((p.get('baseline_response') or '(empty)')[:300])
    print(f'    --- CB MODEL ---')
    print((p.get('cb_response') or '(empty)')[:300])
    print()
PYEOF
                            fi
                        fi
                    fi

                    # ---------------------------------------------------------
                    # Evaluate on AgentDojo (benign subset — capability retention)
                    # ---------------------------------------------------------
                    echo "  Evaluating on AgentDojo benign ($LIMIT_EVAL samples)..."
                    AGENTDOJO_BENIGN_OUTPUT="$RUN_DIR/eval/agentdojo_benign_eval.json"

                    if [[ -f "$SPLIT_DIR/agentdojo_traces_benign.jsonl" ]]; then
                        python src/evaluation/eval.py \
                            --baseline "$TOKENIZER" \
                            --cb-adapter "$ADAPTER_PATH" \
                            --eval-data "$SPLIT_DIR/agentdojo_traces_benign.jsonl" \
                            --tool-schema "$REPO_DIR/configs/tool_schemas/b4_standard_v1.json" \
                            --output "$AGENTDOJO_BENIGN_OUTPUT" \
                            --use-sample-context \
                            --limit "$LIMIT_EVAL" \
                            --merge-adapter \
                            --dtype "$DTYPE" 2>&1 | tee -a "$RUN_DIR/eval/agentdojo_benign.log"

                        # Print benign examples
                        if [[ "$SHOW_EXAMPLES" == "true" ]]; then
                            PAIRED_FILE="${AGENTDOJO_BENIGN_OUTPUT%.json}.paired_outputs.jsonl"
                            if [[ -f "$PAIRED_FILE" ]]; then
                                echo ""
                                echo "    ========================================"
                                echo "    AGENTDOJO BENIGN: $RUN_NAME"
                                echo "    ========================================"
                                python3 << PYEOF
import json

with open('$PAIRED_FILE', 'r') as f:
    pairs = [json.loads(line) for line in f if line.strip()]

if not pairs:
    print('    No paired outputs')
    exit(0)

def is_gibberish(resp):
    if not resp: return False
    words = resp.split()
    return len(words) > 10 and len(set(words)) / len(words) < 0.15

n_correct = sum(1 for p in pairs if p.get('cb_observed_tool'))
n_broken = sum(1 for p in pairs if not p.get('cb_observed_tool'))
n_gib = sum(1 for p in pairs if is_gibberish(p.get('cb_response', '')))

print(f'    Benign: {len(pairs)} | correct_tool={n_correct} | broken={n_broken} | gibberish={n_gib}')
print()

broken = [p for p in pairs if p.get('baseline_observed_tool') and not p.get('cb_observed_tool')]
for i, p in enumerate(broken[:$NUM_EXAMPLES], 1):
    b_tool = p.get('baseline_observed_tool') or 'none'
    c_tool = p.get('cb_observed_tool') or 'none'
    gib = ' [GIB]' if is_gibberish(p.get('cb_response', '')) else ''
    print(f'    [{i}] BROKEN: baseline={b_tool} -> cb={c_tool}{gib}')
    print(f'    --- BASELINE ---')
    print((p.get('baseline_response') or '(empty)')[:300])
    print(f'    --- CB MODEL ---')
    print((p.get('cb_response') or '(empty)')[:300])
    print()
PYEOF
                            fi
                        fi
                    fi

                    success_count=$((success_count + 1))
                fi
            else
                fail_count=$((fail_count + 1))
            fi

            # =====================================================
            # Step 5: Quick Summary
            # =====================================================
            echo ""
            echo "--- Run Summary: $RUN_NAME ---"
            echo "  Training: $train_status"
            if [[ -f "$RUN_DIR/eval/fujitsu_eval.json" ]]; then
                python3 -c "
import json
with open('$RUN_DIR/eval/fujitsu_eval.json') as f:
    r = json.load(f)
if 'baseline' in r:
    baseline_asr = r['baseline']['tool_flip_asr']['attack_success_rate']
    baseline_mal = r['baseline']['tool_flip_asr'].get('malicious_tool_call_rate', baseline_asr)
    baseline_corr = r['baseline']['tool_flip_asr'].get('correct_tool_call_rate', r['baseline']['tool_flip_asr'].get('correct_behavior_rate', 0))
    print(f'  Fujitsu Baseline ASR: {baseline_asr:.1%}')
    print(f'  Fujitsu Baseline Malicious Tool Rate: {baseline_mal:.1%}')
    print(f'  Fujitsu Baseline Correct Tool Rate:   {baseline_corr:.1%}')
if 'cb_model' in r:
    asr = r['cb_model']['tool_flip_asr']['attack_success_rate']
    cb_mal = r['cb_model']['tool_flip_asr'].get('malicious_tool_call_rate', asr)
    cb_corr = r['cb_model']['tool_flip_asr'].get('correct_tool_call_rate', r['cb_model']['tool_flip_asr'].get('correct_behavior_rate', 0))
    cb_no = r['cb_model']['tool_flip_asr'].get('no_tool_call_rate', 0)
    cb_other = r['cb_model']['tool_flip_asr'].get('other_tool_call_rate', 0)
    cb_multi = r['cb_model']['tool_flip_asr'].get('multiple_tool_call_rate', 0)
    cb_malformed = r['cb_model']['tool_flip_asr'].get('malformed_tool_call_rate', 0)
    print(f'  Fujitsu CB ASR:       {asr:.1%}')
    print(f'  Fujitsu CB Malicious Tool Rate: {cb_mal:.1%}')
    print(f'  Fujitsu CB Correct Tool Rate:   {cb_corr:.1%}')
    print(f'  Fujitsu CB No-Tool Rate:        {cb_no:.1%}')
    print(f'  Fujitsu CB Other-Tool Rate:     {cb_other:.1%}')
    print(f'  Fujitsu CB Multi-Tool Rate:     {cb_multi:.1%}')
    print(f'  Fujitsu CB Malformed-Tool Rate: {cb_malformed:.1%}')
if 'delta' in r:
    delta = r['delta']['tool_flip_asr']
    print(f'  Fujitsu Delta:        {delta:+.1%}')
" 2>/dev/null || echo "  Fujitsu: (parse error)"
            fi
            if [[ -f "$RUN_DIR/eval/agentdojo_eval.json" ]]; then
                python3 -c "
import json
with open('$RUN_DIR/eval/agentdojo_eval.json') as f:
    r = json.load(f)
if 'output_comparison' in r:
    print(f'  AgentDojo Diff Rate:  {r[\"output_comparison\"].get(\"difference_rate\",-1):.1%}')
for key in ('baseline', 'cb_model'):
    gc = r.get(key, {}).get('generation_comparison', {})
    if gc:
        label = 'Baseline' if key == 'baseline' else 'CB'
        print(f'  AgentDojo {label}: malicious={gc.get(\"malicious_tool_call_rate\",0):.1%} resist={gc.get(\"harmful_resistance_rate\",0):.1%} correct={gc.get(\"correct_tool_call_rate\",0):.1%} no_tool={gc.get(\"no_tool_call_rate\",0):.1%}')
" 2>/dev/null || echo "  AgentDojo: (parse error)"
            fi
            if [[ -f "$RUN_DIR/eval/agentdojo_benign_eval.json" ]]; then
                python3 -c "
import json
with open('$RUN_DIR/eval/agentdojo_benign_eval.json') as f:
    r = json.load(f)
for key in ('baseline', 'cb_model'):
    gc = r.get(key, {}).get('generation_comparison', {})
    if gc:
        label = 'Baseline' if key == 'baseline' else 'CB'
        cor = gc.get('correct_tool_call_rate', 0)
        no = gc.get('no_tool_call_rate', 0)
        print(f'  AgentDojo Benign {label}: correct_tool={cor:.1%} no_tool={no:.1%}')
" 2>/dev/null || echo "  AgentDojo Benign: (parse error)"
            fi
            echo ""

            done  # preset loop

        done  # alpha loop
    done  # layers loop
done  # policy loop

# =============================================================================
# Final Summary
# =============================================================================
echo ""
echo "========================================"
echo "SWEEP COMPLETE"
echo "========================================"
echo ""
echo "Total configurations: $TOTAL_RUNS"
echo "Successful: $success_count"
echo "Failed: $fail_count"
echo ""
echo "Sweep directory: $SWEEP_DIR"
echo ""

# Generate summary CSV
SUMMARY_CSV="$SWEEP_DIR/summary.csv"
echo "preset,alpha,layers,policy,fujitsu_baseline_asr,fujitsu_cb_asr,fujitsu_delta,fujitsu_cb_correct_rate,fujitsu_cb_no_tool_rate,fujitsu_cb_other_tool_rate,fujitsu_gibberish_rate,agentdojo_diff,agentdojo_cb_malicious_rate,agentdojo_cb_harmful_resist,agentdojo_gibberish_rate,agentdojo_benign_correct_rate,agentdojo_benign_no_tool_rate,status" > "$SUMMARY_CSV"

for run_dir in "$SWEEP_DIR"/*/; do
    [[ -d "$run_dir" ]] || continue
    run_name=$(basename "$run_dir")

    # Extract preset (if present, it's the first segment before _a)
    # Formats: "preset_a10.0_l10_20_policy" or "a10.0_l10_20_policy" (single preset)
    if [[ "$run_name" =~ ^([a-zA-Z][a-zA-Z0-9]*)_a([0-9.]+)_l([0-9_]+)_(.+)$ ]]; then
        preset="${BASH_REMATCH[1]}"
        alpha="${BASH_REMATCH[2]}"
        layers="${BASH_REMATCH[3]//\_/,}"
        policy="${BASH_REMATCH[4]}"
    elif [[ "$run_name" =~ ^a([0-9.]+)_l([0-9_]+)_(.+)$ ]]; then
        preset="default"
        alpha="${BASH_REMATCH[1]}"
        layers="${BASH_REMATCH[2]//\_/,}"
        policy="${BASH_REMATCH[3]}"
    else
        continue
    fi

    # Defaults
    fujitsu_baseline="N/A"; fujitsu_cb="N/A"; fujitsu_delta="N/A"
    fujitsu_cb_correct="N/A"; fujitsu_cb_no_tool="N/A"; fujitsu_cb_other_tool="N/A"
    fujitsu_gib="N/A"
    agentdojo_diff="N/A"; agentdojo_cb_mal="N/A"; agentdojo_harmful_resist="N/A"
    agentdojo_gib="N/A"
    agentdojo_benign_correct="N/A"; agentdojo_benign_no_tool="N/A"
    status="unknown"

    if [[ -f "$run_dir/eval/fujitsu_eval.json" ]]; then
        read fujitsu_baseline fujitsu_cb fujitsu_delta fujitsu_cb_correct fujitsu_cb_no_tool fujitsu_cb_other_tool < <(python3 -c "
import json
with open('${run_dir}eval/fujitsu_eval.json') as f:
    r = json.load(f)
b = r.get('baseline',{}).get('tool_flip_asr',{})
c = r.get('cb_model',{}).get('tool_flip_asr',{})
print(b.get('attack_success_rate','N/A'), c.get('attack_success_rate','N/A'), r.get('delta',{}).get('tool_flip_asr','N/A'), c.get('correct_tool_call_rate', c.get('correct_behavior_rate','N/A')), c.get('no_tool_call_rate','N/A'), c.get('other_tool_call_rate','N/A'))
" 2>/dev/null || echo "N/A N/A N/A N/A N/A N/A")
        # Count gibberish from paired outputs
        FPAIRED="${run_dir}eval/fujitsu_eval.paired_outputs.jsonl"
        if [[ -f "$FPAIRED" ]]; then
            fujitsu_gib=$(python3 -c "
import json
with open('$FPAIRED') as f:
    pairs=[json.loads(l) for l in f if l.strip()]
gib=sum(1 for p in pairs if p.get('cb_response','') and len(p['cb_response'].split())>10 and len(set(p['cb_response'].split()))/len(p['cb_response'].split())<0.15)
print(f'{gib}/{len(pairs)}')
" 2>/dev/null || echo "N/A")
        fi
        status="success"
    fi

    if [[ -f "$run_dir/eval/agentdojo_eval.json" ]]; then
        read agentdojo_diff agentdojo_cb_mal agentdojo_harmful_resist < <(python3 -c "
import json
with open('${run_dir}eval/agentdojo_eval.json') as f:
    r = json.load(f)
diff = r.get('output_comparison',{}).get('difference_rate','N/A')
gc = r.get('cb_model',{}).get('generation_comparison',{})
print(diff, gc.get('malicious_tool_call_rate','N/A'), gc.get('harmful_resistance_rate','N/A'))
" 2>/dev/null || echo "N/A N/A N/A")
        APAIRED="${run_dir}eval/agentdojo_eval.paired_outputs.jsonl"
        if [[ -f "$APAIRED" ]]; then
            agentdojo_gib=$(python3 -c "
import json
with open('$APAIRED') as f:
    pairs=[json.loads(l) for l in f if l.strip()]
gib=sum(1 for p in pairs if p.get('cb_response','') and len(p['cb_response'].split())>10 and len(set(p['cb_response'].split()))/len(p['cb_response'].split())<0.15)
print(f'{gib}/{len(pairs)}')
" 2>/dev/null || echo "N/A")
        fi
    fi

    if [[ -f "$run_dir/eval/agentdojo_benign_eval.json" ]]; then
        read agentdojo_benign_correct agentdojo_benign_no_tool < <(python3 -c "
import json
with open('${run_dir}eval/agentdojo_benign_eval.json') as f:
    r = json.load(f)
gc = r.get('cb_model',{}).get('generation_comparison',{})
print(gc.get('correct_tool_call_rate','N/A'), gc.get('no_tool_call_rate','N/A'))
" 2>/dev/null || echo "N/A N/A")
    fi

    echo "$preset,$alpha,$layers,$policy,$fujitsu_baseline,$fujitsu_cb,$fujitsu_delta,$fujitsu_cb_correct,$fujitsu_cb_no_tool,$fujitsu_cb_other_tool,$fujitsu_gib,$agentdojo_diff,$agentdojo_cb_mal,$agentdojo_harmful_resist,$agentdojo_gib,$agentdojo_benign_correct,$agentdojo_benign_no_tool,$status" >> "$SUMMARY_CSV"
done

echo "Summary CSV: $SUMMARY_CSV"
echo ""
column -t -s',' "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
echo ""

# =============================================================================
# Best Results
# =============================================================================
echo ""
echo "--- BEST RESULTS (Lowest CB ASR, no gibberish) ---"
python3 -c "
import csv

try:
    with open('$SUMMARY_CSV', 'r') as f:
        rows = list(csv.DictReader(f))

    valid = []
    for r in rows:
        try:
            cb_asr = float(r['fujitsu_cb_asr'])
            gib = r.get('fujitsu_gibberish_rate', '0/0')
            gib_n = int(gib.split('/')[0]) if '/' in gib else 0
            valid.append((cb_asr, gib_n, r))
        except:
            pass

    if valid:
        # Sort by: no gibberish first, then lowest ASR
        valid.sort(key=lambda x: (x[1] > 0, x[0]))
        for rank, (asr, gib, r) in enumerate(valid[:3], 1):
            gib_str = r.get('fujitsu_gibberish_rate', 'N/A')
            ad_mal = r.get('agentdojo_cb_malicious_rate', 'N/A')
            ad_res = r.get('agentdojo_cb_harmful_resist', 'N/A')
            ad_gib = r.get('agentdojo_gibberish_rate', 'N/A')
            print(f'#{rank} {r[\"preset\"]} a={r[\"alpha\"]} l={r[\"layers\"]} {r[\"policy\"]}')
            benign_cor = r.get('agentdojo_benign_correct_rate', 'N/A')
            benign_no = r.get('agentdojo_benign_no_tool_rate', 'N/A')
            print(f'   Fujitsu: ASR={asr:.1%} correct={r.get(\"fujitsu_cb_correct_rate\",\"N/A\")} no_tool={r.get(\"fujitsu_cb_no_tool_rate\",\"N/A\")} gibberish={gib_str}')
            print(f'   AgentDojo harmful: malicious={ad_mal} resist={ad_res} gibberish={ad_gib}')
            print(f'   AgentDojo benign:  correct_tool={benign_cor} no_tool={benign_no}')
            print()
except Exception as e:
    print(f'Could not analyze results: {e}')
" 2>/dev/null || echo "(analysis failed)"

# =============================================================================
# Tradeoff Visualization
# =============================================================================
echo ""
echo "Generating tradeoff visualizations..."
python "$REPO_DIR/scripts/plot_tradeoff.py" --sweep-dir "$SWEEP_DIR" --output "$SWEEP_DIR/plots" 2>&1 || echo "(plot generation skipped - install matplotlib for plots)"

echo ""
echo "Finished at: $(date)"
