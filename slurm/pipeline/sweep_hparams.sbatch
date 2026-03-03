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
# CANONICAL PIPELINE ENTRYPOINT: HPARAM SWEEP
# =============================================================================
#
# This is the sole sweep/entrypoint script (sweep_hparams_simple.sbatch was
# merged into this file). It handles data auto-generation, ETL_B, training,
# evaluation, and summary generation.
#
# Sweep dimensions:
#   - POLICIES        (lossmask policy)
#   - LAYER_CONFIGS   (CB target layers)
#   - ALPHAS          (alpha_max for legacy schedules)
#   - GAMMA_KLS       (triplet KL weight — stability tuning)
#   - POOLING_PRESETS  (pooling mode / distance / margins)
#
# For each configuration: ETL_B (policy) -> train -> eval -> run_summary.
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
#    ALPHAS=10.0 LAYER_CONFIGS="20,21,22,23,24,25,26,27,28,29,30,31" \
#    sbatch slurm/pipeline/sweep_hparams.sbatch
#
# 4) Gamma_kl sweep:
#    GAMMA_KLS="0.9,5.0,20.0" \
#    sbatch slurm/pipeline/sweep_hparams.sbatch
#
# 5) Legacy loss sweep:
#    LOSS_MODES=legacy_cb LOSS_WEIGHTING=dual \
#    sbatch slurm/pipeline/sweep_hparams.sbatch
#
# -----------------------------------------------------------------------------
# Config knobs (pass with `VAR=value sbatch ...`)
# -----------------------------------------------------------------------------
# Sweep controls:
#   QUICK_TEST, TOTAL_STEPS, ALPHAS, GAMMA_KLS, LOSS_MODES, LAYER_CONFIGS, POLICIES
#
# Eval/logging:
#   LIMIT_EVAL, NUM_EXAMPLES, SHOW_EXAMPLES, VIS_CHECKPOINT_STEPS
#
# Core training:
#   LOSS_WEIGHTING, BATCH_SIZE, LEARNING_RATE, WARMUP_STEPS,
#   MAX_SEQ_LENGTH, MAX_MODEL_LEN, DTYPE, LORA_R, LORA_ALPHA
#
# Triplet/loss parameters:
#   TRIPLET_ALPHA_BENIGN, TRIPLET_BETA_HARMFUL,
#   TRIPLET_MIX_L2_WEIGHT, TRIPLET_MIX_COS_WEIGHT
#
# Pooling / mask:
#   POOLING_PRESETS, POOLING_POLICY, POOLING_MASK_POLICY
#
# Data generation:
#   LIMIT_GEN (for generate_completions.py)
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
# Suppress tqdm progress bars in batch mode (they produce noisy multi-line output)
export TQDM_DISABLE=1

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
    echo "QUICK TEST MODE - Reduced configurations"
    DEFAULT_STEPS=150
    DEFAULT_ALPHAS="10.0"
    DEFAULT_LAYER_CONFIGS="20,21,22,23,24,25,26,27,28,29,30,31"
    DEFAULT_POLICIES="assistant_only"
    DEFAULT_LIMIT_EVAL=20
    DEFAULT_NUM_EXAMPLES=5
else
    # Paper defaults (TRIPLET.tex appendix):
    #   Layers 20-31: deep semantic layers (last 12 of 32 for Llama 8B)
    #   1100 steps, LR=1e-5, LoRA alpha=16, batch_size=16, d_mix for all distances
    DEFAULT_STEPS=1100
    DEFAULT_ALPHAS="10.0"
    DEFAULT_LAYER_CONFIGS="20,21,22,23,24,25,26,27,28,29,30,31"
    DEFAULT_POLICIES="cb_full_sequence"
    DEFAULT_LIMIT_EVAL=50
    DEFAULT_NUM_EXAMPLES=10
fi

TOTAL_STEPS=150
ALPHAS="${ALPHAS:-$DEFAULT_ALPHAS}"
LAYER_CONFIGS="${LAYER_CONFIGS:-$DEFAULT_LAYER_CONFIGS}"
POLICIES="${POLICIES:-$DEFAULT_POLICIES}"
LIMIT_EVAL="${LIMIT_EVAL:-$DEFAULT_LIMIT_EVAL}"
NUM_EXAMPLES="${NUM_EXAMPLES:-$DEFAULT_NUM_EXAMPLES}"
SHOW_EXAMPLES="${SHOW_EXAMPLES:-true}"

# Training params (paper: batch=16, LR=1e-5, LoRA alpha=16)
# Single-GPU: BATCH_SIZE=4 + GRAD_ACCUM=4 = effective batch 16 (matches paper)
BATCH_SIZE=1
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DTYPE="${DTYPE:-bfloat16}"
LOSS_WEIGHTING="${LOSS_WEIGHTING:-dual}"
TRIPLET_ALPHA_BENIGN="${TRIPLET_ALPHA_BENIGN:-0.5}"
TRIPLET_BETA_HARMFUL="${TRIPLET_BETA_HARMFUL:-0.4}"
TRIPLET_MIX_L2_WEIGHT="${TRIPLET_MIX_L2_WEIGHT:-0.5}"
TRIPLET_MIX_COS_WEIGHT="${TRIPLET_MIX_COS_WEIGHT:-0.5}"

# Loss mode sweep: comma-separated loss modes.
# "l2_simple" uses L2 distance (non-vanishing gradients, works with LoRA).
# "cosine_simple" has vanishing gradients when cos≈1 — broken with LoRA.
# "triplet_full" uses margin-based triplet (hard to calibrate).
LOSS_MODES="${LOSS_MODES:-l2_simple}"

# Gamma KL sweep: comma-separated values for triplet_gamma_kl.
# Scale depends on loss mode:
#   l2_simple:     losses bounded [0,1], KL~7 → gamma_kl 0.05-0.3
#   cosine_simple: BROKEN with LoRA (vanishing gradients at cos≈1)
#   triplet_full:  losses ~500-1500,   KL~7 → gamma_kl 50-200 needed
GAMMA_KLS="${GAMMA_KLS:-0.05,0.1,0.3}"

# Pooling presets: each is "name:pooling_mode:distance:margin_b:margin_h"
# Multiple presets separated by spaces. Swept as innermost dimension.
# "correct" pooling uses all 4096 dims (dense); "legacy" uses ~2% (sparse, ineffective for CB).
# Margins/distance don't matter for cosine_simple (only used by triplet_full).
POOLING_PRESETS="${POOLING_PRESETS:-cor:correct:dmix:500:1500}"
POOLING_POLICY="${POOLING_POLICY:-cb_full_sequence}"
POOLING_MASK_POLICY="${POOLING_MASK_POLICY:-auto}"
STOP_AFTER_ETL_B="${STOP_AFTER_ETL_B:-false}"
STOP_AFTER_SPLIT="${STOP_AFTER_SPLIT:-false}"

# Data generation params (for auto-generating missing DS/DR traces)
LIMIT_GEN="${LIMIT_GEN:-5000}"
VIS_CHECKPOINT_STEPS="${VIS_CHECKPOINT_STEPS:-50,100}"

echo "Configuration:"
echo "  TOTAL_STEPS: $TOTAL_STEPS"
echo "  ALPHAS: $ALPHAS"
echo "  GAMMA_KLS: $GAMMA_KLS"
echo "  LAYER_CONFIGS: $LAYER_CONFIGS"
echo "  POLICIES: $POLICIES"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  GRAD_ACCUM: $GRAD_ACCUM"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  LOSS_MODES: $LOSS_MODES"
echo "  LIMIT_EVAL: $LIMIT_EVAL"
echo "  NUM_EXAMPLES: $NUM_EXAMPLES"
echo "  SHOW_EXAMPLES: $SHOW_EXAMPLES"
echo "  QUICK_TEST: $QUICK_TEST"
echo "  POOLING_PRESETS: $POOLING_PRESETS"
echo "  POOLING_POLICY: ${POOLING_POLICY:-<none>}"
echo "  POOLING_MASK_POLICY: $POOLING_MASK_POLICY"
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
FUJITSU_SKELETON="$TRACES_DIR/fujitsu_b4_skeletons.jsonl"
TOOL_SCHEMA="configs/tool_schemas/b4_standard_v1.json"

echo "Input traces (ALL used for training):"

# Auto-generate missing Fujitsu DS/DR from skeletons if needed
if [[ ! -f "$DS_TRACES" || ! -f "$DR_TRACES" ]]; then
    if [[ -f "$FUJITSU_SKELETON" ]]; then
        echo "  Auto-generating missing Fujitsu data..."
        if [[ ! -f "$DS_TRACES" ]]; then
            echo "    Generating DS (Harmful)..."
            python src/data_generation/generate_completions.py \
                --traces "$FUJITSU_SKELETON" \
                --output "$DS_TRACES" \
                --model "$TOKENIZER" \
                --tool-schema "$TOOL_SCHEMA" \
                --mode ds \
                --temperature-ds 0.7 \
                --strict-simulated-match true \
                --use-vllm \
                --limit "$LIMIT_GEN"
        fi
        if [[ ! -f "$DR_TRACES" ]]; then
            echo "    Generating DR (Benign)..."
            python src/data_generation/generate_completions.py \
                --ds-data "$DS_TRACES" \
                --output "$DR_TRACES" \
                --model "$TOKENIZER" \
                --tool-schema "$TOOL_SCHEMA" \
                --mode dr \
                --temperature-dr 0.3 \
                --use-vllm \
                --limit "$LIMIT_GEN"
        fi
    else
        echo "  WARNING: Fujitsu skeleton not found at $FUJITSU_SKELETON"
    fi
fi

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
    IFS=',' read -ra GAMMA_KL_ARRAY <<< "$GAMMA_KLS"
    IFS=',' read -ra LOSS_MODE_ARRAY <<< "$LOSS_MODES"
    IFS='|' read -ra LAYER_ARRAY <<< "$LAYER_CONFIGS"
    IFS=',' read -ra POLICY_ARRAY <<< "$POLICIES"
    read -ra PRESET_ARRAY <<< "$POOLING_PRESETS"

    for _a in "${ALPHA_ARRAY[@]}"; do
        for _g in "${GAMMA_KL_ARRAY[@]}"; do
            for _lm in "${LOSS_MODE_ARRAY[@]}"; do
                for _l in "${LAYER_ARRAY[@]}"; do
                    for _pol in "${POLICY_ARRAY[@]}"; do
                        for _p in "${PRESET_ARRAY[@]}"; do
                            count=$((count + 1))
                        done
                    done
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
IFS=',' read -ra GAMMA_KL_ARRAY <<< "$GAMMA_KLS"
IFS=',' read -ra LOSS_MODE_ARRAY <<< "$LOSS_MODES"
IFS='|' read -ra LAYER_ARRAY <<< "$LAYER_CONFIGS"
IFS=',' read -ra POLICY_ARRAY <<< "$POLICIES"
read -ra PRESET_ARRAY <<< "$POOLING_PRESETS"

for policy in "${POLICY_ARRAY[@]}"; do
    for layers in "${LAYER_ARRAY[@]}"; do
        for alpha in "${ALPHA_ARRAY[@]}"; do
            # Create base name for ETL_B sharing (shared across gamma_kl + presets)
            # Compact layer notation: "20,21,...,31" → "20-31", else "4_8_14"
            IFS=',' read -ra _LAYER_NUMS <<< "$layers"
            _n_layers=${#_LAYER_NUMS[@]}
            _first=${_LAYER_NUMS[0]}
            _last=${_LAYER_NUMS[$((_n_layers - 1))]}
            if [[ $_n_layers -gt 3 && $((_last - _first + 1)) -eq $_n_layers ]]; then
                layers_short="${_first}-${_last}"
            else
                layers_short="${layers//,/_}"
            fi
            BASE_NAME="a${alpha}_l${layers_short}_${policy}"
            BASE_DIR="$SWEEP_DIR/$BASE_NAME"

            RUN_NAME="$BASE_NAME"
            RUN_DIR="$BASE_DIR"

            echo "========================================"
            echo "ETL_B: $RUN_NAME"
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
                ETL_B_POOLING_ARGS=()
                if [[ -n "$POOLING_POLICY" ]]; then
                    ETL_B_POOLING_ARGS+=(--pooling-policy-override "$POOLING_POLICY")
                fi
                python src/schemas/tools/ETL_B.py \
                    --traces "$trace_file" \
                    --render-out "$render_out" \
                    --lossmask-out "$lossmask_out" \
                    --tokenizer "$TOKENIZER" \
                    --max-length 4096 \
                    --policy-override "$policy" \
                    "${ETL_B_POOLING_ARGS[@]}" \
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

            # =====================================================
            # Step 2c: Data Diagnostic (once per BASE_DIR)
            # =====================================================
            echo "Step 2c: Running data diagnostic"
            DIAG_HARMFUL_RENDERS=(
                "$BASE_DIR/renders/fujitsu_b4_ds.jsonl"
                "$SPLIT_DIR/agentdojo_renders_harmful.jsonl"
            )
            DIAG_HARMFUL_LOSSMASKS=(
                "$BASE_DIR/lossmasks/fujitsu_b4_ds.jsonl"
                "$SPLIT_DIR/agentdojo_lossmasks_harmful.jsonl"
            )
            DIAG_BENIGN_RENDERS=(
                "$BASE_DIR/renders/fujitsu_b4_dr.jsonl"
                "$SPLIT_DIR/agentdojo_renders_benign.jsonl"
            )
            DIAG_BENIGN_LOSSMASKS=(
                "$BASE_DIR/lossmasks/fujitsu_b4_dr.jsonl"
                "$SPLIT_DIR/agentdojo_lossmasks_benign.jsonl"
            )
            if [[ "$HAS_AGENTHARM" == "true" ]]; then
                [[ -f "$AGENTHARM_SPLIT_DIR/agentharm_renders_harmful.jsonl" ]] && \
                    DIAG_HARMFUL_RENDERS+=("$AGENTHARM_SPLIT_DIR/agentharm_renders_harmful.jsonl") && \
                    DIAG_HARMFUL_LOSSMASKS+=("$AGENTHARM_SPLIT_DIR/agentharm_lossmasks_harmful.jsonl")
                [[ -f "$AGENTHARM_SPLIT_DIR/agentharm_renders_benign.jsonl" ]] && \
                    DIAG_BENIGN_RENDERS+=("$AGENTHARM_SPLIT_DIR/agentharm_renders_benign.jsonl") && \
                    DIAG_BENIGN_LOSSMASKS+=("$AGENTHARM_SPLIT_DIR/agentharm_lossmasks_benign.jsonl")
            fi

            python scripts/diagnose_training_data.py \
                --tokenizer "$TOKENIZER" \
                --harmful-renders "${DIAG_HARMFUL_RENDERS[@]}" \
                --harmful-lossmasks "${DIAG_HARMFUL_LOSSMASKS[@]}" \
                --benign-renders "${DIAG_BENIGN_RENDERS[@]}" \
                --benign-lossmasks "${DIAG_BENIGN_LOSSMASKS[@]}" \
                --output "$BASE_DIR/data_diagnostic.txt" \
                --num-samples 3 2>&1 | tee -a "$BASE_DIR/diagnostic.log"

            echo "  Diagnostic written to: $BASE_DIR/data_diagnostic.txt"
            echo ""

            # Parse layers to array (shared by all presets and gamma_kl values)
            IFS=',' read -ra CB_LAYER_ARRAY <<< "$layers"

            for gamma_kl in "${GAMMA_KL_ARRAY[@]}"; do
            for loss_mode in "${LOSS_MODE_ARRAY[@]}"; do
            for preset_str in "${PRESET_ARRAY[@]}"; do
            # Parse preset: name:pooling_mode:distance:margin_b:margin_h
            IFS=':' read -r _PNAME _PMODE _PDIST _PMB _PMH <<< "$preset_str"

            # Compute loss mode short tag (only included when sweeping multiple)
            _LM_TAG=""
            if [[ ${#LOSS_MODE_ARRAY[@]} -gt 1 ]]; then
                case "$loss_mode" in
                    triplet_full) _LM_TAG="tf" ;;
                    cosine_simple) _LM_TAG="cs" ;;
                    *) _LM_TAG="${loss_mode:0:2}" ;;
                esac
            fi

            # Build run name: include gamma_kl, loss_mode tag, and preset as needed
            # Format: [preset_]a{alpha}_g{gamma}[_lm]_l{layers}_{policy}
            if [[ -n "$_LM_TAG" ]]; then
                _RUN_CORE="a${alpha}_g${gamma_kl}_${_LM_TAG}_l${layers_short}_${policy}"
            else
                _RUN_CORE="a${alpha}_g${gamma_kl}_l${layers_short}_${policy}"
            fi
            if [[ ${#PRESET_ARRAY[@]} -gt 1 || ${#GAMMA_KL_ARRAY[@]} -gt 1 || ${#LOSS_MODE_ARRAY[@]} -gt 1 ]]; then
                RUN_NAME="${_PNAME}_${_RUN_CORE}"
                RUN_DIR="$SWEEP_DIR/$RUN_NAME"
                mkdir -p "$RUN_DIR"/{model,eval}
            else
                RUN_NAME="$_RUN_CORE"
                RUN_DIR="$SWEEP_DIR/$RUN_NAME"
                mkdir -p "$RUN_DIR"/{model,eval}
            fi

            run_idx=$((run_idx + 1))
            echo ""
            echo "========================================"
            echo "[$run_idx/$TOTAL_RUNS] $RUN_NAME"
            echo "========================================"
            echo "  Alpha: $alpha  Gamma KL: $gamma_kl  Loss Mode: $loss_mode"
            echo "  Layers: $layers  Policy: $policy"
            echo "  Preset: $_PNAME (${_PMODE}/${_PDIST}/${_PMB}/${_PMH})"
            echo "  Output: $RUN_DIR"
            echo ""

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
  "loss_mode": "$loss_mode",
  "loss_weighting": "$LOSS_WEIGHTING",
  "pooling_policy": "${POOLING_POLICY:-}",
  "pooling_mask_policy": "$POOLING_MASK_POLICY",
  "triplet_alpha_benign": $TRIPLET_ALPHA_BENIGN,
  "triplet_beta_harmful": $TRIPLET_BETA_HARMFUL,
  "triplet_gamma_kl": $gamma_kl,
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
            echo "Step 3: Training with alpha=$alpha, gamma_kl=$gamma_kl, loss_mode=$loss_mode, layers=$layers, policy=$policy, preset=$_PNAME (${_PMODE}/${_PDIST}/${_PMB}/${_PMH})"

            # NOTE: Renders/lossmasks/splits live in BASE_DIR (shared across presets/gamma_kl).
            # RUN_DIR may differ from BASE_DIR for multi-config sweeps.
            HARMFUL_RENDER_ARGS=(
                "$BASE_DIR/renders/fujitsu_b4_ds.jsonl"
                "$SPLIT_DIR/agentdojo_renders_harmful.jsonl"
            )
            HARMFUL_LOSSMASK_ARGS=(
                "$BASE_DIR/lossmasks/fujitsu_b4_ds.jsonl"
                "$SPLIT_DIR/agentdojo_lossmasks_harmful.jsonl"
            )
            BENIGN_RENDER_ARGS=(
                "$BASE_DIR/renders/fujitsu_b4_dr.jsonl"
                "$SPLIT_DIR/agentdojo_renders_benign.jsonl"
            )
            BENIGN_LOSSMASK_ARGS=(
                "$BASE_DIR/lossmasks/fujitsu_b4_dr.jsonl"
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
                --loss-mode "$loss_mode"
                --loss-weighting "$LOSS_WEIGHTING"
                --triplet-alpha-benign "$TRIPLET_ALPHA_BENIGN"
                --triplet-beta-harmful "$TRIPLET_BETA_HARMFUL"
                --triplet-gamma-kl "$gamma_kl"
                --triplet-margin-benign "$_PMB"
                --triplet-margin-harmful "$_PMH"
                --triplet-benign-positive-distance "$_PDIST"
                --triplet-benign-negative-distance "$_PDIST"
                --triplet-harmful-positive-distance "$_PDIST"
                --triplet-harmful-negative-distance "$_PDIST"
                --triplet-mix-l2-weight "$TRIPLET_MIX_L2_WEIGHT"
                --triplet-mix-cos-weight "$TRIPLET_MIX_COS_WEIGHT"
                --pooling-mode "$_PMODE"
                --pooling-mask-policy "$POOLING_MASK_POLICY"
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
            echo "  Training... (log: $RUN_DIR/train.log)"
            if python src/training/train_schema.py "${TRAIN_ARGS[@]}" > "$RUN_DIR/train.log" 2>&1; then
                train_status="SUCCESS"
            else
                train_status="FAILED"
            fi
            train_end=$(date +%s)
            train_mins=$(( (train_end - train_start) / 60 ))
            echo "  Training $train_status (${train_mins}m)"
            if [[ "$train_status" == "FAILED" ]]; then
                echo "  Last 5 lines of train.log:"
                tail -5 "$RUN_DIR/train.log" | sed 's/^/    /'
            fi
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
                        --dtype "$DTYPE" >> "$CKPT_EVAL_DIR/fujitsu.log" 2>&1

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
                            --dtype "$DTYPE" >> "$CKPT_EVAL_DIR/agentdojo.log" 2>&1
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
                            --dtype "$DTYPE" >> "$CKPT_EVAL_DIR/agentdojo_benign.log" 2>&1
                    fi

                    # Compact checkpoint summary
                    if [[ -f "$CKPT_EVAL_DIR/fujitsu_eval.json" ]]; then
                        python3 -c "
import json
with open('$CKPT_EVAL_DIR/fujitsu_eval.json') as f: r = json.load(f)
cb = r.get('cb_model',{}).get('tool_flip_asr',{})
cap = r.get('cb_model',{}).get('capability_retention',{})
if cb:
    print(f'    {\"$ckpt_name\"}: ASR={cb.get(\"attack_success_rate\",0):.0%} correct={cb.get(\"correct_tool_call_rate\",0):.0%} no_tool={cb.get(\"no_tool_call_rate\",0):.0%} cap={cap.get(\"capability_retention\",0):.0%}')
"
                    fi
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
                        --dtype "$DTYPE" >> "$RUN_DIR/eval/fujitsu.log" 2>&1

                    # Print compact Fujitsu summary from JSON
                    if [[ -f "$FUJITSU_OUTPUT" ]]; then
                        python3 -c "
import json, sys
with open('$FUJITSU_OUTPUT') as f: r = json.load(f)
for lbl, k in [('Baseline','baseline'),('CB','cb_model')]:
    tf = r.get(k,{}).get('tool_flip_asr',{})
    cap = r.get(k,{}).get('capability_retention',{})
    if not tf: continue
    asr = tf.get('attack_success_rate',0)
    cor = tf.get('correct_tool_call_rate',0)
    nt = tf.get('no_tool_call_rate',0)
    cr = cap.get('capability_retention',0)
    n = tf.get('total_samples',0)
    print(f'    {lbl}: ASR={asr:.0%} correct={cor:.0%} no_tool={nt:.0%} cap_ret={cr:.0%} (n={n})')
d = r.get('delta',{})
if d:
    print(f'    Delta: ASR={d.get(\"tool_flip_asr\",0):+.0%} cap={d.get(\"capability_retention\",0):+.0%}')
    g = r.get('stage1_gates',{})
    failed = [k for k,v in g.items() if not v]
    if failed: print(f'    FAILED gates: {\" \".join(failed)}')
    else: print(f'    All gates PASSED')
"
                    fi

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
                            --dtype "$DTYPE" >> "$RUN_DIR/eval/agentdojo.log" 2>&1

                        # Print compact AgentDojo summary
                        if [[ -f "$AGENTDOJO_OUTPUT" ]]; then
                            python3 -c "
import json
with open('$AGENTDOJO_OUTPUT') as f: r = json.load(f)
cb = r.get('cb_model',{}).get('generation_comparison',{})
if cb:
    resist = cb.get('harmful_resistance_rate',0)
    mal = cb.get('malicious_tool_call_rate',0)
    n = cb.get('total_samples',0)
    print(f'    AgentDojo: resist={resist:.0%} malicious={mal:.0%} (n={n})')
"
                        fi

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
                            --dtype "$DTYPE" >> "$RUN_DIR/eval/agentdojo_benign.log" 2>&1

                        # Print compact benign summary
                        if [[ -f "$AGENTDOJO_BENIGN_OUTPUT" ]]; then
                            python3 -c "
import json
with open('$AGENTDOJO_BENIGN_OUTPUT') as f: r = json.load(f)
cb = r.get('cb_model',{}).get('generation_comparison',{})
if cb:
    cor = cb.get('correct_tool_call_rate',0)
    nt = cb.get('no_tool_call_rate',0)
    n = cb.get('total_samples',0)
    print(f'    AD Benign: correct={cor:.0%} no_tool={nt:.0%} (n={n})')
"
                        fi

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
            # Step 5: Quick Summary (compact, metrics already printed above)
            # =====================================================
            run_end=$(date +%s)
            run_mins=$(( (run_end - train_start) / 60 ))
            echo ""
            echo "--- $RUN_NAME: $train_status (${run_mins}m total) ---"
            echo ""

            # Generate per-run summary
            python scripts/generate_run_summary.py --run-dir "$RUN_DIR" 2>&1 || echo "(run summary generation failed)"

            done  # preset loop
            done  # loss_mode loop
            done  # gamma_kl loop

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
echo "preset,alpha,gamma_kl,loss_mode,layers,policy,fujitsu_baseline_asr,fujitsu_cb_asr,fujitsu_delta,fujitsu_cb_correct_rate,fujitsu_cb_no_tool_rate,fujitsu_cb_other_tool_rate,fujitsu_gibberish_rate,agentdojo_diff,agentdojo_cb_malicious_rate,agentdojo_cb_harmful_resist,agentdojo_gibberish_rate,agentdojo_benign_correct_rate,agentdojo_benign_no_tool_rate,status" > "$SUMMARY_CSV"

for run_dir in "$SWEEP_DIR"/*/; do
    [[ -d "$run_dir" ]] || continue
    run_name=$(basename "$run_dir")

    # Extract config from run directory name.
    # Formats (newest to oldest):
    #   "preset_a10.0_g5.0_tf_l20-31_policy"   (with loss mode tag, range layers)
    #   "preset_a10.0_g5.0_l20-31_policy"      (without loss mode tag)
    #   "a10.0_g5.0_l20-31_policy"             (no preset)
    #   "preset_a10.0_l10_20_policy"           (legacy, no gamma, underscore layers)
    #   "a10.0_l10_20_policy"                  (legacy, no preset/gamma)
    gamma_kl_csv="0.9"  # default for legacy dirs
    loss_mode_csv="triplet_full"  # default
    if [[ "$run_name" =~ ^([a-zA-Z][a-zA-Z0-9]*)_a([0-9.]+)_g([0-9.]+)_([a-z]{2})_l([0-9_-]+)_(.+)$ ]]; then
        # New format with loss mode tag: preset_a10.0_g5.0_tf_l20-31_policy
        preset="${BASH_REMATCH[1]}"
        alpha="${BASH_REMATCH[2]}"
        gamma_kl_csv="${BASH_REMATCH[3]}"
        _lm_tag="${BASH_REMATCH[4]}"
        layers="${BASH_REMATCH[5]//\_/,}"
        policy="${BASH_REMATCH[6]}"
        case "$_lm_tag" in
            tf) loss_mode_csv="triplet_full" ;;
            cs) loss_mode_csv="cosine_simple" ;;
            l2) loss_mode_csv="l2_simple" ;;
            *) loss_mode_csv="$_lm_tag" ;;
        esac
    elif [[ "$run_name" =~ ^([a-zA-Z][a-zA-Z0-9]*)_a([0-9.]+)_g([0-9.]+)_l([0-9_-]+)_(.+)$ ]]; then
        preset="${BASH_REMATCH[1]}"
        alpha="${BASH_REMATCH[2]}"
        gamma_kl_csv="${BASH_REMATCH[3]}"
        layers="${BASH_REMATCH[4]//\_/,}"
        policy="${BASH_REMATCH[5]}"
    elif [[ "$run_name" =~ ^a([0-9.]+)_g([0-9.]+)_([a-z]{2})_l([0-9_-]+)_(.+)$ ]]; then
        # No preset, with loss mode tag
        preset="default"
        alpha="${BASH_REMATCH[1]}"
        gamma_kl_csv="${BASH_REMATCH[2]}"
        _lm_tag="${BASH_REMATCH[3]}"
        layers="${BASH_REMATCH[4]//\_/,}"
        policy="${BASH_REMATCH[5]}"
        case "$_lm_tag" in
            tf) loss_mode_csv="triplet_full" ;;
            cs) loss_mode_csv="cosine_simple" ;;
            l2) loss_mode_csv="l2_simple" ;;
            *) loss_mode_csv="$_lm_tag" ;;
        esac
    elif [[ "$run_name" =~ ^a([0-9.]+)_g([0-9.]+)_l([0-9_-]+)_(.+)$ ]]; then
        preset="default"
        alpha="${BASH_REMATCH[1]}"
        gamma_kl_csv="${BASH_REMATCH[2]}"
        layers="${BASH_REMATCH[3]//\_/,}"
        policy="${BASH_REMATCH[4]}"
    elif [[ "$run_name" =~ ^([a-zA-Z][a-zA-Z0-9]*)_a([0-9.]+)_l([0-9_-]+)_(.+)$ ]]; then
        preset="${BASH_REMATCH[1]}"
        alpha="${BASH_REMATCH[2]}"
        layers="${BASH_REMATCH[3]//\_/,}"
        policy="${BASH_REMATCH[4]}"
    elif [[ "$run_name" =~ ^a([0-9.]+)_l([0-9_-]+)_(.+)$ ]]; then
        preset="default"
        alpha="${BASH_REMATCH[1]}"
        layers="${BASH_REMATCH[2]//\_/,}"
        policy="${BASH_REMATCH[3]}"
    else
        continue
    fi

    # Try to get gamma_kl and loss_mode from run_config.json (more reliable than name parsing)
    if [[ -f "$run_dir/run_config.json" ]]; then
        _cfg_gamma=$(python3 -c "import json; print(json.load(open('${run_dir}run_config.json')).get('triplet_gamma_kl', '$gamma_kl_csv'))" 2>/dev/null || echo "$gamma_kl_csv")
        gamma_kl_csv="$_cfg_gamma"
        _cfg_lm=$(python3 -c "import json; print(json.load(open('${run_dir}run_config.json')).get('loss_mode', '$loss_mode_csv'))" 2>/dev/null || echo "$loss_mode_csv")
        loss_mode_csv="$_cfg_lm"
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

    echo "$preset,$alpha,$gamma_kl_csv,$loss_mode_csv,$layers,$policy,$fujitsu_baseline,$fujitsu_cb,$fujitsu_delta,$fujitsu_cb_correct,$fujitsu_cb_no_tool,$fujitsu_cb_other_tool,$fujitsu_gib,$agentdojo_diff,$agentdojo_cb_mal,$agentdojo_harmful_resist,$agentdojo_gib,$agentdojo_benign_correct,$agentdojo_benign_no_tool,$status" >> "$SUMMARY_CSV"
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
            print(f'#{rank} {r[\"preset\"]} a={r[\"alpha\"]} g={r.get(\"gamma_kl\",\"?\")} lm={r.get(\"loss_mode\",\"?\")} l={r[\"layers\"]} {r[\"policy\"]}')
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

# =============================================================================
# Sweep Summary (aggregates all run summaries)
# =============================================================================
echo ""
echo "Generating sweep summary..."
python scripts/generate_sweep_summary.py --sweep-dir "$SWEEP_DIR" 2>&1 || echo "(sweep summary generation failed)"

echo ""
echo "Checkpoint-aware visualization command:"
echo "  python scripts/visualize_sweep_results.py \"$SWEEP_DIR\" \\"
echo "    --compare-checkpoints --checkpoint-steps \"$VIS_CHECKPOINT_STEPS\" --no-color \\"
echo "    --csv \"$SWEEP_DIR/eval_summary.csv\""
echo ""
echo "Finished at: $(date)"
