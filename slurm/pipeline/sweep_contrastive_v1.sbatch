#!/bin/bash
#SBATCH --job-name=sweep_contrastive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=14:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# CONTRASTIVE TRAINING SWEEP (Sweep 5)
# =============================================================================
#
# Purpose: Test 3 improvements toward perfect performance:
#   1. LoRA layer restriction (0-20 vs 0-31)
#   2. gamma_kl increase (0.1 vs 0.3)
#   3. Contrastive pair training + injection_aware lossmask
#
# Grid: 6 runs, each progressive:
#   Run 1: baseline (LoRA 0-31, gamma=0.1, 150 steps)
#   Run 2: LoRA 0-20
#   Run 3: LoRA 0-20 + gamma=0.3
#   Run 4: LoRA 0-20 + gamma=0.3 + 300 steps
#   Run 5: LoRA 0-20 + gamma=0.3 + 300 steps + contrastive pairs
#   Run 6: LoRA 0-20 + gamma=0.3 + 300 steps + contrastive + injection_aware
#
# All use: triplet_full, cb_full_sequence, l10_20, dual, correct pooling
# Eval: next_tool_prediction only
#
# Usage:
#   sbatch slurm/pipeline/sweep_contrastive_v1.sbatch
#
#   # Quick sanity (20 steps, 5 eval samples):
#   QUICK_TEST=true sbatch slurm/pipeline/sweep_contrastive_v1.sbatch
#
#   # Run specific subset:
#   RUN_IDS="1,2,3" sbatch slurm/pipeline/sweep_contrastive_v1.sbatch
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
export TQDM_DISABLE=1

echo "========================================================"
echo "CONTRASTIVE TRAINING SWEEP (Sweep 5)"
echo "========================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Date: $(date)"
echo ""

# =============================================================================
# Configuration
# =============================================================================
QUICK_TEST="${QUICK_TEST:-false}"

if [[ "$QUICK_TEST" == "true" ]]; then
    echo "QUICK TEST MODE"
    DEFAULT_STEPS_SHORT=20
    DEFAULT_STEPS_LONG=30
    DEFAULT_LIMIT_EVAL=5
    DEFAULT_NUM_EXAMPLES=2
else
    DEFAULT_STEPS_SHORT=150
    DEFAULT_STEPS_LONG=300
    DEFAULT_LIMIT_EVAL=100
    DEFAULT_NUM_EXAMPLES=5
fi

LIMIT_EVAL="${LIMIT_EVAL:-$DEFAULT_LIMIT_EVAL}"
NUM_EXAMPLES="${NUM_EXAMPLES:-$DEFAULT_NUM_EXAMPLES}"

# Which runs to execute (default: all)
RUN_IDS="${RUN_IDS:-1,2,3,4,5,6}"

# Fixed hparams
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
DTYPE="${DTYPE:-bfloat16}"
LOSS_WEIGHTING="${LOSS_WEIGHTING:-dual}"
SAVE_STEPS="${SAVE_STEPS:-9999}"
TRIPLET_ALPHA_BENIGN="${TRIPLET_ALPHA_BENIGN:-0.5}"
TRIPLET_BETA_HARMFUL="${TRIPLET_BETA_HARMFUL:-0.4}"
TRIPLET_MIX_L2_WEIGHT="${TRIPLET_MIX_L2_WEIGHT:-0.5}"
TRIPLET_MIX_COS_WEIGHT="${TRIPLET_MIX_COS_WEIGHT:-0.5}"

# Fixed choices (settled from Sweeps 2-4)
LOSS_MODE="triplet_full"
POLICY="cb_full_sequence"
CB_LAYERS="10,20"
EVAL_MODE="next_tool_prediction"
POOLING_MODE="${POOLING_MODE:-correct}"
POOLING_DISTANCE="${POOLING_DISTANCE:-dcos}"
POOLING_MARGIN_B="${POOLING_MARGIN_B:-1.2}"
POOLING_MARGIN_H="${POOLING_MARGIN_H:-1.6}"
POOLING_MASK_POLICY="${POOLING_MASK_POLICY:-auto}"
POOLING_POLICY="${POOLING_POLICY:-cb_full_sequence}"

echo "Configuration:"
echo "  LOSS_MODE:      $LOSS_MODE (fixed)"
echo "  POLICY:         $POLICY (fixed)"
echo "  CB_LAYERS:      $CB_LAYERS (fixed)"
echo "  EVAL_MODE:      $EVAL_MODE (fixed)"
echo "  POOLING:        $POOLING_MODE / $POOLING_DISTANCE / $POOLING_MARGIN_B / $POOLING_MARGIN_H"
echo "  LIMIT_EVAL:     $LIMIT_EVAL"
echo "  RUN_IDS:        $RUN_IDS"
echo "  QUICK_TEST:     $QUICK_TEST"
echo ""

# =============================================================================
# Define the 6 runs
# =============================================================================
# Format: RUN_NAME|LORA_LAYERS|GAMMA_KL|STEPS|USE_CONTRASTIVE|HARMFUL_POLICY
declare -a RUNS=(
    "baseline_lora31_g01_s150|0-31|0.1|$DEFAULT_STEPS_SHORT|false|cb_full_sequence"
    "lora20_g01_s150|0-20|0.1|$DEFAULT_STEPS_SHORT|false|cb_full_sequence"
    "lora20_g03_s150|0-20|0.3|$DEFAULT_STEPS_SHORT|false|cb_full_sequence"
    "lora20_g03_s300|0-20|0.3|$DEFAULT_STEPS_LONG|false|cb_full_sequence"
    "lora20_g03_s300_contrastive|0-20|0.3|$DEFAULT_STEPS_LONG|true|cb_full_sequence"
    "lora20_g03_s300_contr_injaware|0-20|0.3|$DEFAULT_STEPS_LONG|true|injection_aware"
)

IFS=',' read -ra SELECTED_RUNS <<< "$RUN_IDS"

# =============================================================================
# Input Traces
# =============================================================================
TRACES_DIR="$CB_SCRATCH/data/traces"
DS_TRACES="$TRACES_DIR/fujitsu_b4_ds.jsonl"
DR_TRACES="$TRACES_DIR/fujitsu_b4_dr.jsonl"
TOOL_SCHEMA="$REPO_DIR/configs/tool_schemas/b4_standard_v1.json"

AGENTDOJO_AUGMENTED_SRC="$REPO_DIR/data/traces/agentdojo_augmented.jsonl"
AGENTDOJO_TRACES="$TRACES_DIR/agentdojo_augmented.jsonl"
CONTRASTIVE_PAIRS_SRC="$REPO_DIR/data/traces/contrastive_pairs.jsonl"
CONTRASTIVE_PAIRS="$TRACES_DIR/contrastive_pairs.jsonl"

# Copy data to scratch
if [[ ! -f "$AGENTDOJO_TRACES" ]]; then
    if [[ -f "$AGENTDOJO_AUGMENTED_SRC" ]]; then
        echo "Copying agentdojo_augmented.jsonl to scratch..."
        cp "$AGENTDOJO_AUGMENTED_SRC" "$AGENTDOJO_TRACES"
    else
        echo "ERROR: agentdojo_augmented.jsonl not found at $AGENTDOJO_AUGMENTED_SRC"
        exit 1
    fi
fi

# Copy contrastive pairs if available
if [[ -f "$CONTRASTIVE_PAIRS_SRC" ]]; then
    cp "$CONTRASTIVE_PAIRS_SRC" "$CONTRASTIVE_PAIRS"
    echo "Contrastive pairs: $(wc -l < "$CONTRASTIVE_PAIRS") pairs"
elif echo "$RUN_IDS" | grep -qE '(5|6)'; then
    echo "WARNING: contrastive_pairs.jsonl not found at $CONTRASTIVE_PAIRS_SRC"
    echo "  Runs 5,6 will fall back to standard SchemaDataset."
fi

echo ""
echo "Input traces:"
for trace in "$DS_TRACES" "$DR_TRACES" "$AGENTDOJO_TRACES"; do
    if [[ -f "$trace" ]]; then
        echo "  $(basename "$trace"): $(wc -l < "$trace") lines"
    else
        echo "  $(basename "$trace"): NOT FOUND"
        exit 1
    fi
done
echo ""

# =============================================================================
# Resolve Tokenizer
# =============================================================================
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
PRESET="${PRESET:-llama-3.1-8b-instruct}"
TOKENIZER="$MODEL_ID"

MODEL_CACHE_NAME=$(echo "$MODEL_ID" | sed 's/\//--/g')
SNAPSHOT_ROOT="$CACHE_DIR/hf/hub/models--${MODEL_CACHE_NAME}/snapshots"

if [[ -d "$SNAPSHOT_ROOT" ]]; then
    SNAPSHOT_PATH=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1)
    if [[ -n "$SNAPSHOT_PATH" ]]; then
        TOKENIZER="$SNAPSHOT_PATH"
        echo "Tokenizer: $SNAPSHOT_PATH"
    else
        echo "ERROR: Snapshot empty at $SNAPSHOT_ROOT"; exit 1
    fi
else
    echo "ERROR: Model not cached. Run slurm/cache_models.sh first."; exit 1
fi
echo ""

# =============================================================================
# Create Sweep Directory
# =============================================================================
SWEEP_ID="contrastive_$(date +%Y%m%d_%H%M%S)"
export SWEEP_DIR="$CB_SCRATCH/sweeps/$SWEEP_ID"
mkdir -p "$SWEEP_DIR"
echo "Sweep directory: $SWEEP_DIR"
echo ""

# =============================================================================
# Parse CB layers
# =============================================================================
IFS=',' read -ra CB_LAYER_ARRAY <<< "$CB_LAYERS"

# =============================================================================
# Main Sweep Loop
# =============================================================================
run_idx=0
success_count=0
fail_count=0
total_selected=${#SELECTED_RUNS[@]}

for run_num in "${SELECTED_RUNS[@]}"; do
    # Array is 0-indexed, run_num is 1-indexed
    arr_idx=$((run_num - 1))
    if [[ $arr_idx -lt 0 || $arr_idx -ge ${#RUNS[@]} ]]; then
        echo "WARNING: Invalid run number $run_num, skipping"
        continue
    fi

    IFS='|' read -r RUN_NAME LORA_LAYERS GAMMA_KL TOTAL_STEPS USE_CONTRASTIVE HARMFUL_POLICY <<< "${RUNS[$arr_idx]}"

    run_idx=$((run_idx + 1))
    RUN_DIR="$SWEEP_DIR/$RUN_NAME"

    echo "========================================================"
    echo "[$run_idx/$total_selected] Run $run_num: $RUN_NAME"
    echo "========================================================"
    echo "  LoRA layers:     $LORA_LAYERS"
    echo "  gamma_kl:        $GAMMA_KL"
    echo "  Steps:           $TOTAL_STEPS"
    echo "  Contrastive:     $USE_CONTRASTIVE"
    echo "  Harmful policy:  $HARMFUL_POLICY"
    echo ""

    mkdir -p "$RUN_DIR"/{renders,lossmasks,model,eval}

    # =========================================================================
    # Step 1: ETL_B
    # =========================================================================
    echo "Step 1: ETL_B (policy=$HARMFUL_POLICY for harmful, cb_full_sequence for benign)"
    etl_start=$(date +%s)

    for trace_file in "$DS_TRACES" "$DR_TRACES" "$AGENTDOJO_TRACES"; do
        basename_f=$(basename "$trace_file" .jsonl)
        render_out="$RUN_DIR/renders/${basename_f}.jsonl"
        lossmask_out="$RUN_DIR/lossmasks/${basename_f}.jsonl"

        ETL_B_EXTRA=()
        if [[ "$basename_f" == "fujitsu_b4_ds" ]]; then
            ETL_B_EXTRA+=(--allow-skeleton --skeleton-policy cb_full_sequence)
        fi

        # Use harmful policy for DS/harmful data, cb_full_sequence for DR/benign
        if [[ "$basename_f" == "fujitsu_b4_dr" ]]; then
            ETL_POLICY="cb_full_sequence"
        else
            ETL_POLICY="$HARMFUL_POLICY"
        fi

        echo "  Processing: $basename_f (policy=$ETL_POLICY)"
        python src/schemas/tools/ETL_B.py \
            --traces "$trace_file" \
            --render-out "$render_out" \
            --lossmask-out "$lossmask_out" \
            --tokenizer "$TOKENIZER" \
            --max-length "$MAX_SEQ_LENGTH" \
            --policy-override "$ETL_POLICY" \
            --pooling-policy-override "$POOLING_POLICY" \
            "${ETL_B_EXTRA[@]}" 2>&1 | tail -3
    done

    etl_end=$(date +%s)
    echo "  ETL_B done ($(( etl_end - etl_start ))s)"

    # Mask coverage summary
    echo ""
    echo "  Lossmask coverage:"
    for lm in "$RUN_DIR/lossmasks"/*.jsonl; do
        python3 -c "
import json, sys
total_samples = 0; zero_samples = 0; total_ratio = 0
with open('$lm') as f:
    for line in f:
        if not line.strip(): continue
        d = json.loads(line)
        s = d.get('stats', {})
        total_samples += 1
        ratio = s.get('mask_ratio', 0)
        total_ratio += ratio
        if ratio == 0:
            zero_samples += 1
avg = total_ratio / total_samples if total_samples > 0 else 0
print(f'    $(basename "$lm"): {total_samples} samples, avg_coverage={avg:.1%}, zero_mask={zero_samples}')
" 2>/dev/null
    done
    echo ""

    # =========================================================================
    # Step 2: Split AgentDojo
    # =========================================================================
    echo "Step 2: Splitting AgentDojo"
    SPLIT_DIR="$RUN_DIR/agentdojo_split"
    mkdir -p "$SPLIT_DIR"

    python "$REPO_DIR/scripts/split_agentdojo.py" \
        --traces "$AGENTDOJO_TRACES" \
        --renders "$RUN_DIR/renders/agentdojo_augmented.jsonl" \
        --lossmasks "$RUN_DIR/lossmasks/agentdojo_augmented.jsonl" \
        --output-dir "$SPLIT_DIR" \
        --legacy-harmful-naming 2>&1 | tail -5

    echo "  Split results:"
    for f in "$SPLIT_DIR"/*.jsonl; do
        bn=$(basename "$f")
        n=$(wc -l < "$f")
        echo "    $bn: $n lines"
    done
    echo ""

    # =========================================================================
    # Step 3: Save config
    # =========================================================================
    cat > "$RUN_DIR/run_config.json" << RUNCFG
{
  "run_number": $run_num,
  "run_name": "$RUN_NAME",
  "loss_mode": "$LOSS_MODE",
  "policy": "$HARMFUL_POLICY",
  "layers": "$CB_LAYERS",
  "lora_target_layers": "$LORA_LAYERS",
  "total_steps": $TOTAL_STEPS,
  "batch_size": $BATCH_SIZE,
  "grad_accum": $GRAD_ACCUM,
  "effective_batch": $(( BATCH_SIZE * GRAD_ACCUM )),
  "learning_rate": "$LEARNING_RATE",
  "loss_weighting": "$LOSS_WEIGHTING",
  "gamma_kl": $GAMMA_KL,
  "pooling_mode": "$POOLING_MODE",
  "pooling_distance": "$POOLING_DISTANCE",
  "pooling_margin_b": $POOLING_MARGIN_B,
  "pooling_margin_h": $POOLING_MARGIN_H,
  "pooling_policy": "$POOLING_POLICY",
  "pooling_mask_policy": "$POOLING_MASK_POLICY",
  "lora_r": $LORA_R,
  "lora_alpha": $LORA_ALPHA,
  "alpha_benign": $TRIPLET_ALPHA_BENIGN,
  "beta_harmful": $TRIPLET_BETA_HARMFUL,
  "use_contrastive": $USE_CONTRASTIVE,
  "eval_mode": "$EVAL_MODE",
  "model": "$TOKENIZER",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "job_id": "${SLURM_JOB_ID:-local}"
}
RUNCFG

    # =========================================================================
    # Step 4: Training
    # =========================================================================
    echo "Step 3: Training ($LOSS_MODE, $TOTAL_STEPS steps, LoRA=$LORA_LAYERS, gamma=$GAMMA_KL)"

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

    TRAIN_ARGS=(
        --preset "$PRESET"
        --model "$TOKENIZER"
        --output-dir "$RUN_DIR/model"
        --total-steps "$TOTAL_STEPS"
        --batch-size "$BATCH_SIZE"
        --learning-rate "$LEARNING_RATE"
        --warmup-steps "$WARMUP_STEPS"
        --alpha-max 10.0
        --max-seq-length "$MAX_SEQ_LENGTH"
        --gradient-accumulation-steps "$GRAD_ACCUM"
        --loss-mode "$LOSS_MODE"
        --loss-weighting "$LOSS_WEIGHTING"
        --triplet-alpha-benign "$TRIPLET_ALPHA_BENIGN"
        --triplet-beta-harmful "$TRIPLET_BETA_HARMFUL"
        --triplet-gamma-kl "$GAMMA_KL"
        --triplet-margin-benign "$POOLING_MARGIN_B"
        --triplet-margin-harmful "$POOLING_MARGIN_H"
        --triplet-benign-positive-distance "$POOLING_DISTANCE"
        --triplet-benign-negative-distance "$POOLING_DISTANCE"
        --triplet-harmful-positive-distance "$POOLING_DISTANCE"
        --triplet-harmful-negative-distance "$POOLING_DISTANCE"
        --triplet-mix-l2-weight "$TRIPLET_MIX_L2_WEIGHT"
        --triplet-mix-cos-weight "$TRIPLET_MIX_COS_WEIGHT"
        --pooling-mode "$POOLING_MODE"
        --pooling-mask-policy "$POOLING_MASK_POLICY"
        --lora-r "$LORA_R"
        --lora-alpha "$LORA_ALPHA"
        --lora-dropout 0.05
        --lora-target-layers "$LORA_LAYERS"
        --logging-steps 10
        --save-steps "$SAVE_STEPS"
        --no-wandb
        --cb-target-layers "${CB_LAYER_ARRAY[@]}"
        --mode mixed
        --harmful-renders "${HARMFUL_RENDER_ARGS[@]}"
        --harmful-lossmasks "${HARMFUL_LOSSMASK_ARGS[@]}"
        --benign-renders "${BENIGN_RENDER_ARGS[@]}"
        --benign-lossmasks "${BENIGN_LOSSMASK_ARGS[@]}"
    )

    # Add contrastive pairs if enabled
    if [[ "$USE_CONTRASTIVE" == "true" && -f "$CONTRASTIVE_PAIRS" ]]; then
        TRAIN_ARGS+=(--contrastive-pairs "$CONTRASTIVE_PAIRS")
        echo "  Using contrastive pairs: $CONTRASTIVE_PAIRS"
    fi

    train_start=$(date +%s)
    if python src/training/train_schema.py "${TRAIN_ARGS[@]}" > "$RUN_DIR/train.log" 2>&1; then
        train_status="SUCCESS"
    else
        train_status="FAILED"
    fi
    train_end=$(date +%s)
    train_mins=$(( (train_end - train_start) / 60 ))
    echo "  Training $train_status (${train_mins}m)"

    if [[ "$train_status" == "FAILED" ]]; then
        echo "  Last 10 lines of train.log:"
        tail -10 "$RUN_DIR/train.log" | sed 's/^/    /'
        fail_count=$((fail_count + 1))
        echo ""
        continue
    fi

    # Training diagnostics
    echo ""
    echo "  Training diagnostics:"
    python3 << 'TRAINDIAG'
import re, sys

log_path = sys.argv[1] if len(sys.argv) > 1 else ""
if not log_path:
    sys.exit(0)

steps = []
with open(log_path) as f:
    for line in f:
        m = re.search(r'step[= ]+(\d+).*?loss[= ]+([0-9.e+-]+)', line, re.IGNORECASE)
        if m:
            step = int(m.group(1))
            loss = float(m.group(2))
            steps.append((step, loss, line.strip()))

if steps:
    print(f"    Steps logged: {len(steps)}")
    for label, idx in [("First", 0), ("Mid", len(steps)//2), ("Last", -1)]:
        s, l, raw = steps[idx]
        print(f"    {label} (step {s}): {raw[:120]}")
TRAINDIAG
    echo ""

    # =========================================================================
    # Step 5: Evaluation (NTP only)
    # =========================================================================
    echo "Step 4: Evaluation ($EVAL_MODE)"

    # Find adapter
    if [[ -d "$RUN_DIR/model/final" ]]; then
        ADAPTER_PATH="$RUN_DIR/model/final"
    elif [[ -f "$RUN_DIR/model/adapter_config.json" ]]; then
        ADAPTER_PATH="$RUN_DIR/model"
    else
        echo "  WARNING: No adapter found, skipping eval"
        fail_count=$((fail_count + 1))
        continue
    fi

    EVAL_SUBDIR="$RUN_DIR/eval/$EVAL_MODE"
    mkdir -p "$EVAL_SUBDIR"

    # --- Fujitsu eval ---
    echo "  Fujitsu eval ($LIMIT_EVAL samples)..."
    FUJITSU_OUTPUT="$EVAL_SUBDIR/fujitsu_eval.json"
    python src/evaluation/eval.py \
        --baseline "$TOKENIZER" \
        --cb-adapter "$ADAPTER_PATH" \
        --eval-data "$DS_TRACES" \
        --tool-schema "$TOOL_SCHEMA" \
        --output "$FUJITSU_OUTPUT" \
        --limit "$LIMIT_EVAL" \
        --merge-adapter \
        --eval-mode "$EVAL_MODE" \
        --dtype "$DTYPE" >> "$EVAL_SUBDIR/fujitsu.log" 2>&1

    if [[ -f "$FUJITSU_OUTPUT" ]]; then
        python3 -c "
import json
with open('$FUJITSU_OUTPUT') as f: r = json.load(f)
for lbl, k in [('Baseline','baseline'),('CB','cb_model')]:
    tf = r.get(k,{}).get('tool_flip_asr',{})
    if not tf: continue
    print(f'    {lbl}: ASR={tf.get(\"attack_success_rate\",0):.0%} correct={tf.get(\"correct_tool_call_rate\",0):.0%} no_tool={tf.get(\"no_tool_call_rate\",0):.0%}')
d = r.get('delta',{})
if d:
    print(f'    Delta: ASR={d.get(\"tool_flip_asr\",0):+.0%}')
"
    fi

    # --- AgentDojo harmful eval ---
    echo "  AgentDojo harmful eval ($LIMIT_EVAL samples)..."
    AD_HARMFUL_OUTPUT="$EVAL_SUBDIR/agentdojo_eval.json"
    if [[ -f "$SPLIT_DIR/agentdojo_traces_harmful.jsonl" ]]; then
        python src/evaluation/eval.py \
            --baseline "$TOKENIZER" \
            --cb-adapter "$ADAPTER_PATH" \
            --eval-data "$SPLIT_DIR/agentdojo_traces_harmful.jsonl" \
            --tool-schema "$TOOL_SCHEMA" \
            --output "$AD_HARMFUL_OUTPUT" \
            --use-sample-context \
            --limit "$LIMIT_EVAL" \
            --merge-adapter \
            --eval-mode "$EVAL_MODE" \
            --dtype "$DTYPE" >> "$EVAL_SUBDIR/agentdojo.log" 2>&1

        if [[ -f "$AD_HARMFUL_OUTPUT" ]]; then
            python3 -c "
import json
with open('$AD_HARMFUL_OUTPUT') as f: r = json.load(f)
cb = r.get('cb_model',{}).get('generation_comparison',{})
if cb:
    print(f'    AD Harmful: resist={cb.get(\"harmful_resistance_rate\",0):.0%} malicious={cb.get(\"malicious_tool_call_rate\",0):.0%} (n={cb.get(\"total_samples\",0)})')
"
        fi
    fi

    # --- AgentDojo benign eval ---
    echo "  AgentDojo benign eval ($LIMIT_EVAL samples)..."
    AD_BENIGN_OUTPUT="$EVAL_SUBDIR/agentdojo_benign_eval.json"
    if [[ -f "$SPLIT_DIR/agentdojo_traces_benign.jsonl" ]]; then
        python src/evaluation/eval.py \
            --baseline "$TOKENIZER" \
            --cb-adapter "$ADAPTER_PATH" \
            --eval-data "$SPLIT_DIR/agentdojo_traces_benign.jsonl" \
            --tool-schema "$TOOL_SCHEMA" \
            --output "$AD_BENIGN_OUTPUT" \
            --use-sample-context \
            --limit "$LIMIT_EVAL" \
            --merge-adapter \
            --eval-mode "$EVAL_MODE" \
            --dtype "$DTYPE" >> "$EVAL_SUBDIR/agentdojo_benign.log" 2>&1

        if [[ -f "$AD_BENIGN_OUTPUT" ]]; then
            python3 -c "
import json
with open('$AD_BENIGN_OUTPUT') as f: r = json.load(f)
cb = r.get('cb_model',{}).get('generation_comparison',{})
if cb:
    cor = cb.get('correct_tool_call_rate',0)
    nt = cb.get('no_tool_call_rate',0)
    n = cb.get('total_samples',0)
    print(f'    AD Benign:  correct={cor:.0%} no_tool={nt:.0%} (n={n})')
"
        fi
    fi

    # =========================================================================
    # Step 6: Examples
    # =========================================================================
    echo ""
    echo "  ========================================"
    echo "  EXAMPLES: $RUN_NAME"
    echo "  ========================================"

    # Fujitsu examples
    FUJITSU_PAIRED="$EVAL_SUBDIR/fujitsu_eval.paired_outputs.jsonl"
    if [[ -f "$FUJITSU_PAIRED" ]]; then
        python3 << PYEX
import json

with open('$FUJITSU_PAIRED') as f:
    pairs = [json.loads(l) for l in f if l.strip()]

def gib(r):
    if not r: return False
    w = r.split()
    return len(w) > 10 and len(set(w))/len(w) < 0.15

imps = [p for p in pairs if p.get('baseline_outcome')=='attack_success' and p.get('cb_outcome')!='attack_success']
regs = [p for p in pairs if p.get('baseline_outcome')!='attack_success' and p.get('cb_outcome')=='attack_success']
n_cor = sum(1 for p in pairs if p.get('cb_outcome')=='correct_behavior')
n_atk = sum(1 for p in pairs if p.get('cb_outcome')=='attack_success')
n_nt = sum(1 for p in pairs if p.get('cb_outcome')=='no_tool_call')
n_gib = sum(1 for p in pairs if gib(p.get('cb_response','')))

print(f'  Fujitsu: {len(pairs)} pairs | improved={len(imps)} regressed={len(regs)}')
print(f'  CB outcomes: correct={n_cor} attack={n_atk} no_tool={n_nt} gib={n_gib}')
print()

for i, p in enumerate((imps+regs)[:$NUM_EXAMPLES], 1):
    lbl = 'IMP' if p.get('baseline_outcome')=='attack_success' else 'REG'
    g = ' [GIB]' if gib(p.get('cb_response','')) else ''
    print(f'  [{i}] {lbl}: {p.get("baseline_outcome","?")} -> {p.get("cb_outcome","?")}{g}')
    print(f'      expected={p.get("expected_tool","?")} simulated={p.get("simulated_tool","?")}')
    print(f'  BASELINE: {(p.get("baseline_response") or "(empty)")[:250]}')
    print(f'  CB MODEL: {(p.get("cb_response") or "(empty)")[:250]}')
    print()
PYEX
    fi

    # AgentDojo benign examples
    AD_BENIGN_PAIRED="$EVAL_SUBDIR/agentdojo_benign_eval.paired_outputs.jsonl"
    if [[ -f "$AD_BENIGN_PAIRED" ]]; then
        python3 << PYEX2
import json

with open('$AD_BENIGN_PAIRED') as f:
    pairs = [json.loads(l) for l in f if l.strip()]

n_ok = sum(1 for p in pairs if p.get('cb_observed_tool'))
n_brk = sum(1 for p in pairs if not p.get('cb_observed_tool'))

print(f'  AD Benign: {len(pairs)} pairs | tool_ok={n_ok} broken={n_brk}')
print()

broken = [p for p in pairs if p.get('baseline_observed_tool') and not p.get('cb_observed_tool')]
if broken:
    print(f'  Broken examples (baseline had tool, CB lost it):')
    for i, p in enumerate(broken[:$NUM_EXAMPLES], 1):
        print(f'  [{i}] BROKEN: {p.get("baseline_observed_tool","?")} -> none')
        print(f'  CB MODEL: {(p.get("cb_response") or "(empty)")[:250]}')
        print()
PYEX2
    fi

    # Generate per-run summary
    python scripts/generate_run_summary.py --run-dir "$RUN_DIR" 2>&1 || true

    success_count=$((success_count + 1))
    echo ""

done  # run loop

# =============================================================================
# Consolidated Results
# =============================================================================
echo ""
echo "========================================================"
echo "CONSOLIDATED RESULTS"
echo "========================================================"
echo ""

python3 << 'RESULTS'
import json, os, sys

sweep_dir = os.environ.get("SWEEP_DIR", "")
if not sweep_dir:
    sys.exit(0)

rows = []
for run_name in sorted(os.listdir(sweep_dir)):
    run_dir = os.path.join(sweep_dir, run_name)
    config_path = os.path.join(run_dir, "run_config.json")
    if not os.path.isfile(config_path):
        continue

    with open(config_path) as f:
        cfg = json.load(f)

    row = {"run": run_name, "cfg": cfg}

    eval_dir = os.path.join(run_dir, "eval", "next_tool_prediction")
    if not os.path.isdir(eval_dir):
        rows.append(row)
        continue

    fuj = os.path.join(eval_dir, "fujitsu_eval.json")
    if os.path.isfile(fuj):
        with open(fuj) as f:
            r = json.load(f)
        cb_tf = r.get("cb_model", {}).get("tool_flip_asr", {})
        row["fuj_asr"] = cb_tf.get("attack_success_rate", -1)
        row["fuj_cor"] = cb_tf.get("correct_tool_call_rate", -1)
        row["fuj_nt"] = cb_tf.get("no_tool_call_rate", -1)

    adh = os.path.join(eval_dir, "agentdojo_eval.json")
    if os.path.isfile(adh):
        with open(adh) as f:
            r = json.load(f)
        cb = r.get("cb_model", {}).get("generation_comparison", {})
        row["ad_res"] = cb.get("harmful_resistance_rate", -1)
        row["ad_mal"] = cb.get("malicious_tool_call_rate", -1)

    adb = os.path.join(eval_dir, "agentdojo_benign_eval.json")
    if os.path.isfile(adb):
        with open(adb) as f:
            r = json.load(f)
        cb = r.get("cb_model", {}).get("generation_comparison", {})
        row["ben_cor"] = cb.get("correct_tool_call_rate", -1)
        row["ben_nt"] = cb.get("no_tool_call_rate", -1)

    rows.append(row)

if not rows:
    print("  No results found.")
    sys.exit(0)

fmt = lambda v: f"{v:.0%}" if isinstance(v, (int, float)) and v >= 0 else "n/a"

print(f"{'#':>2} {'Run':<40} | {'Fuj ASR':>7} | {'AD Res':>6} | {'Ben Cor':>7} {'Ben NT':>6} | {'LoRA':>5} {'g_kl':>4} {'Steps':>5}")
print("-" * 105)
for row in rows:
    r = row["run"]
    if len(r) > 39:
        r = r[:36] + "..."
    c = row.get("cfg", {})
    num = c.get("run_number", "?")
    lora = c.get("lora_target_layers", "?")
    gkl = c.get("gamma_kl", "?")
    steps = c.get("total_steps", "?")

    print(f"{num:>2} {r:<40} | {fmt(row.get('fuj_asr', -1)):>7} | {fmt(row.get('ad_res', -1)):>6} | {fmt(row.get('ben_cor', -1)):>7} {fmt(row.get('ben_nt', -1)):>6} | {lora:>5} {gkl:>4} {steps:>5}")

print()
print("Fuj ASR = Fujitsu attack success rate (lower is better)")
print("AD Res = AgentDojo harmful resistance (higher is better)")
print("Ben Cor = benign correct tool (higher is better)")
print("Ben NT = benign no_tool rate (lower is better)")
RESULTS

# Generate sweep summary
python scripts/generate_sweep_summary.py --sweep-dir "$SWEEP_DIR" 2>&1 || true

echo ""
echo "========================================================"
echo "SWEEP COMPLETE"
echo "========================================================"
echo "Runs attempted: $run_idx | Success: $success_count | Failed: $fail_count"
echo "Output: $SWEEP_DIR"
echo "Date: $(date)"
