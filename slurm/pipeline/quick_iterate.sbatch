#!/bin/bash
#SBATCH --job-name=quick_iter
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# QUICK ITERATE: Single-config train + eval
# =============================================================================
#
# Fast iteration on a single training config. Defaults to best known
# config (trip_cfs_l10_20) with configurable hparams.
#
# Usage:
#   # Default: 50 steps, 30 eval samples, trip_cfs_l10_20
#   sbatch slurm/pipeline/quick_iterate.sbatch
#
#   # Sweep gamma_kl:
#   GAMMA_KL=0.3 sbatch slurm/pipeline/quick_iterate.sbatch
#   GAMMA_KL=0.5 sbatch slurm/pipeline/quick_iterate.sbatch
#
#   # Full 150-step run:
#   TOTAL_STEPS=150 LIMIT_EVAL=100 sbatch slurm/pipeline/quick_iterate.sbatch
#
#   # Try different loss mode:
#   LOSS_MODE=legacy_cb BETA_KL=0.5 sbatch slurm/pipeline/quick_iterate.sbatch
#
#   # Baseline-only eval (no training):
#   BASELINE_ONLY=true sbatch slurm/pipeline/quick_iterate.sbatch
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

# =============================================================================
# Configuration — all overridable via env vars
# =============================================================================
TOTAL_STEPS="${TOTAL_STEPS:-100}"
LIMIT_EVAL="${LIMIT_EVAL:-50}"
NUM_EXAMPLES="${NUM_EXAMPLES:-3}"
BASELINE_ONLY="${BASELINE_ONLY:-false}"

# Single config (defaults to best: trip_cfs_l10_20)
LOSS_MODE="${LOSS_MODE:-triplet_full}"
POLICY="${POLICY:-cb_full_sequence}"
LAYERS="${LAYERS:-10,20}"
EVAL_MODE="${EVAL_MODE:-next_tool_prediction}"

# Training hparams
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
DTYPE="${DTYPE:-bfloat16}"
LOSS_WEIGHTING="${LOSS_WEIGHTING:-dual}"
GAMMA_KL="${GAMMA_KL:-0.1}"
BETA_KL="${BETA_KL:-0.5}"
TRIPLET_ALPHA_BENIGN="${TRIPLET_ALPHA_BENIGN:-0.5}"
TRIPLET_BETA_HARMFUL="${TRIPLET_BETA_HARMFUL:-0.4}"
TRIPLET_MIX_L2_WEIGHT="${TRIPLET_MIX_L2_WEIGHT:-0.5}"
TRIPLET_MIX_COS_WEIGHT="${TRIPLET_MIX_COS_WEIGHT:-0.5}"
SAVE_STEPS="${SAVE_STEPS:-9999}"

# Pooling
POOLING_MODE="${POOLING_MODE:-correct}"
POOLING_DISTANCE="${POOLING_DISTANCE:-dcos}"
POOLING_MARGIN_B="${POOLING_MARGIN_B:-1.2}"
POOLING_MARGIN_H="${POOLING_MARGIN_H:-1.6}"
POOLING_MASK_POLICY="${POOLING_MASK_POLICY:-auto}"
POOLING_POLICY="${POOLING_POLICY:-cb_full_sequence}"

# Run tag (auto-generated from key params)
RUN_TAG="${RUN_TAG:-}"
if [[ -z "$RUN_TAG" ]]; then
    lm_short="trip"
    [[ "$LOSS_MODE" == "legacy_cb" ]] && lm_short="lcb"
    pol_short="${POLICY}"
    [[ "$POLICY" == "cb_full_sequence" ]] && pol_short="cfs"
    [[ "$POLICY" == "post_injection_only" ]] && pol_short="pio"
    RUN_TAG="${lm_short}_${pol_short}_s${TOTAL_STEPS}_gkl${GAMMA_KL}"
fi

echo "========================================================"
echo "QUICK ITERATE: $RUN_TAG"
echo "========================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Date: $(date)"
echo ""
echo "Config:"
echo "  LOSS_MODE:      $LOSS_MODE"
echo "  POLICY:         $POLICY"
echo "  LAYERS:         $LAYERS"
echo "  TOTAL_STEPS:    $TOTAL_STEPS"
echo "  GAMMA_KL:       $GAMMA_KL"
echo "  EVAL_MODE:      $EVAL_MODE"
echo "  LIMIT_EVAL:     $LIMIT_EVAL"
echo "  BASELINE_ONLY:  $BASELINE_ONLY"
echo ""

# =============================================================================
# Input Traces
# =============================================================================
TRACES_DIR="$CB_SCRATCH/data/traces"
DS_TRACES="$TRACES_DIR/fujitsu_b4_ds.jsonl"
DR_TRACES="$TRACES_DIR/fujitsu_b4_dr.jsonl"
TOOL_SCHEMA="$REPO_DIR/configs/tool_schemas/b4_standard_v1.json"

AGENTDOJO_AUGMENTED_SRC="$REPO_DIR/data/traces/agentdojo_augmented.jsonl"
AGENTDOJO_TRACES="$TRACES_DIR/agentdojo_augmented.jsonl"

if [[ ! -f "$AGENTDOJO_TRACES" ]]; then
    if [[ -f "$AGENTDOJO_AUGMENTED_SRC" ]]; then
        cp "$AGENTDOJO_AUGMENTED_SRC" "$AGENTDOJO_TRACES"
    else
        echo "ERROR: agentdojo_augmented.jsonl not found"; exit 1
    fi
fi

for trace in "$DS_TRACES" "$DR_TRACES" "$AGENTDOJO_TRACES"; do
    if [[ ! -f "$trace" ]]; then
        echo "ERROR: $(basename "$trace") not found"; exit 1
    fi
done

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
    else
        echo "ERROR: Snapshot empty"; exit 1
    fi
else
    echo "ERROR: Model not cached"; exit 1
fi

# =============================================================================
# Create Run Directory
# =============================================================================
RUN_DIR="$CB_SCRATCH/quick/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"/{renders,lossmasks,model,eval}
echo "Output: $RUN_DIR"
echo ""

# =============================================================================
# Baseline-only eval (skip training)
# =============================================================================
if [[ "$BASELINE_ONLY" == "true" ]]; then
    echo "========================================================"
    echo "BASELINE-ONLY EVAL (no training)"
    echo "========================================================"

    # Fujitsu
    echo "Fujitsu eval ($LIMIT_EVAL samples, $EVAL_MODE)..."
    python src/evaluation/eval.py \
        --baseline "$TOKENIZER" \
        --eval-data "$DS_TRACES" \
        --tool-schema "$TOOL_SCHEMA" \
        --output "$RUN_DIR/eval/fujitsu_eval.json" \
        --limit "$LIMIT_EVAL" \
        --eval-mode "$EVAL_MODE" \
        --dtype "$DTYPE" 2>&1 | tail -5

    if [[ -f "$RUN_DIR/eval/fujitsu_eval.json" ]]; then
        python3 -c "
import json
with open('$RUN_DIR/eval/fujitsu_eval.json') as f: r = json.load(f)
b = r.get('baseline',{}).get('tool_flip_asr',{})
if b:
    print(f'  Baseline: ASR={b.get(\"attack_success_rate\",0):.0%} correct={b.get(\"correct_tool_call_rate\",0):.0%} no_tool={b.get(\"no_tool_call_rate\",0):.0%}')
"
    fi

    # Split AgentDojo traces only (no renders/lossmasks needed for baseline eval)
    SPLIT_DIR="$RUN_DIR/agentdojo_split"
    mkdir -p "$SPLIT_DIR"
    python3 -c "
import json
benign, harmful = [], []
with open('$AGENTDOJO_TRACES') as f:
    for line in f:
        if not line.strip(): continue
        d = json.loads(line)
        cat = d.get('category', d.get('metadata', {}).get('category', ''))
        if cat in ('benign', 'resisted'):
            benign.append(line)
        else:
            harmful.append(line)
with open('$SPLIT_DIR/agentdojo_traces_benign.jsonl', 'w') as f: f.writelines(benign)
with open('$SPLIT_DIR/agentdojo_traces_harmful.jsonl', 'w') as f: f.writelines(harmful)
print(f'  Split: {len(benign)} benign, {len(harmful)} harmful')
"

    # AD benign
    if [[ -f "$SPLIT_DIR/agentdojo_traces_benign.jsonl" ]]; then
        echo "AgentDojo benign eval ($LIMIT_EVAL samples, $EVAL_MODE)..."
        python src/evaluation/eval.py \
            --baseline "$TOKENIZER" \
            --eval-data "$SPLIT_DIR/agentdojo_traces_benign.jsonl" \
            --tool-schema "$TOOL_SCHEMA" \
            --output "$RUN_DIR/eval/agentdojo_benign_eval.json" \
            --use-sample-context \
            --limit "$LIMIT_EVAL" \
            --eval-mode "$EVAL_MODE" \
            --dtype "$DTYPE" 2>&1 | tail -5

        if [[ -f "$RUN_DIR/eval/agentdojo_benign_eval.json" ]]; then
            python3 -c "
import json
with open('$RUN_DIR/eval/agentdojo_benign_eval.json') as f: r = json.load(f)
b = r.get('baseline',{}).get('generation_comparison',{})
if b:
    print(f'  Baseline Benign: correct={b.get(\"correct_tool_call_rate\",0):.0%} no_tool={b.get(\"no_tool_call_rate\",0):.0%} (n={b.get(\"total_samples\",0)})')
"
        fi
    fi

    # AD harmful
    if [[ -f "$SPLIT_DIR/agentdojo_traces_harmful.jsonl" ]]; then
        echo "AgentDojo harmful eval ($LIMIT_EVAL samples, $EVAL_MODE)..."
        python src/evaluation/eval.py \
            --baseline "$TOKENIZER" \
            --eval-data "$SPLIT_DIR/agentdojo_traces_harmful.jsonl" \
            --tool-schema "$TOOL_SCHEMA" \
            --output "$RUN_DIR/eval/agentdojo_eval.json" \
            --use-sample-context \
            --limit "$LIMIT_EVAL" \
            --eval-mode "$EVAL_MODE" \
            --dtype "$DTYPE" 2>&1 | tail -5

        if [[ -f "$RUN_DIR/eval/agentdojo_eval.json" ]]; then
            python3 -c "
import json
with open('$RUN_DIR/eval/agentdojo_eval.json') as f: r = json.load(f)
b = r.get('baseline',{}).get('generation_comparison',{})
if b:
    print(f'  Baseline Harmful: resist={b.get(\"harmful_resistance_rate\",0):.0%} malicious={b.get(\"malicious_tool_call_rate\",0):.0%} (n={b.get(\"total_samples\",0)})')
"
        fi
    fi

    echo ""
    echo "Baseline eval complete. Output: $RUN_DIR"
    exit 0
fi

# =============================================================================
# ETL_B
# =============================================================================
echo "Step 1: ETL_B (policy=$POLICY)"
etl_start=$(date +%s)

IFS=',' read -ra CB_LAYER_ARRAY <<< "$LAYERS"

for trace_file in "$DS_TRACES" "$DR_TRACES" "$AGENTDOJO_TRACES"; do
    basename=$(basename "$trace_file" .jsonl)
    ETL_B_EXTRA=()
    [[ "$basename" == "fujitsu_b4_ds" ]] && ETL_B_EXTRA+=(--allow-skeleton --skeleton-policy cb_full_sequence)

    python src/schemas/tools/ETL_B.py \
        --traces "$trace_file" \
        --render-out "$RUN_DIR/renders/${basename}.jsonl" \
        --lossmask-out "$RUN_DIR/lossmasks/${basename}.jsonl" \
        --tokenizer "$TOKENIZER" \
        --max-length "$MAX_SEQ_LENGTH" \
        --policy-override "$POLICY" \
        --pooling-policy-override "$POOLING_POLICY" \
        "${ETL_B_EXTRA[@]}" 2>&1 | tail -1
done

etl_end=$(date +%s)
echo "  ETL_B done ($(( etl_end - etl_start ))s)"

# =============================================================================
# Split AgentDojo
# =============================================================================
SPLIT_DIR="$RUN_DIR/agentdojo_split"
mkdir -p "$SPLIT_DIR"

python "$REPO_DIR/scripts/split_agentdojo.py" \
    --traces "$AGENTDOJO_TRACES" \
    --renders "$RUN_DIR/renders/agentdojo_augmented.jsonl" \
    --lossmasks "$RUN_DIR/lossmasks/agentdojo_augmented.jsonl" \
    --output-dir "$SPLIT_DIR" \
    --legacy-harmful-naming 2>&1 | tail -3

# =============================================================================
# Save config
# =============================================================================
cat > "$RUN_DIR/run_config.json" << RUNCFG
{
  "loss_mode": "$LOSS_MODE",
  "policy": "$POLICY",
  "layers": "$LAYERS",
  "layers_short": "${LAYERS//,/_}",
  "total_steps": $TOTAL_STEPS,
  "batch_size": $BATCH_SIZE,
  "grad_accum": $GRAD_ACCUM,
  "effective_batch": $(( BATCH_SIZE * GRAD_ACCUM )),
  "learning_rate": "$LEARNING_RATE",
  "loss_weighting": "$LOSS_WEIGHTING",
  "gamma_kl": $GAMMA_KL,
  "beta_kl": $BETA_KL,
  "pooling_mode": "$POOLING_MODE",
  "pooling_distance": "$POOLING_DISTANCE",
  "pooling_margin_b": $POOLING_MARGIN_B,
  "pooling_margin_h": $POOLING_MARGIN_H,
  "pooling_policy": "$POOLING_POLICY",
  "pooling_mask_policy": "$POOLING_MASK_POLICY",
  "lora_r": $LORA_R,
  "lora_alpha": $LORA_ALPHA,
  "preset": "$PRESET",
  "eval_mode": "$EVAL_MODE",
  "model": "$TOKENIZER",
  "run_name": "$RUN_TAG",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "job_id": "${SLURM_JOB_ID:-local}"
}
RUNCFG

# =============================================================================
# Training
# =============================================================================
echo ""
echo "Step 2: Training ($LOSS_MODE, $TOTAL_STEPS steps, gamma_kl=$GAMMA_KL)"

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
    --logging-steps 10
    --save-steps "$SAVE_STEPS"
    --no-wandb
    --cb-target-layers "${CB_LAYER_ARRAY[@]}"
    --mode mixed
    --harmful-renders "$RUN_DIR/renders/fujitsu_b4_ds.jsonl" "$SPLIT_DIR/agentdojo_renders_harmful.jsonl"
    --harmful-lossmasks "$RUN_DIR/lossmasks/fujitsu_b4_ds.jsonl" "$SPLIT_DIR/agentdojo_lossmasks_harmful.jsonl"
    --benign-renders "$RUN_DIR/renders/fujitsu_b4_dr.jsonl" "$SPLIT_DIR/agentdojo_renders_benign.jsonl"
    --benign-lossmasks "$RUN_DIR/lossmasks/fujitsu_b4_dr.jsonl" "$SPLIT_DIR/agentdojo_lossmasks_benign.jsonl"
)

[[ "$LOSS_MODE" == "legacy_cb" ]] && TRAIN_ARGS+=(--beta-kl "$BETA_KL")

train_start=$(date +%s)
if python src/training/train_schema.py "${TRAIN_ARGS[@]}" > "$RUN_DIR/train.log" 2>&1; then
    train_status="SUCCESS"
else
    train_status="FAILED"
fi
train_end=$(date +%s)
echo "  Training $train_status ($((( train_end - train_start ) / 60))m)"

if [[ "$train_status" == "FAILED" ]]; then
    echo "  Last 10 lines:"
    tail -10 "$RUN_DIR/train.log" | sed 's/^/    /'
    exit 1
fi

# Quick training curve
grep -E "^Step [0-9]+:" "$RUN_DIR/train.log" | head -3
echo "  ..."
grep -E "^Step [0-9]+:" "$RUN_DIR/train.log" | tail -3

# =============================================================================
# Evaluation
# =============================================================================
echo ""
echo "Step 3: Eval ($EVAL_MODE, $LIMIT_EVAL samples)"

ADAPTER_PATH="$RUN_DIR/model/final"
[[ ! -d "$ADAPTER_PATH" ]] && ADAPTER_PATH="$RUN_DIR/model"

# Fujitsu
python src/evaluation/eval.py \
    --baseline "$TOKENIZER" \
    --cb-adapter "$ADAPTER_PATH" \
    --eval-data "$DS_TRACES" \
    --tool-schema "$TOOL_SCHEMA" \
    --output "$RUN_DIR/eval/fujitsu_eval.json" \
    --limit "$LIMIT_EVAL" \
    --merge-adapter \
    --eval-mode "$EVAL_MODE" \
    --dtype "$DTYPE" >> "$RUN_DIR/eval/fujitsu.log" 2>&1

if [[ -f "$RUN_DIR/eval/fujitsu_eval.json" ]]; then
    python3 -c "
import json
with open('$RUN_DIR/eval/fujitsu_eval.json') as f: r = json.load(f)
for lbl, k in [('Baseline','baseline'),('CB','cb_model')]:
    tf = r.get(k,{}).get('tool_flip_asr',{})
    if not tf: continue
    print(f'  {lbl}: ASR={tf.get(\"attack_success_rate\",0):.0%} correct={tf.get(\"correct_tool_call_rate\",0):.0%} no_tool={tf.get(\"no_tool_call_rate\",0):.0%}')
d = r.get('delta',{})
if d: print(f'  Delta: ASR={d.get(\"tool_flip_asr\",0):+.0%}')
"
fi

# AD harmful
if [[ -f "$SPLIT_DIR/agentdojo_traces_harmful.jsonl" ]]; then
    python src/evaluation/eval.py \
        --baseline "$TOKENIZER" \
        --cb-adapter "$ADAPTER_PATH" \
        --eval-data "$SPLIT_DIR/agentdojo_traces_harmful.jsonl" \
        --tool-schema "$TOOL_SCHEMA" \
        --output "$RUN_DIR/eval/agentdojo_eval.json" \
        --use-sample-context \
        --limit "$LIMIT_EVAL" \
        --merge-adapter \
        --eval-mode "$EVAL_MODE" \
        --dtype "$DTYPE" >> "$RUN_DIR/eval/agentdojo.log" 2>&1

    if [[ -f "$RUN_DIR/eval/agentdojo_eval.json" ]]; then
        python3 -c "
import json
with open('$RUN_DIR/eval/agentdojo_eval.json') as f: r = json.load(f)
cb = r.get('cb_model',{}).get('generation_comparison',{})
if cb: print(f'  AD Harmful: resist={cb.get(\"harmful_resistance_rate\",0):.0%} malicious={cb.get(\"malicious_tool_call_rate\",0):.0%}')
"
    fi
fi

# AD benign
if [[ -f "$SPLIT_DIR/agentdojo_traces_benign.jsonl" ]]; then
    python src/evaluation/eval.py \
        --baseline "$TOKENIZER" \
        --cb-adapter "$ADAPTER_PATH" \
        --eval-data "$SPLIT_DIR/agentdojo_traces_benign.jsonl" \
        --tool-schema "$TOOL_SCHEMA" \
        --output "$RUN_DIR/eval/agentdojo_benign_eval.json" \
        --use-sample-context \
        --limit "$LIMIT_EVAL" \
        --merge-adapter \
        --eval-mode "$EVAL_MODE" \
        --dtype "$DTYPE" >> "$RUN_DIR/eval/agentdojo_benign.log" 2>&1

    if [[ -f "$RUN_DIR/eval/agentdojo_benign_eval.json" ]]; then
        python3 -c "
import json
with open('$RUN_DIR/eval/agentdojo_benign_eval.json') as f: r = json.load(f)
cb = r.get('cb_model',{}).get('generation_comparison',{})
if cb:
    print(f'  AD Benign: correct={cb.get(\"correct_tool_call_rate\",0):.0%} no_tool={cb.get(\"no_tool_call_rate\",0):.0%}')
"
    fi
fi

# =============================================================================
# Examples
# =============================================================================
echo ""
echo "========================================"
echo "EXAMPLES ($NUM_EXAMPLES)"
echo "========================================"

FUJITSU_PAIRED="$RUN_DIR/eval/fujitsu_eval.paired_outputs.jsonl"
if [[ -f "$FUJITSU_PAIRED" ]]; then
    python3 << PYEX
import json
with open('$FUJITSU_PAIRED') as f:
    pairs = [json.loads(l) for l in f if l.strip()]
imps = [p for p in pairs if p.get('baseline_outcome')=='attack_success' and p.get('cb_outcome')!='attack_success']
regs = [p for p in pairs if p.get('baseline_outcome')!='attack_success' and p.get('cb_outcome')=='attack_success']
n_cor = sum(1 for p in pairs if p.get('cb_outcome')=='correct_behavior')
n_atk = sum(1 for p in pairs if p.get('cb_outcome')=='attack_success')
n_nt = sum(1 for p in pairs if p.get('cb_outcome')=='no_tool_call')
print(f'Fujitsu: {len(pairs)} pairs | improved={len(imps)} regressed={len(regs)} | correct={n_cor} attack={n_atk} no_tool={n_nt}')
for i, p in enumerate((imps+regs)[:$NUM_EXAMPLES], 1):
    lbl = 'IMP' if p.get('baseline_outcome')=='attack_success' else 'REG'
    print(f'  [{i}] {lbl}: {p.get("baseline_outcome","?")} -> {p.get("cb_outcome","?")}')
    print(f'  BASELINE: {(p.get("baseline_response") or "(empty)")[:200]}')
    print(f'  CB MODEL: {(p.get("cb_response") or "(empty)")[:200]}')
    print()
PYEX
fi

AD_BENIGN_PAIRED="$RUN_DIR/eval/agentdojo_benign_eval.paired_outputs.jsonl"
if [[ -f "$AD_BENIGN_PAIRED" ]]; then
    python3 << PYEX2
import json
with open('$AD_BENIGN_PAIRED') as f:
    pairs = [json.loads(l) for l in f if l.strip()]
n_ok = sum(1 for p in pairs if p.get('cb_observed_tool'))
n_brk = sum(1 for p in pairs if not p.get('cb_observed_tool'))
print(f'AD Benign: {len(pairs)} pairs | tool_ok={n_ok} broken={n_brk}')
broken = [p for p in pairs if p.get('baseline_observed_tool') and not p.get('cb_observed_tool')]
for i, p in enumerate(broken[:$NUM_EXAMPLES], 1):
    print(f'  [{i}] BROKEN: {p.get("baseline_observed_tool","?")} -> none')
    print(f'  CB MODEL: {(p.get("cb_response") or "(empty)")[:200]}')
    print()
PYEX2
fi

# Generate run summary
python scripts/generate_run_summary.py --run-dir "$RUN_DIR" 2>&1 || true

echo ""
echo "========================================================"
echo "DONE: $RUN_TAG"
echo "========================================================"
echo "Output: $RUN_DIR"
echo "Date: $(date)"
