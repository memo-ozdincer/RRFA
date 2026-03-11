#!/bin/bash
#SBATCH --job-name=sweep_kl_restore
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# SWEEP: KL Strength Restoration Verification
# =============================================================================
#
# 5 runs testing KL restoration strength across original_rr and per_token_cb:
#
#   1. orig_rr_kl5_const   - original_rr, gamma_kl=5.0, constant KL
#   2. orig_rr_kl05_cr     - original_rr, gamma_kl=0.5, cr-scaled KL
#   3. ptcb_kl5_mb0        - per_token_cb, gamma_kl=5.0, constant KL, margin_benign=0
#   4. ptcb_kl05_cr_mb0    - per_token_cb, gamma_kl=0.5, cr-scaled KL, margin_benign=0
#   5. ptcb_kl5_mb0_mh1    - per_token_cb, gamma_kl=5.0, constant KL, margin_harmful=1
#
# Data: reuses shared ETL from sweep_completed_v1 (cb_full_sequence policy)
#
# Usage:
#   sbatch slurm/pipeline/sweep_kl_restore.sbatch
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
# Fixed configuration (shared across all 5 runs)
# =============================================================================
BATCH_SIZE=4
GRAD_ACCUM=2
LEARNING_RATE=5e-5
WARMUP_STEPS=20
MAX_SEQ_LENGTH=4096
LORA_R=16
LORA_ALPHA=32
CB_LAYERS="10 20"
ALPHA_MAX=10.0
POLICY=cb_full_sequence
POOLING_POLICY=cb_full_sequence
EVAL_MODE=next_tool_prediction
PRESET=llama-3.1-8b-instruct
LIMIT_EVAL=100
NUM_EXAMPLES=5

echo "========================================================"
echo "SWEEP: KL Strength Restoration (5 configs)"
echo "========================================================"
echo "Job: ${SLURM_JOB_ID:-local} | $(date)"
echo ""

# =============================================================================
# Resolve tokenizer
# =============================================================================
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
MODEL_CACHE_NAME=$(echo "$MODEL_ID" | sed 's/\//--/g')
SNAPSHOT_ROOT="$CACHE_DIR/hf/hub/models--${MODEL_CACHE_NAME}/snapshots"
TOKENIZER=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1)
if [[ -z "$TOKENIZER" ]]; then
    echo "ERROR: Model not cached. Run slurm/cache_models.sh first."; exit 1
fi
echo "Tokenizer: $TOKENIZER"

# =============================================================================
# Input data (raw sources)
# =============================================================================
TRACES_DIR="$CB_SCRATCH/data/traces"
FUJITSU_DS="$TRACES_DIR/fujitsu_b4_ds.jsonl"
FUJITSU_DR="$TRACES_DIR/fujitsu_b4_dr.jsonl"
AGENTDOJO_RAW="$TRACES_DIR/agentdojo_augmented.jsonl"
PAIRS_SRC="$TRACES_DIR/contrastive_pairs.jsonl"
TOOL_SCHEMA="$REPO_DIR/configs/tool_schemas/b4_standard_v1.json"
AD_TOOL_SCHEMA="$REPO_DIR/configs/tool_schemas/agentdojo_v1.json"

# Copy from repo if not on scratch yet (agentdojo)
for src_name in agentdojo_augmented.jsonl; do
    if [[ ! -f "$TRACES_DIR/$src_name" ]] && [[ -f "$REPO_DIR/data/traces/$src_name" ]]; then
        echo "  Copying $src_name to scratch..."
        cp "$REPO_DIR/data/traces/$src_name" "$TRACES_DIR/$src_name"
    fi
done

# Generate contrastive pairs if missing (gitignored, won't be in repo)
if [[ ! -f "$PAIRS_SRC" ]]; then
    echo "  Generating contrastive pairs..."
    python "$REPO_DIR/scripts/augment_agentdojo.py" \
        --traces-input "$AGENTDOJO_RAW" \
        --operations removal,contrastive \
        --contrastive-output "$PAIRS_SRC" \
        --output "$CB_SCRATCH/data/traces/_augment_temp.jsonl" 2>&1 | tail -5
    rm -f "$CB_SCRATCH/data/traces/_augment_temp.jsonl"
fi

for f in "$FUJITSU_DS" "$FUJITSU_DR" "$AGENTDOJO_RAW" "$PAIRS_SRC"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing $(basename "$f") — not on scratch and not in repo"; exit 1
    fi
    echo "  $(basename "$f"): $(wc -l < "$f") lines"
done
echo ""

# =============================================================================
# Shared data directory (reuse from sweep_completed_v1)
# =============================================================================
SHARED_DATA="$CB_SCRATCH/sweeps/shared_data_${POLICY}_ts"
echo "Shared data: $SHARED_DATA"

# Verify shared data exists (ETL_B must have been run by sweep_completed_v1)
for required in \
    "$SHARED_DATA/renders/fujitsu_b4_ds.jsonl" \
    "$SHARED_DATA/renders/fujitsu_b4_dr.jsonl" \
    "$SHARED_DATA/lossmasks/fujitsu_b4_ds.jsonl" \
    "$SHARED_DATA/lossmasks/fujitsu_b4_dr.jsonl" \
    "$SHARED_DATA/split_agentdojo/agentdojo_renders_harmful.jsonl" \
    "$SHARED_DATA/split_agentdojo/agentdojo_renders_benign.jsonl" \
    "$SHARED_DATA/split_agentdojo/agentdojo_lossmasks_harmful.jsonl" \
    "$SHARED_DATA/split_agentdojo/agentdojo_lossmasks_benign.jsonl"; do
    if [[ ! -f "$required" ]] || [[ ! -s "$required" ]]; then
        echo "ERROR: Missing shared data: $required"
        echo "Run sweep_completed_v1.sbatch first to generate ETL_B data."
        exit 1
    fi
done
echo "  Shared data verified (all renders/lossmasks/splits present)"
echo ""

# =============================================================================
# Config arrays (5 runs)
# =============================================================================
RUN_NAMES=(
    "orig_rr_kl5_const"
    "orig_rr_kl05_cr"
    "ptcb_kl5_mb0"
    "ptcb_kl05_cr_mb0"
    "ptcb_kl5_mb0_mh1"
)
LOSS_MODES=(
    "original_rr"
    "original_rr"
    "per_token_cb"
    "per_token_cb"
    "per_token_cb"
)
GAMMA_KLS=(
    5.0
    0.5
    5.0
    0.5
    5.0
)
KL_SCALINGS=(
    "constant"
    "cr"
    "constant"
    "cr"
    "constant"
)
DISTANCES=(
    "dl2sq"
    "dl2sq"
    "dl2rc"
    "dl2rc"
    "dl2rc"
)
MARGIN_HARMFULS=(
    20.0
    20.0
    20.0
    20.0
    1.0
)
MARGIN_BENIGNS=(
    0.0
    0.0
    0.0
    0.0
    0.0
)
MARGIN_FREES=(
    "true"
    "true"
    "false"
    "false"
    "false"
)
DECAY_MULTS=(
    1.0
    1.0
    2.0
    2.0
    2.0
)
TOTAL_STEPS_ARR=(
    300
    300
    300
    300
    300
)

NUM_CONFIGS=${#RUN_NAMES[@]}

echo "========================================================"
echo "Running $NUM_CONFIGS configs"
echo "========================================================"
for i in $(seq 0 $((NUM_CONFIGS - 1))); do
    echo "  [$((i+1))] ${RUN_NAMES[$i]}: loss=${LOSS_MODES[$i]} gamma_kl=${GAMMA_KLS[$i]} kl_scaling=${KL_SCALINGS[$i]} dist=${DISTANCES[$i]} mh=${MARGIN_HARMFULS[$i]} mb=${MARGIN_BENIGNS[$i]} mf=${MARGIN_FREES[$i]} decay=${DECAY_MULTS[$i]} steps=${TOTAL_STEPS_ARR[$i]}"
done
echo ""

# =============================================================================
# Main loop: train + eval for each config
# =============================================================================
for i in $(seq 0 $((NUM_CONFIGS - 1))); do
    RUN_NAME="${RUN_NAMES[$i]}"
    LOSS_MODE="${LOSS_MODES[$i]}"
    GAMMA_KL="${GAMMA_KLS[$i]}"
    KL_SCALING="${KL_SCALINGS[$i]}"
    DISTANCE="${DISTANCES[$i]}"
    MARGIN_HARMFUL="${MARGIN_HARMFULS[$i]}"
    MARGIN_BENIGN="${MARGIN_BENIGNS[$i]}"
    MARGIN_FREE="${MARGIN_FREES[$i]}"
    DECAY_MULT="${DECAY_MULTS[$i]}"
    TOTAL_STEPS="${TOTAL_STEPS_ARR[$i]}"

    echo ""
    echo "================================================================"
    echo "CONFIG [$((i+1))/$NUM_CONFIGS]: $RUN_NAME"
    echo "================================================================"
    echo "  loss=$LOSS_MODE gamma_kl=$GAMMA_KL kl_scaling=$KL_SCALING"
    echo "  distance=$DISTANCE margin_h=$MARGIN_HARMFUL margin_b=$MARGIN_BENIGN margin_free=$MARGIN_FREE"
    echo "  decay_mult=$DECAY_MULT steps=$TOTAL_STEPS"
    echo ""

    RUN_DIR="$CB_SCRATCH/sweeps/kl_restore_${RUN_NAME}_$(date +%Y%m%d_%H%M%S)/run"
    mkdir -p "$RUN_DIR"/{model,eval}

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    echo "  Training ($LOSS_MODE, $TOTAL_STEPS steps)..."
    train_start=$(date +%s)

    python src/training/train_schema.py \
        --preset "$PRESET" \
        --model "$TOKENIZER" \
        --output-dir "$RUN_DIR/model" \
        --total-steps "$TOTAL_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --warmup-steps "$WARMUP_STEPS" \
        --alpha-max "$ALPHA_MAX" \
        --max-seq-length "$MAX_SEQ_LENGTH" \
        --gradient-accumulation-steps "$GRAD_ACCUM" \
        --loss-mode "$LOSS_MODE" \
        --loss-weighting dual \
        --triplet-alpha-benign 0.5 \
        --triplet-beta-harmful 0.4 \
        --triplet-gamma-kl "$GAMMA_KL" \
        --triplet-margin-benign "$MARGIN_BENIGN" \
        --triplet-margin-harmful "$MARGIN_HARMFUL" \
        --triplet-benign-positive-distance "$DISTANCE" \
        --triplet-benign-negative-distance "$DISTANCE" \
        --triplet-harmful-positive-distance "$DISTANCE" \
        --triplet-harmful-negative-distance "$DISTANCE" \
        --triplet-mix-l2-weight 0.5 \
        --triplet-mix-cos-weight 0.5 \
        --pooling-mode correct \
        --pooling-mask-policy auto \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA" \
        --lora-dropout 0.05 \
        --logging-steps 10 \
        --save-steps 9999 \
        --no-wandb \
        --cb-target-layers $CB_LAYERS \
        --mode mixed \
        --kl-scaling "$KL_SCALING" \
        --alpha-decay-multiplier "$DECAY_MULT" \
        --harmful-renders "$SHARED_DATA/renders/fujitsu_b4_ds.jsonl" \
                          "$SHARED_DATA/split_agentdojo/agentdojo_renders_harmful.jsonl" \
        --harmful-lossmasks "$SHARED_DATA/lossmasks/fujitsu_b4_ds.jsonl" \
                            "$SHARED_DATA/split_agentdojo/agentdojo_lossmasks_harmful.jsonl" \
        --benign-renders "$SHARED_DATA/renders/fujitsu_b4_dr.jsonl" \
                         "$SHARED_DATA/split_agentdojo/agentdojo_renders_benign.jsonl" \
        --benign-lossmasks "$SHARED_DATA/lossmasks/fujitsu_b4_dr.jsonl" \
                           "$SHARED_DATA/split_agentdojo/agentdojo_lossmasks_benign.jsonl" \
        --contrastive-pairs "$PAIRS_SRC" \
        $( [[ "$MARGIN_FREE" == "true" ]] && echo "--margin-free" ) \
        > "$RUN_DIR/train.log" 2>&1

    train_status=$?
    train_mins=$(( ($(date +%s) - train_start) / 60 ))
    if [[ $train_status -eq 0 ]]; then
        echo "  Training SUCCESS (${train_mins}m)"
    else
        echo "  Training FAILED (${train_mins}m)"
        tail -20 "$RUN_DIR/train.log" | sed 's/^/    /'
        echo "  Skipping eval for $RUN_NAME, continuing to next config..."
        continue
    fi

    # Training history summary
    python3 - "$RUN_DIR/train.log" << 'TRAINDIAG'
import re, sys

log_path = sys.argv[1] if len(sys.argv) > 1 else ""
if not log_path:
    sys.exit(0)

steps = []
with open(log_path) as f:
    for line in f:
        m = re.search(r'\[Step (\d+)\](.+)', line)
        if m:
            step = int(m.group(1))
            rest = m.group(2).strip()
            steps.append((step, rest))
        gm = re.search(r'grad_norm=([0-9.e+-]+)', line)
        if gm and steps:
            steps[-1] = (steps[-1][0], steps[-1][1] + f" grad={gm.group(1)}")

if not steps:
    with open(log_path) as f:
        for line in f:
            m = re.search(r'step[= ]+(\d+).*?loss[= ]+([0-9.e+-]+)', line, re.IGNORECASE)
            if m:
                steps.append((int(m.group(1)), f"loss={m.group(2)}"))

if steps:
    print(f"    Steps logged: {len(steps)}")
    for label, idx in [("First", 0), ("Mid", len(steps)//2), ("Last", -1)]:
        s, info = steps[idx]
        print(f"    {label:5s} (step {s:>4d}): {info[:150]}")
else:
    print("    No step diagnostics found in log")
TRAINDIAG

    # Save run config
    cat > "$RUN_DIR/run_config.json" << RUNCFG
{
  "run_name": "$RUN_NAME",
  "loss_mode": "$LOSS_MODE",
  "distance": "$DISTANCE",
  "gamma_kl": $GAMMA_KL,
  "kl_scaling": "$KL_SCALING",
  "margin_harmful": $MARGIN_HARMFUL,
  "margin_benign": $MARGIN_BENIGN,
  "margin_free": "$MARGIN_FREE",
  "decay_mult": $DECAY_MULT,
  "policy": "$POLICY",
  "pooling_policy": "$POOLING_POLICY",
  "cb_layers": "$CB_LAYERS",
  "lora_layers": "all",
  "total_steps": $TOTAL_STEPS,
  "batch_size": $BATCH_SIZE,
  "grad_accum": $GRAD_ACCUM,
  "effective_batch": $(( BATCH_SIZE * GRAD_ACCUM )),
  "learning_rate": "$LEARNING_RATE",
  "alpha_max": $ALPHA_MAX,
  "lora_r": $LORA_R,
  "lora_alpha": $LORA_ALPHA,
  "eval_mode": "$EVAL_MODE",
  "model": "$TOKENIZER",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "job_id": "${SLURM_JOB_ID:-local}"
}
RUNCFG

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    echo ""
    echo "  Evaluation ($EVAL_MODE + full_trace, n=$LIMIT_EVAL)..."

    # Find adapter
    if [[ -d "$RUN_DIR/model/final" ]]; then
        ADAPTER_PATH="$RUN_DIR/model/final"
    elif [[ -f "$RUN_DIR/model/adapter_config.json" ]]; then
        ADAPTER_PATH="$RUN_DIR/model"
    else
        echo "  ERROR: No adapter found for $RUN_NAME, skipping eval..."
        continue
    fi

    EVAL_DIR="$RUN_DIR/eval/$EVAL_MODE"
    mkdir -p "$EVAL_DIR"

    # Use original (non-truncated) data for eval
    AGENTDOJO_EVAL_SRC="$TRACES_DIR/agentdojo_augmented.jsonl"
    DS_TRACES="$TRACES_DIR/fujitsu_b4_ds.jsonl"

    # --- Fujitsu harmful eval ---
    echo "    Fujitsu eval..."
    python src/evaluation/eval.py \
        --baseline "$TOKENIZER" \
        --cb-adapter "$ADAPTER_PATH" \
        --eval-data "$DS_TRACES" \
        --tool-schema "$TOOL_SCHEMA" \
        --output "$EVAL_DIR/fujitsu_eval.json" \
        --limit "$LIMIT_EVAL" \
        --merge-adapter \
        --eval-mode "$EVAL_MODE" \
        --dtype bfloat16 >> "$EVAL_DIR/fujitsu.log" 2>&1

    if [[ -f "$EVAL_DIR/fujitsu_eval.json" ]]; then
        python3 -c "
import json
with open('$EVAL_DIR/fujitsu_eval.json') as f: r = json.load(f)
for lbl, k in [('Baseline','baseline'),('CB','cb_model')]:
    tf = r.get(k,{}).get('tool_flip_asr',{})
    if tf:
        print(f'    {lbl}: ASR={tf.get(\"attack_success_rate\",0):.0%} correct={tf.get(\"correct_tool_call_rate\",0):.0%} no_tool={tf.get(\"no_tool_call_rate\",0):.0%}')
"
    fi

    # --- AgentDojo harmful/benign split ---
    python3 -c "
import json
harmful, benign = [], []
for line in open('$AGENTDOJO_EVAL_SRC'):
    t = json.loads(line)
    cat = t.get('labels',{}).get('category','')
    if cat == 'harmful': harmful.append(line)
    else: benign.append(line)
with open('$EVAL_DIR/ad_harmful_traces.jsonl','w') as f: f.writelines(harmful)
with open('$EVAL_DIR/ad_benign_traces.jsonl','w') as f: f.writelines(benign)
print(f'    AD split: {len(harmful)} harmful, {len(benign)} benign')
"

    # --- AgentDojo harmful eval (next_tool_prediction) ---
    echo "    AgentDojo harmful eval..."
    python src/evaluation/eval.py \
        --baseline "$TOKENIZER" \
        --cb-adapter "$ADAPTER_PATH" \
        --eval-data "$EVAL_DIR/ad_harmful_traces.jsonl" \
        --tool-schema "$AD_TOOL_SCHEMA" \
        --output "$EVAL_DIR/agentdojo_eval.json" \
        --use-sample-context \
        --limit "$LIMIT_EVAL" \
        --merge-adapter \
        --eval-mode "$EVAL_MODE" \
        --dtype bfloat16 >> "$EVAL_DIR/agentdojo.log" 2>&1

    if [[ -f "$EVAL_DIR/agentdojo_eval.json" ]]; then
        python3 -c "
import json
with open('$EVAL_DIR/agentdojo_eval.json') as f: r = json.load(f)
cb = r.get('cb_model',{}).get('generation_comparison',{})
if cb: print(f'    AD Harmful: resist={cb.get(\"harmful_resistance_rate\",0):.0%} malicious={cb.get(\"malicious_tool_call_rate\",0):.0%}')
"
    fi

    # --- AgentDojo benign eval (next_tool_prediction) ---
    echo "    AgentDojo benign eval..."
    python src/evaluation/eval.py \
        --baseline "$TOKENIZER" \
        --cb-adapter "$ADAPTER_PATH" \
        --eval-data "$EVAL_DIR/ad_benign_traces.jsonl" \
        --tool-schema "$AD_TOOL_SCHEMA" \
        --output "$EVAL_DIR/agentdojo_benign_eval.json" \
        --use-sample-context \
        --limit "$LIMIT_EVAL" \
        --merge-adapter \
        --eval-mode "$EVAL_MODE" \
        --dtype bfloat16 >> "$EVAL_DIR/agentdojo_benign.log" 2>&1

    if [[ -f "$EVAL_DIR/agentdojo_benign_eval.json" ]]; then
        python3 -c "
import json
with open('$EVAL_DIR/agentdojo_benign_eval.json') as f: r = json.load(f)
cb = r.get('cb_model',{}).get('generation_comparison',{})
if cb: print(f'    AD Benign:  correct={cb.get(\"correct_tool_call_rate\",0):.0%} no_tool={cb.get(\"no_tool_call_rate\",0):.0%}')
"
    fi

    # --- AgentDojo harmful eval (full_trace) ---
    echo "    AgentDojo full_trace eval..."
    FULL_TRACE_DIR="$RUN_DIR/eval/full_trace"
    mkdir -p "$FULL_TRACE_DIR"

    python src/evaluation/eval.py \
        --baseline "$TOKENIZER" \
        --cb-adapter "$ADAPTER_PATH" \
        --eval-data "$EVAL_DIR/ad_harmful_traces.jsonl" \
        --tool-schema "$AD_TOOL_SCHEMA" \
        --output "$FULL_TRACE_DIR/agentdojo_full_trace_eval.json" \
        --use-sample-context \
        --limit "$LIMIT_EVAL" \
        --merge-adapter \
        --eval-mode full_trace \
        --dtype bfloat16 >> "$FULL_TRACE_DIR/agentdojo_full_trace.log" 2>&1

    if [[ -f "$FULL_TRACE_DIR/agentdojo_full_trace_eval.json" ]]; then
        python3 -c "
import json
with open('$FULL_TRACE_DIR/agentdojo_full_trace_eval.json') as f: r = json.load(f)
cb = r.get('cb_model',{}).get('generation_comparison',{})
if cb: print(f'    AD Full Trace: resist={cb.get(\"harmful_resistance_rate\",0):.0%} malicious={cb.get(\"malicious_tool_call_rate\",0):.0%}')
"
    fi

    # -----------------------------------------------------------------
    # Examples (paired output comparisons)
    # -----------------------------------------------------------------
    echo ""
    echo "  EXAMPLES ($RUN_NAME)"

    # Fujitsu examples
    FUJITSU_PAIRED="$EVAL_DIR/fujitsu_eval.paired_outputs.jsonl"
    if [[ -f "$FUJITSU_PAIRED" ]]; then
        python3 - "$FUJITSU_PAIRED" "$NUM_EXAMPLES" << 'FUJEX'
import json, sys

paired_path = sys.argv[1]
num_ex = int(sys.argv[2]) if len(sys.argv) > 2 else 3

with open(paired_path) as f:
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

for i, p in enumerate((imps+regs)[:num_ex], 1):
    lbl = 'IMP' if p.get('baseline_outcome')=='attack_success' else 'REG'
    g = ' [GIB]' if gib(p.get('cb_response','')) else ''
    print(f'  [{i}] {lbl}: {p.get("baseline_outcome","?")} -> {p.get("cb_outcome","?")}{g}')
    print(f'      expected={p.get("expected_tool","?")} simulated={p.get("simulated_tool","?")}')
    print(f'  BASELINE: {(p.get("baseline_response") or "(empty)")[:250]}')
    print(f'  CB MODEL: {(p.get("cb_response") or "(empty)")[:250]}')
    print()
FUJEX
    fi

    # AgentDojo benign examples
    AD_BENIGN_PAIRED="$EVAL_DIR/agentdojo_benign_eval.paired_outputs.jsonl"
    if [[ -f "$AD_BENIGN_PAIRED" ]]; then
        python3 - "$AD_BENIGN_PAIRED" "$NUM_EXAMPLES" << 'ADEX'
import json, sys

paired_path = sys.argv[1]
num_ex = int(sys.argv[2]) if len(sys.argv) > 2 else 3

with open(paired_path) as f:
    pairs = [json.loads(l) for l in f if l.strip()]

n_ok = sum(1 for p in pairs if p.get('cb_observed_tool'))
n_brk = sum(1 for p in pairs if not p.get('cb_observed_tool'))

print(f'  AD Benign: {len(pairs)} pairs | tool_ok={n_ok} broken={n_brk}')
print()

broken = [p for p in pairs if p.get('baseline_observed_tool') and not p.get('cb_observed_tool')]
if broken:
    print(f'  Broken examples (baseline had tool, CB lost it):')
    for i, p in enumerate(broken[:num_ex], 1):
        print(f'  [{i}] BROKEN: {p.get("baseline_observed_tool","?")} -> none')
        print(f'  CB MODEL: {(p.get("cb_response") or "(empty)")[:250]}')
        print()
ADEX
    fi

    # Generate run summary if script exists
    if [[ -f "$REPO_DIR/scripts/generate_run_summary.py" ]]; then
        python "$REPO_DIR/scripts/generate_run_summary.py" --run-dir "$RUN_DIR" 2>&1 || true
    fi

    # -----------------------------------------------------------------
    # Run summary
    # -----------------------------------------------------------------
    echo ""
    echo "  --------------------------------------------------------"
    echo "  RESULTS: $RUN_NAME"
    echo "  --------------------------------------------------------"

    python3 - "$RUN_DIR" "$RUN_NAME" "$LOSS_MODE" "$DISTANCE" "$GAMMA_KL" "$KL_SCALING" "$TOTAL_STEPS" << 'SUMMARY'
import json, os, sys

run_dir = sys.argv[1]
run_name = sys.argv[2]
loss_mode = sys.argv[3]
distance = sys.argv[4]
gamma_kl = sys.argv[5]
kl_scaling = sys.argv[6]
steps = sys.argv[7]

fmt = lambda v: f"{v:.0%}" if isinstance(v, (int, float)) and v >= 0 else "n/a"

print(f"  Config: {run_name} | {loss_mode} | dist={distance} | gamma_kl={gamma_kl} | kl_scaling={kl_scaling} | steps={steps}")
print()

# Fujitsu
fuj_path = os.path.join(run_dir, "eval", "next_tool_prediction", "fujitsu_eval.json")
if os.path.isfile(fuj_path):
    with open(fuj_path) as f:
        r = json.load(f)
    bl = r.get("baseline", {}).get("tool_flip_asr", {})
    cb = r.get("cb_model", {}).get("tool_flip_asr", {})
    if bl and cb:
        print(f"  Fujitsu:")
        print(f"    Baseline: ASR={fmt(bl.get('attack_success_rate', -1))} correct={fmt(bl.get('correct_tool_call_rate', -1))}")
        print(f"    CB Model: ASR={fmt(cb.get('attack_success_rate', -1))} correct={fmt(cb.get('correct_tool_call_rate', -1))} no_tool={fmt(cb.get('no_tool_call_rate', -1))}")
        delta = cb.get('attack_success_rate', 0) - bl.get('attack_success_rate', 0)
        print(f"    Delta:    ASR {delta:+.0%}")

# AgentDojo harmful
adh_path = os.path.join(run_dir, "eval", "next_tool_prediction", "agentdojo_eval.json")
if os.path.isfile(adh_path):
    with open(adh_path) as f:
        r = json.load(f)
    cb = r.get("cb_model", {}).get("generation_comparison", {})
    if cb:
        print(f"  AgentDojo Harmful:")
        print(f"    Resist={fmt(cb.get('harmful_resistance_rate', -1))} Malicious={fmt(cb.get('malicious_tool_call_rate', -1))}")

# AgentDojo benign
adb_path = os.path.join(run_dir, "eval", "next_tool_prediction", "agentdojo_benign_eval.json")
if os.path.isfile(adb_path):
    with open(adb_path) as f:
        r = json.load(f)
    cb = r.get("cb_model", {}).get("generation_comparison", {})
    if cb:
        print(f"  AgentDojo Benign:")
        print(f"    Correct={fmt(cb.get('correct_tool_call_rate', -1))} No_tool={fmt(cb.get('no_tool_call_rate', -1))}")

# Full trace
ft_path = os.path.join(run_dir, "eval", "full_trace", "agentdojo_full_trace_eval.json")
if os.path.isfile(ft_path):
    with open(ft_path) as f:
        r = json.load(f)
    cb = r.get("cb_model", {}).get("generation_comparison", {})
    if cb:
        print(f"  AgentDojo Full Trace:")
        print(f"    Resist={fmt(cb.get('harmful_resistance_rate', -1))} Malicious={fmt(cb.get('malicious_tool_call_rate', -1))}")

print()
SUMMARY

    echo "  Output: $RUN_DIR"
    echo ""

done

# =============================================================================
# Final summary table
# =============================================================================
echo ""
echo "========================================================"
echo "SWEEP COMPLETE: KL Strength Restoration"
echo "========================================================"
echo "Job: ${SLURM_JOB_ID:-local} | $(date)"
echo ""
echo "All run directories under: $CB_SCRATCH/sweeps/kl_restore_*"
echo ""
