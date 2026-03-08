#!/bin/bash
#SBATCH --job-name=sweep_completed
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# SWEEP 7: Bug fixes + new loss modes (per technical plan)
# =============================================================================
#
# Runs (submit separately):
#   LOSS_MODE=original_rr    sbatch slurm/pipeline/sweep_completed_v1.sbatch  # PAPER BASELINE
#   LOSS_MODE=triplet_full   sbatch slurm/pipeline/sweep_completed_v1.sbatch
#   LOSS_MODE=per_token_cb   sbatch slurm/pipeline/sweep_completed_v1.sbatch
#   LOSS_MODE=simplified_pooled sbatch slurm/pipeline/sweep_completed_v1.sbatch
#
# original_rr: Paper's exact loss (Zou et al. 2024, Algorithm 1)
#   ReLU(cos_sim) reroute + L2 retain, paper coefficient schedule
#   No margins, no triplet, no dl2rc. alpha=10, 150 steps.
#   With GAMMA_KL=0 (default): paper exact. GAMMA_KL=0.3: adds KL preservation.
#
# Bug fixes applied to triplet/per_token/simplified modes:
#   1. Distance: dl2rc (L2 + 10*ReLU(cos)) instead of dcos
#   2. KL fix: benign KL uses attention_mask, not loss_mask
#   3. Fujitsu truncated to 3 messages (no template garbage)
#
# Data:
#   fujitsu_b4_completed.jsonl  - 26,492 (13,246 harmful + 13,246 benign)
#   agentdojo_truncated.jsonl   - 4,144 (1,000 harmful + 2,894 benign + 250 resisted)
#   contrastive_pairs.jsonl     - 1,247 pairs (generated if missing)
#
# Usage:
#   sbatch slurm/pipeline/sweep_completed_v1.sbatch
#   QUICK_TEST=true sbatch slurm/pipeline/sweep_completed_v1.sbatch
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
# Configuration
# =============================================================================
QUICK_TEST="${QUICK_TEST:-false}"
LOSS_MODE="${LOSS_MODE:-original_rr}"

if [[ "$QUICK_TEST" == "true" ]]; then
    TOTAL_STEPS=20; LIMIT_EVAL=5; NUM_EXAMPLES=2
    echo "*** QUICK TEST MODE ***"
elif [[ "$LOSS_MODE" == "original_rr" ]]; then
    # Paper settings: 150 steps, alpha=10, batch=16 (4x4)
    TOTAL_STEPS=150; LIMIT_EVAL=100; NUM_EXAMPLES=5
else
    TOTAL_STEPS=300; LIMIT_EVAL=100; NUM_EXAMPLES=5
fi

BATCH_SIZE=4
GRAD_ACCUM=4
LEARNING_RATE=1e-5
WARMUP_STEPS=20
MAX_SEQ_LENGTH=4096
LORA_R=16
LORA_ALPHA=16
LORA_LAYERS="0-20"
CB_LAYERS="10 20"
DISTANCE=dl2rc
POLICY=injection_aware
POOLING_POLICY=cb_full_sequence
EVAL_MODE=next_tool_prediction
PRESET=llama-3.1-8b-instruct

# Mode-specific defaults
if [[ "$LOSS_MODE" == "original_rr" ]]; then
    ALPHA_MAX=10.0
    GAMMA_KL="${GAMMA_KL:-0.3}"   # 0 = paper exact, 0.3 = paper + KL
else
    ALPHA_MAX=10.0
    GAMMA_KL="${GAMMA_KL:-0.3}"
fi

echo "========================================================"
echo "SWEEP 7: Bug fixes + $LOSS_MODE"
echo "========================================================"
echo "Job: ${SLURM_JOB_ID:-local} | $(date)"
echo "Steps: $TOTAL_STEPS | Loss: $LOSS_MODE | Distance: $DISTANCE"
echo "Policy: $POLICY | LoRA: $LORA_LAYERS | γ_kl: $GAMMA_KL"
echo ""

# =============================================================================
# Resolve tokenizer
# =============================================================================
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_CACHE_NAME=$(echo "$MODEL_ID" | sed 's/\//--/g')
SNAPSHOT_ROOT="$CACHE_DIR/hf/hub/models--${MODEL_CACHE_NAME}/snapshots"
TOKENIZER=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1)
if [[ -z "$TOKENIZER" ]]; then
    echo "ERROR: Model not cached. Run slurm/cache_models.sh first."; exit 1
fi
echo "Tokenizer: $TOKENIZER"

# =============================================================================
# Input data (raw sources — on scratch from previous sweeps)
# =============================================================================
TRACES_DIR="$CB_SCRATCH/data/traces"
FUJITSU_RAW="$TRACES_DIR/fujitsu_b4_ds.jsonl"
AGENTDOJO_RAW="$TRACES_DIR/agentdojo_augmented.jsonl"
PAIRS_SRC="$TRACES_DIR/contrastive_pairs.jsonl"
TOOL_SCHEMA="$REPO_DIR/configs/tool_schemas/b4_standard_v1.json"

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

for f in "$FUJITSU_RAW" "$AGENTDOJO_RAW" "$PAIRS_SRC"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing $(basename "$f") — not on scratch and not in repo"; exit 1
    fi
    echo "  $(basename "$f"): $(wc -l < "$f") lines"
done
echo ""

# =============================================================================
# Shared data directory (reused across concurrent loss-mode runs)
# =============================================================================
SHARED_DATA="$CB_SCRATCH/sweeps/bugfix_shared_data"
SWEEP_ID="bugfix_${LOSS_MODE}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$CB_SCRATCH/sweeps/$SWEEP_ID/run"
mkdir -p "$RUN_DIR"/{model,eval}
mkdir -p "$SHARED_DATA"/{renders,lossmasks,split_fujitsu,split_agentdojo}
echo "Shared data: $SHARED_DATA"
echo "Run output:  $RUN_DIR"
echo ""

# =============================================================================
# Step 0: Generate completed Fujitsu + truncated AgentDojo (shared, skip if exists)
# =============================================================================
echo "========================================================"
echo "Step 0: Data generation"
echo "========================================================"
gen_start=$(date +%s)

FUJITSU_SRC="$SHARED_DATA/fujitsu_b4_completed.jsonl"
AGENTDOJO_SRC="$SHARED_DATA/agentdojo_truncated.jsonl"

if [[ -f "$FUJITSU_SRC" ]] && [[ -s "$FUJITSU_SRC" ]]; then
    echo "  Fujitsu completed: REUSING existing ($(wc -l < "$FUJITSU_SRC") lines)"
elif [[ -f "$FUJITSU_RAW" ]] && [[ -s "$FUJITSU_RAW" ]]; then
    # Option B: fujitsu_b4_ds.jsonl is already completed — use directly
    echo "  Using $(basename "$FUJITSU_RAW") directly (already completed)..."
    cp "$FUJITSU_RAW" "$FUJITSU_SRC"
    echo "  Copied $(wc -l < "$FUJITSU_SRC") lines"
else
    # Option A fallback: complete from raw skeleton traces
    FUJITSU_SKEL="$TRACES_DIR/fujitsu_b4_traces.jsonl"
    if [[ ! -f "$FUJITSU_SKEL" ]] && [[ -f "$REPO_DIR/data/traces/fujitsu_b4_traces.jsonl" ]]; then
        cp "$REPO_DIR/data/traces/fujitsu_b4_traces.jsonl" "$FUJITSU_SKEL"
    fi
    echo "  Completing Fujitsu skeletons (3-message truncated)..."
    python "$REPO_DIR/scripts/augment_llm.py" \
        --operation complete_fujitsu \
        --input "$FUJITSU_SKEL" \
        --output "$FUJITSU_SRC" \
        --dry-run 2>&1 | tail -5
fi

if [[ -f "$AGENTDOJO_SRC" ]] && [[ -s "$AGENTDOJO_SRC" ]]; then
    echo "  AgentDojo truncated: REUSING existing ($(wc -l < "$AGENTDOJO_SRC") lines)"
else
    echo "  Truncating AgentDojo to injection window..."
    python "$REPO_DIR/scripts/truncate_to_injection_window.py" \
        --input "$AGENTDOJO_RAW" \
        --output "$AGENTDOJO_SRC" 2>&1 | tail -5
fi

for f in "$FUJITSU_SRC" "$AGENTDOJO_SRC"; do
    if [[ ! -f "$f" ]] || [[ ! -s "$f" ]]; then
        echo "ERROR: Data generation failed for $(basename "$f")"; exit 1
    fi
    echo "  $(basename "$f"): $(wc -l < "$f") lines"
done

echo "  Data generation done ($(( $(date +%s) - gen_start ))s)"
echo ""

# =============================================================================
# Step 1: ETL_B (render + lossmask) — shared, skip if exists
# =============================================================================
echo "========================================================"
echo "Step 1: ETL_B"
echo "========================================================"
etl_start=$(date +%s)

for src in "$FUJITSU_SRC" "$AGENTDOJO_SRC"; do
    bn=$(basename "$src" .jsonl)
    render_out="$SHARED_DATA/renders/${bn}.jsonl"
    mask_out="$SHARED_DATA/lossmasks/${bn}.jsonl"
    if [[ -f "$render_out" ]] && [[ -s "$render_out" ]] && [[ -f "$mask_out" ]] && [[ -s "$mask_out" ]]; then
        echo "  $bn: REUSING existing ($(wc -l < "$render_out") renders)"
    else
        echo "  $bn (policy=$POLICY, pool=$POOLING_POLICY)..."
        python src/schemas/tools/ETL_B.py \
            --traces "$src" \
            --render-out "$render_out" \
            --lossmask-out "$mask_out" \
            --tokenizer "$TOKENIZER" \
            --max-length "$MAX_SEQ_LENGTH" \
            --policy-override "$POLICY" \
            --pooling-policy-override "$POOLING_POLICY" 2>&1 | tail -3
    fi
done

echo "  ETL_B done ($(( $(date +%s) - etl_start ))s)"

echo ""

# =============================================================================
# Step 2: Split by category — shared, skip if exists
# =============================================================================
echo "========================================================"
echo "Step 2: Split harmful/benign"
echo "========================================================"

# Split Fujitsu (harmful vs benign by category in trace data)
if [[ -f "$SHARED_DATA/split_fujitsu/renders_harmful.jsonl" ]] && [[ -s "$SHARED_DATA/split_fujitsu/renders_harmful.jsonl" ]]; then
    echo "  Fujitsu split: REUSING existing"
else
python3 - "$FUJITSU_SRC" "$SHARED_DATA/renders/fujitsu_b4_completed.jsonl" "$SHARED_DATA/lossmasks/fujitsu_b4_completed.jsonl" "$SHARED_DATA/split_fujitsu" << 'SPLIT_FUJITSU'
import json, sys

traces_path = sys.argv[1]
renders_path = sys.argv[2]
lossmasks_path = sys.argv[3]
out_dir = sys.argv[4]

# Index traces by ID
traces = {}
with open(traces_path) as f:
    for line in f:
        t = json.loads(line)
        traces[t["id"]] = t

# Split renders
harmful_r, benign_r = [], []
with open(renders_path) as f:
    for line in f:
        r = json.loads(line)
        tid = r.get("trace_id", "")
        t = traces.get(tid, {})
        cat = t.get("labels", {}).get("category", "")
        if cat == "harmful":
            harmful_r.append(line)
        else:
            benign_r.append(line)

# Split lossmasks
harmful_l, benign_l = [], []
with open(lossmasks_path) as f:
    for line in f:
        lm = json.loads(line)
        tid = lm.get("trace_id", "")
        t = traces.get(tid, {})
        cat = t.get("labels", {}).get("category", "")
        if cat == "harmful":
            harmful_l.append(line)
        else:
            benign_l.append(line)

for name, data in [("renders_harmful", harmful_r), ("renders_benign", benign_r),
                    ("lossmasks_harmful", harmful_l), ("lossmasks_benign", benign_l)]:
    path = f"{out_dir}/{name}.jsonl"
    with open(path, "w") as f:
        f.writelines(data)
    print(f"  fujitsu {name}: {len(data)}")
SPLIT_FUJITSU
fi

# Split AgentDojo (reuse existing script)
if [[ -f "$SHARED_DATA/split_agentdojo/agentdojo_renders_harmful.jsonl" ]] && [[ -s "$SHARED_DATA/split_agentdojo/agentdojo_renders_harmful.jsonl" ]]; then
    echo "  AgentDojo split: REUSING existing"
else
python "$REPO_DIR/scripts/split_agentdojo.py" \
    --traces "$AGENTDOJO_SRC" \
    --renders "$SHARED_DATA/renders/agentdojo_truncated.jsonl" \
    --lossmasks "$SHARED_DATA/lossmasks/agentdojo_truncated.jsonl" \
    --output-dir "$SHARED_DATA/split_agentdojo" \
    --legacy-harmful-naming 2>&1 | tail -5
fi

echo ""

# =============================================================================
# Step 3: Training
# =============================================================================
echo "========================================================"
echo "Step 3: Training ($LOSS_MODE, $TOTAL_STEPS steps)"
echo "========================================================"
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
    --triplet-margin-benign 1.2 \
    --triplet-margin-harmful 1.6 \
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
    --lora-target-layers "$LORA_LAYERS" \
    --logging-steps 10 \
    --save-steps 9999 \
    --no-wandb \
    --cb-target-layers $CB_LAYERS \
    --mode mixed \
    --harmful-renders "$SHARED_DATA/split_fujitsu/renders_harmful.jsonl" \
                      "$SHARED_DATA/split_agentdojo/agentdojo_renders_harmful.jsonl" \
    --harmful-lossmasks "$SHARED_DATA/split_fujitsu/lossmasks_harmful.jsonl" \
                        "$SHARED_DATA/split_agentdojo/agentdojo_lossmasks_harmful.jsonl" \
    --benign-renders "$SHARED_DATA/split_fujitsu/renders_benign.jsonl" \
                     "$SHARED_DATA/split_agentdojo/agentdojo_renders_benign.jsonl" \
    --benign-lossmasks "$SHARED_DATA/split_fujitsu/lossmasks_benign.jsonl" \
                       "$SHARED_DATA/split_agentdojo/agentdojo_lossmasks_benign.jsonl" \
    --contrastive-pairs "$PAIRS_SRC" \
    > "$RUN_DIR/train.log" 2>&1

train_status=$?
train_mins=$(( ($(date +%s) - train_start) / 60 ))
if [[ $train_status -eq 0 ]]; then
    echo "  Training SUCCESS (${train_mins}m)"
else
    echo "  Training FAILED (${train_mins}m)"
    tail -20 "$RUN_DIR/train.log" | sed 's/^/    /'
    exit 1
fi
echo ""

# =============================================================================
# Step 4: Evaluation
# =============================================================================
echo "========================================================"
echo "Step 4: Evaluation ($EVAL_MODE, n=$LIMIT_EVAL)"
echo "========================================================"

# Find adapter
if [[ -d "$RUN_DIR/model/final" ]]; then
    ADAPTER_PATH="$RUN_DIR/model/final"
elif [[ -f "$RUN_DIR/model/adapter_config.json" ]]; then
    ADAPTER_PATH="$RUN_DIR/model"
else
    echo "ERROR: No adapter found"; exit 1
fi

EVAL_DIR="$RUN_DIR/eval/$EVAL_MODE"
mkdir -p "$EVAL_DIR"

# Use original (non-truncated) data for eval to match Sweep 5
AGENTDOJO_EVAL_SRC="$TRACES_DIR/agentdojo_augmented.jsonl"
DS_TRACES="$TRACES_DIR/fujitsu_b4_ds.jsonl"

# Fujitsu eval
echo "  Fujitsu eval..."
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

# AgentDojo harmful eval — split the eval source on the fly
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
print(f'  AD split: {len(harmful)} harmful, {len(benign)} benign')
"

echo "  AgentDojo harmful eval..."
python src/evaluation/eval.py \
    --baseline "$TOKENIZER" \
    --cb-adapter "$ADAPTER_PATH" \
    --eval-data "$EVAL_DIR/ad_harmful_traces.jsonl" \
    --tool-schema "$TOOL_SCHEMA" \
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

echo "  AgentDojo benign eval..."
python src/evaluation/eval.py \
    --baseline "$TOKENIZER" \
    --cb-adapter "$ADAPTER_PATH" \
    --eval-data "$EVAL_DIR/ad_benign_traces.jsonl" \
    --tool-schema "$TOOL_SCHEMA" \
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

# =============================================================================
# Done
# =============================================================================
echo ""
echo "========================================================"
echo "COMPLETE ($LOSS_MODE)"
echo "========================================================"
echo "Output: $RUN_DIR"
echo "Date: $(date)"
