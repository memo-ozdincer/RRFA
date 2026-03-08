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
# SWEEP 6: Completed Fujitsu + Truncated AgentDojo
# =============================================================================
#
# Single run with all settled hyperparameters from Sweep 5.
# Only change: completed Fujitsu traces (not skeletons) and truncated AD traces.
#
# Data:
#   fujitsu_b4_completed.jsonl  - 26,492 (13,246 harmful + 13,246 benign)
#   agentdojo_truncated.jsonl   - 4,144 (1,000 harmful + 2,894 benign + 250 resisted)
#   contrastive_pairs.jsonl     - 1,247 pairs
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
# Configuration (ALL SETTLED from Sweep 5)
# =============================================================================
QUICK_TEST="${QUICK_TEST:-false}"
if [[ "$QUICK_TEST" == "true" ]]; then
    TOTAL_STEPS=20; LIMIT_EVAL=5; NUM_EXAMPLES=2
    echo "*** QUICK TEST MODE ***"
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
GAMMA_KL=0.3
LOSS_MODE=triplet_full
POLICY=injection_aware
POOLING_POLICY=cb_full_sequence
EVAL_MODE=next_tool_prediction
PRESET=llama-3.1-8b-instruct

echo "========================================================"
echo "SWEEP 6: Completed Fujitsu + Truncated AgentDojo"
echo "========================================================"
echo "Job: ${SLURM_JOB_ID:-local} | $(date)"
echo "Steps: $TOTAL_STEPS | Policy: $POLICY | LoRA: $LORA_LAYERS | γ_kl: $GAMMA_KL"
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

# Copy from repo if not on scratch yet (agentdojo, contrastive pairs)
for src_name in agentdojo_augmented.jsonl contrastive_pairs.jsonl; do
    if [[ ! -f "$TRACES_DIR/$src_name" ]] && [[ -f "$REPO_DIR/data/traces/$src_name" ]]; then
        echo "  Copying $src_name to scratch..."
        cp "$REPO_DIR/data/traces/$src_name" "$TRACES_DIR/$src_name"
    fi
done

for f in "$FUJITSU_RAW" "$AGENTDOJO_RAW" "$PAIRS_SRC"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing $(basename "$f") — not on scratch and not in repo"; exit 1
    fi
    echo "  $(basename "$f"): $(wc -l < "$f") lines"
done
echo ""

# =============================================================================
# Output directory
# =============================================================================
SWEEP_ID="completed_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$CB_SCRATCH/sweeps/$SWEEP_ID/run"
mkdir -p "$RUN_DIR"/{renders,lossmasks,split_fujitsu,split_agentdojo,model,eval}
echo "Output: $RUN_DIR"
echo ""

# =============================================================================
# Step 0: Generate completed Fujitsu + truncated AgentDojo
# =============================================================================
echo "========================================================"
echo "Step 0: Data generation"
echo "========================================================"
gen_start=$(date +%s)

FUJITSU_SRC="$RUN_DIR/fujitsu_b4_completed.jsonl"
AGENTDOJO_SRC="$RUN_DIR/agentdojo_truncated.jsonl"

echo "  Completing Fujitsu skeletons..."
python "$REPO_DIR/scripts/augment_llm.py" \
    --operation complete_fujitsu \
    --input "$FUJITSU_RAW" \
    --output "$FUJITSU_SRC" 2>&1 | tail -5

echo "  Truncating AgentDojo to injection window..."
python "$REPO_DIR/scripts/truncate_to_injection_window.py" \
    --input "$AGENTDOJO_RAW" \
    --output "$AGENTDOJO_SRC" 2>&1 | tail -5

for f in "$FUJITSU_SRC" "$AGENTDOJO_SRC"; do
    if [[ ! -f "$f" ]] || [[ ! -s "$f" ]]; then
        echo "ERROR: Data generation failed for $(basename "$f")"; exit 1
    fi
    echo "  $(basename "$f"): $(wc -l < "$f") lines"
done

echo "  Data generation done ($(( $(date +%s) - gen_start ))s)"
echo ""

# =============================================================================
# Step 1: ETL_B (render + lossmask)
# =============================================================================
echo "========================================================"
echo "Step 1: ETL_B"
echo "========================================================"
etl_start=$(date +%s)

for src in "$FUJITSU_SRC" "$AGENTDOJO_SRC"; do
    bn=$(basename "$src" .jsonl)
    echo "  $bn (policy=$POLICY, pool=$POOLING_POLICY)..."
    python src/schemas/tools/ETL_B.py \
        --traces "$src" \
        --render-out "$RUN_DIR/renders/${bn}.jsonl" \
        --lossmask-out "$RUN_DIR/lossmasks/${bn}.jsonl" \
        --tokenizer "$TOKENIZER" \
        --max-length "$MAX_SEQ_LENGTH" \
        --policy-override "$POLICY" \
        --pooling-policy-override "$POOLING_POLICY" 2>&1 | tail -3
done

echo "  ETL_B done ($(( $(date +%s) - etl_start ))s)"

echo ""

# =============================================================================
# Step 2: Split by category
# =============================================================================
echo "========================================================"
echo "Step 2: Split harmful/benign"
echo "========================================================"

# Split Fujitsu (harmful vs benign by category in trace data)
python3 << 'SPLIT_FUJITSU'
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
"$FUJITSU_SRC" "$RUN_DIR/renders/fujitsu_b4_completed.jsonl" "$RUN_DIR/lossmasks/fujitsu_b4_completed.jsonl" "$RUN_DIR/split_fujitsu"

# Split AgentDojo (reuse existing script)
python "$REPO_DIR/scripts/split_agentdojo.py" \
    --traces "$AGENTDOJO_SRC" \
    --renders "$RUN_DIR/renders/agentdojo_truncated.jsonl" \
    --lossmasks "$RUN_DIR/lossmasks/agentdojo_truncated.jsonl" \
    --output-dir "$RUN_DIR/split_agentdojo" \
    --legacy-harmful-naming 2>&1 | tail -5

echo ""

# =============================================================================
# Step 3: Training
# =============================================================================
echo "========================================================"
echo "Step 3: Training ($TOTAL_STEPS steps)"
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
    --alpha-max 10.0 \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --loss-mode "$LOSS_MODE" \
    --loss-weighting dual \
    --triplet-alpha-benign 0.5 \
    --triplet-beta-harmful 0.4 \
    --triplet-gamma-kl "$GAMMA_KL" \
    --triplet-margin-benign 1.2 \
    --triplet-margin-harmful 1.6 \
    --triplet-benign-positive-distance dcos \
    --triplet-benign-negative-distance dcos \
    --triplet-harmful-positive-distance dcos \
    --triplet-harmful-negative-distance dcos \
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
    --harmful-renders "$RUN_DIR/split_fujitsu/renders_harmful.jsonl" \
                      "$RUN_DIR/split_agentdojo/agentdojo_renders_harmful.jsonl" \
    --harmful-lossmasks "$RUN_DIR/split_fujitsu/lossmasks_harmful.jsonl" \
                        "$RUN_DIR/split_agentdojo/agentdojo_lossmasks_harmful.jsonl" \
    --benign-renders "$RUN_DIR/split_fujitsu/renders_benign.jsonl" \
                     "$RUN_DIR/split_agentdojo/agentdojo_renders_benign.jsonl" \
    --benign-lossmasks "$RUN_DIR/split_fujitsu/lossmasks_benign.jsonl" \
                       "$RUN_DIR/split_agentdojo/agentdojo_lossmasks_benign.jsonl" \
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
echo "COMPLETE"
echo "========================================================"
echo "Output: $RUN_DIR"
echo "Date: $(date)"
