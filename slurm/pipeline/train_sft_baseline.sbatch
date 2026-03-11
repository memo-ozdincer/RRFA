#!/bin/bash
#SBATCH --job-name=sft_baseline
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# SFT Refusal Baseline
# =============================================================================
#
# Trains a standard SFT adapter (next-token prediction on refusal responses)
# instead of representation rerouting. This is the "just teach it to refuse"
# baseline for comparison against CB training.
#
# Pipeline:
#   1. Generate refusal traces from harmful data (replace malicious assistant
#      response with refusal text)
#   2. Run ETL_B on refusal traces to get tokenized renders + loss masks
#   3. Train with causal LM loss on refusal response tokens (loss_mask > 0)
#   4. Evaluate with same eval suite as CB runs
#
# Usage:
#   sbatch slurm/pipeline/train_sft_baseline.sbatch
#   QUICK_TEST=true sbatch slurm/pipeline/train_sft_baseline.sbatch
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

if [[ "$QUICK_TEST" == "true" ]]; then
    TOTAL_STEPS=20; LIMIT_EVAL=5; NUM_EXAMPLES=2
    echo "*** QUICK TEST MODE ***"
else
    TOTAL_STEPS="${TOTAL_STEPS:-300}"; LIMIT_EVAL=100; NUM_EXAMPLES=5
fi

BATCH_SIZE=4
GRAD_ACCUM=2
LEARNING_RATE=5e-5
WARMUP_STEPS=20
MAX_SEQ_LENGTH=4096
LORA_R=16
LORA_ALPHA=32
EVAL_MODE=next_tool_prediction
POLICY="${POLICY:-cb_full_sequence}"
POOLING_POLICY="${POOLING_POLICY:-cb_full_sequence}"

echo "========================================================"
echo "SFT Refusal Baseline"
echo "========================================================"
echo "Job: ${SLURM_JOB_ID:-local} | $(date)"
echo "Steps: $TOTAL_STEPS | Batch: ${BATCH_SIZE}x${GRAD_ACCUM}=$((BATCH_SIZE*GRAD_ACCUM))"
echo "LR: $LEARNING_RATE | LoRA: r=$LORA_R alpha=$LORA_ALPHA"
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
# Input data
# =============================================================================
TRACES_DIR="$CB_SCRATCH/data/traces"
FUJITSU_DS="$TRACES_DIR/fujitsu_b4_ds.jsonl"
AGENTDOJO_RAW="$TRACES_DIR/agentdojo_augmented.jsonl"
TOOL_SCHEMA="$REPO_DIR/configs/tool_schemas/b4_standard_v1.json"
AD_TOOL_SCHEMA="$REPO_DIR/configs/tool_schemas/agentdojo_v1.json"

# Copy from repo if not on scratch yet (agentdojo)
for src_name in agentdojo_augmented.jsonl; do
    if [[ ! -f "$TRACES_DIR/$src_name" ]] && [[ -f "$REPO_DIR/data/traces/$src_name" ]]; then
        echo "  Copying $src_name to scratch..."
        cp "$REPO_DIR/data/traces/$src_name" "$TRACES_DIR/$src_name"
    fi
done

for f in "$FUJITSU_DS" "$AGENTDOJO_RAW"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing $(basename "$f") — not on scratch and not in repo"; exit 1
    fi
    echo "  $(basename "$f"): $(wc -l < "$f") lines"
done
echo ""

# =============================================================================
# Output directories
# =============================================================================
SWEEP_ID="sft_baseline_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$CB_SCRATCH/sweeps/$SWEEP_ID/run"
DATA_DIR="$RUN_DIR/data"
mkdir -p "$RUN_DIR"/{model,eval}
mkdir -p "$DATA_DIR"/{traces,renders,lossmasks}
echo "Run output: $RUN_DIR"
echo ""

# =============================================================================
# Step 0: Generate refusal traces
# =============================================================================
echo "========================================================"
echo "Step 0: Generate refusal traces"
echo "========================================================"
gen_start=$(date +%s)

# Truncate AgentDojo to injection window (same as sweep_completed_v1)
AGENTDOJO_TRUNC="$DATA_DIR/traces/agentdojo_truncated.jsonl"
echo "  Truncating AgentDojo..."
python "$REPO_DIR/scripts/truncate_to_injection_window.py" \
    --input "$AGENTDOJO_RAW" \
    --output "$AGENTDOJO_TRUNC" 2>&1 | tail -5

# Generate refusal traces from truncated AgentDojo harmful
echo "  Generating AgentDojo refusal traces..."
python scripts/generate_refusal_traces.py \
    --traces "$AGENTDOJO_TRUNC" \
    --output "$DATA_DIR/traces/agentdojo_refusal.jsonl"

# Generate refusal traces from Fujitsu DS (harmful)
echo "  Generating Fujitsu refusal traces..."
python scripts/generate_refusal_traces.py \
    --traces "$FUJITSU_DS" \
    --output "$DATA_DIR/traces/fujitsu_refusal.jsonl"

for f in "$DATA_DIR/traces/agentdojo_refusal.jsonl" "$DATA_DIR/traces/fujitsu_refusal.jsonl"; do
    if [[ ! -f "$f" ]] || [[ ! -s "$f" ]]; then
        echo "ERROR: Refusal trace generation failed for $(basename "$f")"; exit 1
    fi
    echo "  $(basename "$f"): $(wc -l < "$f") lines"
done

echo "  Data generation done ($(( $(date +%s) - gen_start ))s)"
echo ""

# =============================================================================
# Step 1: ETL_B on refusal traces
# =============================================================================
echo "========================================================"
echo "Step 1: ETL_B on refusal traces"
echo "========================================================"
etl_start=$(date +%s)

for src in "$DATA_DIR/traces/agentdojo_refusal.jsonl" "$DATA_DIR/traces/fujitsu_refusal.jsonl"; do
    bn=$(basename "$src" .jsonl)
    render_out="$DATA_DIR/renders/${bn}.jsonl"
    mask_out="$DATA_DIR/lossmasks/${bn}.jsonl"

    # Select correct tool schema per source
    if [[ "$bn" == *fujitsu* ]]; then
        ETL_SCHEMA="$TOOL_SCHEMA"
    else
        ETL_SCHEMA="$AD_TOOL_SCHEMA"
    fi

    echo "  $bn (policy=$POLICY)..."
    python src/schemas/tools/ETL_B.py \
        --traces "$src" \
        --render-out "$render_out" \
        --lossmask-out "$mask_out" \
        --tokenizer "$TOKENIZER" \
        --max-length "$MAX_SEQ_LENGTH" \
        --policy-override "$POLICY" \
        --pooling-policy-override "$POOLING_POLICY" \
        --tool-schema "$ETL_SCHEMA" 2>&1 | tail -3
done

echo "  ETL_B done ($(( $(date +%s) - etl_start ))s)"

# Lossmask coverage summary
echo ""
echo "  Lossmask coverage:"
for lm in "$DATA_DIR/lossmasks"/*.jsonl; do
    if [[ -f "$lm" ]]; then
        python3 -c "
import json
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
    fi
done
echo ""

# =============================================================================
# Step 2: Training (SFT on refusal tokens)
# =============================================================================
echo "========================================================"
echo "Step 2: SFT Training ($TOTAL_STEPS steps)"
echo "========================================================"
train_start=$(date +%s)

python scripts/train_sft_refusal.py \
    --model "$TOKENIZER" \
    --renders "$DATA_DIR/renders/agentdojo_refusal.jsonl" \
              "$DATA_DIR/renders/fujitsu_refusal.jsonl" \
    --lossmasks "$DATA_DIR/lossmasks/agentdojo_refusal.jsonl" \
                "$DATA_DIR/lossmasks/fujitsu_refusal.jsonl" \
    --output-dir "$RUN_DIR/model" \
    --total-steps "$TOTAL_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --lr "$LEARNING_RATE" \
    --warmup-steps "$WARMUP_STEPS" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --logging-steps 10 \
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

# Training history summary
echo ""
echo "  Training history:"
python3 - "$RUN_DIR/train.log" << 'TRAINDIAG'
import re, sys

log_path = sys.argv[1] if len(sys.argv) > 1 else ""
if not log_path:
    sys.exit(0)

steps = []
with open(log_path) as f:
    for line in f:
        m = re.search(r'\[Step\s+(\d+)/\d+\]\s+loss=([0-9.e+-]+)\s+lr=([0-9.e+-]+)', line)
        if m:
            steps.append((int(m.group(1)), f"loss={m.group(2)} lr={m.group(3)}"))

if steps:
    print(f"    Steps logged: {len(steps)}")
    for label, idx in [("First", 0), ("Mid", len(steps)//2), ("Last", -1)]:
        s, info = steps[idx]
        print(f"    {label:5s} (step {s:>4d}): {info}")
else:
    print("    No step diagnostics found in log")
TRAINDIAG
echo ""

# Save run config
cat > "$RUN_DIR/run_config.json" << RUNCFG
{
  "method": "sft_refusal",
  "total_steps": $TOTAL_STEPS,
  "batch_size": $BATCH_SIZE,
  "grad_accum": $GRAD_ACCUM,
  "effective_batch": $(( BATCH_SIZE * GRAD_ACCUM )),
  "learning_rate": "$LEARNING_RATE",
  "lora_r": $LORA_R,
  "lora_alpha": $LORA_ALPHA,
  "eval_mode": "$EVAL_MODE",
  "model": "$TOKENIZER",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "job_id": "${SLURM_JOB_ID:-local}"
}
RUNCFG

# =============================================================================
# Step 3: Evaluation (identical to sweep_completed_v1)
# =============================================================================
echo "========================================================"
echo "Step 3: Evaluation ($EVAL_MODE, n=$LIMIT_EVAL)"
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

# Use original (non-truncated) data for eval to match sweep_completed_v1
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
for lbl, k in [('Baseline','baseline'),('SFT','cb_model')]:
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

echo "  AgentDojo benign eval..."
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

# =============================================================================
# Examples (paired output comparisons)
# =============================================================================
echo ""
echo "========================================================"
echo "EXAMPLES (SFT Refusal Baseline)"
echo "========================================================"

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
print(f'  SFT outcomes: correct={n_cor} attack={n_atk} no_tool={n_nt} gib={n_gib}')
print()

for i, p in enumerate((imps+regs)[:num_ex], 1):
    lbl = 'IMP' if p.get('baseline_outcome')=='attack_success' else 'REG'
    g = ' [GIB]' if gib(p.get('cb_response','')) else ''
    print(f'  [{i}] {lbl}: {p.get("baseline_outcome","?")} -> {p.get("cb_outcome","?")}{g}')
    print(f'      expected={p.get("expected_tool","?")} simulated={p.get("simulated_tool","?")}')
    print(f'  BASELINE: {(p.get("baseline_response") or "(empty)")[:250]}')
    print(f'  SFT:      {(p.get("cb_response") or "(empty)")[:250]}')
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
    print(f'  Broken examples (baseline had tool, SFT lost it):')
    for i, p in enumerate(broken[:num_ex], 1):
        print(f'  [{i}] BROKEN: {p.get("baseline_observed_tool","?")} -> none')
        print(f'  SFT: {(p.get("cb_response") or "(empty)")[:250]}')
        print()
ADEX
fi

# Generate run summary if script exists
if [[ -f "$REPO_DIR/scripts/generate_run_summary.py" ]]; then
    python "$REPO_DIR/scripts/generate_run_summary.py" --run-dir "$RUN_DIR" 2>&1 || true
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================================"
echo "RESULTS SUMMARY (SFT Refusal Baseline)"
echo "========================================================"

python3 - "$RUN_DIR" "$TOTAL_STEPS" << 'SUMMARY'
import json, os, sys

run_dir = sys.argv[1]
steps = sys.argv[2]

fmt = lambda v: f"{v:.0%}" if isinstance(v, (int, float)) and v >= 0 else "n/a"

print(f"  Config: SFT refusal baseline | steps={steps}")
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
        print(f"    SFT:      ASR={fmt(cb.get('attack_success_rate', -1))} correct={fmt(cb.get('correct_tool_call_rate', -1))} no_tool={fmt(cb.get('no_tool_call_rate', -1))}")
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

print()
SUMMARY

echo "Output: $RUN_DIR"
echo "Date: $(date)"
