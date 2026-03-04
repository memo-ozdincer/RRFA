#!/bin/bash
#SBATCH --job-name=sweep_lmfix
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# LOSSMASK FIX VALIDATION SWEEP
# =============================================================================
#
# Purpose: Validate the post_injection_only fix (all-zero fallback) and the
# new post_injection_tool_calls_only policy against the cb_full_sequence
# baseline. Tests two layer configs (original CB paper vs RRFA paper).
#
# Key changes from sweep_hparams.sbatch:
# 1. Uses agentdojo_augmented.jsonl (has injection spans, etl_a_v3)
# 2. Runs lossmask diagnostics before training (coverage stats, zero-mask audit)
# 3. Focused 2D grid: 3 policies x 2 layer configs = 6 runs
# 4. Consolidated report with mask stats, training curves, eval, examples
#
# Usage:
#   sbatch slurm/pipeline/sweep_lossmask_fix.sbatch
#
#   # Quick sanity (20 steps, 5 eval samples):
#   QUICK_TEST=true sbatch slurm/pipeline/sweep_lossmask_fix.sbatch
#
#   # Custom grid:
#   POLICIES="post_injection_only,cb_full_sequence" \
#   LAYER_CONFIGS="10,20|20,21,22,23,24,25,26,27,28,29,30,31" \
#   sbatch slurm/pipeline/sweep_lossmask_fix.sbatch
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
echo "LOSSMASK FIX VALIDATION SWEEP"
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
    DEFAULT_STEPS=20
    DEFAULT_LIMIT_EVAL=5
    DEFAULT_NUM_EXAMPLES=2
else
    DEFAULT_STEPS=150
    DEFAULT_LIMIT_EVAL=50
    DEFAULT_NUM_EXAMPLES=5
fi

TOTAL_STEPS="${TOTAL_STEPS:-$DEFAULT_STEPS}"
LIMIT_EVAL="${LIMIT_EVAL:-$DEFAULT_LIMIT_EVAL}"
NUM_EXAMPLES="${NUM_EXAMPLES:-$DEFAULT_NUM_EXAMPLES}"

# Sweep dimensions: policies x layers
POLICIES="${POLICIES:-post_injection_only,post_injection_tool_calls_only,cb_full_sequence}"
LAYER_CONFIGS="${LAYER_CONFIGS:-10,20|20,21,22,23,24,25,26,27,28,29,30,31}"

# Fixed training hparams (paper-aligned, single pooling preset)
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
DTYPE="${DTYPE:-bfloat16}"
LOSS_MODE="${LOSS_MODE:-triplet_full}"
LOSS_WEIGHTING="${LOSS_WEIGHTING:-dual}"
GAMMA_KL="${GAMMA_KL:-0.1}"
TRIPLET_ALPHA_BENIGN="${TRIPLET_ALPHA_BENIGN:-0.5}"
TRIPLET_BETA_HARMFUL="${TRIPLET_BETA_HARMFUL:-0.4}"
TRIPLET_MIX_L2_WEIGHT="${TRIPLET_MIX_L2_WEIGHT:-0.5}"
TRIPLET_MIX_COS_WEIGHT="${TRIPLET_MIX_COS_WEIGHT:-0.5}"
SAVE_STEPS="${SAVE_STEPS:-9999}"

# Pooling: use correct mode with dcos (best from prior sweep)
POOLING_MODE="${POOLING_MODE:-correct}"
POOLING_DISTANCE="${POOLING_DISTANCE:-dcos}"
POOLING_MARGIN_B="${POOLING_MARGIN_B:-1.2}"
POOLING_MARGIN_H="${POOLING_MARGIN_H:-1.6}"
POOLING_MASK_POLICY="${POOLING_MASK_POLICY:-auto}"

# The pooling mask in ETL_B: use cb_full_sequence for pooling even when
# loss mask is post_injection_only (ensures pooling covers enough tokens)
POOLING_POLICY="${POOLING_POLICY:-cb_full_sequence}"

echo "Configuration:"
echo "  TOTAL_STEPS:    $TOTAL_STEPS"
echo "  POLICIES:       $POLICIES"
echo "  LAYER_CONFIGS:  $LAYER_CONFIGS"
echo "  GAMMA_KL:       $GAMMA_KL"
echo "  LOSS_MODE:      $LOSS_MODE"
echo "  POOLING:        $POOLING_MODE / $POOLING_DISTANCE / $POOLING_MARGIN_B / $POOLING_MARGIN_H"
echo "  LIMIT_EVAL:     $LIMIT_EVAL"
echo "  NUM_EXAMPLES:   $NUM_EXAMPLES"
echo "  QUICK_TEST:     $QUICK_TEST"
echo ""

# =============================================================================
# Input Traces — Use augmented AgentDojo (has injection spans)
# =============================================================================
TRACES_DIR="$CB_SCRATCH/data/traces"
DS_TRACES="$TRACES_DIR/fujitsu_b4_ds.jsonl"
DR_TRACES="$TRACES_DIR/fujitsu_b4_dr.jsonl"
TOOL_SCHEMA="$REPO_DIR/configs/tool_schemas/b4_standard_v1.json"

# KEY CHANGE: Use agentdojo_augmented (etl_a_v3, HAS injection spans)
# agentdojo_complete was etl_a_v1 with ZERO injection spans.
AGENTDOJO_AUGMENTED_SRC="$REPO_DIR/data/traces/agentdojo_augmented.jsonl"
AGENTDOJO_TRACES="$TRACES_DIR/agentdojo_augmented.jsonl"

# Copy augmented data to scratch if not there
if [[ ! -f "$AGENTDOJO_TRACES" ]]; then
    if [[ -f "$AGENTDOJO_AUGMENTED_SRC" ]]; then
        echo "Copying agentdojo_augmented.jsonl to scratch..."
        cp "$AGENTDOJO_AUGMENTED_SRC" "$AGENTDOJO_TRACES"
    else
        echo "ERROR: agentdojo_augmented.jsonl not found at $AGENTDOJO_AUGMENTED_SRC"
        exit 1
    fi
fi

echo "Input traces:"
for trace in "$DS_TRACES" "$DR_TRACES" "$AGENTDOJO_TRACES"; do
    if [[ -f "$trace" ]]; then
        echo "  ✓ $(basename "$trace"): $(wc -l < "$trace") lines"
    else
        echo "  ✗ $(basename "$trace"): NOT FOUND"
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
SWEEP_ID="lmfix_$(date +%Y%m%d_%H%M%S)"
export SWEEP_DIR="$CB_SCRATCH/sweeps/$SWEEP_ID"
mkdir -p "$SWEEP_DIR"
echo "Sweep directory: $SWEEP_DIR"
echo ""

# =============================================================================
# Parse Grid
# =============================================================================
IFS=',' read -ra POLICY_ARRAY <<< "$POLICIES"
IFS='|' read -ra LAYER_ARRAY <<< "$LAYER_CONFIGS"

TOTAL_RUNS=$(( ${#POLICY_ARRAY[@]} * ${#LAYER_ARRAY[@]} ))
echo "========================================================"
echo "TOTAL CONFIGURATIONS: $TOTAL_RUNS"
echo "  Policies: ${POLICY_ARRAY[*]}"
echo "  Layer configs: ${LAYER_ARRAY[*]}"
echo "========================================================"
echo ""

# =============================================================================
# Pre-sweep: Lossmask Diagnostic (run once per policy)
# =============================================================================
echo "========================================================"
echo "PHASE 0: LOSSMASK DIAGNOSTICS"
echo "========================================================"
echo ""

DIAG_DIR="$SWEEP_DIR/diagnostics"
mkdir -p "$DIAG_DIR"

for policy in "${POLICY_ARRAY[@]}"; do
    echo "--- Policy: $policy ---"

    # Run ETL_B on a sample of AgentDojo to get mask stats
    DIAG_RENDER="$DIAG_DIR/${policy}_renders.jsonl"
    DIAG_MASK="$DIAG_DIR/${policy}_lossmasks.jsonl"

    python src/schemas/tools/ETL_B.py \
        --traces "$AGENTDOJO_TRACES" \
        --render-out "$DIAG_RENDER" \
        --lossmask-out "$DIAG_MASK" \
        --tokenizer "$TOKENIZER" \
        --max-length "$MAX_SEQ_LENGTH" \
        --policy-override "$policy" \
        --pooling-policy-override "$POOLING_POLICY" \
        --include-rendered-text \
        --limit 200 2>&1 | tail -3

    # Analyze mask coverage
    python3 << PYDIAG
import json

policy = "$policy"
mask_file = "$DIAG_MASK"
render_file = "$DIAG_RENDER"

masks = []
with open(mask_file) as f:
    for line in f:
        if line.strip():
            masks.append(json.loads(line))

renders = {}
with open(render_file) as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            renders[d["render_id"]] = d

# Stats by category
harmful_masks = []
benign_masks = []
zero_masks = []
total = len(masks)

for m in masks:
    loss_mask = m["loss_mask"]
    stats = m.get("stats", {})
    unmasked = stats.get("unmasked_tokens", sum(1 for x in loss_mask if x > 0))
    total_tok = stats.get("total_tokens", len(loss_mask))
    ratio = unmasked / total_tok if total_tok > 0 else 0

    # Check if harmful or benign from trace_id
    tid = m.get("trace_id", "")

    if unmasked == 0:
        zero_masks.append(m)

    # Use ratio to classify
    if ratio > 0:
        harmful_masks.append((ratio, unmasked, total_tok))
    else:
        benign_masks.append((ratio, unmasked, total_tok))

    # Check pooling mask too
    pool = m.get("pooling_mask")
    pool_stats = m.get("pooling_stats", {})

print(f"  Policy: {policy}")
print(f"  Total samples: {total}")
print(f"  Zero-mask samples: {len(zero_masks)} ({100*len(zero_masks)/total:.0f}%)")
print(f"  Nonzero-mask samples: {len(harmful_masks)} ({100*len(harmful_masks)/total:.0f}%)")
if harmful_masks:
    ratios = [r for r, _, _ in harmful_masks]
    avg_ratio = sum(ratios) / len(ratios)
    avg_unmasked = sum(u for _, u, _ in harmful_masks) / len(harmful_masks)
    print(f"  Avg mask coverage (nonzero only): {avg_ratio:.1%} ({avg_unmasked:.0f} tokens)")
    min_r = min(ratios)
    max_r = max(ratios)
    print(f"  Coverage range: {min_r:.1%} — {max_r:.1%}")

# Compare with pooling mask if present
pool_present = sum(1 for m in masks if m.get("pooling_mask"))
if pool_present > 0:
    pool_nonzero = sum(1 for m in masks if m.get("pooling_stats", {}).get("unmasked_tokens", 0) > 0)
    print(f"  Pooling mask present: {pool_present}/{total}, nonzero: {pool_nonzero}")

print()

# Print first 3 zero-mask examples (if any) for debugging
if zero_masks and policy != "cb_full_sequence":
    print(f"  Zero-mask sample IDs (first 3):")
    for m in zero_masks[:3]:
        rid = m.get("render_id", "?")
        tid = m.get("trace_id", "?")
        r = renders.get(rid, {})
        has_inj = bool(r.get("signals", {}).get("injection_spans"))
        n_asst = len(r.get("alignment", {}).get("assistant_spans", []))
        print(f"    {tid[:50]} has_injection={has_inj} n_assistant_spans={n_asst}")
    print()
PYDIAG

done

echo ""

# =============================================================================
# Main Sweep Loop
# =============================================================================
run_idx=0
success_count=0
fail_count=0

for policy in "${POLICY_ARRAY[@]}"; do
for layers in "${LAYER_ARRAY[@]}"; do

    # Compact layer notation
    IFS=',' read -ra CB_LAYER_ARRAY <<< "$layers"
    _n_layers=${#CB_LAYER_ARRAY[@]}
    _first=${CB_LAYER_ARRAY[0]}
    _last=${CB_LAYER_ARRAY[$((_n_layers - 1))]}
    if [[ $_n_layers -gt 3 && $((_last - _first + 1)) -eq $_n_layers ]]; then
        layers_short="${_first}-${_last}"
    else
        layers_short="${layers//,/_}"
    fi

    RUN_NAME="${policy}_l${layers_short}"
    RUN_DIR="$SWEEP_DIR/$RUN_NAME"
    run_idx=$((run_idx + 1))

    echo "========================================================"
    echo "[$run_idx/$TOTAL_RUNS] $RUN_NAME"
    echo "========================================================"
    echo "  Policy: $policy"
    echo "  Layers: $layers ($layers_short)"
    echo ""

    mkdir -p "$RUN_DIR"/{renders,lossmasks,model,eval}

    # =========================================================================
    # Step 1: ETL_B
    # =========================================================================
    echo "Step 1: ETL_B (policy=$policy)"
    etl_start=$(date +%s)

    for trace_file in "$DS_TRACES" "$DR_TRACES" "$AGENTDOJO_TRACES"; do
        basename=$(basename "$trace_file" .jsonl)
        render_out="$RUN_DIR/renders/${basename}.jsonl"
        lossmask_out="$RUN_DIR/lossmasks/${basename}.jsonl"

        ETL_B_EXTRA=()
        # Fujitsu DS traces are skeletons — allow with cb_full_sequence
        if [[ "$basename" == "fujitsu_b4_ds" ]]; then
            ETL_B_EXTRA+=(--allow-skeleton --skeleton-policy cb_full_sequence)
        fi

        echo "  Processing: $basename"
        python src/schemas/tools/ETL_B.py \
            --traces "$trace_file" \
            --render-out "$render_out" \
            --lossmask-out "$lossmask_out" \
            --tokenizer "$TOKENIZER" \
            --max-length "$MAX_SEQ_LENGTH" \
            --policy-override "$policy" \
            --pooling-policy-override "$POOLING_POLICY" \
            "${ETL_B_EXTRA[@]}" 2>&1 | tail -3
    done

    etl_end=$(date +%s)
    echo "  ETL_B done ($(( etl_end - etl_start ))s)"

    # Post-ETL_B: mask coverage summary per file
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

    # Show split sizes + zero-mask counts
    echo "  Split results:"
    for f in "$SPLIT_DIR"/*.jsonl; do
        bn=$(basename "$f")
        n=$(wc -l < "$f")
        if [[ "$bn" == *lossmask* ]]; then
            zc=$(python3 -c "
import json
z=0
with open('$f') as fh:
    for line in fh:
        if not line.strip(): continue
        d=json.loads(line)
        if d.get('stats',{}).get('mask_ratio',0)==0: z+=1
print(z)
" 2>/dev/null)
            echo "    $bn: $n lines (zero_mask=$zc)"
        else
            echo "    $bn: $n lines"
        fi
    done
    echo ""

    # =========================================================================
    # Step 3: Save config
    # =========================================================================
    cat > "$RUN_DIR/run_config.json" << RUNCFG
{
  "policy": "$policy",
  "layers": "$layers",
  "layers_short": "$layers_short",
  "total_steps": $TOTAL_STEPS,
  "batch_size": $BATCH_SIZE,
  "grad_accum": $GRAD_ACCUM,
  "effective_batch": $(( BATCH_SIZE * GRAD_ACCUM )),
  "learning_rate": "$LEARNING_RATE",
  "loss_mode": "$LOSS_MODE",
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
  "agentdojo_source": "agentdojo_augmented (etl_a_v3, has injection spans)",
  "model": "$TOKENIZER",
  "run_name": "$RUN_NAME",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "job_id": "${SLURM_JOB_ID:-local}"
}
RUNCFG

    # =========================================================================
    # Step 4: Training
    # =========================================================================
    echo "Step 3: Training ($TOTAL_STEPS steps)"

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

    # Extract key training metrics from log
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
        # Parse step lines: "Step X: loss=... benign_loss=... harmful_loss=..."
        m = re.search(r'step[= ]+(\d+).*?loss[= ]+([0-9.e+-]+)', line, re.IGNORECASE)
        if m:
            step = int(m.group(1))
            loss = float(m.group(2))
            steps.append((step, loss, line.strip()))

if steps:
    print(f"    Steps logged: {len(steps)}")
    # Show first, middle, last
    for label, idx in [("First", 0), ("Mid", len(steps)//2), ("Last", -1)]:
        s, l, raw = steps[idx]
        # Truncate raw line for readability
        print(f"    {label} (step {s}): {raw[:120]}")
TRAINDIAG
    echo ""

    # =========================================================================
    # Step 5: Evaluation
    # =========================================================================
    echo "Step 4: Evaluation"

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

    # --- Fujitsu eval ---
    echo "  Fujitsu eval ($LIMIT_EVAL samples)..."
    FUJITSU_OUTPUT="$RUN_DIR/eval/fujitsu_eval.json"
    python src/evaluation/eval.py \
        --baseline "$TOKENIZER" \
        --cb-adapter "$ADAPTER_PATH" \
        --eval-data "$DS_TRACES" \
        --tool-schema "$TOOL_SCHEMA" \
        --output "$FUJITSU_OUTPUT" \
        --limit "$LIMIT_EVAL" \
        --merge-adapter \
        --dtype "$DTYPE" >> "$RUN_DIR/eval/fujitsu.log" 2>&1

    if [[ -f "$FUJITSU_OUTPUT" ]]; then
        python3 -c "
import json
with open('$FUJITSU_OUTPUT') as f: r = json.load(f)
for lbl, k in [('Baseline','baseline'),('CB','cb_model')]:
    tf = r.get(k,{}).get('tool_flip_asr',{})
    cap = r.get(k,{}).get('capability_retention',{})
    if not tf: continue
    print(f'    {lbl}: ASR={tf.get(\"attack_success_rate\",0):.0%} correct={tf.get(\"correct_tool_call_rate\",0):.0%} no_tool={tf.get(\"no_tool_call_rate\",0):.0%} cap={cap.get(\"capability_retention\",0):.0%}')
d = r.get('delta',{})
if d:
    print(f'    Delta: ASR={d.get(\"tool_flip_asr\",0):+.0%} cap={d.get(\"capability_retention\",0):+.0%}')
"
    fi

    # --- AgentDojo harmful eval ---
    echo "  AgentDojo harmful eval ($LIMIT_EVAL samples)..."
    AD_HARMFUL_OUTPUT="$RUN_DIR/eval/agentdojo_eval.json"
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
            --dtype "$DTYPE" >> "$RUN_DIR/eval/agentdojo.log" 2>&1

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

    # --- AgentDojo benign eval (THE KEY METRIC) ---
    echo "  AgentDojo benign eval ($LIMIT_EVAL samples)..."
    AD_BENIGN_OUTPUT="$RUN_DIR/eval/agentdojo_benign_eval.json"
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
            --dtype "$DTYPE" >> "$RUN_DIR/eval/agentdojo_benign.log" 2>&1

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
    if cor > 0:
        print(f'    >>> BENIGN_CORRECT > 0%! This is the key improvement. <<<')
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
    FUJITSU_PAIRED="${FUJITSU_OUTPUT%.json}.paired_outputs.jsonl"
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

    # AgentDojo benign examples (most important — did we fix 0% benign_correct?)
    AD_BENIGN_PAIRED="${AD_BENIGN_OUTPUT%.json}.paired_outputs.jsonl"
    if [[ -f "$AD_BENIGN_PAIRED" ]]; then
        python3 << PYEX2
import json

with open('$AD_BENIGN_PAIRED') as f:
    pairs = [json.loads(l) for l in f if l.strip()]

def gib(r):
    if not r: return False
    w = r.split()
    return len(w) > 10 and len(set(w))/len(w) < 0.15

n_ok = sum(1 for p in pairs if p.get('cb_observed_tool'))
n_brk = sum(1 for p in pairs if not p.get('cb_observed_tool'))
n_gib = sum(1 for p in pairs if gib(p.get('cb_response','')))

print(f'  AD Benign: {len(pairs)} pairs | tool_ok={n_ok} broken={n_brk} gib={n_gib}')
print()

broken = [p for p in pairs if p.get('baseline_observed_tool') and not p.get('cb_observed_tool')]
working = [p for p in pairs if p.get('cb_observed_tool')]

if broken:
    print(f'  Broken examples (baseline had tool, CB lost it):')
    for i, p in enumerate(broken[:$NUM_EXAMPLES], 1):
        g = ' [GIB]' if gib(p.get('cb_response','')) else ''
        print(f'  [{i}] BROKEN: {p.get("baseline_observed_tool","?")} -> none{g}')
        print(f'  BASELINE: {(p.get("baseline_response") or "(empty)")[:250]}')
        print(f'  CB MODEL: {(p.get("cb_response") or "(empty)")[:250]}')
        print()

if working:
    print(f'  Working examples (CB preserved tool calling):')
    for i, p in enumerate(working[:min(3, $NUM_EXAMPLES)], 1):
        print(f'  [{i}] OK: tool={p.get("cb_observed_tool","?")}')
        print(f'  CB MODEL: {(p.get("cb_response") or "(empty)")[:250]}')
        print()
PYEX2
    fi

    # Generate per-run summary
    python scripts/generate_run_summary.py --run-dir "$RUN_DIR" 2>&1 || true

    success_count=$((success_count + 1))
    echo ""

done  # layers
done  # policy

# =============================================================================
# Final Consolidated Report
# =============================================================================
echo ""
echo "========================================================"
echo "CONSOLIDATED RESULTS"
echo "========================================================"
echo ""

python "$REPO_DIR/scripts/generate_results_report.py" \
    --sweep-dir "$SWEEP_DIR" \
    --include-mask-stats

# Also generate sweep summary
python scripts/generate_sweep_summary.py --sweep-dir "$SWEEP_DIR" 2>&1 || true

echo ""
echo "========================================================"
echo "SWEEP COMPLETE"
echo "========================================================"
echo "Total: $TOTAL_RUNS | Success: $success_count | Failed: $fail_count"
echo "Output: $SWEEP_DIR"
echo "Report: $REPORT_FILE"
echo "Date: $(date)"
