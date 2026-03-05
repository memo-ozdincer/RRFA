#!/bin/bash
#SBATCH --job-name=sweep_balanced
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=14:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# BALANCED TRAINING SWEEP (Sweep 6)
# =============================================================================
#
# Root cause from Sweep 5: 33/98 Fujitsu failures are all "hard cases":
#   - 17/19 Comment-based system override (search_web->retrieve) = 89% fail
#   - 11/40 Hidden-JSON (retrieve->search_web) = 28% fail
#   - 5/39 Over-broad web fan-out = 13% fail
#
# Training data imbalance:
#   - retrieve_multimodal_docs -> search_web: 12,501 (94.4%)
#   - search_web -> retrieve_multimodal_docs: 745 (5.6%) [Comment-based only]
#   - Self-delegation retrieval loop: 2,466 (18.6%) [0 eval samples]
#
# Fixes tested:
#   1. Upsample Comment-based traces 10x (745 -> 7450)
#   2. Increase margin_h from 1.6 to 3.2 (keep gradient flowing)
#   3. LoRA rank 32 (more capacity for diverse patterns)
#   4. More CB layers: 5,10,15,20 (broader intervention)
#   5. 600 steps (more time for minority direction)
#
# Grid: 4 runs, each adds one fix on top of Sweep 5 best:
#   Run 1: Sweep 5 best repro (baseline for comparison)
#   Run 2: + upsample 10x
#   Run 3: + upsample 10x + margin_h=3.2 + LoRA r=32
#   Run 4: + upsample 10x + margin_h=3.2 + LoRA r=32 + layers 5,10,15,20 + 600 steps
#
# Usage:
#   sbatch slurm/pipeline/sweep_balanced_v1.sbatch
#   QUICK_TEST=true sbatch slurm/pipeline/sweep_balanced_v1.sbatch
#   RUN_IDS="1,4" sbatch slurm/pipeline/sweep_balanced_v1.sbatch
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
echo "BALANCED TRAINING SWEEP (Sweep 6)"
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
    DEFAULT_STEPS=30
    DEFAULT_STEPS_LONG=40
    DEFAULT_LIMIT_EVAL=5
    DEFAULT_NUM_EXAMPLES=2
else
    DEFAULT_STEPS=300
    DEFAULT_STEPS_LONG=600
    DEFAULT_LIMIT_EVAL=100
    DEFAULT_NUM_EXAMPLES=5
fi

LIMIT_EVAL="${LIMIT_EVAL:-$DEFAULT_LIMIT_EVAL}"
NUM_EXAMPLES="${NUM_EXAMPLES:-$DEFAULT_NUM_EXAMPLES}"
RUN_IDS="${RUN_IDS:-1,2,3,4}"

# Fixed hparams from Sweep 5 best
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
DTYPE="${DTYPE:-bfloat16}"
LOSS_WEIGHTING="${LOSS_WEIGHTING:-dual}"
SAVE_STEPS="${SAVE_STEPS:-9999}"
TRIPLET_ALPHA_BENIGN="${TRIPLET_ALPHA_BENIGN:-0.5}"
TRIPLET_BETA_HARMFUL="${TRIPLET_BETA_HARMFUL:-0.4}"
TRIPLET_MIX_L2_WEIGHT="${TRIPLET_MIX_L2_WEIGHT:-0.5}"
TRIPLET_MIX_COS_WEIGHT="${TRIPLET_MIX_COS_WEIGHT:-0.5}"

# Fixed choices (settled from Sweeps 2-5)
LOSS_MODE="triplet_full"
POLICY="injection_aware"
EVAL_MODE="next_tool_prediction"
POOLING_MODE="${POOLING_MODE:-correct}"
POOLING_DISTANCE="${POOLING_DISTANCE:-dcos}"
POOLING_MARGIN_B="${POOLING_MARGIN_B:-1.2}"
POOLING_MASK_POLICY="${POOLING_MASK_POLICY:-auto}"
POOLING_POLICY="${POOLING_POLICY:-cb_full_sequence}"

echo "Configuration:"
echo "  LOSS_MODE:      $LOSS_MODE (fixed)"
echo "  POLICY:         $POLICY (fixed)"
echo "  EVAL_MODE:      $EVAL_MODE (fixed)"
echo "  POOLING:        $POOLING_MODE / $POOLING_DISTANCE"
echo "  LIMIT_EVAL:     $LIMIT_EVAL"
echo "  RUN_IDS:        $RUN_IDS"
echo "  QUICK_TEST:     $QUICK_TEST"
echo ""

# =============================================================================
# Define the 4 runs
# =============================================================================
# Format: RUN_NAME|LORA_LAYERS|GAMMA_KL|STEPS|LORA_R|MARGIN_H|CB_LAYERS|UPSAMPLE_FACTOR
declare -a RUNS=(
    "s5best_repro|0-20|0.3|$DEFAULT_STEPS|16|1.6|10,20|1"
    "upsample10x|0-20|0.3|$DEFAULT_STEPS|16|1.6|10,20|10"
    "ups_margin_r32|0-20|0.3|$DEFAULT_STEPS|32|3.2|10,20|10"
    "ups_mar_r32_l4_s600|0-20|0.3|$DEFAULT_STEPS_LONG|32|3.2|5,10,15,20|10"
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

if [[ -f "$CONTRASTIVE_PAIRS_SRC" ]]; then
    cp "$CONTRASTIVE_PAIRS_SRC" "$CONTRASTIVE_PAIRS"
    echo "Contrastive pairs: $(wc -l < "$CONTRASTIVE_PAIRS") pairs"
fi

echo ""
echo "Input traces:"
for trace in "$DS_TRACES" "$DR_TRACES" "$AGENTDOJO_TRACES"; do
    if [[ -f "$trace" ]]; then
        echo "  $(basename "$trace"): $(wc -l < "$trace") lines"
    else
        echo "  $(basename "$trace"): NOT FOUND"
    fi
done

# Tokenizer path
TOKENIZER=$(find "$HF_HUB_CACHE" -name "config.json" -path "*/Llama-3.1-8B-Instruct/*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
if [[ -z "$TOKENIZER" ]]; then
    TOKENIZER="meta-llama/Llama-3.1-8B-Instruct"
fi
echo "Tokenizer: $TOKENIZER"

# =============================================================================
# Sweep directory
# =============================================================================
SWEEP_ID="balanced_$(date +%Y%m%d_%H%M%S)"
export SWEEP_DIR="$CB_SCRATCH/sweeps/$SWEEP_ID"
mkdir -p "$SWEEP_DIR"
echo "Sweep directory: $SWEEP_DIR"
echo ""

# =============================================================================
# Run loop
# =============================================================================
run_idx=0
success_count=0
fail_count=0

for run_spec in "${RUNS[@]}"; do
    run_idx=$((run_idx + 1))

    # Check if this run is selected
    selected=false
    for sel in "${SELECTED_RUNS[@]}"; do
        if [[ "$sel" == "$run_idx" ]]; then selected=true; fi
    done
    if [[ "$selected" == "false" ]]; then continue; fi

    # Parse run spec
    IFS='|' read -r RUN_NAME LORA_LAYERS GAMMA_KL TOTAL_STEPS LORA_R MARGIN_H CB_LAYERS UPSAMPLE_FACTOR <<< "$run_spec"

    RUN_DIR="$SWEEP_DIR/$RUN_NAME"
    PRESET="default"

    # Parse CB layers
    IFS=',' read -ra CB_LAYER_ARRAY <<< "$CB_LAYERS"

    echo "========================================================"
    echo "RUN $run_idx: $RUN_NAME"
    echo "========================================================"
    echo "  LoRA layers:     $LORA_LAYERS"
    echo "  LoRA rank:       $LORA_R"
    echo "  gamma_kl:        $GAMMA_KL"
    echo "  margin_h:        $MARGIN_H"
    echo "  CB layers:       $CB_LAYERS"
    echo "  Steps:           $TOTAL_STEPS"
    echo "  Upsample factor: ${UPSAMPLE_FACTOR}x"
    echo "  Contrastive:     true"
    echo "  Policy:          $POLICY"
    echo ""

    mkdir -p "$RUN_DIR"/{renders,lossmasks,model,eval}

    # =========================================================================
    # Step 1: ETL_B
    # =========================================================================
    echo "Step 1: ETL_B (policy=$POLICY for harmful, cb_full_sequence for benign)"
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
            ETL_POLICY="$POLICY"
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

    # =========================================================================
    # Step 1.5: Upsample underrepresented Fujitsu variants
    # =========================================================================
    if [[ "$UPSAMPLE_FACTOR" -gt 1 ]]; then
        echo ""
        echo "Step 1.5: Upsampling Comment-based system override traces ${UPSAMPLE_FACTOR}x"

        RENDER_DS="$RUN_DIR/renders/fujitsu_b4_ds.jsonl"
        LOSSMASK_DS="$RUN_DIR/lossmasks/fujitsu_b4_ds.jsonl"

        python3 << UPSAMPLE_PY
import json, sys

factor = $UPSAMPLE_FACTOR

for fpath in ["$RENDER_DS", "$LOSSMASK_DS"]:
    lines = []
    minority_lines = []

    with open(fpath) as f:
        for line in f:
            lines.append(line)
            try:
                d = json.loads(line)
                # Identify Comment-based system override traces by the expected_tool direction
                # These have search_web as expected (the minority direction)
                trace_id = d.get("trace_id", d.get("id", ""))
                # Check if this is a minority-direction trace via injection content
                user_content = ""
                if "messages" in d:
                    # render file has messages
                    for m in d.get("messages", []):
                        if m.get("role") == "user":
                            user_content = m.get("content", "")
                            break
                elif "input_ids" in d:
                    # This is a tokenized render - check via trace metadata
                    pass

                # For lossmask files: check the variant via metadata or trace_id pattern
                # Simpler approach: check sample_weight or add marker
                # Actually, for skeleton traces (DS), all have the tool_attack info in the original trace
                # We'll use a helper: read the DS traces to find minority IDs
            except:
                pass

    # Re-read from DS_TRACES to identify minority trace IDs
    minority_ids = set()
    try:
        with open("$DS_TRACES") as f:
            for line in f:
                t = json.loads(line)
                ta = t.get("tool_attack", {})
                task = t.get("task", {})
                variant = task.get("variant", "")
                if "Comment" in variant or "comment" in variant:
                    minority_ids.add(t.get("id", ""))
    except Exception as e:
        print(f"  WARNING: Could not read DS traces: {e}", file=sys.stderr)

    print(f"  {fpath.split('/')[-1]}: minority_ids={len(minority_ids)}")

    # Now rescan the file and duplicate minority lines
    upsampled = []
    original_count = 0
    added_count = 0

    with open(fpath) as f:
        for line in f:
            original_count += 1
            upsampled.append(line)
            try:
                d = json.loads(line)
                tid = d.get("trace_id", d.get("id", ""))
                if tid in minority_ids:
                    # Add (factor-1) copies
                    for _ in range(factor - 1):
                        upsampled.append(line)
                        added_count += 1
            except:
                pass

    with open(fpath, "w") as f:
        for line in upsampled:
            f.write(line)

    print(f"    {fpath.split('/')[-1]}: {original_count} -> {original_count + added_count} (+{added_count} upsampled)")

UPSAMPLE_PY
    fi

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

    echo ""

    # =========================================================================
    # Step 3: Save run config
    # =========================================================================
    cat > "$RUN_DIR/run_config.json" << RUNCFG
{
  "run_number": $run_idx,
  "run_name": "$RUN_NAME",
  "loss_mode": "$LOSS_MODE",
  "policy": "$POLICY",
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
  "pooling_margin_h": $MARGIN_H,
  "pooling_policy": "$POOLING_POLICY",
  "pooling_mask_policy": "$POOLING_MASK_POLICY",
  "lora_r": $LORA_R,
  "lora_alpha": $LORA_R,
  "alpha_benign": $TRIPLET_ALPHA_BENIGN,
  "beta_harmful": $TRIPLET_BETA_HARMFUL,
  "use_contrastive": true,
  "upsample_factor": $UPSAMPLE_FACTOR,
  "eval_mode": "$EVAL_MODE",
  "model": "$TOKENIZER",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "job_id": "${SLURM_JOB_ID:-local}"
}
RUNCFG

    # =========================================================================
    # Step 4: Training
    # =========================================================================
    echo "Step 3: Training ($LOSS_MODE, $TOTAL_STEPS steps, LoRA=$LORA_LAYERS r=$LORA_R, gamma=$GAMMA_KL, margin_h=$MARGIN_H, CB=$CB_LAYERS)"

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
        --triplet-margin-harmful "$MARGIN_H"
        --triplet-benign-positive-distance "$POOLING_DISTANCE"
        --triplet-benign-negative-distance "$POOLING_DISTANCE"
        --triplet-harmful-positive-distance "$POOLING_DISTANCE"
        --triplet-harmful-negative-distance "$POOLING_DISTANCE"
        --triplet-mix-l2-weight "$TRIPLET_MIX_L2_WEIGHT"
        --triplet-mix-cos-weight "$TRIPLET_MIX_COS_WEIGHT"
        --pooling-mode "$POOLING_MODE"
        --pooling-mask-policy "$POOLING_MASK_POLICY"
        --lora-r "$LORA_R"
        --lora-alpha "$LORA_R"
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

    # Always use contrastive pairs
    if [[ -f "$CONTRASTIVE_PAIRS" ]]; then
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

# Show per-variant breakdown
from collections import Counter
variant_results = Counter()
for p in pairs:
    ctx = p.get('sample_context', {})
    if isinstance(ctx, str):
        import json as j
        ctx = j.loads(ctx)
    variant = ctx.get('task_variant', '?')
    outcome = p.get('cb_outcome', '?')
    variant_results[(variant, outcome)] += 1

variants = sorted(set(v for v, _ in variant_results.keys()))
for v in variants:
    atk = variant_results.get((v, 'attack_success'), 0)
    cor = variant_results.get((v, 'correct_behavior'), 0)
    total = atk + cor
    if total > 0:
        print(f'  {v}: {cor}/{total} correct ({cor/total:.0%})')

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

print(f"{'#':>2} {'Run':<30} | {'Fuj ASR':>7} | {'AD Res':>6} | {'Ben Cor':>7} {'Ben NT':>6} | {'LoRA':>5} {'r':>3} {'m_h':>4} {'CB':>12} {'Steps':>5} {'Ups':>3}")
print("-" * 120)
for row in rows:
    r = row["run"]
    if len(r) > 29:
        r = r[:26] + "..."
    c = row.get("cfg", {})
    num = c.get("run_number", "?")
    lora = c.get("lora_target_layers", "?")
    lr = c.get("lora_r", "?")
    mh = c.get("pooling_margin_h", "?")
    cb = c.get("layers", "?")
    steps = c.get("total_steps", "?")
    ups = c.get("upsample_factor", 1)

    print(f"{num:>2} {r:<30} | {fmt(row.get('fuj_asr', -1)):>7} | {fmt(row.get('ad_res', -1)):>6} | {fmt(row.get('ben_cor', -1)):>7} {fmt(row.get('ben_nt', -1)):>6} | {lora:>5} {lr:>3} {mh:>4} {cb:>12} {steps:>5} {ups:>3}")

print()
print("Fuj ASR = Fujitsu attack success rate (lower is better)")
print("AD Res = AgentDojo harmful resistance (higher is better)")
print("Ben Cor = benign correct tool (higher is better)")
print("Ben NT = benign no_tool rate (lower is better)")
print("Ups = upsample factor for Comment-based traces")
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
