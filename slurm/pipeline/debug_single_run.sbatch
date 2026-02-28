#!/bin/bash
#SBATCH --job-name=debug_single_run
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

set -euo pipefail

# =============================================================================
# DEBUG SINGLE RUN
# =============================================================================
# Mirrors sweep_hparams_simple.sbatch but runs ONE configuration only.
# Use this to quickly verify training produces coherent output.
#
# Usage:
#   sbatch slurm/pipeline/debug_single_run.sbatch
#
#   # Override any param:
#   TOTAL_STEPS=50 ALPHA=10.0 sbatch slurm/pipeline/debug_single_run.sbatch
# =============================================================================

PROJECT_DIR="/project/def-zhijing/memoozd"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

if [[ -d "$REPO_DIR" ]]; then
    cd "$REPO_DIR"
    echo "Changed directory to: $(pwd)"
else
    echo "WARNING: Repo directory not found at $REPO_DIR. Assuming current directory is correct."
fi

module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated venv: $VENV_DIR"
else
    echo "WARNING: Venv not found at $VENV_DIR"
fi

# =============================================================================
# Cache Setup (Offline Mode)
# =============================================================================
CB_SCRATCH="/scratch/memoozd/cb-scratch"
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,wandb,triton}

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

# =============================================================================
# Single Run Configuration
# =============================================================================
POOLING_MODE="${POOLING_MODE:-legacy}"
ALPHA="${ALPHA:-10.0}"
LAYERS="${LAYERS:-10,20}"
POLICY="${POLICY:-assistant_only}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"
LIMIT_EVAL="${LIMIT_EVAL:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
DTYPE="${DTYPE:-bfloat16}"
LOSS_MODE="${LOSS_MODE:-triplet_full}"
LOSS_WEIGHTING="${LOSS_WEIGHTING:-dual}"

# Triplet hyperparameters
TRIPLET_ALPHA_BENIGN="${TRIPLET_ALPHA_BENIGN:-0.5}"
TRIPLET_BETA_HARMFUL="${TRIPLET_BETA_HARMFUL:-0.4}"
TRIPLET_GAMMA_KL="${TRIPLET_GAMMA_KL:-0.9}"

# Auto-configure distance and margins for correct pooling mode
# Correct pooling uses cosine-only distance (bounded [0,2], no scale issues)
# with smaller margins calibrated for dense representations
if [[ "$POOLING_MODE" == "correct" ]]; then
    _DEFAULT_DIST="dcos"
    _DEFAULT_MARGIN_B="0.3"
    _DEFAULT_MARGIN_H="0.8"
else
    _DEFAULT_DIST="dmix"
    _DEFAULT_MARGIN_B="500.0"
    _DEFAULT_MARGIN_H="1500.0"
fi

TRIPLET_MARGIN_BENIGN="${TRIPLET_MARGIN_BENIGN:-$_DEFAULT_MARGIN_B}"
TRIPLET_MARGIN_HARMFUL="${TRIPLET_MARGIN_HARMFUL:-$_DEFAULT_MARGIN_H}"
TRIPLET_BENIGN_POS_DISTANCE="${TRIPLET_BENIGN_POS_DISTANCE:-$_DEFAULT_DIST}"
TRIPLET_BENIGN_NEG_DISTANCE="${TRIPLET_BENIGN_NEG_DISTANCE:-$_DEFAULT_DIST}"
TRIPLET_HARMFUL_POS_DISTANCE="${TRIPLET_HARMFUL_POS_DISTANCE:-$_DEFAULT_DIST}"
TRIPLET_HARMFUL_NEG_DISTANCE="${TRIPLET_HARMFUL_NEG_DISTANCE:-$_DEFAULT_DIST}"
TRIPLET_MIX_L2_WEIGHT="${TRIPLET_MIX_L2_WEIGHT:-0.5}"
TRIPLET_MIX_COS_WEIGHT="${TRIPLET_MIX_COS_WEIGHT:-0.5}"

echo "========================================"
echo "DEBUG SINGLE RUN"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Date: $(date)"
echo ""
echo "Configuration:"
echo "  ALPHA: $ALPHA"
echo "  LAYERS: $LAYERS"
echo "  POLICY: $POLICY"
echo "  TOTAL_STEPS: $TOTAL_STEPS"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  GRAD_ACCUM: $GRAD_ACCUM"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  LOSS_MODE: $LOSS_MODE"
echo "  LOSS_WEIGHTING: $LOSS_WEIGHTING"
echo "  POOLING_MODE: $POOLING_MODE"
echo "  DISTANCE: $TRIPLET_BENIGN_POS_DISTANCE"
echo "  TRIPLET_MARGIN_BENIGN: $TRIPLET_MARGIN_BENIGN"
echo "  TRIPLET_MARGIN_HARMFUL: $TRIPLET_MARGIN_HARMFUL"
echo "  TRIPLET_BETA_HARMFUL: $TRIPLET_BETA_HARMFUL"
echo "  TRIPLET_GAMMA_KL: $TRIPLET_GAMMA_KL"
echo "  LIMIT_EVAL: $LIMIT_EVAL"
echo ""

# =============================================================================
# Resolve Model
# =============================================================================
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
PRESET="${PRESET:-llama-3.1-8b-instruct}"

MODEL_CACHE_NAME=$(echo "$MODEL_ID" | sed 's/\//--/g')
MODEL_CACHE_DIR="$HF_HUB_CACHE/models--${MODEL_CACHE_NAME}"
SNAPSHOT_ROOT="$MODEL_CACHE_DIR/snapshots"

if [[ -d "$SNAPSHOT_ROOT" ]]; then
    TOKENIZER=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1)
    if [[ -z "$TOKENIZER" ]]; then
        echo "ERROR: Snapshot directory empty: $SNAPSHOT_ROOT"
        exit 1
    fi
    echo "Model snapshot: $TOKENIZER"
else
    echo "ERROR: Model not cached at: $MODEL_CACHE_DIR"
    exit 1
fi

# =============================================================================
# Input Traces
# =============================================================================
TRACES_DIR="$CB_SCRATCH/data/traces"
DS_TRACES="$TRACES_DIR/fujitsu_b4_ds.jsonl"
DR_TRACES="$TRACES_DIR/fujitsu_b4_dr.jsonl"
AGENTDOJO_TRACES="$TRACES_DIR/agentdojo_complete.jsonl"
AGENTHARM_TRACES="$TRACES_DIR/agentharm_mixed.jsonl"
TOOL_SCHEMA="$REPO_DIR/configs/tool_schemas/b4_standard_v1.json"

echo "Input traces:"
for trace in "$DS_TRACES" "$DR_TRACES"; do
    if [[ -f "$trace" ]]; then
        echo "  OK $(basename "$trace"): $(wc -l < "$trace") lines"
    else
        echo "  MISSING $(basename "$trace")"
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
    elif [[ $(wc -l < "$AGENTDOJO_TRACES") -lt $AGENTDOJO_MIN_LINES ]]; then
        regen=true
        echo "  AgentDojo traces too small ($(wc -l < "$AGENTDOJO_TRACES") < $AGENTDOJO_MIN_LINES). Regenerating..."
    fi
    if [[ "$regen" == "true" ]]; then
        python src/schemas/tools/ETL_A.py \
            --agentdojo "$AGENTDOJO_RAW" \
            --output "$AGENTDOJO_TRACES" \
            --split train
        echo "  OK AgentDojo regenerated: $(wc -l < "$AGENTDOJO_TRACES") traces"
    else
        echo "  OK $(basename "$AGENTDOJO_TRACES"): $(wc -l < "$AGENTDOJO_TRACES") lines"
    fi
else
    echo "  MISSING AgentDojo raw file: $AGENTDOJO_RAW"
    exit 1
fi

HAS_AGENTHARM=false
if [[ -f "$AGENTHARM_TRACES" ]]; then
    HAS_AGENTHARM=true
    echo "  OK $(basename "$AGENTHARM_TRACES"): $(wc -l < "$AGENTHARM_TRACES") lines"
fi

echo ""

# =============================================================================
# Output Directory
# =============================================================================
layers_short="${LAYERS//,/_}"
RUN_NAME="debug_a${ALPHA}_l${layers_short}_${POLICY}"
RUN_DIR="$CB_SCRATCH/debug_runs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"/{renders,lossmasks,model,eval}

echo "Output: $RUN_DIR"
echo ""

# =============================================================================
# Step 1: ETL_B
# =============================================================================
echo "========================================"
echo "STEP 1: ETL_B (policy=$POLICY)"
echo "========================================"

TRACE_FILES=("$DS_TRACES" "$DR_TRACES" "$AGENTDOJO_TRACES")
if [[ "$HAS_AGENTHARM" == "true" ]]; then
    TRACE_FILES+=("$AGENTHARM_TRACES")
fi
for trace_file in "${TRACE_FILES[@]}"; do
    basename=$(basename "$trace_file" .jsonl)
    render_out="$RUN_DIR/renders/${basename}.jsonl"
    lossmask_out="$RUN_DIR/lossmasks/${basename}.jsonl"

    ETL_B_EXTRA_ARGS=()
    if [[ "$basename" == agentharm_* ]]; then
        ETL_B_EXTRA_ARGS+=(--allow-skeleton)
    fi

    echo "  Processing: $basename"
    python src/schemas/tools/ETL_B.py \
        --traces "$trace_file" \
        --render-out "$render_out" \
        --lossmask-out "$lossmask_out" \
        --tokenizer "$TOKENIZER" \
        --max-length 4096 \
        --policy-override "$POLICY" \
        "${ETL_B_EXTRA_ARGS[@]}" 2>&1 | tee -a "$RUN_DIR/etl_b.log"
done

# Verify lossmasks
echo ""
echo "Lossmask verification (first 5 samples each):"
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
    total_lines=$(wc -l < "$lm")
    echo "  $(basename "$lm"): $nonzero non-zero tokens (first 5), $total_lines total samples"
done
echo ""

# =============================================================================
# Step 2: Split AgentDojo
# =============================================================================
echo "========================================"
echo "STEP 2: Split AgentDojo"
echo "========================================"

SPLIT_DIR="$RUN_DIR/agentdojo_split"
mkdir -p "$SPLIT_DIR"

python "$REPO_DIR/scripts/split_agentdojo.py" \
    --traces "$AGENTDOJO_TRACES" \
    --renders "$RUN_DIR/renders/agentdojo_complete.jsonl" \
    --lossmasks "$RUN_DIR/lossmasks/agentdojo_complete.jsonl" \
    --output-dir "$SPLIT_DIR" \
    --legacy-harmful-naming 2>&1 | tee -a "$RUN_DIR/split.log"

AGENTHARM_SPLIT_DIR="$RUN_DIR/agentharm_split"
if [[ "$HAS_AGENTHARM" == "true" ]]; then
    mkdir -p "$AGENTHARM_SPLIT_DIR"
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

# =============================================================================
# Step 3: Training
# =============================================================================
echo "========================================"
echo "STEP 3: Training"
echo "========================================"

IFS=',' read -ra CB_LAYER_ARRAY <<< "$LAYERS"

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

echo "Harmful renders: ${#HARMFUL_RENDER_ARGS[@]} files"
echo "Benign renders:  ${#BENIGN_RENDER_ARGS[@]} files"
echo ""

# Save run configuration for reproducibility and devkit integration
cat > "$RUN_DIR/run_config.json" <<RUNCFG
{
  "preset": "debug",
  "pooling_mode": "$POOLING_MODE",
  "distance": "$TRIPLET_BENIGN_POS_DISTANCE",
  "margin_benign": $TRIPLET_MARGIN_BENIGN,
  "margin_harmful": $TRIPLET_MARGIN_HARMFUL,
  "alpha": $ALPHA,
  "layers": "$LAYERS",
  "policy": "$POLICY",
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
echo "Saved run_config.json"

# Print sample counts per file
echo "Training data sizes:"
for f in "${HARMFUL_RENDER_ARGS[@]}"; do
    if [[ -f "$f" ]]; then
        echo "  harmful: $(wc -l < "$f") $(basename "$f")"
    else
        echo "  harmful: MISSING $(basename "$f")"
    fi
done
for f in "${BENIGN_RENDER_ARGS[@]}"; do
    if [[ -f "$f" ]]; then
        echo "  benign:  $(wc -l < "$f") $(basename "$f")"
    else
        echo "  benign:  MISSING $(basename "$f")"
    fi
done
echo ""

TRAIN_ARGS=(
    --preset "$PRESET"
    --model "$TOKENIZER"
    --output-dir "$RUN_DIR/model"
    --total-steps "$TOTAL_STEPS"
    --batch-size "$BATCH_SIZE"
    --learning-rate "$LEARNING_RATE"
    --warmup-steps "$WARMUP_STEPS"
    --alpha-max "$ALPHA"
    --max-seq-length "$MAX_SEQ_LENGTH"
    --gradient-accumulation-steps "$GRAD_ACCUM"
    --loss-mode "$LOSS_MODE"
    --loss-weighting "$LOSS_WEIGHTING"
    --triplet-alpha-benign "$TRIPLET_ALPHA_BENIGN"
    --triplet-beta-harmful "$TRIPLET_BETA_HARMFUL"
    --triplet-gamma-kl "$TRIPLET_GAMMA_KL"
    --triplet-margin-benign "$TRIPLET_MARGIN_BENIGN"
    --triplet-margin-harmful "$TRIPLET_MARGIN_HARMFUL"
    --triplet-benign-positive-distance "$TRIPLET_BENIGN_POS_DISTANCE"
    --triplet-benign-negative-distance "$TRIPLET_BENIGN_NEG_DISTANCE"
    --triplet-harmful-positive-distance "$TRIPLET_HARMFUL_POS_DISTANCE"
    --triplet-harmful-negative-distance "$TRIPLET_HARMFUL_NEG_DISTANCE"
    --triplet-mix-l2-weight "$TRIPLET_MIX_L2_WEIGHT"
    --triplet-mix-cos-weight "$TRIPLET_MIX_COS_WEIGHT"
    --pooling-mode "$POOLING_MODE"
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

echo "Training command:"
echo "  python src/training/train_schema.py \\"
for arg in "${TRAIN_ARGS[@]}"; do
    echo "    $arg \\"
done
echo ""

train_start=$(date +%s)
if python src/training/train_schema.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$RUN_DIR/train.log"; then
    train_status="SUCCESS"
else
    train_status="FAILED"
fi
train_end=$(date +%s)
echo ""
echo "Training $train_status in $((train_end - train_start))s"
echo ""

if [[ "$train_status" != "SUCCESS" ]]; then
    echo "Training failed. Check $RUN_DIR/train.log"
    exit 1
fi

# =============================================================================
# Step 4a: Checkpoint Evaluation (intermediate checkpoints)
# =============================================================================
echo "========================================"
echo "STEP 4a: Checkpoint Evaluation"
echo "========================================"

for ckpt_path in "$RUN_DIR/model/checkpoint-"*; do
    [[ -d "$ckpt_path" ]] || continue
    ckpt_name=$(basename "$ckpt_path")
    CKPT_EVAL_DIR="$RUN_DIR/eval/checkpoints/$ckpt_name"
    mkdir -p "$CKPT_EVAL_DIR"
    echo "Evaluating $ckpt_name..."

    # Fujitsu checkpoint eval
    python src/evaluation/eval.py \
        --baseline "$TOKENIZER" \
        --cb-adapter "$ckpt_path" \
        --eval-data "$DS_TRACES" \
        --tool-schema "$TOOL_SCHEMA" \
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
            --tool-schema "$TOOL_SCHEMA" \
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
            --tool-schema "$TOOL_SCHEMA" \
            --output "$CKPT_EVAL_DIR/agentdojo_benign_eval.json" \
            --use-sample-context \
            --limit "$LIMIT_EVAL" \
            --merge-adapter \
            --dtype "$DTYPE" 2>&1 | tee -a "$CKPT_EVAL_DIR/agentdojo_benign.log"
    fi

    echo "$ckpt_name eval complete."
    echo ""
done

# =============================================================================
# Step 4b: Final Evaluation
# =============================================================================
echo "========================================"
echo "STEP 4b: Final Evaluation"
echo "========================================"

if [[ -d "$RUN_DIR/model/final" ]]; then
    ADAPTER_PATH="$RUN_DIR/model/final"
elif [[ -f "$RUN_DIR/model/adapter_config.json" ]]; then
    ADAPTER_PATH="$RUN_DIR/model"
else
    echo "WARNING: No adapter found, skipping evaluation"
    exit 1
fi

echo "Adapter: $ADAPTER_PATH"
echo ""

# --- Fujitsu eval ---
echo "Evaluating on Fujitsu ($LIMIT_EVAL samples)..."
FUJITSU_OUTPUT="$RUN_DIR/eval/fujitsu_eval.json"

python src/evaluation/eval.py \
    --baseline "$TOKENIZER" \
    --cb-adapter "$ADAPTER_PATH" \
    --eval-data "$DS_TRACES" \
    --tool-schema "$TOOL_SCHEMA" \
    --output "$FUJITSU_OUTPUT" \
    --limit "$LIMIT_EVAL" \
    --merge-adapter \
    --dtype "$DTYPE" 2>&1 | tee -a "$RUN_DIR/eval/fujitsu.log"

# --- AgentDojo eval ---
if [[ -f "$SPLIT_DIR/agentdojo_traces_harmful.jsonl" ]]; then
    echo ""
    echo "Evaluating on AgentDojo ($LIMIT_EVAL samples)..."
    AGENTDOJO_OUTPUT="$RUN_DIR/eval/agentdojo_eval.json"

    python src/evaluation/eval.py \
        --baseline "$TOKENIZER" \
        --cb-adapter "$ADAPTER_PATH" \
        --eval-data "$SPLIT_DIR/agentdojo_traces_harmful.jsonl" \
        --tool-schema "$TOOL_SCHEMA" \
        --output "$AGENTDOJO_OUTPUT" \
        --use-sample-context \
        --limit "$LIMIT_EVAL" \
        --merge-adapter \
        --dtype "$DTYPE" 2>&1 | tee -a "$RUN_DIR/eval/agentdojo.log"
fi

# --- AgentDojo benign eval (capability retention) ---
if [[ -f "$SPLIT_DIR/agentdojo_traces_benign.jsonl" ]]; then
    echo ""
    echo "Evaluating on AgentDojo benign ($LIMIT_EVAL samples)..."
    AGENTDOJO_BENIGN_OUTPUT="$RUN_DIR/eval/agentdojo_benign_eval.json"

    python src/evaluation/eval.py \
        --baseline "$TOKENIZER" \
        --cb-adapter "$ADAPTER_PATH" \
        --eval-data "$SPLIT_DIR/agentdojo_traces_benign.jsonl" \
        --tool-schema "$TOOL_SCHEMA" \
        --output "$AGENTDOJO_BENIGN_OUTPUT" \
        --use-sample-context \
        --limit "$LIMIT_EVAL" \
        --merge-adapter \
        --dtype "$DTYPE" 2>&1 | tee -a "$RUN_DIR/eval/agentdojo_benign.log"
fi

# =============================================================================
# Step 5: Print Examples (the key debug output — final checkpoint)
# =============================================================================
echo ""
echo "========================================"
echo "EXAMPLE OUTPUTS"
echo "========================================"

# Fujitsu examples
FUJITSU_PAIRED="${FUJITSU_OUTPUT%.json}.paired_outputs.jsonl"
if [[ -f "$FUJITSU_PAIRED" ]]; then
    echo ""
    echo "--- FUJITSU ---"
    python3 - "$FUJITSU_PAIRED" << 'PYEOF'
import json, sys

paired_file = sys.argv[1]
with open(paired_file, 'r') as f:
    pairs = [json.loads(line) for line in f if line.strip()]

if not pairs:
    print("No paired outputs")
    sys.exit(0)

# Classify each CB response
def is_gibberish(resp):
    if not resp:
        return False
    words = resp.split()
    return len(words) > 10 and len(set(words)) / len(words) < 0.15

improvements = [p for p in pairs if p.get('baseline_outcome') == 'attack_success' and p.get('cb_outcome') != 'attack_success']
regressions = [p for p in pairs if p.get('baseline_outcome') != 'attack_success' and p.get('cb_outcome') == 'attack_success']

# Quality breakdown of CB model outputs
n_gibberish = sum(1 for p in pairs if is_gibberish(p.get('cb_response', '')))
n_correct = sum(1 for p in pairs if p.get('cb_outcome') == 'correct_behavior')
n_no_tool = sum(1 for p in pairs if p.get('cb_outcome') == 'no_tool_call')
n_attack = sum(1 for p in pairs if p.get('cb_outcome') == 'attack_success')
n_other = sum(1 for p in pairs if p.get('cb_outcome') == 'other_tool')
n_malformed = sum(1 for p in pairs if p.get('cb_outcome') == 'malformed_tool_call')

print(f"Pairs: {len(pairs)} | Improved: {len(improvements)} | Regressed: {len(regressions)}")
print(f"  CB outcomes: correct={n_correct} | no_tool={n_no_tool} | attack={n_attack} | other={n_other} | malformed={n_malformed}")
if n_gibberish > 0:
    print(f"  WARNING: {n_gibberish}/{len(pairs)} gibberish ({100*n_gibberish/len(pairs):.0f}%)")
else:
    print(f"  Gibberish: NONE")

print()
for i, p in enumerate((improvements + regressions)[:10], 1):
    b_out = p.get('baseline_outcome', '?')
    c_out = p.get('cb_outcome', '?')
    label = 'IMP' if b_out == 'attack_success' else 'REG'
    gib = ' [GIBBERISH]' if is_gibberish(p.get('cb_response', '')) else ''
    print(f"[{i}] {label}: {b_out} -> {c_out}{gib}")
    print(f"    expected={p.get('expected_tool','?')} simulated={p.get('simulated_tool','?')}")
    print(f"--- BASELINE ---")
    resp = p.get('baseline_response', '(empty)')
    print(resp[:300])
    print(f"--- CB MODEL ---")
    resp = p.get('cb_response', '(empty)')
    print(resp[:300])
    print()
PYEOF
fi

# AgentDojo examples
AGENTDOJO_PAIRED="${AGENTDOJO_OUTPUT:-}.paired_outputs.jsonl"
AGENTDOJO_PAIRED="${AGENTDOJO_PAIRED%.json.paired_outputs.jsonl}.paired_outputs.jsonl"
if [[ -n "${AGENTDOJO_OUTPUT:-}" ]]; then
    AGENTDOJO_PAIRED="${AGENTDOJO_OUTPUT%.json}.paired_outputs.jsonl"
fi
if [[ -f "$AGENTDOJO_PAIRED" ]]; then
    echo ""
    echo "--- AGENTDOJO ---"
    python3 - "$AGENTDOJO_PAIRED" "${AGENTDOJO_OUTPUT:-}" << 'PYEOF'
import json, sys

paired_file = sys.argv[1]
eval_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

with open(paired_file, 'r') as f:
    pairs = [json.loads(line) for line in f if line.strip()]

if not pairs:
    print("No paired outputs")
    sys.exit(0)

def is_gibberish(resp):
    if not resp:
        return False
    words = resp.split()
    return len(words) > 10 and len(set(words)) / len(words) < 0.15

differ = [p for p in pairs if p.get('responses_differ')]
same = [p for p in pairs if not p.get('responses_differ')]
n_gibberish = sum(1 for p in pairs if is_gibberish(p.get('cb_response', '')))

# Classify CB responses
harmful_pairs = [p for p in pairs if p.get('category') == 'harmful']
benign_pairs = [p for p in pairs if p.get('category') in ('benign', 'resisted')]
n_harmful_still_tool = sum(1 for p in harmful_pairs if p.get('cb_observed_tool'))
n_harmful_refused = sum(1 for p in harmful_pairs if not p.get('cb_observed_tool'))
n_benign_correct = sum(1 for p in benign_pairs if p.get('cb_observed_tool'))
n_benign_broken = sum(1 for p in benign_pairs if not p.get('cb_observed_tool'))

print(f"Pairs: {len(pairs)} | Different: {len(differ)} ({100*len(differ)/len(pairs):.0f}%) | Same: {len(same)}")
print(f"  Harmful ({len(harmful_pairs)}): refused={n_harmful_refused} | still_called_tool={n_harmful_still_tool}")
if benign_pairs:
    print(f"  Benign ({len(benign_pairs)}): correct_tool={n_benign_correct} | broken={n_benign_broken}")
if n_gibberish > 0:
    print(f"  WARNING: {n_gibberish}/{len(pairs)} gibberish ({100*n_gibberish/len(pairs):.0f}%)")
else:
    print(f"  Gibberish: NONE")

# Surface generation_comparison metrics from eval JSON
if eval_file:
    try:
        with open(eval_file) as f:
            ev = json.load(f)
        for key in ('baseline', 'cb_model'):
            gc = ev.get(key, {}).get('generation_comparison', {})
            if gc:
                label = 'Baseline' if key == 'baseline' else 'CB Model'
                mal_rate = gc.get('malicious_tool_call_rate', 0)
                res_rate = gc.get('harmful_resistance_rate', 0)
                cor_rate = gc.get('correct_tool_call_rate', 0)
                no_rate = gc.get('no_tool_call_rate', 0)
                print(f"  {label}: malicious={mal_rate:.0%} resist={res_rate:.0%} correct_tool={cor_rate:.0%} no_tool={no_rate:.0%}")
    except Exception:
        pass

print()
for i, p in enumerate(differ[:10], 1):
    cat = p.get('category', '?')
    b_tool = p.get('baseline_observed_tool') or 'none'
    c_tool = p.get('cb_observed_tool') or 'none'
    gib = ' [GIBBERISH]' if is_gibberish(p.get('cb_response', '')) else ''
    print(f"[{i}] {cat}: baseline={b_tool} -> cb={c_tool}{gib}")
    print(f"--- BASELINE ---")
    resp = p.get('baseline_response', '(empty)')
    print(resp[:300])
    print(f"--- CB MODEL ---")
    resp = p.get('cb_response', '(empty)')
    print(resp[:300])
    print()
PYEOF
fi

# AgentDojo benign examples (capability retention)
AGENTDOJO_BENIGN_PAIRED="${AGENTDOJO_BENIGN_OUTPUT:-}.paired_outputs.jsonl"
if [[ -n "${AGENTDOJO_BENIGN_OUTPUT:-}" ]]; then
    AGENTDOJO_BENIGN_PAIRED="${AGENTDOJO_BENIGN_OUTPUT%.json}.paired_outputs.jsonl"
fi
if [[ -f "$AGENTDOJO_BENIGN_PAIRED" ]]; then
    echo ""
    echo "--- AGENTDOJO BENIGN (capability retention) ---"
    python3 - "$AGENTDOJO_BENIGN_PAIRED" "${AGENTDOJO_BENIGN_OUTPUT:-}" << 'PYEOF'
import json, sys

paired_file = sys.argv[1]
eval_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

with open(paired_file, 'r') as f:
    pairs = [json.loads(line) for line in f if line.strip()]

if not pairs:
    print("No paired outputs")
    sys.exit(0)

def is_gibberish(resp):
    if not resp: return False
    words = resp.split()
    return len(words) > 10 and len(set(words)) / len(words) < 0.15

n_correct = sum(1 for p in pairs if p.get('cb_observed_tool'))
n_broken = sum(1 for p in pairs if not p.get('cb_observed_tool'))
n_gibberish = sum(1 for p in pairs if is_gibberish(p.get('cb_response', '')))

print(f"Benign pairs: {len(pairs)} | correct_tool={n_correct} | broken={n_broken} | gibberish={n_gibberish}")

# Show generation_comparison metrics from benign eval
if eval_file:
    try:
        with open(eval_file) as f:
            ev = json.load(f)
        for key in ('baseline', 'cb_model'):
            gc = ev.get(key, {}).get('generation_comparison', {})
            if gc:
                label = 'Baseline' if key == 'baseline' else 'CB Model'
                cor_rate = gc.get('correct_tool_call_rate', 0)
                no_rate = gc.get('no_tool_call_rate', 0)
                print(f"  {label}: correct_tool={cor_rate:.0%} no_tool={no_rate:.0%}")
    except Exception:
        pass

# Show examples where CB broke a benign tool call
broken = [p for p in pairs if p.get('baseline_observed_tool') and not p.get('cb_observed_tool')]
print()
for i, p in enumerate(broken[:5], 1):
    b_tool = p.get('baseline_observed_tool') or 'none'
    c_tool = p.get('cb_observed_tool') or 'none'
    gib = ' [GIBBERISH]' if is_gibberish(p.get('cb_response', '')) else ''
    print(f"[{i}] BROKEN: baseline={b_tool} -> cb={c_tool}{gib}")
    print(f"--- BASELINE ---")
    print((p.get('baseline_response') or '(empty)')[:300])
    print(f"--- CB MODEL ---")
    print((p.get('cb_response') or '(empty)')[:300])
    print()
PYEOF
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "DEBUG RUN COMPLETE"
echo "========================================"
echo ""
echo "Output: $RUN_DIR"
echo ""

# Print metrics summary
if [[ -f "$FUJITSU_OUTPUT" ]]; then
    python3 -c "
import json
with open('$FUJITSU_OUTPUT') as f:
    r = json.load(f)
if 'baseline' in r:
    b = r['baseline']['tool_flip_asr']
    print(f'Fujitsu Baseline ASR: {b[\"attack_success_rate\"]:.1%}')
if 'cb_model' in r:
    c = r['cb_model']['tool_flip_asr']
    print(f'Fujitsu CB ASR:       {c[\"attack_success_rate\"]:.1%}')
    print(f'Fujitsu CB No-Tool:   {c.get(\"no_tool_call_rate\", 0):.1%}')
if 'delta' in r:
    print(f'Fujitsu Delta:        {r[\"delta\"][\"tool_flip_asr\"]:+.1%}')
" 2>/dev/null || echo "(parse error)"
fi

if [[ -f "${AGENTDOJO_OUTPUT:-/dev/null}" ]]; then
    python3 -c "
import json
with open('${AGENTDOJO_OUTPUT}') as f:
    r = json.load(f)
if 'output_comparison' in r:
    print(f'AgentDojo Diff Rate:  {r[\"output_comparison\"][\"difference_rate\"]:.1%}')
for key in ('baseline', 'cb_model'):
    gc = r.get(key, {}).get('generation_comparison', {})
    if gc:
        label = 'Baseline' if key == 'baseline' else 'CB Model'
        print(f'AgentDojo {label}: malicious={gc.get(\"malicious_tool_call_rate\",0):.1%} resist={gc.get(\"harmful_resistance_rate\",0):.1%} correct_tool={gc.get(\"correct_tool_call_rate\",0):.1%} no_tool={gc.get(\"no_tool_call_rate\",0):.1%}')
" 2>/dev/null || echo "(parse error)"
fi

if [[ -f "${AGENTDOJO_BENIGN_OUTPUT:-/dev/null}" ]]; then
    python3 -c "
import json
with open('${AGENTDOJO_BENIGN_OUTPUT}') as f:
    r = json.load(f)
for key in ('baseline', 'cb_model'):
    gc = r.get(key, {}).get('generation_comparison', {})
    if gc:
        label = 'Baseline' if key == 'baseline' else 'CB Model'
        cor = gc.get('correct_tool_call_rate', 0)
        no = gc.get('no_tool_call_rate', 0)
        print(f'AgentDojo Benign {label}: correct_tool={cor:.1%} no_tool={no:.1%}')
" 2>/dev/null || echo "(parse error)"
fi

echo ""
echo "Finished at: $(date)"
