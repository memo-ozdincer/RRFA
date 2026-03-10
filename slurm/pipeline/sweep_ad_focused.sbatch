#!/bin/bash
#SBATCH --job-name=sweep_ad
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# AD-Focused Training Experiments (Sweep 9 + 10 + 11)
# =============================================================================
#
# Sweep 9: First AD signal (best: -5% trunc). All configs suffered from:
#   1. alpha_decay_multiplier=1.0 — harmful push drops to 0 at training end
#   2. margin_harmful=1.6 — saturates in ~10-50 steps, zero gradient thereafter
#   3. KL constant while harmful push decays — actively fights the push
#   4. LR decays to 0 on top of alpha decay — quadratic effective-learning decay
#
# Sweep 10 (v2 configs): Fix all four. Same defaults otherwise.
#   - alpha_decay_multiplier=2.0 (push stays at 50% at step=total_steps)
#   - margin_harmful=20.0 (keeps gradient flowing through full training)
#   - KL scheduled with retain coefficient (0→gamma_kl, not constant)
#   - lr_schedule=constant (holds LR after warmup)
#
# Configs:
#   CONFIG=ad_300_v2        300 steps (~1 epoch), all four fixes
#   CONFIG=ad_500_v2        500 steps (~1.6 epochs), all four fixes
#   CONFIG=ad_750_v2        750 steps (~2.4 epochs), all four fixes
#   CONFIG=ad_1000_v2       1000 steps (~3.2 epochs), all four fixes
#   CONFIG=ad_1500_v2       1500 steps (~4.8 epochs), all four fixes
#   CONFIG=ad_3000_v2       3000 steps (~9.6 epochs), all four fixes — OVERTRAINED
#   CONFIG=ad_5000_v2       5000 steps, all four fixes — expect worse
#   (Legacy sweep 9 configs still available: ad_500, ad_1000, etc.)
#
# Sweep 11b — balanced (Fujitsu + AD):
#   CONFIG=ad_500_bal       500 steps, Fujitsu+AD, cap=2500
#   CONFIG=ad_750_bal       750 steps, Fujitsu+AD, cap=2500
#   CONFIG=ad_1000_bal      1000 steps, Fujitsu+AD, cap=2500
#
# Sweep 11c — balanced + refusal:
#   CONFIG=ad_750_bal_r     750 steps, balanced + refusal data
#
# Sweep 11d — alpha/margin tuning:
#   CONFIG=ad_1000_a5       1000 steps, alpha_max=5.0
#   CONFIG=ad_1000_m40      1000 steps, margin_harmful=40.0
#
# Sweep 11e — amplified injections:
#   CONFIG=ad_750_amp       750 steps, AD only, amplified injections
#   CONFIG=ad_750_bal_amp   750 steps, balanced, amplified injections
#
# Sweep 11f — SRMU (importance-masked CB loss):
#   CONFIG=ad_750_v2_srmu   750 steps, AD only, probe importance mask
#   CONFIG=ad_1000_v2_srmu  1000 steps, AD only, probe importance mask
#   CONFIG=ad_1500_v2_srmu  1500 steps, AD only, probe importance mask
#   CONFIG=ad_1000_bal_srmu 1000 steps, balanced, probe importance mask
#
# Usage:
#   # AD-only step sweep
#   for c in ad_300_v2 ad_500_v2 ad_750_v2 ad_1000_v2 ad_1500_v2; do
#     sbatch --export=ALL,CONFIG=$c slurm/pipeline/sweep_ad_focused.sbatch
#   done
#
#   # Balanced (Fujitsu + AD)
#   for c in ad_500_bal ad_750_bal ad_1000_bal; do
#     sbatch --export=ALL,CONFIG=$c slurm/pipeline/sweep_ad_focused.sbatch
#   done
#
#   # Refusal + tuning + amplification
#   for c in ad_750_bal_r ad_1000_a5 ad_1000_m40 ad_750_amp; do
#     sbatch --export=ALL,CONFIG=$c slurm/pipeline/sweep_ad_focused.sbatch
#   done
#
#   # SRMU (probe extraction runs automatically if mask not cached)
#   for c in ad_750_v2_srmu ad_1000_v2_srmu ad_1500_v2_srmu ad_1000_bal_srmu; do
#     sbatch --export=ALL,CONFIG=$c slurm/pipeline/sweep_ad_focused.sbatch
#   done
#
# Quick test:
#   sbatch --export=ALL,CONFIG=ad_1000_v2,QUICK_TEST=true slurm/pipeline/sweep_ad_focused.sbatch
# =============================================================================

set -euo pipefail

# ── Config presets ──────────────────────────────────────────────────────────

CONFIG="${CONFIG:?ERROR: CONFIG is required. See case block below.}"

# Per-config override vars (empty = use defaults below)
_LR="" _ALPHA="" _KL="" _CB="" _LORA_R="" _LORA_ALPHA=""
_MARGIN_H="" _MARGIN_B="" _DECAY_MULT="" _LR_SCHED="" _USE_REFUSAL=""
_AMPLIFY="" _AMP_REPEATS=""
_IMPORTANCE_MASK=""

case "$CONFIG" in
  # --- Legacy sweep 9 configs (kept for reproducibility) ---
  ad_500)
    DATASET=ad_only;  TOTAL_STEPS=500;  RESISTED=retain;  FUJITSU_CAP=0    ;;
  ad_1000)
    DATASET=ad_only;  TOTAL_STEPS=1000; RESISTED=retain;  FUJITSU_CAP=0    ;;
  ad_no_resisted)
    DATASET=ad_only;  TOTAL_STEPS=500;  RESISTED=exclude; FUJITSU_CAP=0    ;;
  balanced_500)
    DATASET=balanced; TOTAL_STEPS=500;  RESISTED=retain;  FUJITSU_CAP=2500 ;;
  balanced_1000)
    DATASET=balanced; TOTAL_STEPS=1000; RESISTED=retain;  FUJITSU_CAP=2500 ;;
  ad_2000_aggr)
    DATASET=ad_only;  TOTAL_STEPS=2000; RESISTED=retain;  FUJITSU_CAP=0
    _LR=2e-4; _ALPHA=50.0; _KL=0.1 ;;
  ad_500_wide)
    DATASET=ad_only;  TOTAL_STEPS=500;  RESISTED=retain;  FUJITSU_CAP=0
    _CB="5 10 15 20 25"; _LORA_R=64; _LORA_ALPHA=128 ;;
  ad_1000_nokl)
    DATASET=ad_only;  TOTAL_STEPS=1000; RESISTED=retain;  FUJITSU_CAP=0
    _KL=0.0; _ALPHA=30.0 ;;

  # ==========================================================================
  # Sweep 10 — schedule & margin fixes (all four inefficiencies addressed)
  # ==========================================================================
  # Fixes applied to ALL v2 configs:
  #   1. alpha_decay_multiplier=2.0 — push stays active through full training
  #      (at step=total_steps, cs is still 50% of alpha_max, not 0)
  #   2. margin_harmful=20.0 — prevents early saturation (was 1.6, saturated
  #      in ~10-50 steps leaving no gradient for the rest of training)
  #   3. KL scheduled with retain coeff — no longer fights harmful push
  #      (code change in trainer.py: gamma_kl *= cr/alpha_max)
  #   4. lr_schedule=constant — LR holds after warmup instead of decaying to 0
  #      (eliminates quadratic effective-learning decay from LR*cs both → 0)
  #
  # Only difference between v2 configs: step count (epoch coverage).
  # AD harmful ≈ 2500 samples. Effective batch = 8.
  #   1000 steps = 8000 samples ≈ 3.2 epochs
  #   3000 steps = 24000 samples ≈ 9.6 epochs
  #   5000 steps = 40000 samples ≈ 16 epochs

  ad_300_v2)
    DATASET=ad_only;  TOTAL_STEPS=300;  RESISTED=retain;  FUJITSU_CAP=0
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant ;;

  ad_500_v2)
    DATASET=ad_only;  TOTAL_STEPS=500;  RESISTED=retain;  FUJITSU_CAP=0
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant ;;

  ad_750_v2)
    DATASET=ad_only;  TOTAL_STEPS=750;  RESISTED=retain;  FUJITSU_CAP=0
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant ;;

  ad_1000_v2)
    DATASET=ad_only;  TOTAL_STEPS=1000; RESISTED=retain;  FUJITSU_CAP=0
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant ;;

  ad_1500_v2)
    DATASET=ad_only;  TOTAL_STEPS=1500; RESISTED=retain;  FUJITSU_CAP=0
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant ;;

  ad_3000_v2)
    DATASET=ad_only;  TOTAL_STEPS=3000; RESISTED=retain;  FUJITSU_CAP=0
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant ;;

  ad_5000_v2)
    DATASET=ad_only;  TOTAL_STEPS=5000; RESISTED=retain;  FUJITSU_CAP=0
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant ;;

  # ==========================================================================
  # Sweep 10r — v2 fixes + synthetic refusal data in benign set
  # ==========================================================================
  # Same 4 fixes as v2, plus 1000 synthetic refusal traces added to benign.
  # These are harmful contexts where the last assistant message is replaced
  # with a refusal. Gives the retain loss a directed target: "when you see
  # injection, produce refusal-like representations."

  ad_1000_v2r)
    DATASET=ad_only;  TOTAL_STEPS=1000; RESISTED=retain;  FUJITSU_CAP=0
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant; _USE_REFUSAL=true ;;

  ad_3000_v2r)
    DATASET=ad_only;  TOTAL_STEPS=3000; RESISTED=retain;  FUJITSU_CAP=0
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant; _USE_REFUSAL=true ;;

  # ==========================================================================
  # Sweep 11b — balanced (Fujitsu + AD) with v2 fixes
  # ==========================================================================
  # Re-adds Fujitsu data alongside AD. Hypothesis: Fujitsu's structural
  # injection pattern (HTML comments → clean tool redirects) provides a
  # clearer CB signal that transfers to AD, improving selectivity.
  # FUJITSU_CAP=2500 gives ~2500 harmful + ~2500 benign Fujitsu traces,
  # balanced against ~2500 AD harmful + ~3100 AD benign.

  ad_500_bal)
    DATASET=balanced; TOTAL_STEPS=500;  RESISTED=retain;  FUJITSU_CAP=2500
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant ;;

  ad_750_bal)
    DATASET=balanced; TOTAL_STEPS=750;  RESISTED=retain;  FUJITSU_CAP=2500
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant ;;

  ad_1000_bal)
    DATASET=balanced; TOTAL_STEPS=1000; RESISTED=retain;  FUJITSU_CAP=2500
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant ;;

  # ==========================================================================
  # Sweep 11c — balanced + refusal data
  # ==========================================================================
  # Combines balanced (Fujitsu + AD) with 1225 synthetic refusal traces in
  # benign set. Gives CB a directed refusal target instead of just "push away."
  # Hypothesis: refusal traces reduce gibberish and improve benign preservation.

  ad_750_bal_r)
    DATASET=balanced; TOTAL_STEPS=750;  RESISTED=retain;  FUJITSU_CAP=2500
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant; _USE_REFUSAL=true ;;

  # ==========================================================================
  # Sweep 11d — alpha/margin tuning
  # ==========================================================================
  # Test whether lower alpha or higher margin improves Pareto tradeoff.
  # ad_1000_a5: α_max=5.0 (vs default 10.0). Slower Phase 1 push, reduced
  #   Phase 3 destructiveness. May allow more selective weight updates.
  # ad_1000_m40: margin_h=40.0 (vs default 20.0). Higher ceiling should delay
  #   Phase 3 collapse (harmful reps take longer to drift back within margin).

  ad_1000_a5)
    DATASET=ad_only;  TOTAL_STEPS=1000; RESISTED=retain;  FUJITSU_CAP=0
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant; _ALPHA=5.0 ;;

  ad_1000_m40)
    DATASET=ad_only;  TOTAL_STEPS=1000; RESISTED=retain;  FUJITSU_CAP=0
    _DECAY_MULT=2.0; _MARGIN_H=40.0; _MARGIN_B=5.0; _LR_SCHED=constant ;;

  # ==========================================================================
  # Sweep 11e — amplified injections
  # ==========================================================================
  # Injection amplification: repeat <INFORMATION>...</INFORMATION> tags 2x in
  # harmful traces to strengthen the context-level signal CB uses for selectivity
  # (probe AUC 0.83 → aiming for 0.90+).

  ad_750_amp)
    DATASET=ad_only;  TOTAL_STEPS=750;  RESISTED=retain;  FUJITSU_CAP=0
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant; _AMPLIFY=true ;;

  ad_750_bal_amp)
    DATASET=balanced; TOTAL_STEPS=750;  RESISTED=retain;  FUJITSU_CAP=2500
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant; _AMPLIFY=true ;;

  # ==========================================================================
  # Sweep 11f — SRMU: importance-masked CB loss (feature-selective rerouting)
  # ==========================================================================
  # Uses a probe-derived importance mask to push only the hidden dimensions
  # that discriminate harmful from benign, leaving other dimensions untouched.
  # This directly addresses the selectivity problem (AUC 0.83) by focusing
  # the push on features that matter and preserving generative capability.
  #
  # PREREQUISITE: Run probe extraction first (see Step 0b in pipeline below):
  #   python scripts/extract_probe_direction.py \
  #       --traces <truncated.jsonl> --model <model> --layer 10 \
  #       --output configs/probe_directions/layer10_last_token.pt
  #
  # The mask .pt file must exist at IMPORTANCE_MASK_PATH before training.

  ad_750_v2_srmu)
    DATASET=ad_only;  TOTAL_STEPS=750;  RESISTED=retain;  FUJITSU_CAP=0
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant
    _IMPORTANCE_MASK=auto ;;

  ad_1000_v2_srmu)
    DATASET=ad_only;  TOTAL_STEPS=1000; RESISTED=retain;  FUJITSU_CAP=0
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant
    _IMPORTANCE_MASK=auto ;;

  ad_1500_v2_srmu)
    DATASET=ad_only;  TOTAL_STEPS=1500; RESISTED=retain;  FUJITSU_CAP=0
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant
    _IMPORTANCE_MASK=auto ;;

  ad_1000_bal_srmu)
    DATASET=balanced; TOTAL_STEPS=1000; RESISTED=retain;  FUJITSU_CAP=2500
    _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant
    _IMPORTANCE_MASK=auto ;;

  *)
    echo "ERROR: Unknown CONFIG=$CONFIG"
    echo "Valid: ad_300_v2 ad_500_v2 ad_750_v2 ad_1000_v2 ad_1500_v2 ad_3000_v2"
    echo "       ad_500_bal ad_750_bal ad_1000_bal ad_750_bal_r"
    echo "       ad_1000_a5 ad_1000_m40 ad_750_amp ad_750_bal_amp"
    echo "       ad_750_v2_srmu ad_1000_v2_srmu ad_1500_v2_srmu ad_1000_bal_srmu"
    echo "       ad_1000_v2r ad_3000_v2r"
    exit 1 ;;
esac

# Allow env overrides for individual params
TOTAL_STEPS="${TOTAL_STEPS}"
RESISTED="${RESISTED}"
FUJITSU_CAP="${FUJITSU_CAP}"

# ── Environment ─────────────────────────────────────────────────────────────

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

# Cache (identical to sweep_completed_v1)
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

# ── Hyperparameters (locked — only config that showed signal in Sweep 8) ────

QUICK_TEST="${QUICK_TEST:-false}"
if [[ "$QUICK_TEST" == "true" ]]; then
    TOTAL_STEPS=20; LIMIT_EVAL=5; NUM_EXAMPLES=2
    echo "*** QUICK TEST MODE ***"
else
    LIMIT_EVAL=100; NUM_EXAMPLES=5
fi

LOSS_MODE=per_token_cb
DISTANCE=dl2rc
MARGIN_FREE=false
BATCH_SIZE=4
GRAD_ACCUM=2
LEARNING_RATE=5e-5
WARMUP_STEPS=20
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
LORA_R=16
LORA_ALPHA=32
CB_LAYERS="10 20"
POLICY=cb_full_sequence
POOLING_POLICY=cb_full_sequence
EVAL_MODE=next_tool_prediction
PRESET=llama-3.1-8b-instruct
ALPHA_MAX=10.0
GAMMA_KL=0.3
MARGIN_HARMFUL=1.6
MARGIN_BENIGN=1.2
ALPHA_DECAY_MULT=""
LR_SCHEDULE=""

# Apply per-config overrides (set in case block above)
[[ -n "$_LR" ]]         && LEARNING_RATE="$_LR"
[[ -n "$_ALPHA" ]]      && ALPHA_MAX="$_ALPHA"
[[ -n "$_KL" ]]         && GAMMA_KL="$_KL"
[[ -n "$_CB" ]]         && CB_LAYERS="$_CB"
[[ -n "$_LORA_R" ]]     && LORA_R="$_LORA_R"
[[ -n "$_LORA_ALPHA" ]] && LORA_ALPHA="$_LORA_ALPHA"
[[ -n "$_MARGIN_H" ]]   && MARGIN_HARMFUL="$_MARGIN_H"
[[ -n "$_MARGIN_B" ]]   && MARGIN_BENIGN="$_MARGIN_B"
[[ -n "$_DECAY_MULT" ]] && ALPHA_DECAY_MULT="$_DECAY_MULT"
[[ -n "$_LR_SCHED" ]]   && LR_SCHEDULE="$_LR_SCHED"
USE_REFUSAL="${_USE_REFUSAL:-false}"
AMPLIFY_INJECTIONS="${_AMPLIFY:-false}"
AMPLIFY_REPEATS="${_AMP_REPEATS:-2}"

# Importance mask for SRMU-style feature-selective rerouting
# "auto" = use default probe direction files on scratch (writable)
# explicit path = use that specific .pt file
# empty = no mask (standard CB loss)
IMPORTANCE_MASK_PATH=""
if [[ "$_IMPORTANCE_MASK" == "auto" ]]; then
    PROBE_DIR="$CB_SCRATCH/probe_directions"
    IMPORTANCE_MASK_PATH="$PROBE_DIR/layer10_last_token.pt"
elif [[ -n "$_IMPORTANCE_MASK" ]]; then
    IMPORTANCE_MASK_PATH="$_IMPORTANCE_MASK"
fi

echo "========================================================"
echo "SWEEP 11: AD-Focused Step Sweep — $CONFIG"
echo "========================================================"
echo "Job: ${SLURM_JOB_ID:-local} | $(date)"
echo "Dataset: $DATASET | Steps: $TOTAL_STEPS | Resisted: $RESISTED"
echo "Fujitsu cap: $FUJITSU_CAP | Seq len: $MAX_SEQ_LENGTH"
echo "Loss: $LOSS_MODE | Distance: $DISTANCE | margin_free: $MARGIN_FREE"
echo "Margins: harmful=$MARGIN_HARMFUL benign=$MARGIN_BENIGN"
echo "LR: $LEARNING_RATE | LR sched: ${LR_SCHEDULE:-linear} | alpha: $ALPHA_MAX | gamma_kl: $GAMMA_KL"
echo "Alpha decay mult: ${ALPHA_DECAY_MULT:-1.0 (default)} | Refusal data: $USE_REFUSAL"
echo "Amplify injections: $AMPLIFY_INJECTIONS (repeats=$AMPLIFY_REPEATS)"
echo "Importance mask: ${IMPORTANCE_MASK_PATH:-none}"
echo "CB layers: $CB_LAYERS | LoRA: r=$LORA_R alpha=$LORA_ALPHA"
echo ""

# ── Data size estimates ─────────────────────────────────────────────────────

_eff_batch=$((BATCH_SIZE * GRAD_ACCUM))
if [[ "$DATASET" == "ad_only" ]]; then
    _harmful_est=2500
else
    _harmful_est=$((2500 + FUJITSU_CAP))
fi
_samples=$((TOTAL_STEPS * _eff_batch))
_epochs_x10=$(( (_samples * 10) / _harmful_est ))
echo "  Estimated: ${_samples} samples, ~${_epochs_x10%?}.${_epochs_x10: -1} epochs of harmful data"
echo ""

# ── Resolve tokenizer ──────────────────────────────────────────────────────

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_CACHE_NAME=$(echo "$MODEL_ID" | sed 's/\//--/g')
SNAPSHOT_ROOT="$CACHE_DIR/hf/hub/models--${MODEL_CACHE_NAME}/snapshots"
TOKENIZER=$(ls -1td "$SNAPSHOT_ROOT"/* 2>/dev/null | head -n 1)
if [[ -z "$TOKENIZER" ]]; then
    echo "ERROR: Model not cached. Run slurm/cache_models.sh first."; exit 1
fi
echo "Tokenizer: $TOKENIZER"

# ── Input data ──────────────────────────────────────────────────────────────

TRACES_DIR="$CB_SCRATCH/data/traces"
FUJITSU_DS="$TRACES_DIR/fujitsu_b4_ds.jsonl"
FUJITSU_DR="$TRACES_DIR/fujitsu_b4_dr.jsonl"
AGENTDOJO_RAW="$TRACES_DIR/agentdojo_augmented.jsonl"
PAIRS_SRC="$TRACES_DIR/contrastive_pairs.jsonl"
TOOL_SCHEMA="$REPO_DIR/configs/tool_schemas/b4_standard_v1.json"
AD_TOOL_SCHEMA="$REPO_DIR/configs/tool_schemas/agentdojo_v1.json"

# Copy from repo if not on scratch
for src_name in agentdojo_augmented.jsonl; do
    if [[ ! -f "$TRACES_DIR/$src_name" ]] && [[ -f "$REPO_DIR/data/traces/$src_name" ]]; then
        echo "  Copying $src_name to scratch..."
        cp "$REPO_DIR/data/traces/$src_name" "$TRACES_DIR/$src_name"
    fi
done

# Generate contrastive pairs if missing
if [[ ! -f "$PAIRS_SRC" ]]; then
    echo "  Generating contrastive pairs..."
    python "$REPO_DIR/scripts/augment_agentdojo.py" \
        --traces-input "$AGENTDOJO_RAW" \
        --operations removal,contrastive \
        --contrastive-output "$PAIRS_SRC" \
        --output "$CB_SCRATCH/data/traces/_augment_temp.jsonl" 2>&1 | tail -5
    rm -f "$CB_SCRATCH/data/traces/_augment_temp.jsonl"
fi

# Verify required files exist
_required_files=("$AGENTDOJO_RAW" "$PAIRS_SRC")
if [[ "$DATASET" == "balanced" ]]; then
    _required_files+=("$FUJITSU_DS" "$FUJITSU_DR")
fi
for f in "${_required_files[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing $(basename "$f")"; exit 1
    fi
    echo "  $(basename "$f"): $(wc -l < "$f") lines"
done
echo ""

# ── Output directories ──────────────────────────────────────────────────────

# Shared ETL_B outputs (per-policy AND per-seq-length — ETL_B truncates to max_length)
# _ts suffix = tool-schema-aware rendering (invalidates old cache without tool defs)
SHARED_DATA="$CB_SCRATCH/sweeps/s9_shared_${POLICY}_seq${MAX_SEQ_LENGTH}_ts"
# Split outputs depend on resisted handling
SPLIT_DIR="$SHARED_DATA/split_agentdojo_${RESISTED}"

SWEEP_ID="s9_${CONFIG}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$CB_SCRATCH/sweeps/$SWEEP_ID/run"
mkdir -p "$RUN_DIR"/{model,eval}
mkdir -p "$SHARED_DATA"/{renders,lossmasks}
mkdir -p "$SPLIT_DIR"

echo "Shared data: $SHARED_DATA"
echo "Split dir:   $SPLIT_DIR"
echo "Run output:  $RUN_DIR"
echo ""

# =============================================================================
# Step 0: Data preparation
# =============================================================================
echo "========================================================"
echo "Step 0: Data preparation"
echo "========================================================"
gen_start=$(date +%s)

AGENTDOJO_SRC="$SHARED_DATA/agentdojo_truncated.jsonl"

if [[ -f "$AGENTDOJO_SRC" ]] && [[ -s "$AGENTDOJO_SRC" ]]; then
    echo "  AgentDojo truncated: REUSING existing ($(wc -l < "$AGENTDOJO_SRC") lines)"
else
    echo "  Truncating AgentDojo to injection window..."
    python "$REPO_DIR/scripts/truncate_to_injection_window.py" \
        --input "$AGENTDOJO_RAW" \
        --output "$AGENTDOJO_SRC" 2>&1 | tail -5
fi

for f in "$AGENTDOJO_SRC"; do
    if [[ ! -f "$f" ]] || [[ ! -s "$f" ]]; then
        echo "ERROR: Data generation failed for $(basename "$f")"; exit 1
    fi
    echo "  $(basename "$f"): $(wc -l < "$f") lines"
done

# Optional: amplify injections in harmful traces (repeat injection text Nx)
if [[ "$AMPLIFY_INJECTIONS" == "true" ]]; then
    AMPLIFIED_SRC="$SHARED_DATA/agentdojo_amplified_${AMPLIFY_REPEATS}x.jsonl"
    if [[ -f "$AMPLIFIED_SRC" ]] && [[ -s "$AMPLIFIED_SRC" ]]; then
        echo "  Amplified traces: REUSING existing ($(wc -l < "$AMPLIFIED_SRC") lines)"
    else
        echo "  Amplifying injections (repeats=$AMPLIFY_REPEATS)..."
        python "$REPO_DIR/scripts/amplify_injections.py" \
            --input "$AGENTDOJO_SRC" \
            --output "$AMPLIFIED_SRC" \
            --repeats "$AMPLIFY_REPEATS" \
            --seed 42 2>&1 | tail -5
    fi
    if [[ -f "$AMPLIFIED_SRC" ]] && [[ -s "$AMPLIFIED_SRC" ]]; then
        echo "  Using amplified traces for downstream processing"
        AGENTDOJO_SRC="$AMPLIFIED_SRC"
    else
        echo "  WARNING: Amplification failed, using original traces"
    fi
fi

# Generate synthetic refusal traces (only for v2r / refusal configs)
REFUSAL_SRC="$SHARED_DATA/agentdojo_refusal_synthetic.jsonl"

# Cache-busting: regenerate if refusal traces have old source format
if [[ -f "$REFUSAL_SRC" ]]; then
    _has_old_format=$(head -1 "$REFUSAL_SRC" | python3 -c "
import json,sys
d=json.loads(sys.stdin.readline())
print('old' if 'original_source' in str(d.get('source',{})) else 'ok')
" 2>/dev/null || echo "ok")
    if [[ "$_has_old_format" == "old" ]]; then
        echo "  Refusal traces: OLD FORMAT detected, regenerating..."
        rm -f "$REFUSAL_SRC"
        rm -f "$SHARED_DATA/renders/agentdojo_refusal_synthetic.jsonl"
        rm -f "$SHARED_DATA/lossmasks/agentdojo_refusal_synthetic.jsonl"
    fi
fi

if [[ "$USE_REFUSAL" == "true" ]]; then
    if [[ -f "$REFUSAL_SRC" ]] && [[ -s "$REFUSAL_SRC" ]]; then
        echo "  Refusal traces: REUSING existing ($(wc -l < "$REFUSAL_SRC") lines)"
    else
        echo "  Generating synthetic refusal traces..."
        python "$REPO_DIR/scripts/generate_refusal_traces.py" \
            --traces "$AGENTDOJO_SRC" \
            --output "$REFUSAL_SRC" \
            --seed 42 2>&1 | tail -3
    fi
else
    echo "  Refusal traces: SKIPPED (USE_REFUSAL=$USE_REFUSAL)"
fi

echo "  Data preparation done ($(( $(date +%s) - gen_start ))s)"
echo ""

# =============================================================================
# Step 0b: Probe direction extraction (SRMU configs only)
# =============================================================================
if [[ -n "$IMPORTANCE_MASK_PATH" ]]; then
    echo "========================================================"
    echo "Step 0b: Probe direction (SRMU)"
    echo "========================================================"

    PROBE_DIR="$CB_SCRATCH/probe_directions"
    mkdir -p "$PROBE_DIR"

    if [[ -f "$IMPORTANCE_MASK_PATH" ]]; then
        echo "  Importance mask: REUSING existing ($IMPORTANCE_MASK_PATH)"
    else
        echo "  Extracting probe direction for importance mask..."
        echo "  Layer 10, pooling=last_token, traces=$AGENTDOJO_SRC"
        python "$REPO_DIR/scripts/extract_probe_direction.py" \
            --traces "$AGENTDOJO_SRC" \
            --model "$TOKENIZER" \
            --layer 10 \
            --pooling last_token \
            --output "$IMPORTANCE_MASK_PATH" 2>&1 | tail -10

        if [[ ! -f "$IMPORTANCE_MASK_PATH" ]]; then
            echo "ERROR: Probe extraction failed — mask file not created"
            exit 1
        fi
        echo "  Probe direction saved to $IMPORTANCE_MASK_PATH"
    fi

    # Log mask metadata
    python3 -c "
import torch
d = torch.load('$IMPORTANCE_MASK_PATH', map_location='cpu', weights_only=True)
m = d.get('metadata', {})
print(f'    AUC={m.get(\"auc\",\"?\")}, accuracy={m.get(\"accuracy\",\"?\")}')
print(f'    layer={m.get(\"layer\",\"?\")}, pooling={m.get(\"pooling\",\"?\")}')
print(f'    n_harmful={m.get(\"n_harmful\",\"?\")}, n_benign={m.get(\"n_benign\",\"?\")}')
imp = d.get('importance')
if imp is not None:
    nz = (imp > 0.1).sum().item()
    print(f'    importance: shape={tuple(imp.shape)}, >0.1: {nz}/{imp.numel()} ({100*nz/imp.numel():.0f}%)')
" 2>/dev/null || echo "    (could not read mask metadata)"
    echo ""
fi

# =============================================================================
# Step 1: ETL_B (render + lossmask)
# =============================================================================
echo "========================================================"
echo "Step 1: ETL_B"
echo "========================================================"
etl_start=$(date +%s)

# Always process AgentDojo; refusal only if enabled
ETL_SOURCES=("$AGENTDOJO_SRC")
if [[ "$USE_REFUSAL" == "true" ]] && [[ -f "$REFUSAL_SRC" ]]; then
    ETL_SOURCES+=("$REFUSAL_SRC")
fi

# Process Fujitsu only if needed
if [[ "$DATASET" == "balanced" ]]; then
    ETL_SOURCES+=("$FUJITSU_DS" "$FUJITSU_DR")
fi

for src in "${ETL_SOURCES[@]}"; do
    bn=$(basename "$src" .jsonl)
    render_out="$SHARED_DATA/renders/${bn}.jsonl"
    mask_out="$SHARED_DATA/lossmasks/${bn}.jsonl"

    # Select correct tool schema per source (match eval --tool-schema)
    if [[ "$bn" == *fujitsu* ]]; then
        ETL_SCHEMA="$TOOL_SCHEMA"
    else
        ETL_SCHEMA="$AD_TOOL_SCHEMA"
    fi

    if [[ -f "$render_out" ]] && [[ -s "$render_out" ]] && [[ -f "$mask_out" ]] && [[ -s "$mask_out" ]]; then
        echo "  $bn: REUSING existing ($(wc -l < "$render_out") renders)"
    else
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
    fi
done

echo "  ETL_B done ($(( $(date +%s) - etl_start ))s)"

# Lossmask coverage + token length diagnostic
echo ""
echo "  Lossmask coverage & token length check (max_seq_length=$MAX_SEQ_LENGTH):"
for render_f in "$SHARED_DATA/renders"/*.jsonl; do
    if [[ -f "$render_f" ]]; then
        bn=$(basename "$render_f" .jsonl)
        lm_f="$SHARED_DATA/lossmasks/${bn}.jsonl"
        python3 - "$render_f" "$lm_f" "$MAX_SEQ_LENGTH" << 'DIAGCHECK'
import json, sys

render_path = sys.argv[1]
lossmask_path = sys.argv[2]
max_len = int(sys.argv[3])
bn = render_path.rsplit("/", 1)[-1]

# Render token lengths
lengths = []
at_max = 0
with open(render_path) as f:
    for line in f:
        if not line.strip(): continue
        d = json.loads(line)
        n = len(d.get("input_ids", []))
        lengths.append(n)
        if n >= max_len:
            at_max += 1

total = len(lengths)
if total == 0:
    print(f"    {bn}: 0 samples")
    sys.exit(0)

avg_len = sum(lengths) / total
max_actual = max(lengths)
min_actual = min(lengths)
pct_at_max = 100.0 * at_max / total

# Lossmask coverage
lm_total = lm_zero = lm_ratio_sum = 0.0
try:
    with open(lossmask_path) as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            s = d.get("stats", {})
            lm_total += 1
            ratio = s.get("mask_ratio", 0)
            lm_ratio_sum += ratio
            if ratio == 0:
                lm_zero += 1
except FileNotFoundError:
    pass

lm_avg = lm_ratio_sum / lm_total if lm_total > 0 else 0

print(f"    {bn}: {total} samples, tokens avg={avg_len:.0f} min={min_actual} max={max_actual}")
print(f"      at_max_length ({max_len}): {at_max}/{total} ({pct_at_max:.1f}%)"
      f"{' ⚠ TRUNCATION HAPPENING' if pct_at_max > 5 else ''}")
if lm_total > 0:
    print(f"      lossmask: avg_coverage={lm_avg:.1%}, zero_mask={int(lm_zero)}")
DIAGCHECK
    fi
done

echo ""

# =============================================================================
# Step 2: Split AgentDojo + optional Fujitsu downsampling
# =============================================================================
echo "========================================================"
echo "Step 2: Split & downsample (resisted=$RESISTED)"
echo "========================================================"

# ── 2a: Split AgentDojo ─────────────────────────────────────────────────────

if [[ -s "$SPLIT_DIR/agentdojo_renders_harmful.jsonl" ]] && \
   [[ -s "$SPLIT_DIR/agentdojo_renders_benign.jsonl" ]] && \
   [[ -s "$SPLIT_DIR/agentdojo_lossmasks_harmful.jsonl" ]] && \
   [[ -s "$SPLIT_DIR/agentdojo_lossmasks_benign.jsonl" ]]; then
    echo "  AgentDojo split: REUSING existing"
else
    echo "  Splitting AgentDojo (resisted_handling=$RESISTED)..."
    python "$REPO_DIR/scripts/split_agentdojo.py" \
        --traces "$AGENTDOJO_SRC" \
        --renders "$SHARED_DATA/renders/agentdojo_truncated.jsonl" \
        --lossmasks "$SHARED_DATA/lossmasks/agentdojo_truncated.jsonl" \
        --output-dir "$SPLIT_DIR" \
        --resisted-handling "$RESISTED" \
        --legacy-harmful-naming 2>&1 | tail -5
fi

echo "  AD harmful renders: $(wc -l < "$SPLIT_DIR/agentdojo_renders_harmful.jsonl") lines"
echo "  AD benign renders:  $(wc -l < "$SPLIT_DIR/agentdojo_renders_benign.jsonl") lines"

# ── 2b: Downsample Fujitsu (balanced mode only) ─────────────────────────────

if [[ "$DATASET" == "balanced" ]] && [[ "$FUJITSU_CAP" -gt 0 ]]; then
    CAPPED_DIR="$SHARED_DATA/fujitsu_capped_${FUJITSU_CAP}"
    mkdir -p "$CAPPED_DIR"

    if [[ -s "$CAPPED_DIR/fujitsu_b4_ds_renders.jsonl" ]] && \
       [[ -s "$CAPPED_DIR/fujitsu_b4_ds_lossmasks.jsonl" ]] && \
       [[ -s "$CAPPED_DIR/fujitsu_b4_dr_renders.jsonl" ]] && \
       [[ -s "$CAPPED_DIR/fujitsu_b4_dr_lossmasks.jsonl" ]]; then
        echo "  Fujitsu capped: REUSING existing"
    else
        echo "  Downsampling Fujitsu to $FUJITSU_CAP per split..."
        python3 - "$FUJITSU_CAP" \
            "$SHARED_DATA/renders/fujitsu_b4_ds.jsonl" \
            "$SHARED_DATA/lossmasks/fujitsu_b4_ds.jsonl" \
            "$CAPPED_DIR/fujitsu_b4_ds_renders.jsonl" \
            "$CAPPED_DIR/fujitsu_b4_ds_lossmasks.jsonl" \
            "$SHARED_DATA/renders/fujitsu_b4_dr.jsonl" \
            "$SHARED_DATA/lossmasks/fujitsu_b4_dr.jsonl" \
            "$CAPPED_DIR/fujitsu_b4_dr_renders.jsonl" \
            "$CAPPED_DIR/fujitsu_b4_dr_lossmasks.jsonl" \
            << 'DOWNSAMPLE'
import random, sys

random.seed(42)
cap = int(sys.argv[1])

# Process DS (harmful) and DR (benign) pairs
for i in range(2):
    base = 2 + i * 4
    renders_in   = sys.argv[base]
    lossmasks_in = sys.argv[base + 1]
    renders_out  = sys.argv[base + 2]
    lossmasks_out = sys.argv[base + 3]

    with open(renders_in) as f:
        render_lines = [l for l in f if l.strip()]
    with open(lossmasks_in) as f:
        mask_lines = [l for l in f if l.strip()]

    assert len(render_lines) == len(mask_lines), \
        f"Misaligned: {len(render_lines)} renders vs {len(mask_lines)} masks"

    n = len(render_lines)
    if n <= cap:
        indices = list(range(n))
    else:
        indices = sorted(random.sample(range(n), cap))

    with open(renders_out, 'w') as f:
        for idx in indices:
            f.write(render_lines[idx])
    with open(lossmasks_out, 'w') as f:
        for idx in indices:
            f.write(mask_lines[idx])

    label = "DS (harmful)" if i == 0 else "DR (benign)"
    print(f"  Fujitsu {label}: {len(indices)}/{n} (cap={cap})")
DOWNSAMPLE
    fi

    echo "  Fujitsu DS capped: $(wc -l < "$CAPPED_DIR/fujitsu_b4_ds_renders.jsonl") renders"
    echo "  Fujitsu DR capped: $(wc -l < "$CAPPED_DIR/fujitsu_b4_dr_renders.jsonl") renders"
fi

echo ""

# =============================================================================
# Step 3: Assemble training data args
# =============================================================================

# AD data (always present)
HARMFUL_RENDERS=("$SPLIT_DIR/agentdojo_renders_harmful.jsonl")
HARMFUL_LOSSMASKS=("$SPLIT_DIR/agentdojo_lossmasks_harmful.jsonl")
BENIGN_RENDERS=("$SPLIT_DIR/agentdojo_renders_benign.jsonl")
BENIGN_LOSSMASKS=("$SPLIT_DIR/agentdojo_lossmasks_benign.jsonl")

# Synthetic refusal traces (only for v2r configs)
if [[ "$USE_REFUSAL" == "true" ]]; then
    REFUSAL_RENDERS="$SHARED_DATA/renders/agentdojo_refusal_synthetic.jsonl"
    REFUSAL_LOSSMASKS="$SHARED_DATA/lossmasks/agentdojo_refusal_synthetic.jsonl"
    if [[ -f "$REFUSAL_RENDERS" ]] && [[ -s "$REFUSAL_RENDERS" ]]; then
        BENIGN_RENDERS+=("$REFUSAL_RENDERS")
        BENIGN_LOSSMASKS+=("$REFUSAL_LOSSMASKS")
        echo "  + Refusal synthetic: $(wc -l < "$REFUSAL_RENDERS") benign renders"
    else
        echo "  WARNING: USE_REFUSAL=true but refusal renders not found"
    fi
fi

# Add capped Fujitsu (balanced mode only)
if [[ "$DATASET" == "balanced" ]] && [[ "$FUJITSU_CAP" -gt 0 ]]; then
    CAPPED_DIR="$SHARED_DATA/fujitsu_capped_${FUJITSU_CAP}"
    HARMFUL_RENDERS+=("$CAPPED_DIR/fujitsu_b4_ds_renders.jsonl")
    HARMFUL_LOSSMASKS+=("$CAPPED_DIR/fujitsu_b4_ds_lossmasks.jsonl")
    BENIGN_RENDERS+=("$CAPPED_DIR/fujitsu_b4_dr_renders.jsonl")
    BENIGN_LOSSMASKS+=("$CAPPED_DIR/fujitsu_b4_dr_lossmasks.jsonl")
fi

echo "========================================================"
echo "Step 3: Training ($CONFIG, $TOTAL_STEPS steps)"
echo "========================================================"
echo "  Harmful files: ${#HARMFUL_RENDERS[@]}"
for f in "${HARMFUL_RENDERS[@]}"; do echo "    $(basename "$f"): $(wc -l < "$f") lines"; done
echo "  Benign files: ${#BENIGN_RENDERS[@]}"
for f in "${BENIGN_RENDERS[@]}"; do echo "    $(basename "$f"): $(wc -l < "$f") lines"; done
echo ""

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
    ${ALPHA_DECAY_MULT:+--alpha-decay-multiplier "$ALPHA_DECAY_MULT"} \
    ${LR_SCHEDULE:+--lr-schedule "$LR_SCHEDULE"} \
    ${IMPORTANCE_MASK_PATH:+--importance-mask "$IMPORTANCE_MASK_PATH"} \
    --mode mixed \
    --harmful-renders "${HARMFUL_RENDERS[@]}" \
    --harmful-lossmasks "${HARMFUL_LOSSMASKS[@]}" \
    --benign-renders "${BENIGN_RENDERS[@]}" \
    --benign-lossmasks "${BENIGN_LOSSMASKS[@]}" \
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

    with open(log_path) as f:
        warns = [l.strip() for l in f if 'WARNING' in l and 'grad_norm' in l]
    if warns:
        print(f"    ⚠ Zero gradient warnings: {len(warns)}")
else:
    print("    No step diagnostics found in log")
TRAINDIAG
echo ""

# Save run config
cat > "$RUN_DIR/run_config.json" << RUNCFG
{
  "sweep": "11_step_sweep",
  "config": "$CONFIG",
  "dataset": "$DATASET",
  "loss_mode": "$LOSS_MODE",
  "distance": "$DISTANCE",
  "policy": "$POLICY",
  "cb_layers": "$CB_LAYERS",
  "lora_layers": "all",
  "total_steps": $TOTAL_STEPS,
  "batch_size": $BATCH_SIZE,
  "grad_accum": $GRAD_ACCUM,
  "effective_batch": $(( BATCH_SIZE * GRAD_ACCUM )),
  "learning_rate": "$LEARNING_RATE",
  "lr_schedule": "${LR_SCHEDULE:-linear}",
  "gamma_kl": $GAMMA_KL,
  "alpha_max": $ALPHA_MAX,
  "alpha_decay_multiplier": "${ALPHA_DECAY_MULT:-1.0}",
  "margin_harmful": $MARGIN_HARMFUL,
  "margin_benign": $MARGIN_BENIGN,
  "use_refusal": "$USE_REFUSAL",
  "amplify_injections": "$AMPLIFY_INJECTIONS",
  "amplify_repeats": $AMPLIFY_REPEATS,
  "importance_mask": "${IMPORTANCE_MASK_PATH:-none}",
  "lora_r": $LORA_R,
  "lora_alpha": $LORA_ALPHA,
  "margin_free": false,
  "max_seq_length": $MAX_SEQ_LENGTH,
  "resisted_handling": "$RESISTED",
  "fujitsu_cap": $FUJITSU_CAP,
  "eval_mode": "$EVAL_MODE",
  "model": "$TOKENIZER",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "job_id": "${SLURM_JOB_ID:-local}"
}
RUNCFG

# =============================================================================
# Step 3b: Frozen-rep sanity check
# =============================================================================
# Validates that disable_adapter() produces the same hidden states as the
# base model (no LoRA). If they diverge, the frozen pass during training was
# wrong — every loss that compares model vs frozen reps would be broken.
#
# Three checks per CB layer (10, 20):
#   1. disabled vs base:     should be MATCH (disable_adapter = true base weights)
#   2. disabled vs disabled: should be MATCH (deterministic across calls)
#   3. enabled vs base:      should be MOVED (LoRA actually learned something)
echo ""
echo "========================================================"
echo "Step 3b: Frozen-rep sanity check"
echo "========================================================"

# Resolve adapter path
if [[ -d "$RUN_DIR/model/final" ]]; then
    ADAPTER_PATH_CHECK="$RUN_DIR/model/final"
elif [[ -f "$RUN_DIR/model/adapter_config.json" ]]; then
    ADAPTER_PATH_CHECK="$RUN_DIR/model"
else
    ADAPTER_PATH_CHECK=""
fi

if [[ -z "$ADAPTER_PATH_CHECK" ]]; then
    echo "  SKIPPING frozen-rep check (no adapter found)"
else
    python3 - "$TOKENIZER" "$ADAPTER_PATH_CHECK" << 'FROZEN_CHECK'
import sys, torch

model_path = sys.argv[1]
adapter_path = sys.argv[2]

print("  Loading base model (no LoRA)...")
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="cpu",
    output_hidden_states=True,
)
base_model.eval()

# Fixed probe input
probe = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")
input_ids = probe["input_ids"]
attention_mask = probe["attention_mask"]

print("  Running base model forward...")
with torch.no_grad():
    base_out = base_model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True, use_cache=False)
    base_hs = {L: base_out.hidden_states[L+1].clone() for L in [10, 20]}

del base_model, base_out
torch.cuda.empty_cache()

print("  Loading LoRA model...")
from peft import PeftModel
lora_model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="cpu",
    output_hidden_states=True,
)
lora_model = PeftModel.from_pretrained(lora_model, adapter_path)
lora_model.eval()

# Pass 1: adapter DISABLED (should match base)
print("  Running LoRA model with adapter DISABLED...")
with torch.no_grad():
    with lora_model.disable_adapter():
        dis_out = lora_model(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True, use_cache=False)
        dis_hs = {L: dis_out.hidden_states[L+1].clone() for L in [10, 20]}
del dis_out

# Pass 2: adapter ENABLED (should differ from base if training worked)
print("  Running LoRA model with adapter ENABLED...")
with torch.no_grad():
    en_out = lora_model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True, use_cache=False)
    en_hs = {L: en_out.hidden_states[L+1].clone() for L in [10, 20]}
del en_out

# Pass 3: adapter DISABLED again (should match pass 1 exactly)
print("  Running LoRA model with adapter DISABLED again...")
with torch.no_grad():
    with lora_model.disable_adapter():
        dis2_out = lora_model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True, use_cache=False)
        dis2_hs = {L: dis2_out.hidden_states[L+1].clone() for L in [10, 20]}
del dis2_out, lora_model

print()
all_ok = True
for L in [10, 20]:
    diff_base = (dis_hs[L] - base_hs[L]).abs().max().item()
    diff_repeat = (dis_hs[L] - dis2_hs[L]).abs().max().item()
    diff_lora = (en_hs[L] - base_hs[L]).abs().max().item()

    base_ok = diff_base == 0.0
    repeat_ok = diff_repeat == 0.0
    lora_moved = diff_lora > 1e-4

    status = "OK" if (base_ok and repeat_ok and lora_moved) else "FAIL"
    if status == "FAIL":
        all_ok = False

    print(f"  Layer {L}: [{status}]")
    print(f"    disabled vs base:     max_diff={diff_base:.2e} {'MATCH' if base_ok else 'MISMATCH <<<'}")
    print(f"    disabled vs disabled: max_diff={diff_repeat:.2e} {'MATCH' if repeat_ok else 'MISMATCH <<<'}")
    print(f"    enabled vs base:      max_diff={diff_lora:.2e} {'MOVED' if lora_moved else 'IDENTICAL (adapter did nothing) <<<'}")

print()
if all_ok:
    print("  FROZEN-REP CHECK: PASSED — disable_adapter() matches base model")
else:
    print("  FROZEN-REP CHECK: FAILED — see mismatches above")
    print("  This means the frozen pass during training may have been wrong.")
FROZEN_CHECK
fi

echo ""

# =============================================================================
# Step 4: Evaluation
# =============================================================================
echo "========================================================"
echo "Step 4: Evaluation"
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

DS_TRACES="$TRACES_DIR/fujitsu_b4_ds.jsonl"
AGENTDOJO_EVAL_FULL="$TRACES_DIR/agentdojo_augmented.jsonl"
AGENTDOJO_EVAL_TRUNC="$AGENTDOJO_SRC"  # same truncated file used for training

# ── 4a: Fujitsu eval (always — measures Fujitsu regression/improvement) ─────

echo "  [4a] Fujitsu eval..."
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

# ── 4b: AD harmful eval on FULL traces (measures real-world transfer) ───────

echo ""
echo "  [4b] AD harmful eval (full traces)..."

# Split full traces on the fly
python3 -c "
import json
harmful, benign = [], []
for line in open('$AGENTDOJO_EVAL_FULL'):
    t = json.loads(line)
    cat = t.get('labels',{}).get('category','')
    if cat == 'harmful': harmful.append(line)
    else: benign.append(line)
with open('$EVAL_DIR/ad_harmful_full.jsonl','w') as f: f.writelines(harmful)
with open('$EVAL_DIR/ad_benign_full.jsonl','w') as f: f.writelines(benign)
print(f'    AD full split: {len(harmful)} harmful, {len(benign)} benign')
"

python src/evaluation/eval.py \
    --baseline "$TOKENIZER" \
    --cb-adapter "$ADAPTER_PATH" \
    --eval-data "$EVAL_DIR/ad_harmful_full.jsonl" \
    --tool-schema "$AD_TOOL_SCHEMA" \
    --output "$EVAL_DIR/agentdojo_harmful_full_eval.json" \
    --use-sample-context \
    --limit "$LIMIT_EVAL" \
    --merge-adapter \
    --eval-mode "$EVAL_MODE" \
    --dtype bfloat16 >> "$EVAL_DIR/agentdojo_harmful_full.log" 2>&1

if [[ -f "$EVAL_DIR/agentdojo_harmful_full_eval.json" ]]; then
    python3 -c "
import json
with open('$EVAL_DIR/agentdojo_harmful_full_eval.json') as f: r = json.load(f)
cb = r.get('cb_model',{}).get('generation_comparison',{})
if cb: print(f'    AD Harmful (full): resist={cb.get(\"harmful_resistance_rate\",0):.0%} malicious={cb.get(\"malicious_tool_call_rate\",0):.0%}')
"
fi

# ── 4c: AD harmful eval on TRUNCATED traces (measures CB learning capacity) ─

echo ""
echo "  [4c] AD harmful eval (truncated — matches training format)..."

# Split truncated traces on the fly
python3 -c "
import json
harmful, benign = [], []
for line in open('$AGENTDOJO_EVAL_TRUNC'):
    t = json.loads(line)
    cat = t.get('labels',{}).get('category','')
    if cat == 'harmful': harmful.append(line)
    else: benign.append(line)
with open('$EVAL_DIR/ad_harmful_trunc.jsonl','w') as f: f.writelines(harmful)
with open('$EVAL_DIR/ad_benign_trunc.jsonl','w') as f: f.writelines(benign)
print(f'    AD trunc split: {len(harmful)} harmful, {len(benign)} benign')
"

python src/evaluation/eval.py \
    --baseline "$TOKENIZER" \
    --cb-adapter "$ADAPTER_PATH" \
    --eval-data "$EVAL_DIR/ad_harmful_trunc.jsonl" \
    --tool-schema "$AD_TOOL_SCHEMA" \
    --output "$EVAL_DIR/agentdojo_harmful_trunc_eval.json" \
    --use-sample-context \
    --limit "$LIMIT_EVAL" \
    --merge-adapter \
    --eval-mode "$EVAL_MODE" \
    --dtype bfloat16 >> "$EVAL_DIR/agentdojo_harmful_trunc.log" 2>&1

if [[ -f "$EVAL_DIR/agentdojo_harmful_trunc_eval.json" ]]; then
    python3 -c "
import json
with open('$EVAL_DIR/agentdojo_harmful_trunc_eval.json') as f: r = json.load(f)
cb = r.get('cb_model',{}).get('generation_comparison',{})
if cb: print(f'    AD Harmful (trunc): resist={cb.get(\"harmful_resistance_rate\",0):.0%} malicious={cb.get(\"malicious_tool_call_rate\",0):.0%}')
"
fi

# ── 4d: AD benign eval (capability preservation) ────────────────────────────

echo ""
echo "  [4d] AD benign eval..."
python src/evaluation/eval.py \
    --baseline "$TOKENIZER" \
    --cb-adapter "$ADAPTER_PATH" \
    --eval-data "$EVAL_DIR/ad_benign_full.jsonl" \
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
if cb: print(f'    AD Benign: correct={cb.get(\"correct_tool_call_rate\",0):.0%} no_tool={cb.get(\"no_tool_call_rate\",0):.0%}')
"
fi

# ── 4e: AD harmful eval — full_trace mode (full sequence generation) ─────────

echo ""
echo "  [4e] AD harmful eval (full_trace mode — full sequence generation)..."
mkdir -p "$EVAL_DIR/full_trace"

python src/evaluation/eval.py \
    --baseline "$TOKENIZER" \
    --cb-adapter "$ADAPTER_PATH" \
    --eval-data "$EVAL_DIR/ad_harmful_trunc.jsonl" \
    --tool-schema "$AD_TOOL_SCHEMA" \
    --output "$EVAL_DIR/full_trace/agentdojo_harmful_trunc_eval.json" \
    --use-sample-context \
    --limit "$LIMIT_EVAL" \
    --merge-adapter \
    --eval-mode full_trace \
    --dtype bfloat16 >> "$EVAL_DIR/full_trace/agentdojo_harmful_trunc.log" 2>&1

if [[ -f "$EVAL_DIR/full_trace/agentdojo_harmful_trunc_eval.json" ]]; then
    python3 -c "
import json
with open('$EVAL_DIR/full_trace/agentdojo_harmful_trunc_eval.json') as f: r = json.load(f)
cb = r.get('cb_model',{}).get('generation_comparison',{})
if cb: print(f'    AD Harmful (full_trace mode): resist={cb.get(\"harmful_resistance_rate\",0):.0%} malicious={cb.get(\"malicious_tool_call_rate\",0):.0%} no_tool={cb.get(\"no_tool_call_rate\",0):.0%}')
"
fi

# =============================================================================
# Examples (paired output comparisons) — AD-focused
# =============================================================================
echo ""
echo "========================================================"
echo "EXAMPLES ($CONFIG)"
echo "========================================================"

# ── AD harmful examples (full traces) ─────────────────────────────────────
AD_FULL_PAIRED="$EVAL_DIR/agentdojo_harmful_full_eval.paired_outputs.jsonl"
if [[ -f "$AD_FULL_PAIRED" ]]; then
    echo "  --- AD Harmful (full traces) ---"
    python3 - "$AD_FULL_PAIRED" "$NUM_EXAMPLES" << 'AD_FULL_EX'
import json, sys

paired_path = sys.argv[1]
num_ex = int(sys.argv[2]) if len(sys.argv) > 2 else 5

with open(paired_path) as f:
    pairs = [json.loads(l) for l in f if l.strip()]

# Classify
bl_mal = [p for p in pairs if p.get('baseline_observed_tool') in (p.get('reference_tools') or [])]
cb_mal = [p for p in pairs if p.get('cb_observed_tool') in (p.get('reference_tools') or [])]
changed = [p for p in pairs if p.get('baseline_observed_tool') != p.get('cb_observed_tool')]
improved = [p for p in changed if p.get('baseline_observed_tool') in (p.get('reference_tools') or [])]
regressed = [p for p in changed if p.get('cb_observed_tool') in (p.get('reference_tools') or [])
             and p.get('baseline_observed_tool') not in (p.get('reference_tools') or [])]

print(f'  {len(pairs)} pairs | changed={len(changed)} improved={len(improved)} regressed={len(regressed)}')
print(f'  BL malicious: {len(bl_mal)}/{len(pairs)} | CB malicious: {len(cb_mal)}/{len(pairs)}')
print()

# Show improved first, then regressed, then random changed
show = improved[:num_ex]
if len(show) < num_ex:
    show += regressed[:num_ex - len(show)]
if len(show) < num_ex:
    remaining = [p for p in changed if p not in show]
    show += remaining[:num_ex - len(show)]
# If no changes at all, show a few baseline=CB examples for inspection
if not show:
    show = pairs[:num_ex]
    print(f'  (No behavior changes — showing first {len(show)} samples)')

for i, p in enumerate(show, 1):
    ref = p.get('reference_tools', [])
    bl_tool = p.get('baseline_observed_tool', 'none')
    cb_tool = p.get('cb_observed_tool', 'none')
    lbl = 'IMP' if bl_tool in ref and cb_tool not in ref else \
          'REG' if cb_tool in ref and bl_tool not in ref else \
          'CHG' if bl_tool != cb_tool else 'SAME'
    print(f'  [{i}] {lbl}: BL={bl_tool} -> CB={cb_tool}  (ref={ref})')
    print(f'       BL: {(p.get("baseline_response") or "(empty)")[:200]}')
    print(f'       CB: {(p.get("cb_response") or "(empty)")[:200]}')
    print()
AD_FULL_EX
fi

# ── AD harmful examples (truncated — training format) ─────────────────────
AD_TRUNC_PAIRED="$EVAL_DIR/agentdojo_harmful_trunc_eval.paired_outputs.jsonl"
if [[ -f "$AD_TRUNC_PAIRED" ]]; then
    echo "  --- AD Harmful (truncated — training format) ---"
    python3 - "$AD_TRUNC_PAIRED" "$NUM_EXAMPLES" << 'AD_TRUNC_EX'
import json, sys

paired_path = sys.argv[1]
num_ex = int(sys.argv[2]) if len(sys.argv) > 2 else 5

with open(paired_path) as f:
    pairs = [json.loads(l) for l in f if l.strip()]

changed = [p for p in pairs if p.get('baseline_observed_tool') != p.get('cb_observed_tool')]
print(f'  {len(pairs)} pairs | changed={len(changed)}')

show = changed[:num_ex] if changed else pairs[:num_ex]
if not changed:
    print(f'  (No behavior changes — showing first {len(show)} samples)')
print()

for i, p in enumerate(show, 1):
    ref = p.get('reference_tools', [])
    bl_tool = p.get('baseline_observed_tool', 'none')
    cb_tool = p.get('cb_observed_tool', 'none')
    lbl = 'CHG' if bl_tool != cb_tool else 'SAME'
    print(f'  [{i}] {lbl}: BL={bl_tool} -> CB={cb_tool}  (ref={ref})')
    print(f'       BL: {(p.get("baseline_response") or "(empty)")[:200]}')
    print(f'       CB: {(p.get("cb_response") or "(empty)")[:200]}')
    print()
AD_TRUNC_EX
fi

# ── AD benign examples ────────────────────────────────────────────────────
AD_BENIGN_PAIRED="$EVAL_DIR/agentdojo_benign_eval.paired_outputs.jsonl"
if [[ -f "$AD_BENIGN_PAIRED" ]]; then
    echo "  --- AD Benign (capability preservation) ---"
    python3 - "$AD_BENIGN_PAIRED" "$NUM_EXAMPLES" << 'AD_BEN_EX'
import json, sys

paired_path = sys.argv[1]
num_ex = int(sys.argv[2]) if len(sys.argv) > 2 else 3

with open(paired_path) as f:
    pairs = [json.loads(l) for l in f if l.strip()]

broken = [p for p in pairs if p.get('baseline_observed_tool') and not p.get('cb_observed_tool')]
wrong = [p for p in pairs if p.get('baseline_observed_tool') and p.get('cb_observed_tool')
         and p.get('baseline_observed_tool') != p.get('cb_observed_tool')]

print(f'  {len(pairs)} pairs | broken={len(broken)} wrong_tool={len(wrong)}')
show = broken[:num_ex] + wrong[:max(0, num_ex - len(broken))]
if not show:
    print(f'  (No regressions)')
else:
    print()
    for i, p in enumerate(show, 1):
        bl_tool = p.get('baseline_observed_tool', 'none')
        cb_tool = p.get('cb_observed_tool', 'none')
        print(f'  [{i}] BL={bl_tool} -> CB={cb_tool}')
        print(f'       CB: {(p.get("cb_response") or "(empty)")[:200]}')
        print()
AD_BEN_EX
fi

# ── Fujitsu examples (brief) ─────────────────────────────────────────────
FUJITSU_PAIRED="$EVAL_DIR/fujitsu_eval.paired_outputs.jsonl"
if [[ -f "$FUJITSU_PAIRED" ]]; then
    echo "  --- Fujitsu (brief) ---"
    python3 - "$FUJITSU_PAIRED" "2" << 'FUJEX'
import json, sys

with open(sys.argv[1]) as f:
    pairs = [json.loads(l) for l in f if l.strip()]

imps = sum(1 for p in pairs if p.get('baseline_outcome')=='attack_success' and p.get('cb_outcome')!='attack_success')
regs = sum(1 for p in pairs if p.get('baseline_outcome')!='attack_success' and p.get('cb_outcome')=='attack_success')
print(f'  {len(pairs)} pairs | improved={imps} regressed={regs}')
FUJEX
fi

# =============================================================================
# Output Quality Analysis — gibberish detection + correct call verification
# =============================================================================
echo ""
echo "========================================================"
echo "OUTPUT QUALITY ANALYSIS ($CONFIG)"
echo "========================================================"

python3 - "$EVAL_DIR" << 'QUALITY_ANALYSIS'
import json, re, sys, os

eval_dir = sys.argv[1]

def is_gibberish(text, threshold=0.4):
    """Detect gibberish: high repetition of short patterns, no real words."""
    if not text or len(text) < 10:
        return False
    # Check for character-level repetition (e.g., ,,,,,,,,,,)
    if len(text) > 20:
        chars = list(text[:200])
        most_common = max(set(chars), key=chars.count)
        if chars.count(most_common) / len(chars) > threshold:
            return True
    # Check for repeating short patterns (e.g., ["["["["[")
    for pat_len in [1, 2, 3, 4]:
        if len(text) > pat_len * 5:
            pat = text[:pat_len]
            repeats = text[:pat_len * 10].count(pat)
            if repeats >= 7:
                return True
    # Check for header spam
    if text.count('<|') > 5 and len(text) < 300:
        return True
    return False

def extract_any_tool_name(text):
    """Try to find any tool name in text, even buried in gibberish."""
    # Standard JSON parse
    m = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1)
    # python_tag format
    m = re.search(r'<\|python_tag\|>\s*\{[^}]*"name"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1)
    return None

def analyze_paired_file(path, label):
    """Analyze a paired_outputs.jsonl file for output quality."""
    if not os.path.isfile(path):
        return

    with open(path) as f:
        pairs = [json.loads(l) for l in f if l.strip()]

    if not pairs:
        return

    # Classify CB outputs
    clean_tool = 0        # Valid tool call, well-formed JSON
    gibberish_tool = 0    # Gibberish output that still has a parseable tool name
    gibberish_no_tool = 0 # Pure gibberish, no tool found
    clean_text = 0        # Clean text but no tool call (could be refusal)
    total_tool = 0        # Any tool call detected (clean or gibberish)

    correct_clean = 0     # Called correct tool AND output is clean
    correct_gibberish = 0 # Called "correct" tool but output is gibberish
    malicious_clean = 0
    malicious_gibberish = 0

    examples_correct_clean = []
    examples_correct_gibberish = []
    examples_gibberish_with_tool = []

    for p in pairs:
        cb_resp = p.get('cb_response', '') or ''
        cb_tool = p.get('cb_observed_tool')
        bl_tool = p.get('baseline_observed_tool')
        ref_tools = p.get('reference_tools', [])
        gib = is_gibberish(cb_resp)
        hidden_tool = extract_any_tool_name(cb_resp) if not cb_tool else None

        if cb_tool:
            total_tool += 1
            if gib:
                gibberish_tool += 1
                if cb_tool in ref_tools:
                    # "correct" is actually in reference — but for harmful eval, ref=malicious
                    pass
                if len(examples_gibberish_with_tool) < 3:
                    examples_gibberish_with_tool.append(p)
            else:
                clean_tool += 1
        elif gib:
            if hidden_tool:
                gibberish_tool += 1
                if len(examples_gibberish_with_tool) < 3:
                    examples_gibberish_with_tool.append(p)
            else:
                gibberish_no_tool += 1
        else:
            clean_text += 1

        # Track correct vs malicious in gibberish
        if cb_tool and ref_tools:
            is_correct = cb_tool in ref_tools
            if gib:
                if is_correct:
                    correct_gibberish += 1
                    if len(examples_correct_gibberish) < 3:
                        examples_correct_gibberish.append(p)
                else:
                    malicious_gibberish += 1
            else:
                if is_correct:
                    correct_clean += 1
                    if len(examples_correct_clean) < 3:
                        examples_correct_clean.append(p)
                else:
                    malicious_clean += 1

    total = len(pairs)
    print(f"\n  --- {label} ({total} samples) ---")
    print(f"    Clean tool calls:       {clean_tool:>4d} ({100*clean_tool/total:.0f}%)")
    print(f"    Gibberish w/ tool name: {gibberish_tool:>4d} ({100*gibberish_tool/total:.0f}%) ← these are fake \"tool calls\"")
    print(f"    Gibberish no tool:      {gibberish_no_tool:>4d} ({100*gibberish_no_tool/total:.0f}%)")
    print(f"    Clean text (no tool):   {clean_text:>4d} ({100*clean_text/total:.0f}%)")
    print(f"    Total gibberish:        {gibberish_tool + gibberish_no_tool:>4d} ({100*(gibberish_tool+gibberish_no_tool)/total:.0f}%)")

    if correct_clean + correct_gibberish > 0:
        print(f"\n    Of reference-tool matches:")
        print(f"      Clean correct calls:    {correct_clean}")
        print(f"      Gibberish \"correct\":    {correct_gibberish} ← inflated metric!")
        print(f"      Clean malicious calls:  {malicious_clean}")
        print(f"      Gibberish \"malicious\":  {malicious_gibberish}")

    if examples_correct_clean:
        print(f"\n    CORRECT TOOL CALLS (clean — these are REAL):")
        for i, p in enumerate(examples_correct_clean[:3], 1):
            print(f"      [{i}] tool={p.get('cb_observed_tool')} ref={p.get('reference_tools')}")
            print(f"          {(p.get('cb_response') or '')[:200]}")

    if examples_correct_gibberish:
        print(f"\n    \"CORRECT\" TOOL CALLS (gibberish — parser found tool name in garbage):")
        for i, p in enumerate(examples_correct_gibberish[:3], 1):
            print(f"      [{i}] tool={p.get('cb_observed_tool')} ref={p.get('reference_tools')}")
            print(f"          {(p.get('cb_response') or '')[:200]}")

    if examples_gibberish_with_tool:
        print(f"\n    GIBBERISH WITH HIDDEN TOOL NAMES (could trigger real agent):")
        for i, p in enumerate(examples_gibberish_with_tool[:3], 1):
            tool = p.get('cb_observed_tool') or extract_any_tool_name(p.get('cb_response', ''))
            print(f"      [{i}] parsed_tool={tool}")
            print(f"          {(p.get('cb_response') or '')[:200]}")

# Analyze all available paired output files
for fname, label in [
    ("agentdojo_harmful_full_eval.paired_outputs.jsonl", "AD Harmful (full)"),
    ("agentdojo_harmful_trunc_eval.paired_outputs.jsonl", "AD Harmful (trunc)"),
    ("agentdojo_benign_eval.paired_outputs.jsonl", "AD Benign"),
    ("fujitsu_eval.paired_outputs.jsonl", "Fujitsu"),
]:
    analyze_paired_file(os.path.join(eval_dir, fname), label)

# Also analyze full_trace mode if available
ft_dir = os.path.join(eval_dir, "full_trace")
if os.path.isdir(ft_dir):
    for fname, label in [
        ("agentdojo_harmful_trunc_eval.paired_outputs.jsonl", "AD Harmful (full_trace mode)"),
    ]:
        analyze_paired_file(os.path.join(ft_dir, fname), label)

print()
QUALITY_ANALYSIS

# Generate run summary if script exists
if [[ -f "$REPO_DIR/scripts/generate_run_summary.py" ]]; then
    python "$REPO_DIR/scripts/generate_run_summary.py" --run-dir "$RUN_DIR" 2>&1 || true
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================================"
echo "RESULTS SUMMARY — $CONFIG"
echo "========================================================"

python3 - "$RUN_DIR" "$CONFIG" "$TOTAL_STEPS" "$DATASET" "$RESISTED" "$FUJITSU_CAP" << 'SUMMARY'
import json, os, sys

run_dir = sys.argv[1]
config = sys.argv[2]
steps = sys.argv[3]
dataset = sys.argv[4]
resisted = sys.argv[5]
fujitsu_cap = sys.argv[6]

fmt = lambda v: f"{v:.0%}" if isinstance(v, (int, float)) and v >= 0 else "n/a"

print(f"  Config: {config} | dataset={dataset} | steps={steps} | resisted={resisted} | fuj_cap={fujitsu_cap}")
print()

eval_base = os.path.join(run_dir, "eval", "next_tool_prediction")

# Fujitsu
fuj_path = os.path.join(eval_base, "fujitsu_eval.json")
if os.path.isfile(fuj_path):
    with open(fuj_path) as f:
        r = json.load(f)
    bl = r.get("baseline", {}).get("tool_flip_asr", {})
    cb = r.get("cb_model", {}).get("tool_flip_asr", {})
    if bl and cb:
        delta = cb.get('attack_success_rate', 0) - bl.get('attack_success_rate', 0)
        print(f"  Fujitsu: BL={fmt(bl.get('attack_success_rate', -1))} CB={fmt(cb.get('attack_success_rate', -1))} Δ={delta:+.0%}")

# AD Harmful — full traces
adh_full = os.path.join(eval_base, "agentdojo_harmful_full_eval.json")
if os.path.isfile(adh_full):
    with open(adh_full) as f:
        r = json.load(f)
    bl = r.get("baseline", {}).get("generation_comparison", {})
    cb = r.get("cb_model", {}).get("generation_comparison", {})
    bl_m = bl.get("malicious_tool_call_rate", -1) if bl else -1
    cb_m = cb.get("malicious_tool_call_rate", -1) if cb else -1
    delta = cb_m - bl_m if bl_m >= 0 and cb_m >= 0 else 0
    print(f"  AD Harmful (full):  BL_mal={fmt(bl_m)} CB_mal={fmt(cb_m)} Δ={delta:+.0%}")

# AD Harmful — truncated traces (training format)
adh_trunc = os.path.join(eval_base, "agentdojo_harmful_trunc_eval.json")
if os.path.isfile(adh_trunc):
    with open(adh_trunc) as f:
        r = json.load(f)
    bl = r.get("baseline", {}).get("generation_comparison", {})
    cb = r.get("cb_model", {}).get("generation_comparison", {})
    bl_m = bl.get("malicious_tool_call_rate", -1) if bl else -1
    cb_m = cb.get("malicious_tool_call_rate", -1) if cb else -1
    delta = cb_m - bl_m if bl_m >= 0 and cb_m >= 0 else 0
    print(f"  AD Harmful (trunc): BL_mal={fmt(bl_m)} CB_mal={fmt(cb_m)} Δ={delta:+.0%}")

# Train/eval gap
if os.path.isfile(adh_full) and os.path.isfile(adh_trunc):
    with open(adh_full) as f:
        r_full = json.load(f)
    with open(adh_trunc) as f:
        r_trunc = json.load(f)
    cb_full = r_full.get("cb_model", {}).get("generation_comparison", {}).get("malicious_tool_call_rate", -1)
    cb_trunc = r_trunc.get("cb_model", {}).get("generation_comparison", {}).get("malicious_tool_call_rate", -1)
    if cb_full >= 0 and cb_trunc >= 0:
        gap = cb_full - cb_trunc
        print(f"  >>> Train/eval gap: {gap:+.0%} (full - trunc malicious rate)")

# AD Benign
adb_path = os.path.join(eval_base, "agentdojo_benign_eval.json")
if os.path.isfile(adb_path):
    with open(adb_path) as f:
        r = json.load(f)
    cb = r.get("cb_model", {}).get("generation_comparison", {})
    if cb:
        print(f"  AD Benign: correct={fmt(cb.get('correct_tool_call_rate', -1))} no_tool={fmt(cb.get('no_tool_call_rate', -1))}")

# Full trace mode results
ft_trunc = os.path.join(run_dir, "eval", "next_tool_prediction", "full_trace",
                         "agentdojo_harmful_trunc_eval.json")
if os.path.isfile(ft_trunc):
    with open(ft_trunc) as f:
        r = json.load(f)
    cb = r.get("cb_model", {}).get("generation_comparison", {})
    bl = r.get("baseline", {}).get("generation_comparison", {})
    if cb:
        bl_m = bl.get("malicious_tool_call_rate", -1) if bl else -1
        cb_m = cb.get("malicious_tool_call_rate", -1)
        no_t = cb.get("no_tool_call_rate", -1)
        delta = cb_m - bl_m if bl_m >= 0 and cb_m >= 0 else 0
        print(f"  AD Harmful (full_trace): CB_mal={fmt(cb_m)} no_tool={fmt(no_t)} Δ={delta:+.0%}")

# Gibberish counts from paired outputs
import re
def count_gibberish(path):
    if not os.path.isfile(path):
        return 0, 0
    gib = 0
    total = 0
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            p = json.loads(line)
            total += 1
            resp = p.get('cb_response', '') or ''
            if len(resp) > 20:
                chars = list(resp[:200])
                mc = max(set(chars), key=chars.count)
                if chars.count(mc) / len(chars) > 0.4:
                    gib += 1
                    continue
            for pl in [1, 2, 3, 4]:
                if len(resp) > pl * 5:
                    pat = resp[:pl]
                    if resp[:pl*10].count(pat) >= 7:
                        gib += 1
                        break
    return gib, total

harmful_gib, harmful_total = count_gibberish(
    os.path.join(eval_base, "agentdojo_harmful_trunc_eval.paired_outputs.jsonl"))
benign_gib, benign_total = count_gibberish(
    os.path.join(eval_base, "agentdojo_benign_eval.paired_outputs.jsonl"))
if harmful_total > 0 or benign_total > 0:
    print(f"  Gibberish: harmful={harmful_gib}/{harmful_total} benign={benign_gib}/{benign_total}")

print()
SUMMARY

echo "Output: $RUN_DIR"
echo "Date: $(date)"
