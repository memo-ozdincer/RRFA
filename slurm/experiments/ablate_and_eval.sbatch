#!/bin/bash
#SBATCH --job-name=ablate
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# LoRA Layer Ablation + Full Eval
# =============================================================================
#
# Usage:
#   CB=/scratch/memoozd/cb-scratch
#   sbatch --export=ALL,RUN_DIR=$(ls -d $CB/sweeps/s9_ad_5000_v2_*/run),KEEP_LAYERS=8-22 \
#       slurm/experiments/ablate_and_eval.sbatch
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
SHARED_DATA="$CB_SCRATCH/shared_data_ts"

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

RUN_DIR="${RUN_DIR:?ERROR: Set RUN_DIR}"
KEEP_LAYERS="${KEEP_LAYERS:?ERROR: Set KEEP_LAYERS (e.g., 8-22)}"
TOKENIZER="meta-llama/Llama-3.1-8B-Instruct"
AD_TOOL_SCHEMA="$REPO_DIR/configs/tool_schemas/agentdojo_v1.json"
TOOL_SCHEMA="$REPO_DIR/configs/tool_schemas/b4_standard_v1.json"
EVAL_MODE="next_tool_prediction"
LIMIT_EVAL=100

# Find adapter
ADAPTER_DIR=""
if [[ -d "$RUN_DIR/model/final" ]]; then
    ADAPTER_DIR="$RUN_DIR/model/final"
elif [[ -d "$RUN_DIR/adapter" ]]; then
    ADAPTER_DIR="$RUN_DIR/adapter"
fi
[[ -z "$ADAPTER_DIR" ]] && { echo "ERROR: No adapter in $RUN_DIR"; exit 1; }

KEEP_SAFE="${KEEP_LAYERS//,/_}"
ABLATED_DIR="$RUN_DIR/model/ablated_keep_${KEEP_SAFE}"
EVAL_DIR="$RUN_DIR/eval_ablated_keep_${KEEP_SAFE}/$EVAL_MODE"
mkdir -p "$EVAL_DIR"

echo "========================================================"
echo "LoRA LAYER ABLATION + EVAL"
echo "========================================================"
echo "Job: ${SLURM_JOB_ID:-local} | $(date)"
echo "Source:      $ADAPTER_DIR"
echo "Keep layers: $KEEP_LAYERS"
echo "Ablated:     $ABLATED_DIR"
echo ""

# ── Step 1: Ablate ──────────────────────────────────────────
python scripts/ablate_lora_layers.py \
    --adapter "$ADAPTER_DIR" \
    --keep-layers "$KEEP_LAYERS" \
    --output "$ABLATED_DIR"

ADAPTER_PATH="$ABLATED_DIR"

echo ""
echo "========================================================"
echo "EVALUATION"
echo "========================================================"

# ── Fujitsu ──
TRACES_DIR="$REPO_DIR/data/traces"
DS_TRACES="$TRACES_DIR/fujitsu_b4_ds.jsonl"

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
    --dtype bfloat16 2>&1 | tail -5

python3 -c "
import json
with open('$EVAL_DIR/fujitsu_eval.json') as f: r = json.load(f)
for lbl, k in [('Baseline','baseline'),('CB','cb_model')]:
    tf = r.get(k,{}).get('tool_flip_asr',{})
    if tf:
        print(f'    {lbl}: ASR={tf.get(\"attack_success_rate\",0):.0%} correct={tf.get(\"correct_tool_call_rate\",0):.0%} no_tool={tf.get(\"no_tool_call_rate\",0):.0%}')
" 2>/dev/null || true

# ── AD split ──
AGENTDOJO_EVAL_FULL="$TRACES_DIR/agentdojo_augmented.jsonl"
AGENTDOJO_SRC="$SHARED_DATA/agentdojo_truncated.jsonl"

echo ""
echo "  Splitting AD traces..."
python3 -c "
import json
harmful_f, benign_f, harmful_t, benign_t = [], [], [], []
for line in open('$AGENTDOJO_EVAL_FULL'):
    t = json.loads(line)
    cat = t.get('labels',{}).get('category','')
    if cat == 'harmful': harmful_f.append(line)
    else: benign_f.append(line)
for line in open('$AGENTDOJO_SRC'):
    t = json.loads(line)
    cat = t.get('labels',{}).get('category','')
    if cat == 'harmful': harmful_t.append(line)
    else: benign_t.append(line)
with open('$EVAL_DIR/ad_harmful_full.jsonl','w') as f: f.writelines(harmful_f)
with open('$EVAL_DIR/ad_benign_full.jsonl','w') as f: f.writelines(benign_f)
with open('$EVAL_DIR/ad_harmful_trunc.jsonl','w') as f: f.writelines(harmful_t)
print(f'    Full: {len(harmful_f)} harmful, {len(benign_f)} benign')
print(f'    Trunc: {len(harmful_t)} harmful, {len(benign_t)} benign')
"

# ── AD harmful (trunc) ──
echo ""
echo "  [4c] AD harmful (trunc)..."
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
    --dtype bfloat16 2>&1 | tail -5

python3 -c "
import json
with open('$EVAL_DIR/agentdojo_harmful_trunc_eval.json') as f: r = json.load(f)
cb = r.get('cb_model',{}).get('generation_comparison',{})
if cb: print(f'    AD Harmful (trunc): resist={cb.get(\"harmful_resistance_rate\",0):.0%} malicious={cb.get(\"malicious_tool_call_rate\",0):.0%}')
" 2>/dev/null || true

# ── AD benign ──
echo ""
echo "  [4d] AD benign..."
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
    --dtype bfloat16 2>&1 | tail -5

python3 -c "
import json
with open('$EVAL_DIR/agentdojo_benign_eval.json') as f: r = json.load(f)
cb = r.get('cb_model',{}).get('generation_comparison',{})
if cb: print(f'    AD Benign: correct={cb.get(\"correct_tool_call_rate\",0):.0%} no_tool={cb.get(\"no_tool_call_rate\",0):.0%}')
" 2>/dev/null || true

echo ""
echo "========================================================"
echo "ABLATION COMPLETE | $(date)"
echo "========================================================"
