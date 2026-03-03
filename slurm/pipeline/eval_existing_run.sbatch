#!/bin/bash
#SBATCH --job-name=eval_existing
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=08:30:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

set -euo pipefail

# =============================================================================
# EVAL AN EXISTING RUN (resume after timeout)
# =============================================================================
# Runs eval + examples + summary on a completed training run.
#
# Usage:
#   RUN_DIR=/scratch/memoozd/cb-scratch/debug_runs/debug_a10.0_l20-31_... \
#     sbatch slurm/pipeline/eval_existing_run.sbatch
#
#   # Eval only specific checkpoints:
#   EVAL_CHECKPOINTS="checkpoint-900,final" RUN_DIR=... \
#     sbatch slurm/pipeline/eval_existing_run.sbatch
# =============================================================================

RUN_DIR="${RUN_DIR:-/scratch/memoozd/cb-scratch/debug_runs/debug_a10.0_l20_21_22_23_24_25_26_27_28_29_30_31_cb_full_sequence_20260302_185051}"

if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: RUN_DIR does not exist: $RUN_DIR"
    exit 1
fi

PROJECT_DIR="/project/def-zhijing/memoozd"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

if [[ -d "$REPO_DIR" ]]; then
    cd "$REPO_DIR"
fi

module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
fi

# Cache setup (offline mode)
CB_SCRATCH="/scratch/memoozd/cb-scratch"
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch,triton}

export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export XDG_CACHE_HOME="$CACHE_DIR"
export DS_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
mkdir -p "$TRITON_CACHE_DIR" "$TRITON_CACHE_DIR/autotune"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TQDM_DISABLE=1

LIMIT_EVAL="${LIMIT_EVAL:-50}"
DTYPE="${DTYPE:-bfloat16}"
NUM_EXAMPLES="${NUM_EXAMPLES:-10}"

echo "========================================"
echo "EVAL EXISTING RUN"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Date: $(date)"
echo "RUN_DIR: $RUN_DIR"
echo "LIMIT_EVAL: $LIMIT_EVAL"
echo ""

# =============================================================================
# Resolve Model
# =============================================================================
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"

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
# Resolve eval data paths
# =============================================================================
TRACES_DIR="$CB_SCRATCH/data/traces"
DS_TRACES="$TRACES_DIR/fujitsu_b4_ds.jsonl"
TOOL_SCHEMA="$REPO_DIR/configs/tool_schemas/b4_standard_v1.json"

# Find AgentDojo split dir (could be in RUN_DIR or a parent BASE_DIR)
SPLIT_DIR=""
if [[ -d "$RUN_DIR/agentdojo_split" ]]; then
    SPLIT_DIR="$RUN_DIR/agentdojo_split"
else
    # Try parent directory (sweep layout: BASE_DIR has splits, RUN_DIR is nested)
    _PARENT=$(dirname "$RUN_DIR")
    for _candidate in "$_PARENT"/a*_l*_*/agentdojo_split "$_PARENT/agentdojo_split"; do
        if [[ -d "$_candidate" ]]; then
            SPLIT_DIR="$_candidate"
            break
        fi
    done
fi

echo "Eval data:"
echo "  DS_TRACES: $DS_TRACES"
echo "  SPLIT_DIR: ${SPLIT_DIR:-<not found>}"
echo ""

mkdir -p "$RUN_DIR/eval"

# =============================================================================
# Checkpoint Evaluation
# =============================================================================
echo "========================================"
echo "CHECKPOINT EVALUATION"
echo "========================================"

# Which checkpoints to eval
EVAL_CHECKPOINTS="${EVAL_CHECKPOINTS:-all}"

for ckpt_path in "$RUN_DIR/model/checkpoint-"* "$RUN_DIR/model/final"; do
    [[ -d "$ckpt_path" ]] || continue
    ckpt_name=$(basename "$ckpt_path")

    # Filter if user specified specific checkpoints
    if [[ "$EVAL_CHECKPOINTS" != "all" ]]; then
        if [[ ! ",$EVAL_CHECKPOINTS," == *",$ckpt_name,"* ]]; then
            echo "  Skipping $ckpt_name (not in EVAL_CHECKPOINTS)"
            continue
        fi
    fi

    if [[ "$ckpt_name" == "final" ]]; then
        EVAL_DIR="$RUN_DIR/eval"
        FUJITSU_OUTPUT="$EVAL_DIR/fujitsu_eval.json"
        AGENTDOJO_OUTPUT="$EVAL_DIR/agentdojo_eval.json"
        AGENTDOJO_BENIGN_OUTPUT="$EVAL_DIR/agentdojo_benign_eval.json"
    else
        EVAL_DIR="$RUN_DIR/eval/checkpoints/$ckpt_name"
        FUJITSU_OUTPUT="$EVAL_DIR/fujitsu_eval.json"
        AGENTDOJO_OUTPUT="$EVAL_DIR/agentdojo_eval.json"
        AGENTDOJO_BENIGN_OUTPUT="$EVAL_DIR/agentdojo_benign_eval.json"
    fi
    mkdir -p "$EVAL_DIR"

    echo ""
    echo "--- Evaluating: $ckpt_name ---"
    echo "  Adapter: $ckpt_path"

    # Fujitsu eval
    echo "  Fujitsu ($LIMIT_EVAL samples)..."
    python src/evaluation/eval.py \
        --baseline "$TOKENIZER" \
        --cb-adapter "$ckpt_path" \
        --eval-data "$DS_TRACES" \
        --tool-schema "$TOOL_SCHEMA" \
        --output "$FUJITSU_OUTPUT" \
        --limit "$LIMIT_EVAL" \
        --merge-adapter \
        --dtype "$DTYPE" 2>&1 | tee -a "$EVAL_DIR/fujitsu.log"

    # Print compact summary
    if [[ -f "$FUJITSU_OUTPUT" ]]; then
        python3 -c "
import json
with open('$FUJITSU_OUTPUT') as f: r = json.load(f)
for lbl, k in [('Baseline','baseline'),('CB','cb_model')]:
    tf = r.get(k,{}).get('tool_flip_asr',{})
    cap = r.get(k,{}).get('capability_retention',{})
    if not tf: continue
    asr = tf.get('attack_success_rate',0)
    cor = tf.get('correct_tool_call_rate',0)
    nt = tf.get('no_tool_call_rate',0)
    cr = cap.get('capability_retention',0)
    n = tf.get('total_samples',0)
    print(f'    {lbl}: ASR={asr:.0%} correct={cor:.0%} no_tool={nt:.0%} cap_ret={cr:.0%} (n={n})')
d = r.get('delta',{})
if d:
    print(f'    Delta: ASR={d.get(\"tool_flip_asr\",0):+.0%} cap={d.get(\"capability_retention\",0):+.0%}')
    g = r.get('stage1_gates',{})
    failed = [k for k,v in g.items() if not v]
    if failed: print(f'    FAILED gates: {\" \".join(failed)}')
    else: print(f'    All gates PASSED')
" 2>/dev/null || echo "    (parse error)"
    fi

    # AgentDojo harmful eval
    if [[ -n "$SPLIT_DIR" && -f "$SPLIT_DIR/agentdojo_traces_harmful.jsonl" ]]; then
        echo "  AgentDojo harmful ($LIMIT_EVAL samples)..."
        python src/evaluation/eval.py \
            --baseline "$TOKENIZER" \
            --cb-adapter "$ckpt_path" \
            --eval-data "$SPLIT_DIR/agentdojo_traces_harmful.jsonl" \
            --tool-schema "$TOOL_SCHEMA" \
            --output "$AGENTDOJO_OUTPUT" \
            --use-sample-context \
            --limit "$LIMIT_EVAL" \
            --merge-adapter \
            --dtype "$DTYPE" 2>&1 | tee -a "$EVAL_DIR/agentdojo.log"

        if [[ -f "$AGENTDOJO_OUTPUT" ]]; then
            python3 -c "
import json
with open('$AGENTDOJO_OUTPUT') as f: r = json.load(f)
for key in ('baseline', 'cb_model'):
    gc = r.get(key, {}).get('generation_comparison', {})
    if gc:
        label = 'Baseline' if key == 'baseline' else 'CB Model'
        mal = gc.get('malicious_tool_call_rate', 0)
        res = gc.get('harmful_resistance_rate', 0)
        cor = gc.get('correct_tool_call_rate', 0)
        no = gc.get('no_tool_call_rate', 0)
        print(f'    {label}: malicious={mal:.0%} resist={res:.0%} correct={cor:.0%} no_tool={no:.0%}')
" 2>/dev/null || echo "    (parse error)"
        fi
    fi

    # AgentDojo benign eval
    if [[ -n "$SPLIT_DIR" && -f "$SPLIT_DIR/agentdojo_traces_benign.jsonl" ]]; then
        echo "  AgentDojo benign ($LIMIT_EVAL samples)..."
        python src/evaluation/eval.py \
            --baseline "$TOKENIZER" \
            --cb-adapter "$ckpt_path" \
            --eval-data "$SPLIT_DIR/agentdojo_traces_benign.jsonl" \
            --tool-schema "$TOOL_SCHEMA" \
            --output "$AGENTDOJO_BENIGN_OUTPUT" \
            --use-sample-context \
            --limit "$LIMIT_EVAL" \
            --merge-adapter \
            --dtype "$DTYPE" 2>&1 | tee -a "$EVAL_DIR/agentdojo_benign.log"
    fi

    echo "  $ckpt_name complete."
done

# =============================================================================
# Examples (final model only)
# =============================================================================
echo ""
echo "========================================"
echo "EXAMPLE OUTPUTS"
echo "========================================"

FUJITSU_OUTPUT="$RUN_DIR/eval/fujitsu_eval.json"
AGENTDOJO_OUTPUT="$RUN_DIR/eval/agentdojo_eval.json"
AGENTDOJO_BENIGN_OUTPUT="$RUN_DIR/eval/agentdojo_benign_eval.json"

# Fujitsu examples
FUJITSU_PAIRED="${FUJITSU_OUTPUT%.json}.paired_outputs.jsonl"
if [[ -f "$FUJITSU_PAIRED" ]]; then
    echo ""
    echo "--- FUJITSU ---"
    python3 - "$FUJITSU_PAIRED" "$NUM_EXAMPLES" << 'PYEOF'
import json, sys

paired_file = sys.argv[1]
num_ex = int(sys.argv[2]) if len(sys.argv) > 2 else 10

with open(paired_file, 'r') as f:
    pairs = [json.loads(line) for line in f if line.strip()]

if not pairs:
    print("No paired outputs")
    sys.exit(0)

def is_gibberish(resp):
    if not resp: return False
    words = resp.split()
    return len(words) > 10 and len(set(words)) / len(words) < 0.15

improvements = [p for p in pairs if p.get('baseline_outcome') == 'attack_success' and p.get('cb_outcome') != 'attack_success']
regressions = [p for p in pairs if p.get('baseline_outcome') != 'attack_success' and p.get('cb_outcome') == 'attack_success']
n_gib = sum(1 for p in pairs if is_gibberish(p.get('cb_response', '')))
n_correct = sum(1 for p in pairs if p.get('cb_outcome') == 'correct_behavior')
n_no_tool = sum(1 for p in pairs if p.get('cb_outcome') == 'no_tool_call')
n_attack = sum(1 for p in pairs if p.get('cb_outcome') == 'attack_success')

print(f'Pairs: {len(pairs)} | Improved: {len(improvements)} | Regressed: {len(regressions)}')
print(f'  CB outcomes: correct={n_correct} | no_tool={n_no_tool} | attack={n_attack} | gibberish={n_gib}')
print()

for i, p in enumerate((improvements + regressions)[:num_ex], 1):
    b_out = p.get('baseline_outcome', '?')
    c_out = p.get('cb_outcome', '?')
    label = 'IMP' if b_out == 'attack_success' else 'REG'
    gib = ' [GIBBERISH]' if is_gibberish(p.get('cb_response', '')) else ''
    print(f'[{i}] {label}: {b_out} -> {c_out}{gib}')
    print(f'    expected={p.get("expected_tool","?")} simulated={p.get("simulated_tool","?")}')
    print(f'--- BASELINE ---')
    print((p.get('baseline_response') or '(empty)')[:300])
    print(f'--- CB MODEL ---')
    print((p.get('cb_response') or '(empty)')[:300])
    print()
PYEOF
fi

# AgentDojo examples
if [[ -f "${AGENTDOJO_OUTPUT%.json}.paired_outputs.jsonl" ]]; then
    AGENTDOJO_PAIRED="${AGENTDOJO_OUTPUT%.json}.paired_outputs.jsonl"
    echo ""
    echo "--- AGENTDOJO ---"
    python3 - "$AGENTDOJO_PAIRED" "$AGENTDOJO_OUTPUT" "$NUM_EXAMPLES" << 'PYEOF'
import json, sys

paired_file = sys.argv[1]
eval_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
num_ex = int(sys.argv[3]) if len(sys.argv) > 3 else 10

with open(paired_file, 'r') as f:
    pairs = [json.loads(line) for line in f if line.strip()]

if not pairs:
    print("No paired outputs")
    sys.exit(0)

def is_gibberish(resp):
    if not resp: return False
    words = resp.split()
    return len(words) > 10 and len(set(words)) / len(words) < 0.15

differ = [p for p in pairs if p.get('responses_differ')]
n_gibberish = sum(1 for p in pairs if is_gibberish(p.get('cb_response', '')))

harmful_pairs = [p for p in pairs if p.get('category') == 'harmful']
benign_pairs = [p for p in pairs if p.get('category') in ('benign', 'resisted')]
n_harmful_refused = sum(1 for p in harmful_pairs if not p.get('cb_observed_tool'))
n_benign_correct = sum(1 for p in benign_pairs if p.get('cb_observed_tool'))

print(f"Pairs: {len(pairs)} | Different: {len(differ)} ({100*len(differ)/len(pairs):.0f}%)")
print(f"  Harmful ({len(harmful_pairs)}): refused={n_harmful_refused}")
if benign_pairs:
    print(f"  Benign ({len(benign_pairs)}): correct_tool={n_benign_correct}")
if n_gibberish > 0:
    print(f"  WARNING: {n_gibberish}/{len(pairs)} gibberish ({100*n_gibberish/len(pairs):.0f}%)")

if eval_file:
    try:
        with open(eval_file) as f:
            ev = json.load(f)
        for key in ('baseline', 'cb_model'):
            gc = ev.get(key, {}).get('generation_comparison', {})
            if gc:
                label = 'Baseline' if key == 'baseline' else 'CB Model'
                mal = gc.get('malicious_tool_call_rate', 0)
                res = gc.get('harmful_resistance_rate', 0)
                cor = gc.get('correct_tool_call_rate', 0)
                no = gc.get('no_tool_call_rate', 0)
                print(f"  {label}: malicious={mal:.0%} resist={res:.0%} correct={cor:.0%} no_tool={no:.0%}")
    except Exception:
        pass

print()
for i, p in enumerate(differ[:num_ex], 1):
    cat = p.get('category', '?')
    b_tool = p.get('baseline_observed_tool') or 'none'
    c_tool = p.get('cb_observed_tool') or 'none'
    gib = ' [GIBBERISH]' if is_gibberish(p.get('cb_response', '')) else ''
    print(f"[{i}] {cat}: baseline={b_tool} -> cb={c_tool}{gib}")
    print(f"--- BASELINE ---")
    print((p.get('baseline_response') or '(empty)')[:300])
    print(f"--- CB MODEL ---")
    print((p.get('cb_response') or '(empty)')[:300])
    print()
PYEOF
fi

# =============================================================================
# Generate run summary
# =============================================================================
echo ""
python scripts/generate_run_summary.py --run-dir "$RUN_DIR" 2>&1 || echo "(summary generation failed)"
echo "Run summary: $RUN_DIR/run_summary.txt"

echo ""
echo "========================================"
echo "EVAL COMPLETE"
echo "========================================"
echo "RUN_DIR: $RUN_DIR"
echo "Finished at: $(date)"
