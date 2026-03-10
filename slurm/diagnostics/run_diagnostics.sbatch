#!/bin/bash
#SBATCH --job-name=diag
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Diagnostic Analysis Suite
# =============================================================================
#
# Runs ALL diagnostics (D1, D3, D5, loss mask coverage) on one or more
# completed experiment runs. Compares SRMU vs non-SRMU when both are provided.
#
# Usage:
#   CB=/scratch/memoozd/cb-scratch
#
#   # Single run (resolve glob on login node first):
#   sbatch --export=ALL,RUN_DIR=$(ls -d $CB/sweeps/s9_ad_5000_v2_*/run) \
#       slurm/diagnostics/run_diagnostics.sbatch
#
#   # Compare two runs (SRMU vs baseline):
#   sbatch --export=ALL,RUN_DIR=$(ls -d $CB/sweeps/s9_ad_1000_v2_*/run),COMPARE_DIR=$(ls -d $CB/sweeps/s9_ad_1000_v2_srmu_*/run) \
#       slurm/diagnostics/run_diagnostics.sbatch
#
#   # Batch: all completed sweep 10/11 experiments:
#   sbatch --export=ALL,SWEEP_DIR=$CB/sweeps,SWEEP_PATTERN="s9_ad_*_v2*" \
#       slurm/diagnostics/run_diagnostics.sbatch
#
# =============================================================================

set -euo pipefail

# ── Hardcoded paths (same as sweep_ad_focused.sbatch) ─────────
PROJECT_DIR="/project/def-zhijing/memoozd"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
SHARED_DATA="$CB_SCRATCH/shared_data_ts"
TOKENIZER="meta-llama/Llama-3.1-8B-Instruct"

# Activate venv
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
fi

cd "$REPO_DIR"

# Traces for metadata lookup
AD_TRACES="${AD_TRACES:-$SHARED_DATA/agentdojo_truncated.jsonl}"
FUJ_TRACES="${FUJ_TRACES:-$REPO_DIR/data/traces/fujitsu_b4_traces.jsonl}"

echo "========================================================"
echo "DIAGNOSTIC ANALYSIS SUITE"
echo "========================================================"
echo "Job: ${SLURM_JOB_ID:-local} | $(date)"
echo "Repo: $REPO_DIR"
echo ""

# ── Collect run directories ───────────────────────────────────

RUNS=()

if [[ -n "${SWEEP_DIR:-}" ]] && [[ -n "${SWEEP_PATTERN:-}" ]]; then
    # Batch mode: find all matching runs
    for d in "$SWEEP_DIR"/$SWEEP_PATTERN/run; do
        if [[ -d "$d" ]]; then
            RUNS+=("$d")
        fi
    done
    echo "Found ${#RUNS[@]} runs matching $SWEEP_PATTERN"
elif [[ -n "${RUN_DIR:-}" ]]; then
    RUNS+=("$RUN_DIR")
    if [[ -n "${COMPARE_DIR:-}" ]]; then
        RUNS+=("$COMPARE_DIR")
    fi
fi

if [[ ${#RUNS[@]} -eq 0 ]]; then
    echo "ERROR: No run directories found."
    echo "Set RUN_DIR, or SWEEP_DIR+SWEEP_PATTERN"
    exit 1
fi

echo "Runs to analyze:"
for r in "${RUNS[@]}"; do echo "  $r"; done
echo ""

# ── Helper: run diagnostics on a single run ───────────────────

run_single_diagnostics() {
    local run_dir="$1"
    local run_name
    run_name=$(basename "$(dirname "$run_dir")")
    local diag_dir="$run_dir/diagnostics"
    mkdir -p "$diag_dir"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Analyzing: $run_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── D1: Per-sample error analysis ──
    local eval_dir="$run_dir/eval/next_tool_prediction"
    if [[ -d "$eval_dir" ]] && ls "$eval_dir"/*.json &>/dev/null; then
        echo "  D1: Per-sample error analysis..."
        python "$REPO_DIR/scripts/diagnostics/per_sample_analysis.py" \
            --eval-dir "$eval_dir" \
            --traces "$AD_TRACES" \
            --output "$diag_dir/per_sample_analysis.json" \
            2>&1 | tee "$diag_dir/d1_output.txt"
    else
        echo "  D1: SKIPPED (no eval results at $eval_dir)"
    fi

    # ── D3: LoRA weight norms ──
    local adapter_dir=""
    if [[ -d "$run_dir/adapter" ]]; then
        adapter_dir="$run_dir/adapter"
    else
        # Search for adapter files
        local found
        found=$(find "$run_dir" -name "adapter_model.safetensors" -o -name "adapter_model.bin" 2>/dev/null | head -1)
        if [[ -n "$found" ]]; then
            adapter_dir=$(dirname "$found")
        fi
    fi

    if [[ -n "$adapter_dir" ]]; then
        echo ""
        echo "  D3: LoRA weight norms..."
        python "$REPO_DIR/scripts/diagnostics/lora_weight_norms.py" \
            --adapter "$adapter_dir" \
            --cb-layers 10 20 \
            --output "$diag_dir/lora_norms.json" \
            --plot "$diag_dir/lora_norms.png" \
            2>&1 | tee "$diag_dir/d3_output.txt"
    else
        echo "  D3: SKIPPED (no adapter checkpoint found)"
    fi

    # ── D5: Distance distributions (needs GPU) ──
    if [[ -n "$adapter_dir" ]] && [[ -f "$AD_TRACES" ]]; then
        echo ""
        echo "  D5: Distance distributions (GPU)..."
        python "$REPO_DIR/scripts/diagnostics/distance_distributions.py" \
            --model "$TOKENIZER" \
            --adapter "$adapter_dir" \
            --harmful-traces "$AD_TRACES" \
            --benign-traces "$AD_TRACES" \
            --layers 10 20 \
            --max-samples 50 \
            --output "$diag_dir/distance_distributions.json" \
            --plot "$diag_dir/distance_distributions.png" \
            2>&1 | tee "$diag_dir/d5_output.txt"
    else
        echo "  D5: SKIPPED (adapter or traces not found)"
    fi

    # ── Loss mask coverage ──
    local harmful_masks="" benign_masks=""
    if [[ -f "$run_dir/run_config.json" ]]; then
        harmful_masks=$(python3 -c "
import json
with open('$run_dir/run_config.json') as f:
    d = json.load(f)
v = d.get('harmful_lossmasks', d.get('harmful_lossmask', ''))
if isinstance(v, list): print(' '.join(v))
elif v: print(v)
" 2>/dev/null || echo "")

        benign_masks=$(python3 -c "
import json
with open('$run_dir/run_config.json') as f:
    d = json.load(f)
v = d.get('benign_lossmasks', d.get('benign_lossmask', ''))
if isinstance(v, list): print(' '.join(v))
elif v: print(v)
" 2>/dev/null || echo "")
    fi

    if [[ -n "$harmful_masks" ]] && [[ -n "$benign_masks" ]]; then
        echo ""
        echo "  Loss mask coverage..."
        python "$REPO_DIR/scripts/diagnostics/lossmask_coverage.py" \
            --harmful-lossmasks $harmful_masks \
            --benign-lossmasks $benign_masks \
            2>&1 | tee "$diag_dir/lossmask_coverage.txt"
    else
        echo "  Loss mask: SKIPPED (paths not in run_config.json)"
    fi

    # ── SRMU mask analysis (if this run used importance masking) ──
    local mask_path=""
    if [[ -f "$run_dir/run_config.json" ]]; then
        mask_path=$(python3 -c "
import json
with open('$run_dir/run_config.json') as f:
    d = json.load(f)
v = d.get('importance_mask', 'none')
if v != 'none': print(v)
" 2>/dev/null || echo "")
    fi

    if [[ -n "$mask_path" ]] && [[ -f "$mask_path" ]]; then
        echo ""
        echo "  SRMU mask analysis..."
        python3 -c "
import torch
import json

d = torch.load('$mask_path', map_location='cpu', weights_only=True)
imp = d['importance']
meta = d.get('metadata', {})

print(f'  Mask file: $mask_path')
print(f'  Probe AUC: {meta.get(\"auc\", \"?\"):.3f}')
print(f'  Probe accuracy: {meta.get(\"accuracy\", \"?\"):.1%}')
print(f'  Layer: {meta.get(\"layer\", \"?\")}, Pooling: {meta.get(\"pooling\", \"?\")}')
print(f'  Samples: {meta.get(\"n_harmful\", \"?\")} harmful, {meta.get(\"n_benign\", \"?\")} benign')
print()
print(f'  Importance mask stats:')
print(f'    Shape: {tuple(imp.shape)}')
print(f'    Range: [{imp.min():.4f}, {imp.max():.4f}]')
print(f'    Mean:  {imp.mean():.4f}')
print(f'    >0.5:  {(imp > 0.5).sum().item()}/{imp.numel()} ({(imp > 0.5).float().mean():.1%})')
print(f'    >0.1:  {(imp > 0.1).sum().item()}/{imp.numel()} ({(imp > 0.1).float().mean():.1%})')
print(f'    <0.01: {(imp < 0.01).sum().item()}/{imp.numel()} ({(imp < 0.01).float().mean():.1%})')
print(f'    Top 10 dims: {imp.topk(10).indices.tolist()}')
print(f'    Top 10 vals: {[f\"{v:.3f}\" for v in imp.topk(10).values.tolist()]}')

# Effective dimensionality: how many dims carry 90% of the total weight
sorted_imp, _ = imp.sort(descending=True)
cumsum = sorted_imp.cumsum(0) / sorted_imp.sum()
eff_dims_90 = (cumsum < 0.9).sum().item() + 1
eff_dims_50 = (cumsum < 0.5).sum().item() + 1
print(f'    Effective dims (90% weight): {eff_dims_90}/{imp.numel()} ({eff_dims_90/imp.numel():.1%})')
print(f'    Effective dims (50% weight): {eff_dims_50}/{imp.numel()} ({eff_dims_50/imp.numel():.1%})')
" 2>&1 | tee "$diag_dir/srmu_mask_analysis.txt"
    fi

    echo ""
    echo "  Results in: $diag_dir"
    ls "$diag_dir/" 2>/dev/null | sed 's/^/    /'
    echo ""
}

# ── Run on each directory ─────────────────────────────────────

for run in "${RUNS[@]}"; do
    run_single_diagnostics "$run"
done

# ── Comparison (if two runs provided) ─────────────────────────

if [[ ${#RUNS[@]} -ge 2 ]]; then
    echo "========================================================"
    echo "COMPARISON: $(basename "$(dirname "${RUNS[0]}")") vs $(basename "$(dirname "${RUNS[1]}")")"
    echo "========================================================"

    DIAG_A="${RUNS[0]}/diagnostics"
    DIAG_B="${RUNS[1]}/diagnostics"

    # Compare D5 distance distributions
    if [[ -f "$DIAG_A/distance_distributions.json" ]] && [[ -f "$DIAG_B/distance_distributions.json" ]]; then
        python3 -c "
import json
import numpy as np

with open('$DIAG_A/distance_distributions.json') as f:
    a = json.load(f)
with open('$DIAG_B/distance_distributions.json') as f:
    b = json.load(f)

name_a = '$(basename "$(dirname "${RUNS[0]}")")'.split('_', 2)[-1] if '_' in '$(basename "$(dirname "${RUNS[0]}")")' else 'Run A'
name_b = '$(basename "$(dirname "${RUNS[1]}")")'.split('_', 2)[-1] if '_' in '$(basename "$(dirname "${RUNS[1]}")")' else 'Run B'

for layer in [10, 20]:
    print(f'\nLayer {layer}:')
    for cat in ['harmful', 'benign']:
        d_a = [r['distance'] for r in a if r['layer'] == layer and r['category'] == cat]
        d_b = [r['distance'] for r in b if r['layer'] == layer and r['category'] == cat]
        if d_a and d_b:
            print(f'  {cat:>8} — {name_a}: mean={np.mean(d_a):.2f}  {name_b}: mean={np.mean(d_b):.2f}  delta={np.mean(d_b)-np.mean(d_a):+.2f}')

# Key question: did SRMU increase separation?
for layer in [10, 20]:
    h_a = [r['distance'] for r in a if r['layer'] == layer and r['category'] == 'harmful']
    b_a = [r['distance'] for r in a if r['layer'] == layer and r['category'] == 'benign']
    h_b = [r['distance'] for r in b if r['layer'] == layer and r['category'] == 'harmful']
    b_b = [r['distance'] for r in b if r['layer'] == layer and r['category'] == 'benign']
    if h_a and b_a and h_b and b_b:
        sep_a = np.mean(h_a) - np.mean(b_a)
        sep_b = np.mean(h_b) - np.mean(b_b)
        print(f'\nLayer {layer} separation (harmful_mean - benign_mean):')
        print(f'  {name_a}: {sep_a:.2f}')
        print(f'  {name_b}: {sep_b:.2f}')
        if sep_b > sep_a:
            print(f'  → {name_b} has BETTER separation (+{sep_b - sep_a:.2f})')
        else:
            print(f'  → {name_a} has better separation ({sep_a - sep_b:+.2f})')
" 2>&1
    fi

    # Compare LoRA norms
    if [[ -f "$DIAG_A/lora_norms.json" ]] && [[ -f "$DIAG_B/lora_norms.json" ]]; then
        echo ""
        echo "LoRA norm comparison:"
        python3 -c "
import json
with open('$DIAG_A/lora_norms.json') as f:
    a = json.load(f)
with open('$DIAG_B/lora_norms.json') as f:
    b = json.load(f)
print(f'  Run A — CB norm: {a[\"cb_norm\"]:.4f}, non-CB: {a[\"non_cb_norm\"]:.4f}, ratio: {a[\"non_cb_norm\"]/a[\"cb_norm\"]:.2f}x')
print(f'  Run B — CB norm: {b[\"cb_norm\"]:.4f}, non-CB: {b[\"non_cb_norm\"]:.4f}, ratio: {b[\"non_cb_norm\"]/b[\"cb_norm\"]:.2f}x')
if b['non_cb_norm']/b['cb_norm'] < a['non_cb_norm']/a['cb_norm']:
    print('  → Run B has less collateral damage (lower non-CB/CB ratio)')
else:
    print('  → Run A has less collateral damage')
" 2>&1
    fi
fi

echo ""
echo "========================================================"
echo "DIAGNOSTICS COMPLETE | $(date)"
echo "========================================================"
