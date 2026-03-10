#!/bin/bash
# Run all diagnostic analyses on a completed experiment.
#
# Usage:
#   bash scripts/diagnostics/run_all_diagnostics.sh /path/to/experiment/run/dir
#
# Expects the run dir to contain:
#   - adapter/ (LoRA checkpoint)
#   - eval/next_tool_prediction/ (eval results)
#   - train.log (training log)
#   - run_config.json (experiment config)
#
# Outputs go to run_dir/diagnostics/

set -euo pipefail

RUN_DIR="${1:?Usage: $0 /path/to/run/dir}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DIAG_DIR="$RUN_DIR/diagnostics"
mkdir -p "$DIAG_DIR"

echo "========================================================"
echo "Running Diagnostics for: $RUN_DIR"
echo "========================================================"

# ── D1: Per-sample error analysis ─────────────────────────────
EVAL_DIR="$RUN_DIR/eval/next_tool_prediction"
TRACES="$REPO_DIR/data/traces/agentdojo_truncated.jsonl"

if [[ -d "$EVAL_DIR" ]]; then
    echo ""
    echo "D1: Per-sample error analysis..."
    python "$REPO_DIR/scripts/diagnostics/per_sample_analysis.py" \
        --eval-dir "$EVAL_DIR" \
        --traces "$TRACES" \
        --output "$DIAG_DIR/per_sample_analysis.json" 2>&1 | tee "$DIAG_DIR/d1_output.txt"
else
    echo "D1: SKIPPED (no eval dir at $EVAL_DIR)"
fi

# ── D3: LoRA weight norms ─────────────────────────────────────
ADAPTER_DIR="$RUN_DIR/adapter"
if [[ ! -d "$ADAPTER_DIR" ]]; then
    # Try alternative paths
    ADAPTER_DIR=$(find "$RUN_DIR" -name "adapter_model.safetensors" -o -name "adapter_model.bin" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
fi

if [[ -d "$ADAPTER_DIR" ]]; then
    echo ""
    echo "D3: LoRA weight norms..."
    python "$REPO_DIR/scripts/diagnostics/lora_weight_norms.py" \
        --adapter "$ADAPTER_DIR" \
        --cb-layers 10 20 \
        --output "$DIAG_DIR/lora_norms.json" \
        --plot "$DIAG_DIR/lora_norms.png" 2>&1 | tee "$DIAG_DIR/d3_output.txt"
else
    echo "D3: SKIPPED (no adapter found)"
fi

# ── D5: Distance distributions (requires GPU) ─────────────────
# Only run if GPU is available
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    MODEL=$(python3 -c "
import json
with open('$RUN_DIR/run_config.json') as f:
    print(json.load(f).get('model', json.load(f).get('base_model', '')))
" 2>/dev/null || echo "")

    if [[ -n "$MODEL" ]] && [[ -d "$ADAPTER_DIR" ]]; then
        echo ""
        echo "D5: Distance distributions..."
        python "$REPO_DIR/scripts/diagnostics/distance_distributions.py" \
            --model "$MODEL" \
            --adapter "$ADAPTER_DIR" \
            --harmful-traces "$TRACES" \
            --benign-traces "$TRACES" \
            --layers 10 20 \
            --max-samples 50 \
            --output "$DIAG_DIR/distance_distributions.json" \
            --plot "$DIAG_DIR/distance_distributions.png" 2>&1 | tee "$DIAG_DIR/d5_output.txt"
    else
        echo "D5: SKIPPED (model path or adapter not found)"
    fi
else
    echo "D5: SKIPPED (no GPU available)"
fi

# ── Loss mask coverage ─────────────────────────────────────────
# Find lossmask files from run_config
echo ""
echo "Loss mask coverage analysis..."
HARMFUL_MASKS=$(python3 -c "
import json
with open('$RUN_DIR/run_config.json') as f:
    d = json.load(f)
for k in ['harmful_lossmasks', 'harmful_lossmask']:
    v = d.get(k)
    if v:
        if isinstance(v, list):
            print(' '.join(v))
        else:
            print(v)
        break
" 2>/dev/null || echo "")

BENIGN_MASKS=$(python3 -c "
import json
with open('$RUN_DIR/run_config.json') as f:
    d = json.load(f)
for k in ['benign_lossmasks', 'benign_lossmask']:
    v = d.get(k)
    if v:
        if isinstance(v, list):
            print(' '.join(v))
        else:
            print(v)
        break
" 2>/dev/null || echo "")

if [[ -n "$HARMFUL_MASKS" ]] && [[ -n "$BENIGN_MASKS" ]]; then
    python "$REPO_DIR/scripts/diagnostics/lossmask_coverage.py" \
        --harmful-lossmasks $HARMFUL_MASKS \
        --benign-lossmasks $BENIGN_MASKS 2>&1 | tee "$DIAG_DIR/lossmask_coverage.txt"
else
    echo "  SKIPPED (could not find mask paths in run_config.json)"
fi

echo ""
echo "========================================================"
echo "Diagnostics complete. Results in: $DIAG_DIR"
echo "========================================================"
ls -la "$DIAG_DIR/"
