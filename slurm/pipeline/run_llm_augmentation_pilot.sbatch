#!/bin/bash
#SBATCH --job-name=llm_aug_pilot
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

set -euo pipefail

# =============================================================================
# PILOT RUN: LLM-Based Data Augmentation (Small Batch)
# =============================================================================
# Runs on 100 Fujitsu skeletons + 100 AgentDojo traces.
# Produces rich diagnostics you can show to collaborators for review.
#
# Output: $CB_SCRATCH/pilot_augmentation/
#   - traces/*.jsonl          (all augmented trace files)
#   - diagnostics/            (JSON reports, sample printouts)
#   - renders/*.jsonl         (tokenized for ETL_B verification)
#   - lossmasks/*.jsonl       (mask verification)
#
# Usage:
#   sbatch slurm/pipeline/run_llm_augmentation_pilot.sbatch
#
# After completion, SCP the entire pilot_augmentation/ directory:
#   scp -r cluster:$CB_SCRATCH/pilot_augmentation/ .
# =============================================================================

PROJECT_DIR="/project/def-zhijing/memoozd"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

if [[ -d "$REPO_DIR" ]]; then
    cd "$REPO_DIR"
    echo "Working directory: $(pwd)"
else
    echo "ERROR: Repo not found at $REPO_DIR"
    exit 1
fi

module --force purge || true
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.2

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated venv: $VENV_DIR"
fi

# Cache
CB_SCRATCH="/scratch/memoozd/cb-scratch"
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets,torch}
export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Config
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TOKENIZER="${TOKENIZER:-$MODEL}"
PILOT_LIMIT=100
BATCH_SIZE=32
SEED=42
MAX_SEQ_LENGTH=4096
VLLM_PORT=8899

# Output dirs
PILOT_DIR="$CB_SCRATCH/pilot_augmentation"
TRACES_OUT="$PILOT_DIR/traces"
DIAG_OUT="$PILOT_DIR/diagnostics"
RENDERS_OUT="$PILOT_DIR/renders"
LOSSMASKS_OUT="$PILOT_DIR/lossmasks"
rm -rf "$PILOT_DIR"
mkdir -p "$TRACES_OUT" "$DIAG_OUT" "$RENDERS_OUT" "$LOSSMASKS_OUT"

echo "============================================================"
echo "PILOT RUN: LLM-Based Data Augmentation"
echo "============================================================"
echo "Model:     $MODEL"
echo "Limit:     $PILOT_LIMIT traces per dataset"
echo "Output:    $PILOT_DIR"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# ── STEP 0: Start vLLM ──────────────────────────────────────────────

echo "============================================================"
echo "STEP 0: Starting vLLM Server"
echo "============================================================"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port $VLLM_PORT \
    --max-model-len $MAX_SEQ_LENGTH \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

echo "Waiting for vLLM server..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

VLLM_URL="http://localhost:$VLLM_PORT"

# ── STEP 1: Complete Fujitsu Skeletons ───────────────────────────────

echo ""
echo "============================================================"
echo "STEP 1: Complete Fujitsu Skeletons (limit=$PILOT_LIMIT)"
echo "============================================================"

python scripts/augment_llm.py \
    --operation complete_fujitsu \
    --input data/traces/fujitsu_b4_traces.jsonl \
    --output "$TRACES_OUT/fujitsu_completed.jsonl" \
    --vllm-url "$VLLM_URL" \
    --model "$MODEL" \
    --batch-size $BATCH_SIZE \
    --limit $PILOT_LIMIT \
    --seed $SEED \
    --verbose \
    2>&1 | tee "$DIAG_OUT/step1_complete_fujitsu.log"

echo ""
echo "── Verification ──"
python scripts/augment_llm.py \
    --operation verify \
    --input "$TRACES_OUT/fujitsu_completed.jsonl" \
    --verbose \
    2>&1 | tee "$DIAG_OUT/step1_verify.log"

# ── STEP 2: Canonicalize AgentDojo ───────────────────────────────────

echo ""
echo "============================================================"
echo "STEP 2: Canonicalize AgentDojo (limit=$PILOT_LIMIT)"
echo "============================================================"

python scripts/augment_llm.py \
    --operation canonicalize_ad \
    --input data/traces/agentdojo_augmented.jsonl \
    --output "$TRACES_OUT/agentdojo_canonicalized.jsonl" \
    --limit $PILOT_LIMIT \
    --verbose \
    2>&1 | tee "$DIAG_OUT/step2_canonicalize.log"

# ── STEP 3: Upsample Minority ───────────────────────────────────────

echo ""
echo "============================================================"
echo "STEP 3: Upsample Minority (target=50 for pilot)"
echo "============================================================"

python scripts/augment_llm.py \
    --operation upsample_minority \
    --input data/traces/fujitsu_b4_traces.jsonl \
    --output "$TRACES_OUT/minority_skeletons.jsonl" \
    --minority-target 50 \
    --seed $SEED \
    --verbose \
    2>&1 | tee "$DIAG_OUT/step3_upsample.log"

# ── STEP 3b: Complete minority skeletons ─────────────────────────────

echo ""
echo "============================================================"
echo "STEP 3b: Complete Minority Skeletons"
echo "============================================================"

python scripts/augment_llm.py \
    --operation complete_fujitsu \
    --input "$TRACES_OUT/minority_skeletons.jsonl" \
    --output "$TRACES_OUT/minority_completed.jsonl" \
    --vllm-url "$VLLM_URL" \
    --model "$MODEL" \
    --batch-size $BATCH_SIZE \
    --seed $SEED \
    --verbose \
    2>&1 | tee "$DIAG_OUT/step3b_complete_minority.log"

# ── STEP 4: Cross-Pollinate ──────────────────────────────────────────

echo ""
echo "============================================================"
echo "STEP 4: Cross-Pollinate (max=20 per direction)"
echo "============================================================"

python scripts/augment_llm.py \
    --operation cross_pollinate \
    --input data/traces/fujitsu_b4_traces.jsonl \
    --input2 data/traces/agentdojo_augmented.jsonl \
    --output "$TRACES_OUT/cross_pollinated.jsonl" \
    --cross-max 20 \
    --seed $SEED \
    --verbose \
    2>&1 | tee "$DIAG_OUT/step4_cross_pollinate.log"

# ── STEP 5: Contrastive Pairs ───────────────────────────────────────

echo ""
echo "============================================================"
echo "STEP 5: Build Contrastive Pairs"
echo "============================================================"

python scripts/augment_llm.py \
    --operation contrastive_fujitsu \
    --input "$TRACES_OUT/fujitsu_completed.jsonl" \
    --output /dev/null \
    --pairs-output "$TRACES_OUT/contrastive_pairs.jsonl" \
    2>&1 | tee "$DIAG_OUT/step5_contrastive.log"

echo ""
PAIR_COUNT=$(wc -l < "$TRACES_OUT/contrastive_pairs.jsonl" 2>/dev/null || echo 0)
echo "Contrastive pairs: $PAIR_COUNT"

# ── STEP 6: Merge + Full Verify ──────────────────────────────────────

echo ""
echo "============================================================"
echo "STEP 6: Merge All + Final Verification"
echo "============================================================"

cat "$TRACES_OUT/fujitsu_completed.jsonl" \
    "$TRACES_OUT/agentdojo_canonicalized.jsonl" \
    "$TRACES_OUT/minority_completed.jsonl" \
    "$TRACES_OUT/cross_pollinated.jsonl" \
    > "$TRACES_OUT/pilot_merged.jsonl"

TOTAL=$(wc -l < "$TRACES_OUT/pilot_merged.jsonl")
echo "Merged: $TOTAL traces"
echo ""

python scripts/augment_llm.py \
    --operation verify \
    --input "$TRACES_OUT/pilot_merged.jsonl" \
    --verbose \
    2>&1 | tee "$DIAG_OUT/step6_final_verify.log"

# ── STEP 7: ETL_B Pipeline ──────────────────────────────────────────

echo ""
echo "============================================================"
echo "STEP 7: ETL_B (Render + Lossmask) on Pilot Data"
echo "============================================================"

python -m src.schemas.tools.ETL_B \
    --traces "$TRACES_OUT/pilot_merged.jsonl" \
    --render-out "$RENDERS_OUT/pilot_merged.jsonl" \
    --lossmask-out "$LOSSMASKS_OUT/pilot_merged.jsonl" \
    --tokenizer "$TOKENIZER" \
    --max-length $MAX_SEQ_LENGTH \
    --policy-override injection_aware \
    --pooling-policy-override cb_full_sequence \
    --include-rendered-text \
    --allow-skeleton \
    --skeleton-policy cb_full_sequence \
    2>&1 | tee "$DIAG_OUT/step7_etl_b.log"

RENDER_COUNT=$(wc -l < "$RENDERS_OUT/pilot_merged.jsonl" 2>/dev/null || echo 0)
LOSSMASK_COUNT=$(wc -l < "$LOSSMASKS_OUT/pilot_merged.jsonl" 2>/dev/null || echo 0)
echo "  Renders: $RENDER_COUNT"
echo "  Lossmasks: $LOSSMASK_COUNT"

# ── STEP 8: Rich Diagnostic Report ──────────────────────────────────

echo ""
echo "============================================================"
echo "STEP 8: Generate Diagnostic Report"
echo "============================================================"

python3 << 'DIAGNOSTIC_EOF'
import json, sys, os
from collections import Counter, defaultdict

PILOT_DIR = os.environ.get("PILOT_DIR", "/scratch/memoozd/cb-scratch/pilot_augmentation")
TRACES = f"{PILOT_DIR}/traces/pilot_merged.jsonl"
RENDERS = f"{PILOT_DIR}/renders/pilot_merged.jsonl"
LOSSMASKS = f"{PILOT_DIR}/lossmasks/pilot_merged.jsonl"
DIAG = f"{PILOT_DIR}/diagnostics"

report = {"timestamp": "", "summary": {}, "per_operation": {}, "mask_analysis": {}, "sample_traces": []}

# Load traces
traces = []
with open(TRACES) as f:
    for line in f:
        traces.append(json.loads(line))

# Summary stats
cats = Counter(t.get("labels", {}).get("category", "?") for t in traces)
provs = Counter(
    t.get("raw_metadata", {}).get("source_fields", {}).get("augmentation_provenance", "original")
    for t in traces
)
completeness = Counter(t.get("completeness", "?") for t in traces)

report["summary"] = {
    "total_traces": len(traces),
    "categories": dict(cats),
    "provenance": dict(provs),
    "completeness": dict(completeness),
}

# Injection span coverage
harmful = [t for t in traces if t.get("labels", {}).get("category") == "harmful"]
harmful_with_span = sum(
    1 for t in harmful
    if (t.get("signal_hints") or {}).get("injection_char_span") is not None
)
report["summary"]["harmful_count"] = len(harmful)
report["summary"]["harmful_with_injection_span"] = harmful_with_span
report["summary"]["harmful_span_coverage"] = f"{100*harmful_with_span/len(harmful):.1f}%" if harmful else "N/A"

# Tool call distribution
has_tc = sum(1 for t in traces if any(m.get("tool_calls") for m in t.get("messages", [])))
report["summary"]["traces_with_tool_calls"] = has_tc

# Variant distribution
variants = Counter()
for t in traces:
    v = (t.get("task") or {}).get("variant", "")
    if v:
        exp = (t.get("tool_attack") or {}).get("expected_tool", "?")
        obs = (t.get("tool_attack") or {}).get("observed_tool", "?")
        variants[f"{v} ({exp}->{obs})"] = variants.get(f"{v} ({exp}->{obs})", 0) + 1
report["summary"]["variant_distribution"] = dict(variants)

# Load lossmasks and analyze
if os.path.exists(LOSSMASKS):
    masks = []
    with open(LOSSMASKS) as f:
        for line in f:
            masks.append(json.loads(line))

    # Analyze mask distributions by category
    mask_by_cat = defaultdict(lambda: {"count": 0, "total_tokens": 0, "w05_tokens": 0, "w10_tokens": 0, "zero_masks": 0})

    # Build trace_id -> category mapping
    cat_map = {t["id"]: t.get("labels", {}).get("category", "?") for t in traces}
    prov_map = {t["id"]: t.get("raw_metadata", {}).get("source_fields", {}).get("augmentation_provenance", "?") for t in traces}

    for m in masks:
        tid = m.get("trace_id", "")
        cat = cat_map.get(tid, "?")
        prov = prov_map.get(tid, "?")
        lm = m.get("loss_mask", [])
        pm = m.get("pooling_mask", lm)

        key = f"{cat}/{prov}"
        stats = mask_by_cat[key]
        stats["count"] += 1
        stats["total_tokens"] += len(lm)
        stats["w05_tokens"] += sum(1 for x in lm if x == 0.5)
        stats["w10_tokens"] += sum(1 for x in lm if x == 1.0)
        if all(x == 0 for x in lm):
            stats["zero_masks"] += 1

    for key, stats in mask_by_cat.items():
        n = stats["count"]
        stats["avg_tokens"] = stats["total_tokens"] / n if n else 0
        stats["avg_w05"] = stats["w05_tokens"] / n if n else 0
        stats["avg_w10"] = stats["w10_tokens"] / n if n else 0
        stats["pct_zero_mask"] = f"{100*stats['zero_masks']/n:.1f}%" if n else "N/A"

    report["mask_analysis"] = {k: v for k, v in sorted(mask_by_cat.items())}

# Sample traces (3 per provenance)
sample_by_prov = defaultdict(list)
for t in traces:
    prov = t.get("raw_metadata", {}).get("source_fields", {}).get("augmentation_provenance", "original")
    if len(sample_by_prov[prov]) < 3:
        sample = {
            "id": t["id"][:60],
            "category": t.get("labels", {}).get("category"),
            "completeness": t.get("completeness"),
            "provenance": prov,
            "num_messages": len(t.get("messages", [])),
            "has_tool_calls": any(m.get("tool_calls") for m in t.get("messages", [])),
            "injection_span": (t.get("signal_hints") or {}).get("injection_char_span"),
            "messages_summary": [],
        }
        for j, msg in enumerate(t.get("messages", [])):
            role = msg.get("role", "?")
            content_len = len(msg.get("content") or "")
            tc_names = [c.get("function", {}).get("name", "?") for c in (msg.get("tool_calls") or [])]
            sample["messages_summary"].append({
                "idx": j,
                "role": role,
                "content_chars": content_len,
                "content_preview": (msg.get("content") or "")[:100],
                "tool_calls": tc_names if tc_names else None,
            })
        sample_by_prov[prov].append(sample)

report["sample_traces"] = dict(sample_by_prov)

# Write report
report_path = f"{DIAG}/pilot_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"Diagnostic report written to: {report_path}")

# Print summary for quick viewing
print("\n" + "="*60)
print("PILOT DIAGNOSTIC SUMMARY")
print("="*60)
print(f"Total traces: {report['summary']['total_traces']}")
print(f"\nCategories:")
for k, v in sorted(report["summary"]["categories"].items()):
    print(f"  {k}: {v}")
print(f"\nProvenance:")
for k, v in sorted(report["summary"]["provenance"].items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")
print(f"\nHarmful injection span coverage: {report['summary']['harmful_span_coverage']}")
print(f"Traces with tool calls: {report['summary']['traces_with_tool_calls']}")

if report.get("mask_analysis"):
    print(f"\nLossmask Analysis (injection_aware + cb_full pool):")
    print(f"{'Category/Provenance':<45} {'Count':>5} {'AvgTok':>7} {'Avg0.5':>7} {'Avg1.0':>7} {'%Zero':>7}")
    print("-" * 80)
    for key, stats in sorted(report["mask_analysis"].items()):
        print(f"{key:<45} {stats['count']:>5} {stats['avg_tokens']:>7.0f} {stats['avg_w05']:>7.1f} {stats['avg_w10']:>7.1f} {stats['pct_zero_mask']:>7}")

print(f"\nVariant distribution:")
for k, v in sorted(report["summary"].get("variant_distribution", {}).items(), key=lambda x: -x[1]):
    print(f"  {v:>5} {k}")

DIAGNOSTIC_EOF

# ── STEP 9: Shutdown ─────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "STEP 9: Shutdown vLLM"
echo "============================================================"

kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "vLLM server stopped"

# ── Final Output Summary ─────────────────────────────────────────────

echo ""
echo "============================================================"
echo "PILOT COMPLETE"
echo "============================================================"
echo ""
echo "Output directory: $PILOT_DIR"
echo ""
echo "Files to review:"
echo "  $DIAG_OUT/pilot_report.json         <- Main diagnostic report (JSON)"
echo "  $DIAG_OUT/step1_complete_fujitsu.log <- Fujitsu completion log"
echo "  $DIAG_OUT/step1_verify.log          <- Verification report"
echo "  $DIAG_OUT/step2_canonicalize.log    <- AgentDojo canonicalization"
echo "  $DIAG_OUT/step3_upsample.log        <- Minority upsampling"
echo "  $DIAG_OUT/step6_final_verify.log    <- Final merged verification"
echo "  $DIAG_OUT/step7_etl_b.log           <- ETL_B pipeline log"
echo ""
echo "  $TRACES_OUT/pilot_merged.jsonl      <- All augmented traces"
echo "  $TRACES_OUT/contrastive_pairs.jsonl  <- Fujitsu contrastive pairs"
echo "  $RENDERS_OUT/pilot_merged.jsonl      <- Tokenized renders"
echo "  $LOSSMASKS_OUT/pilot_merged.jsonl    <- Loss masks"
echo ""
echo "SCP command:"
echo "  scp -r cluster:$PILOT_DIR pilot_augmentation/"
echo ""
echo "After review, run full augmentation:"
echo "  sbatch slurm/pipeline/run_llm_augmentation.sbatch"
