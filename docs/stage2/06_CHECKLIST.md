# 06_CHECKLIST.md — Executable Implementation Checklist

**Objective:** Provide a step-by-step, executable checklist for Stage 2 implementation with concrete commands, file paths, and verification steps.

---

## Quick Reference

```bash
# Clone commands for each phase
# Phase 0: cd $REPO && ./scripts/stage2_phase0_audit.sh
# Phase 1: sbatch slurm/Trillium/trillium_stage2_data_gen.sbatch
# Phase 2: python scripts/cb_data_generation/merge_stage2_data.py --validate
# Phase 3: sbatch slurm/Trillium/trillium_stage2_train.sbatch
# Phase 4: sbatch slurm/Trillium/trillium_stage2_eval.sbatch
```

---

## Phase 0: Audit & Schema Lock (0.5 day)

### 0.1 Verify Existing Data Inventory

```bash
# Check raw data exists
ls -la data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl
ls -la data/agent_dojo/agentdojo-*.jsonl | head -5
ls -la data/tau2_repo/data/tau2/domains/*/tasks.json | head -3

# Count records
wc -l data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl
# Expected: 13246
```

- [ ] B4 data: ≥13,000 records
- [ ] AgentDojo data: ≥3,000 records
- [ ] TAU2 data: tasks.json files exist

### 0.2 Verify Tool Schema Frozen

```bash
# Check schema version
jq '.version' configs/tool_schemas/b4_standard_v1.json
# Expected: "b4_standard_v1"

# Verify tools count
jq '.tools | length' configs/tool_schemas/b4_standard_v1.json
# Expected: ≥7
```

- [ ] Schema version is `b4_standard_v1`
- [ ] Schema has ≥7 tools defined

### 0.3 Audit apply_chat_template Compatibility

```bash
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'Hello'},
    {'role': 'assistant', 'content': '<|python_tag|>{\"name\": \"test\"}<|eom_id|>'}
]

rendered = tokenizer.apply_chat_template(messages, tokenize=False)
print(rendered)
print()
print('✅ Contains python_tag:', '<|python_tag|>' in rendered)
print('✅ Contains eom_id:', '<|eom_id|>' in rendered)
"
```

- [ ] `apply_chat_template` preserves `<|python_tag|>`
- [ ] `apply_chat_template` preserves `<|eom_id|>`

### 0.4 Verify Model Cache

```bash
# On HPC login node
ls -la $SCRATCH/cb_cache/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/
ls -la $SCRATCH/cb_cache/hf/hub/models--mlabonne--Meta-Llama-3.1-8B-Instruct-abliterated/
```

- [ ] Llama-3.1-8B-Instruct cached
- [ ] Abliterated model cached (or add to prefetch)

---

## Phase 1: Data Generation (1.5 days)

### 1.1 Generate Ds (Harmful Set) — GPU Required

```bash
# Submit job
sbatch slurm/Trillium/trillium_mvp_generate_ds.sbatch

# Monitor
squeue -u $USER
tail -f /scratch/memoozd/logs/mvp_ds_gen_*.out

# Verify output
wc -l $SCRATCH/cb_mvp_data/ds_stage1.jsonl
# Expected: ≥500
```

- [ ] Job submitted and completed
- [ ] Output has ≥500 samples
- [ ] Spot-check: `head -1 $SCRATCH/cb_mvp_data/ds_stage1.jsonl | jq .`

### 1.2 Generate Adversarial-Safe Dr (CRITICAL) — GPU Required

```bash
# Create sbatch if not exists
cat > slurm/Trillium/trillium_stage2_adversarial_safe.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=stage2_advsafe
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --account=def-zhijing

# ... (standard setup from existing sbatch files)

python scripts/cb_data_generation/generate_adversarial_safe.py \
    --b4-data data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tool-schema configs/tool_schemas/b4_standard_v1.json \
    --output data/circuit_breakers/retain/adversarial_safe.jsonl \
    --target-n 500 \
    --temperature 0.1 \
    --backend vllm \
    --tensor-parallel 4
EOF

# Submit
sbatch slurm/Trillium/trillium_stage2_adversarial_safe.sbatch

# Verify
wc -l data/circuit_breakers/retain/adversarial_safe.jsonl
# Expected: ≥400
grep -c "is_adversarial_safe.*true" data/circuit_breakers/retain/adversarial_safe.jsonl
```

- [ ] Job completed successfully
- [ ] Output has ≥400 samples with `is_adversarial_safe: true`

### 1.3 Split AgentDojo by Security — CPU

```bash
python scripts/cb_data_generation/ingest_agentdojo_splits.py \
    --input-dir data/agent_dojo/ \
    --output-harmful data/circuit_breakers/harmful/agentdojo_failures.jsonl \
    --output-retain data/circuit_breakers/retain/agentdojo_resisted.jsonl \
    --max-each 500

# Verify
wc -l data/circuit_breakers/harmful/agentdojo_failures.jsonl
wc -l data/circuit_breakers/retain/agentdojo_resisted.jsonl
```

- [ ] `agentdojo_failures.jsonl` created
- [ ] `agentdojo_resisted.jsonl` created

### 1.4 Ingest TAU2 Traces — CPU

```bash
python scripts/cb_data_generation/ingest_tau2_traces.py \
    --tau2-path data/tau2_repo/ \
    --output data/circuit_breakers/retain/tau2_traces.jsonl \
    --target-n 500

# Verify
wc -l data/circuit_breakers/retain/tau2_traces.jsonl
# Expected: ≥300
```

- [ ] `tau2_traces.jsonl` created with ≥300 samples

### 1.5 UltraChat (Deferred)

UltraChat is non-agentic; defer for now. Focus on AgentDojo + TAU2.

### 1.6 Ingest XSTest (Optional) — Network Required

```bash
python scripts/cb_data_generation/ingest_xstest.py \
    --output data/circuit_breakers/retain/xstest_borderline.jsonl \
    --target-n 500

# Verify
wc -l data/circuit_breakers/retain/xstest_borderline.jsonl
```

- [ ] `xstest_borderline.jsonl` created (optional)

---

## Phase 2: Validation & Merge (0.5 day)

### 2.1 Validate All Sources

```bash
# Validate harmful sources
python scripts/cb_data_generation/validate_format.py \
    --data data/circuit_breakers/harmful/ \
    --split harmful \
    --strict

# Validate retain sources
python scripts/cb_data_generation/validate_format.py \
    --data data/circuit_breakers/retain/ \
    --split retain \
    --strict

# Generate validation report
python scripts/cb_data_generation/validate_format.py \
    --data-dir data/circuit_breakers/ \
    --report data/circuit_breakers/stage2/validation_report.json
```

- [ ] Harmful validation: 0 errors
- [ ] Retain validation: 0 errors
- [ ] `validation_report.json` created

### 2.2 Merge into Training File

```bash
mkdir -p data/circuit_breakers/stage2/

python scripts/cb_data_generation/merge_stage2_data.py \
    --harmful-dir data/circuit_breakers/harmful/ \
    --retain-dir data/circuit_breakers/retain/ \
    --output data/circuit_breakers/stage2/train.jsonl \
    --eval-output data/circuit_breakers/stage2/eval.jsonl \
    --eval-fraction 0.1 \
    --dr-ratio 4.5 \
    --validate

# Verify
wc -l data/circuit_breakers/stage2/train.jsonl
wc -l data/circuit_breakers/stage2/eval.jsonl

# Check ratio
python -c "
import json
train = [json.loads(l) for l in open('data/circuit_breakers/stage2/train.jsonl')]
ds = sum(1 for s in train if s['labels']['split'] == 'harmful')
dr = sum(1 for s in train if s['labels']['split'] == 'retain')
print(f'Ds: {ds}, Dr: {dr}, Ratio: {dr/ds:.2f}:1')
"
```

- [ ] `train.jsonl` has ≥5000 samples
- [ ] `eval.jsonl` has ≥400 samples
- [ ] Dr:Ds ratio ≥4:1

### 2.3 Preflight Check

```bash
python scripts/circuit_breakers/preflight_check.py \
    --train-data data/circuit_breakers/stage2/train.jsonl \
    --tool-schema configs/tool_schemas/b4_standard_v1.json \
    --strict
```

- [ ] Preflight check passes
- [ ] No tokenization errors
- [ ] No samples exceed max_seq_length

---

## Phase 3: Training (1 day)

### 3.1 Update Config

```bash
# Verify Stage 2 config exists
grep -A5 "CircuitBreakerConfigStage2" scripts/circuit_breakers/config.py

# Check key parameters
grep "alpha_max" scripts/circuit_breakers/config.py | head -3
# Expected: alpha_max: float = 0.5 (for Stage2)
```

- [ ] `CircuitBreakerConfigStage2` exists in `config.py`
- [ ] `alpha_max = 0.5` (not 10.0)

### 3.2 Create/Update Training Sbatch

```bash
# Create Stage 2 training sbatch
cat > slurm/Trillium/trillium_stage2_train.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=stage2_train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err
#SBATCH --account=def-zhijing

# ... (standard setup)

accelerate launch scripts/train_circuit_breaker.py \
    --preset stage2 \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --data-path data/circuit_breakers/stage2/train.jsonl \
    --output-dir outputs/cb_stage2_adapter \
    --loss-weighting dual \
    --alpha-max 0.5 \
    --cb-target-layers 15 \
    --total-steps 300 \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --learning-rate 3e-5 \
    --max-grad-norm 0.5 \
    --wandb-project "circuit-breakers-stage2"
EOF
```

- [ ] Sbatch file created/updated
- [ ] `alpha-max 0.5` (not 10.0)
- [ ] `cb-target-layers 15` (single layer)

### 3.3 Run Training

```bash
sbatch slurm/Trillium/trillium_stage2_train.sbatch

# Monitor
squeue -u $USER
tail -f /scratch/memoozd/logs/stage2_train_*.out

# Check W&B for loss curves (if available)
```

- [ ] Job submitted
- [ ] No NaN/Inf in loss
- [ ] `cos_sim_mean` decreasing

### 3.4 Post-Training Sanity Check

```bash
python scripts/circuit_breakers/sanity_check.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --adapter-path outputs/cb_stage2_adapter/final \
    --test-prompts scripts/circuit_breakers/test_prompts.txt
```

- [ ] Sanity check passes
- [ ] No collapse patterns detected
- [ ] Output is coherent

---

## Phase 4: Evaluation (0.5 day)

### 4.1 Run Full Evaluation

```bash
sbatch slurm/Trillium/trillium_stage2_eval.sbatch

# Or run manually:
python scripts/circuit_breakers/eval_mvp.py \
    --baseline meta-llama/Llama-3.1-8B-Instruct \
    --cb-model outputs/cb_stage2_adapter/final \
    --eval-data data/circuit_breakers/stage2/eval.jsonl \
    --tool-schema configs/tool_schemas/b4_standard_v1.json \
    --output outputs/cb_stage2_adapter/eval_results.json
```

- [ ] Evaluation completed
- [ ] Results saved to `eval_results.json`

### 4.2 Check Gates

```bash
# Parse gate results
jq '.gates' outputs/cb_stage2_adapter/eval_results.json

# Check blocker gates
jq '.gates | to_entries | map(select(.value.severity == "BLOCKER")) | map({(.key): .value.passed})' \
    outputs/cb_stage2_adapter/eval_results.json
```

**Required Gate Results:**

| Gate | Target | Your Result |
|------|--------|-------------|
| `coherent_output` | ≥99% | [ ] ___% |
| `correct_behavior_rate` | ≥70% | [ ] ___% |
| `no_tool_call_rate` | ≤20% | [ ] ___% |
| `capability_retention` | ≥90% | [ ] ___% |

- [ ] All blocker gates pass
- [ ] Results documented

### 4.3 Generate Gate Report

```bash
python -c "
import json
results = json.load(open('outputs/cb_stage2_adapter/eval_results.json'))

print('=== STAGE 2 GATE REPORT ===')
print()
blockers_passed = 0
blockers_total = 0
for gate, data in results['gates'].items():
    if data['severity'] == 'BLOCKER':
        blockers_total += 1
        status = '✅' if data['passed'] else '❌'
        if data['passed']:
            blockers_passed += 1
        print(f'{status} {gate}: {data.get(\"value\", \"N/A\")} (threshold: {data[\"threshold\"]})')

print()
print(f'BLOCKERS: {blockers_passed}/{blockers_total}')
print(f'OVERALL: {\"PASS\" if blockers_passed == blockers_total else \"FAIL\"}')
"
```

- [ ] Gate report generated
- [ ] Overall status: PASS

---

## Rollback Plan

If Stage 2 fails gates:

### If `correct_behavior_rate` < 70%:
1. Increase adversarial-safe samples (target 800)
2. Increase their weight to 2.5
3. Re-train

### If `no_tool_call_rate` > 20%:
1. Check if adversarial-safe samples are properly labeled
2. Reduce `alpha_max` to 0.3
3. Re-train

### If model collapses (gibberish):
1. Reduce `alpha_max` to 0.1
2. Use single layer (already at 15)
3. Reduce `total_steps` to 150
4. Re-train

---

## Final Checklist Summary

```
PHASE 0: AUDIT
[ ] Raw data verified (B4, AgentDojo, TAU2)
[ ] Tool schema frozen at v1
[ ] apply_chat_template compatibility confirmed
[ ] Model cache verified

PHASE 1: DATA GENERATION
[ ] Ds generated (≥500 samples)
[ ] Adversarial-safe Dr generated (≥400 samples) ← CRITICAL
[ ] AgentDojo split complete
[ ] TAU2 traces ingested (≥300 samples)
[ ] UltraChat ingested (~2000 samples)
[ ] XSTest ingested (optional)

PHASE 2: VALIDATION & MERGE
[ ] All sources pass validation (0 errors)
[ ] Training file merged (≥5000 samples)
[ ] Dr:Ds ratio ≥4:1
[ ] Preflight check passes

PHASE 3: TRAINING
[ ] Stage 2 config applied (alpha=0.5)
[ ] Training completes without NaN
[ ] cos_sim_mean decreases
[ ] Sanity check passes (coherent output)

PHASE 4: EVALUATION
[ ] All blocker gates pass
[ ] correct_behavior_rate ≥70%
[ ] no_tool_call_rate ≤20%
[ ] Gate report generated

OVERALL: [ ] STAGE 2 COMPLETE
```
