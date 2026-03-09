# Plan: AD-Focused Circuit Breaker Training

## Problem

CB training across 10+ configurations shows the same result: **AgentDojo is completely immune** (~81% resist, ~19% malicious, unchanged by any CB). The best Fujitsu result (−21.4% ASR) is just shallow tool-name memorization (search_web → retrieve_multimodal_docs), not real representation rerouting.

**Root cause:** Fujitsu data (7661 harmful) overwhelms AD data (~2500 harmful) 3:1. With only 1,600 samples seen per 200-step run, the model sees ~400 AD samples total and learns nothing about AD patterns.

## Goal

Get the CB to actually change AgentDojo harmful behavior (reduce malicious_tool_call_rate from 19%).

---

## Step 1: AD-Only Training Run

**What:** Remove Fujitsu entirely. Train only on AgentDojo data.

**Why:** Isolate whether the CB can learn AD patterns at all, without Fujitsu noise.

**How:** Modify `sweep_completed_v1.sbatch` to accept a `DATASET` env var:

```bash
DATASET="${DATASET:-all}"  # "all", "ad_only", "fujitsu_only"
```

When `DATASET=ad_only`:
- `--harmful-renders` gets ONLY `agentdojo_renders_harmful.jsonl`
- `--harmful-lossmasks` gets ONLY `agentdojo_lossmasks_harmful.jsonl`
- `--benign-renders` gets ONLY `agentdojo_renders_benign.jsonl`
- `--benign-lossmasks` gets ONLY `agentdojo_lossmasks_benign.jsonl`
- Skip `--contrastive-pairs` (pairs are AD-specific anyway, but verify)
- Keep all other settings identical

**Files to modify:**
- `slurm/pipeline/sweep_completed_v1.sbatch` — lines 344-353 (the `--harmful-renders`/`--benign-renders` args). Add conditional logic based on `$DATASET`.

**Data sizes (AD only):**
- Harmful: ~2513 samples (attack_succeeded=true)
- Benign: ~2894 samples (no attack present)
- Resisted: 250 (attack present but failed) — currently goes to benign via `split_agentdojo.py` default behavior

**Submit:**
```bash
DATASET=ad_only LOSS_MODE=per_token_cb DISTANCE=dl2rc MARGIN_FREE=false TOTAL_STEPS=500 sbatch slurm/pipeline/sweep_completed_v1.sbatch
```

---

## Step 2: Increase Steps (Data Coverage)

**What:** Run 500 steps instead of 200.

**Why:** With ~2500 AD harmful samples and batch=4, 200 steps only sees 800 samples (32% of data). At 500 steps, we see 2000 samples (80%). The model needs to see the data.

**How:** `TOTAL_STEPS=500` env var (already supported).

---

## Step 3: Reduce MAX_SEQ_LENGTH to 2048

**What:** Match the old working sweep's sequence length.

**Why:**
- Old sweep (d086ec9) used 2048, not 4096.
- At 4096, the injection tokens are a smaller fraction of the sequence (~3.5% with injection_aware, ~75% with cb_full_sequence). At 2048, signal density doubles.
- Saves memory, allowing larger batch or separate frozen model.
- The `truncate_to_injection_window.py` script already shortens AD traces to the injection decision window — most truncated traces are well under 2048 tokens.

**Files to modify:**
- `slurm/pipeline/sweep_completed_v1.sbatch` — line 94: change `MAX_SEQ_LENGTH=4096` to `MAX_SEQ_LENGTH=2048`.

---

## Step 4: Verify AD Data Quality

**What:** Before training, inspect what the AD harmful training data actually looks like.

**Why:** We need to verify:
1. The truncated AD traces still contain the injection content (not truncated away)
2. The lossmasks (cb_full_sequence) actually cover meaningful tokens
3. The "harmful" AD samples truly contain malicious tool calls (attack_succeeded=true)

**How:** Write a quick diagnostic script or inline bash to:
```bash
# Count categories in truncated AD data
python3 -c "
import json
from collections import Counter
cats = Counter()
for line in open('$SHARED_DATA/split_agentdojo/agentdojo_traces_harmful.jsonl'):
    t = json.loads(line)
    cats[t.get('labels',{}).get('category','')] += 1
    # Check if injection_char_span exists
print(cats)
"

# Check how many harmful AD traces have injection_char_span
python3 -c "
import json
total = span_count = 0
for line in open('$AGENTDOJO_SRC'):
    t = json.loads(line)
    if t.get('labels',{}).get('category') == 'harmful':
        total += 1
        if (t.get('signal_hints') or {}).get('injection_char_span'):
            span_count += 1
print(f'Harmful with injection_char_span: {span_count}/{total}')
"
```

**Relevant code:**
- `scripts/truncate_to_injection_window.py` (lines 26-59) — truncation logic. Harmful traces without `injection_char_span` are DROPPED (line 42-44). Verify none are lost.
- `scripts/split_agentdojo.py` (lines 30-82) — split logic. CB set = `attack_succeeded=True`.

---

## Step 5: Training/Eval Data Alignment Check

**What:** Verify that the TRAINING data and EVAL data use compatible trace formats.

**Why:** Training uses `agentdojo_truncated.jsonl` (truncated to injection window). Eval uses `agentdojo_augmented.jsonl` (full-length, non-truncated). If the truncation changes the input enough that the model doesn't recognize the pattern at eval time, the CB effect won't transfer.

**Concrete concern:** The model trains on short traces (truncated to first post-injection assistant response). At eval time, it sees the full multi-turn conversation. The representations at layers 10/20 might look completely different because the context is different.

**How:** Either:
- (a) Eval on truncated data too (match training), or
- (b) Train on non-truncated data (match eval)

Option (b) is better long-term but increases seq length. Option (a) is a quick diagnostic.

**Files to modify for option (a):**
- `slurm/pipeline/sweep_completed_v1.sbatch` — line 461: change `AGENTDOJO_EVAL_SRC` from `agentdojo_augmented.jsonl` to `$AGENTDOJO_SRC` (truncated).

---

## Step 6: Resisted Traces as Training Signal

**What:** Add the 250 "resisted" AD traces (attack_present but attack_succeeded=false) to harmful training data, not benign.

**Why:** These traces contain injections but the model resisted — they show what correct behavior looks like WHEN an injection is present. Currently they go to benign (default `split_agentdojo.py` behavior). But for CB training, they could be valuable as harmful examples where the loss_mask targets the injection tokens but the model's output was already correct. This teaches the CB "here's what an injection looks like, and this is the right representation to have."

**How:** `split_agentdojo.py` already supports `--resisted-handling separate`. Could add a new mode or just reclassify in the sweep script.

---

## Recommended Run Order

```bash
# Run 1: AD-only, more steps (MOST INFORMATIVE — does CB learn AD at all?)
DATASET=ad_only LOSS_MODE=per_token_cb DISTANCE=dl2rc MARGIN_FREE=false TOTAL_STEPS=500 sbatch ...

# Run 2: AD-only, seq=2048 (does shorter context help?)
DATASET=ad_only LOSS_MODE=per_token_cb DISTANCE=dl2rc MARGIN_FREE=false TOTAL_STEPS=500 MAX_SEQ_LENGTH=2048 sbatch ...

# Run 3: All data, balanced (Fujitsu downsampled to ~2500, equal to AD)
DATASET=balanced LOSS_MODE=per_token_cb DISTANCE=dl2rc MARGIN_FREE=false TOTAL_STEPS=500 sbatch ...
```

Run 1 answers the fundamental question: **can the CB learn AD patterns at all?** If yes, then Run 3 tests balancing. If no, then the problem is deeper (wrong layers, wrong loss, or AD injections are structurally different from what CB can detect).

---

## Key Files Reference

| File | What | Lines |
|------|------|-------|
| `slurm/pipeline/sweep_completed_v1.sbatch` | Main sweep script | 344-353 (training data args), 94 (MAX_SEQ_LENGTH), 461 (eval data) |
| `scripts/split_agentdojo.py` | AD split logic | 30-82 (TraceInfo, label parsing), 239-240 (split semantics) |
| `scripts/truncate_to_injection_window.py` | AD truncation | 26-59 (truncation logic, drops harmful without span) |
| `src/training/train_schema.py` | Dataset/training | 547-580 (_load_samples, mixed mode) |
| `src/evaluation/eval.py` | Eval metrics | 2230-2254 (harmful_resistance calc), 1340-1368 (tool_flip outcome classification) |
| `src/training/losses.py` | Loss functions | 636+ (per_token_cb_loss — the working loss) |
| `src/training/trainer.py` | Forward pass | 1358-1420 (hooks path), 1434-1490 (hidden_states path) |
