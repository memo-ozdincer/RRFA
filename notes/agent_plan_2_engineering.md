# Agent Plan 2: Engineering — Training Pipeline & Data Changes

**FIRST**: Read `notes/shared_context.md` — it contains ALL raw data, current configs, training results, and strategic context.

**Role**: Implement all code changes needed to maximize training signal clarity. Modify training configs, data augmentation, sbatch scripts. Produce runnable configs.

**Worktree**: YES — use `isolation: "worktree"`. Changes touch:
- `slurm/pipeline/sweep_ad_focused.sbatch` (new configs)
- `scripts/amplify_injections.py` (NEW — injection amplification)
- `scripts/generate_hard_negatives.py` (NEW — if time permits)
- `src/training/losses.py` (only if implementing importance-weighted loss — coordinate with Agent 3)

**Does NOT touch**: `paper/`, `notes/experiment_log.tex`, `src/evaluation/eval.py`

---

## Context: What We Know (Updated by Agent 1 Analysis)

**READ FIRST**: `notes/analysis_report.md` — comprehensive analysis with tables, training curve phases, and Pareto frontier.

### The Core Problem
CB rerouting on AgentDojo produces gibberish instead of clean refusals, and destroys benign tool-calling capability. The probe diagnostic shows:
- **Push direction AUC = 0.96** (comply vs refuse at last token) — CB CAN learn which direction to push
- **Selectivity AUC = 0.83** (injection-context vs clean-context at last token) — CB struggles to know WHEN to push
- **Result**: aggressive training pushes ALL tool calls (harmful + benign) away, destroying capability

### Full Sweep 10 Results (3 runs completed)

| Config | Steps | Fujitsu Δ | AD Trunc mal | Benign correct | Benign no_tool | Gibberish (H/B) |
|--------|-------|-----------|-------------|----------------|----------------|-----------------|
| ad_1000_v2 | 1000 | -14% | 30% (-37%) | 41% | 15% | 25/23 |
| ad_3000_v2 | 3000 | -33% | 22% (-45%) | 40% | 52% | 4/10 |
| **ad_5000_v2** | **5000** | **-58%** | **24% (-44%)** | **72%** | **5%** | **10/2** |
| ad_300_v2 *(partial)* | 300 | -69% | ??? | ??? | ??? | — |

**ad_5000_v2 is Pareto-optimal** — dominates both other configs on benign (72% vs 40-41%) and Fujitsu (-58%), with similar AD reduction.

### Three-Phase Training Dynamic (KEY FINDING)
1. **Phase 1 (steps 0-40)**: Harmful margin exceeded in ~30 steps. Push "locks in."
2. **Phase 2 (40 to ~68-72% of training)**: Only benign retain + KL active. Benign anchoring strengthens.
3. **Phase 3 (~68-72% to end)**: Harmful reps drift back within margin, re-activating push. Benign destroyed.

**Critical insight**: More Phase 2 time = stronger benign anchor. ad_5000_v2 had 3470 Phase 2 steps (11.1 epochs) vs ad_3000_v2's 2010 (6.4 epochs), explaining the 72% vs 40% benign gap.

### Output Quality by Step Count
- **1000 steps**: `<|python_tag|>` repetition, "assistantassistant..." spam
- **3000 steps**: Comma/bracket gibberish (`,,,,,{",,,[",,["`)
- **5000 steps**: Semi-coherent garbled JSON (`{"name": "Re: [email"]`)
- **All steps (Fujitsu)**: Clean correct tool calls, zero gibberish, zero regressions

### Current Sweep 11 Configs (already in sbatch)
```
ad_300_v2, ad_500_v2, ad_750_v2, ad_1000_v2, ad_1500_v2
```
All use: decay_mult=2.0, margin_h=20.0, margin_b=3.0, lr_schedule=constant

### Agent 1 Recommendations for Sweep 11 Configs
1. **Optimal step range: 750-1500** to avoid Phase 3 — but note ad_5000_v2 was best overall
2. **Re-add Fujitsu data** (cap 500-1000 per epoch) — Fujitsu outputs are clean, may improve selectivity
3. **Add refusal traces** — gives CB a directed target, may reduce gibberish
4. **Test alpha_max=5.0** — slower Phase 1 push, less Phase 3 destructiveness
5. **Test margin_harmful=40.0** — higher ceiling may delay Phase 3 onset
6. **Save intermediate checkpoints** — evaluate at 300, 500, 750, 1000 to find sweet spot

---

## Task 1: Add Balanced (Fujitsu + AD) Configs

**Why**: Fujitsu produces CLEAN CB outputs (correct tool calls, not gibberish). Adding Fujitsu provides:
1. Structural injection signal (HTML comments — cleaner than AD's semantic injections)
2. More diverse benign data (Fujitsu DR traces anchor tool-calling capability)
3. The model learns both injection STYLES, improving generalization

**Add to `slurm/pipeline/sweep_ad_focused.sbatch`** in the case block:

```bash
# Sweep 11b — balanced (Fujitsu + AD) with v2 fixes
ad_500_bal)
  DATASET=balanced; TOTAL_STEPS=500;  RESISTED=retain; FUJITSU_CAP=2500
  _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant ;;

ad_750_bal)
  DATASET=balanced; TOTAL_STEPS=750;  RESISTED=retain; FUJITSU_CAP=2500
  _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant ;;

ad_1000_bal)
  DATASET=balanced; TOTAL_STEPS=1000; RESISTED=retain; FUJITSU_CAP=2500
  _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant ;;
```

**Notes**:
- FUJITSU_CAP=2500 gives ~2500 harmful + ~2500 benign Fujitsu traces, balanced against ~2500 AD harmful + ~3100 AD benign
- The existing sbatch already handles balanced mode (Step 2b: Fujitsu downsampling). Just need the case entries.
- Update the header comments and the valid configs echo line

## Task 2: Add Refusal + Balanced Configs

Combine refusal data with balanced dataset:

```bash
ad_750_bal_r)
  DATASET=balanced; TOTAL_STEPS=750;  RESISTED=retain; FUJITSU_CAP=2500
  _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant; _USE_REFUSAL=true ;;
```

This adds ~1225 refusal traces to the benign set, giving the retain loss a directed target.

## Task 3: Injection Amplification Script

**Why**: The probe showed selectivity AUC = 0.83 (WEAK). If we make injections MORE obvious in training data, we amplify the context-level signal that CB uses for selectivity.

**Create `scripts/amplify_injections.py`**:

```python
#!/usr/bin/env python3
"""
Amplify injection signal in harmful traces by repeating injection text.

Takes truncated traces, finds the injection span in tool outputs,
and repeats it 2-3x to make the injection MORE prominent. This increases
the context-level signal (probe AUC 0.83 → hopefully 0.90+) that CB uses
to distinguish injection-context from clean-context.

Usage:
    python scripts/amplify_injections.py \
        --input data/traces/agentdojo_truncated.jsonl \
        --output data/traces/agentdojo_amplified.jsonl \
        --repeats 2 \
        --seed 42
"""
```

**Logic**:
1. Read each trace from the truncated JSONL
2. For each message with role="tool" (tool output), search for injection markers:
   - `<INFORMATION>...</INFORMATION>` tags (AgentDojo style)
   - `<!--...-->` (Fujitsu HTML comment style)
   - Any text between common injection delimiters
3. If found, repeat the injection text N times within the tool output
4. Write the modified trace (keep all metadata, labels, etc.)
5. Only modify harmful traces (category=harmful). Pass benign through unchanged.

**Key implementation details**:
- The injection text is typically inside `<INFORMATION>` tags in AgentDojo traces
- Read the traces to understand the format first: `data/traces/agentdojo_truncated.jsonl` (or the shared copy on cluster at `$SHARED_DATA/agentdojo_truncated.jsonl`)
- Preserve the trace ID but add `_amp{N}` suffix
- Keep original source metadata

**Reference files to understand trace format**:
- `scripts/truncate_to_injection_window.py` — shows how traces are structured
- `src/schemas/trace.py` — TraceSource dataclass
- `data/traces/` — sample traces (may need to read from cluster)

## Task 4: Integrate Amplification into sbatch

Add amplification as an optional step in the sbatch pipeline:

1. Add a new variable: `AMPLIFY_INJECTIONS="${_AMPLIFY:-false}"` and `AMPLIFY_REPEATS="${_AMP_REPEATS:-2}"`
2. In Step 0 (data preparation), after truncation:
   ```bash
   if [[ "$AMPLIFY_INJECTIONS" == "true" ]]; then
       AMPLIFIED_SRC="$SHARED_DATA/agentdojo_amplified_${AMPLIFY_REPEATS}x.jsonl"
       if [[ -f "$AMPLIFIED_SRC" ]] && [[ -s "$AMPLIFIED_SRC" ]]; then
           echo "  Amplified traces: REUSING existing"
       else
           python "$REPO_DIR/scripts/amplify_injections.py" \
               --input "$AGENTDOJO_SRC" --output "$AMPLIFIED_SRC" \
               --repeats "$AMPLIFY_REPEATS" --seed 42
       fi
       AGENTDOJO_SRC="$AMPLIFIED_SRC"  # Use amplified for all downstream
   fi
   ```
3. Add amplified configs:
   ```bash
   ad_750_amp)
     DATASET=ad_only; TOTAL_STEPS=750; RESISTED=retain; FUJITSU_CAP=0
     _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant; _AMPLIFY=true ;;

   ad_750_bal_amp)
     DATASET=balanced; TOTAL_STEPS=750; RESISTED=retain; FUJITSU_CAP=2500
     _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant; _AMPLIFY=true ;;
   ```

## Task 5: Fix Stale Refusal Cache

The refusal traces on cluster had a bug (wrong source field). Need to regenerate.

Add to the sbatch a cache-busting check:
```bash
# In Step 0, before "Generate synthetic refusal traces":
# Force regeneration if the source field is wrong (old format had 'original_source')
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
```

## Task 6: Alpha/Margin Tuning (Agent 1 Recommends BOTH)

Based on the analysis report:
- Margin 20.0 was exceeded in ~30 steps across ALL runs
- Phase 3 onset at ~68-72% of training when harmful reps drift back within margin
- ad_5000_v2 (α=10.0) was Pareto-optimal — so α=10 isn't necessarily wrong
- But testing α=5.0 and margin=40.0 may reveal better operating points

**Implement both options**:

**Option A: Lower alpha (α_max=5.0)** — slower Phase 1 push, reduced Phase 3 destructiveness
```bash
ad_1000_a5)
  DATASET=ad_only; TOTAL_STEPS=1000; RESISTED=retain; FUJITSU_CAP=0
  _DECAY_MULT=2.0; _MARGIN_H=20.0; _MARGIN_B=3.0; _LR_SCHED=constant; _ALPHA=5.0 ;;
```

**Option B: Higher margin (margin_h=40.0)** — Phase 3 delayed because larger distance needed to re-enter active zone
```bash
ad_1000_m40)
  DATASET=ad_only; TOTAL_STEPS=1000; RESISTED=retain; FUJITSU_CAP=0
  _DECAY_MULT=2.0; _MARGIN_H=40.0; _MARGIN_B=5.0; _LR_SCHED=constant ;;
```

**Agent 1 notes**: Keep α=10.0 as the control (don't change everything at once). The Pareto analysis shows ad_5000_v2 approaches the theoretical benign ceiling (72% actual vs 83% theoretical), so any improvement needs to come from better selectivity, not just hyperparameter tuning.

---

## File References

### Must Read Before Writing
- `slurm/pipeline/sweep_ad_focused.sbatch` — current full sbatch (you're modifying this)
- `scripts/truncate_to_injection_window.py` — trace format, injection detection
- `scripts/generate_refusal_traces.py` — refusal trace generation pattern
- `src/schemas/trace.py` — TraceSource dataclass (for source field format)

### Reference Only
- `src/training/trainer.py:1880-1920` — schedule implementation (understand cs, cr)
- `src/training/losses.py` — per_token_cb_loss (understand margin/hinge)
- `notes/experiment_log.tex` — full history

### DO NOT MODIFY (Agent 3's territory)
- `src/training/losses.py` (Agent 3 may add importance-weighted loss)
- `paper/` (Agent 3's domain)
- `notes/experiment_log.tex` (Agent 3's domain)

---

## Deliverables

1. Modified `slurm/pipeline/sweep_ad_focused.sbatch` with new configs
2. New `scripts/amplify_injections.py`
3. Summary of all runnable configs with expected command:
   ```bash
   # AD-only step sweep
   for c in ad_300_v2 ad_500_v2 ad_750_v2 ad_1000_v2; do
     sbatch --export=ALL,CONFIG=$c slurm/pipeline/sweep_ad_focused.sbatch
   done

   # Balanced (Fujitsu + AD)
   for c in ad_500_bal ad_750_bal ad_1000_bal; do
     sbatch --export=ALL,CONFIG=$c slurm/pipeline/sweep_ad_focused.sbatch
   done

   # With amplified injections
   for c in ad_750_amp ad_750_bal_amp; do
     sbatch --export=ALL,CONFIG=$c slurm/pipeline/sweep_ad_focused.sbatch
   done

   # With refusal data
   for c in ad_750_bal_r; do
     sbatch --export=ALL,CONFIG=$c slurm/pipeline/sweep_ad_focused.sbatch
   done
   ```
4. Brief description of each config's hypothesis in a comment block
