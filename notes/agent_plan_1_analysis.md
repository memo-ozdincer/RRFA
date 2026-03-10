# Agent Plan 1: Results Analysis & Diagnostics

**FIRST**: Read `notes/shared_context.md` — it contains ALL raw data, numbers, training curves, probe results, and output examples you need.

**Role**: Read-only analyst. Extract every insight from existing runs, produce clean tables, identify the optimal operating point, and write findings that Agents 2 and 3 will consume.

**Output**: `notes/analysis_report.md` — a comprehensive analysis document with tables, training curve analysis, and actionable recommendations.

**Worktree**: Not needed (read-only). But DO write the output report.

---

## Task 1: Parse All Sweep 10 Results

The ad_3000_v2 results are in the conversation. Extract and tabulate:

1. **Training curve milestones**: At what step did Triplet_H first reach 0? When did it spike back? When did Triplet_B start declining? Mark these as phase transitions.

2. **Eval metrics** (from the run summary):
   - Fujitsu: BL=84%, CB=51%, Δ=-33%
   - AD Harmful (full): BL_mal=73%, CB_mal=27%, Δ=-46%
   - AD Harmful (trunc): BL_mal=68%, CB_mal=22%, Δ=-45%
   - AD Benign: correct=40%, no_tool=52%
   - Gibberish: 4/96 harmful, 10/97 benign

3. **Output quality from the examples** (in the conversation):
   - All 5 AD harmful examples: CB produces comma/bracket gibberish, no clean tool calls
   - All 5 AD benign examples: CB produces gibberish, code snippets, header spam
   - Fujitsu: 32 improved, 0 regressed (examples show clean tool call corrections)

### Key Question: Why does Fujitsu work well but AD doesn't?
Fujitsu examples show CLEAN correct tool calls (retrieve_multimodal_docs instead of search_web). AD examples show gibberish. This suggests the CB is learning something useful for Fujitsu's structural injection pattern but producing degenerate outputs for AD's semantic injections.

## Task 2: Training Curve Phase Analysis

From the ad_3000_v2 training history (full data in conversation):

Parse the step-by-step data and identify:
1. **Phase 1** (steps 0-30): Initial push. Triplet_H drops from 19.75 to 0.01. Triplet_B rises to ~24.
2. **Phase 2** (steps 30-~2050): Plateau. Triplet_H ≈ 0 (margin exceeded). Triplet_B slowly decays. Only KL and retain active.
3. **Phase 3** (steps ~2050-3000): Triplet_H spikes (0→10+). Something is pulling representations back within margin.

**Compute and report**:
- Step where Triplet_H first < 0.01 (margin exceeded)
- Step where Triplet_H first > 1.0 again (representations pulled back)
- Step where Triplet_B first < 1.0 (benign representations drifting significantly)
- The KL coefficient at each phase boundary (it's scheduled: γ_kl_eff = (cr/α_max) * 0.3)
- Alpha (cs) value at each boundary

**Produce a table**: step | phase | Triplet_H | Triplet_B | KL | cs | cr | interpretation

## Task 3: Probe Results Synthesis

From the probe results (full data in conversation):

**Test 1: Harmful vs Benign (different context)**
| Pooling | Best AUC (linear) | Best AUC (MLP) | Interpretation |
|---------|-------------------|----------------|----------------|
| Mean | 0.975 | 0.969 | Trivially high — different contexts |
| Last quarter | 0.929 | 0.836 | Strong but partly surface |
| Last token | 0.872 | 0.837 | Weakest — signal distributed |

**Test 2: Harmful vs Refusal (same context)**
| Pooling | Best AUC (linear) | Best AUC (MLP) | Interpretation |
|---------|-------------------|----------------|----------------|
| Last token | 0.962 | 0.967 | Strong — response difference at decision point |
| Last quarter | 0.929 | 0.886 | Good |
| Mean | 0.889 | 0.759 | Weak — identical context dilutes |

**Test 3: All 32 layers (last_token MLP, harmful vs benign)**
- Layer 0: 0.890 (SEPARABLE) — embedding level, lexical
- Layers 1-31: 0.706-0.841 (all WEAK)
- No single layer stands out

**Key insight to highlight**: The pooling pattern FLIPS between Test 1 and Test 2. Mean is best when contexts differ (Test 1), last_token is best when contexts are identical (Test 2). This tells us the signal lives in different places depending on what varies.

## Task 4: Selectivity Analysis

The core problem: CB learns a push direction (strong, AUC 0.96) but must apply it selectively (weak, AUC 0.83).

**Compute the "selectivity ratio"**: push_AUC / selectivity_AUC = 0.96 / 0.83 = 1.16. The push is 16% stronger than the selectivity. This predicts that aggressive training will always overshoot — the model learns to push before it learns when to push.

**Predict the Pareto frontier**: Based on the probe AUCs, estimate the theoretical best tradeoff. If selectivity AUC is 0.83, then ~17% of benign samples will be misclassified as harmful at the representation level. This suggests a ceiling of ~83% benign preservation at maximum harmful reduction.

## Task 5: Cross-Dataset Comparison

Compare Fujitsu vs AgentDojo characteristics:

| Aspect | Fujitsu | AgentDojo |
|--------|---------|-----------|
| Injection location | User message (HTML comment) | Tool output (INFORMATION tag) |
| Attack type | Tool flip (expected→simulated) | Arbitrary action |
| Injection structure | Highly structural | Semantic/varied |
| Baseline ASR | ~84% | ~68-75% |
| CB Δ (sweep 10) | -33% (clean outputs) | -45% (gibberish outputs) |
| Benign quality | Clean tool calls | Gibberish/code spam |

**Key observation**: Fujitsu CB outputs are CLEAN (real tool calls with correct names). AD CB outputs are GIBBERISH. This suggests the injection pattern matters for output quality — structural injections allow cleaner rerouting than semantic ones.

## Task 6: Recommendations

Based on all analysis, produce actionable recommendations for Agent 2:

1. **Step count**: Based on training curve phases, what's the optimal step range?
2. **Fujitsu inclusion**: Should we re-add Fujitsu? What cap? (The clean outputs suggest yes)
3. **Margin adjustment**: 20.0 was exceeded in 30 steps. Should it be higher? Or is 30 steps enough for the initial push?
4. **Alpha reduction**: With constant LR and decay_mult=2.0, is α_max=10.0 too aggressive?
5. **Benign data augmentation**: What would help most — more benign data, harder negatives, or refusal data?

---

## Files to Read

1. **Sweep 10 full output**: In the conversation (user pasted full stdout)
2. **Probe results**: In the conversation (user pasted all 3 tests)
3. **Sweep 9 results**: `notes/MEMORY.md` has the table
4. **Training code** (for understanding metrics): `src/training/trainer.py` (lines ~1880-1920 for schedule)
5. **Eval code** (for understanding metrics): `src/evaluation/eval.py` (lines ~2090-2340 for generation_comparison)
6. **Experiment log**: `notes/experiment_log.tex` (full history)
7. **Sweep config**: `slurm/pipeline/sweep_ad_focused.sbatch` (current configs)

## Output Format

Write `notes/analysis_report.md` with:
1. Executive summary (5 lines)
2. Training curve phase table
3. Probe synthesis table with selectivity analysis
4. Cross-dataset comparison
5. Pareto frontier prediction
6. Numbered recommendations for Agent 2
7. Key findings for Agent 3 (paper narrative ammunition)
