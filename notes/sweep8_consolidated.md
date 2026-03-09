# Sweep 8 Consolidated Notes — Mar 8, 2026

## Part 1: What We Tried, What Failed, and Why

### 1.1 The Problem Statement

Circuit Breaker (CB) training on tool-calling LLMs. The model should:
- **Reroute** harmful representations (injection-attacked tool calls) away from frozen baseline
- **Retain** benign representations (normal tool calls) close to frozen baseline

**Core difficulty**: harmful and benign traces look almost identical (both are tool-calling, differ only by ~144 injection tokens). This is fundamentally different from the original CB paper where harmful = bomb-making, benign = general chat.

---

### 1.2 Timeline of What Was Tried

#### Sweeps 1–4 (Feb 28 – Mar 4): Complex Loss Exploration

**What was tried:**
- `triplet_full` loss with cluster centers, cross-terms, margins
- 5 pooling presets × 2 γ_kl grid
- Layer depth sweep: {10,20} vs {20-31}
- `legacy_cb` (paper-style but broken gradient)
- Various distance functions (dcos, dl2rc, dl2sq)

**Results:**
- Best ASR: 60% (down from 82% baseline) — partial success
- All configs had 0% benign_correct (model destroyed benign ability)
- Pooling had a bug (attention_mask used for both mask and pool)
- Layers 10+20 >> 20-31 (14pp improvement)
- `legacy_cb` gradient ≈ 0 with LoRA (cosine gradient vanishes when a ≈ b at init)

**Key learning:** Layer choice matters. Layers 10+20 match the paper.

#### Sweep 5 (Mar 5): injection_aware Breakthrough

**What was tried:**
- `injection_aware` masking policy: only mask ~144 injection tokens (not full sequence)
- Contrastive pairs (harmful/benign twins from same trace)

**Results:**
- **33.7% ASR** (−27.5pp from 60%) — biggest single improvement
- Contrastive pairs alone didn't help
- injection_aware = the key breakthrough

**But**: Benign data was accidentally filtered out (zero-mask benign dropped), so the low ASR may be partly an artifact of weak retain signal.

#### Sweep 7 (Mar 7–8): Paper Re-implementation (FAILED)

**Motivation:** Too much complexity (8 loss modes, 5 distances, 11 masking policies). Go back to paper's exact algorithm.

**What was tried:**
- `original_rr` loss: ReLU(cos_sim) + L2 retain (paper Algorithm 1)
- `per_token_cb`: per-token distance with margins
- `simplified_pooled`: pooled distance without cluster centers
- Paper's coefficient schedule: cs:0→α/2, cr:α→α/2

**Results:** COMPLETE FAILURE — zero eval effect across all configs.

**Root cause (3 compounding regressions from old working sweep d086ec9):**

| Setting | Old Working (d086ec9) | Sweep 7 | Impact |
|---------|----------------------|---------|--------|
| Schedule | cs:α→0, cr:0→α (reroute HIGH first) | cs:0→α/2, cr:α→α/2 (reroute LOW first) | CRITICAL — zero reroute at init |
| LoRA layers | All 32 | 0–20 only | CRITICAL — frozen 21-31 block propagation |
| LR | 5e-5 | 1e-5 | 5× weaker signal |
| LoRA alpha | 32 (scale 2.0) | 16 (scale 1.0) | 2× weaker LoRA |
| **Combined** | | | **~10× weaker effective signal** |

#### Sweep 8a (Mar 8): Restored Settings + injection_aware (FAILED)

**What was changed:**
- Schedule reverted: cs:α→0, cr:0→α
- LoRA: all 32 layers
- LR=5e-5, LoRA alpha=32, batch=4×2=8 effective
- KL loss chunked (512 tokens) to prevent OOM

**Infrastructure bugs found and fixed:**
1. Missing Fujitsu benign data (DS file only, no DR) → use DS/DR separately
2. DR lossmasks all-zero under injection_aware → switch to cb_full_sequence
3. KL OOM → chunk along seq dim
4. ZeroDivisionError on empty benign → defensive guard
5. Race condition in shared data → check all 4 split files
6. margin_free config field missing → added to dataclass
7. alpha_benign using wrong coefficient (cs not cr) → fixed at 5 call sites

**Results (2 runs with injection_aware, both failed):**
- original_rr: cos_sim=1.0000 (did not move), Fujitsu +4% WORSE, AD unchanged
- per_token_cb + dl2rc: Fujitsu −9.2%, AD unchanged (marginal at best)

**Why injection_aware failed:**
- Harmful mask: only 330/16384 tokens (2%) active → gradient too sparse
- Benign DR mask: 0% coverage (no injection tokens in benign) → retain loss blind to Fujitsu benign
- cos_sim gradient vanishes at LoRA init with such sparse signal

#### Sweep 8b (Mar 8): cb_full_sequence — All Results

Switched to `cb_full_sequence` policy: masks ALL tokens → 75% avg coverage for all files (harmful and benign), zero files with zero mask.

**Results table:**

| Run | Loss | Distance | Margin Free | Fuj Δ ASR | Fuj Imp/Reg | AD Δ | Conclusion |
|-----|------|----------|-------------|-----------|-------------|------|------------|
| per_token_cb | dl2rc | **false** | **−21.4%** (86→64%) | **21/0** | 0 | **Best of sweep, but shallow** |
| per_token_cb | dl2rc | true(?) | −1.0% (84→83%) | ~0/0 | 0 | Margins matter — margin_free kills signal |
| per_token_cb (inj_aware) | dl2rc | false | −9.2% (83→74%) | ~5/0 | barely | injection_aware weaker than cb_full_seq |
| original_rr | dl2sq | — | +2% (85→87%) | 0/? | 0 | **DEAD** — cos_sim=1.0 forever |

**Key training dynamics (best run: per_token_cb + dl2rc + margins):**
```
Step   0: d_h=0.012 d_b=0.011 grad=0.009  (initial — nearly identical reps)
Step  50: d_h=3.406 d_b=1.205 grad=4.361  (harmful pushed away, benign drifting)
Step 100: d_h=3.018 d_b=0.902 grad=2.454  (harmful distance plateauing)
Step 150: d_h=1.970 d_b=0.829 grad=3.916  (harmful distance DECREASING — retain pulling back)
```

**Lossmask coverage (cb_full_sequence):**
```
agentdojo_truncated.jsonl: 5654 samples, avg_coverage=75.2%, zero_mask=0
fujitsu_b4_dr.jsonl:       5652 samples, avg_coverage=68.1%, zero_mask=0
fujitsu_b4_ds.jsonl:       7661 samples, avg_coverage=75.0%, zero_mask=0
```

---

### 1.3 The Real Problem: Nothing Is Actually Working

Despite 8 sweeps, 10+ configurations, and dozens of bug fixes, the fundamental situation is:

**AgentDojo is completely immune to every configuration we've tried.** Across ALL runs:
- Baseline: malicious=19-20%, resist=80-81%
- CB Model: malicious=18-19%, resist=81-82%
- Delta: 0-2% — within noise

**The "best" Fujitsu result (−21.4%) is shallow pattern memorization, not circuit breaking.** All 21 improvements are the exact same substitution: `search_web` → `retrieve_multimodal_docs`. The CB learned one tool-name swap, not injection detection.

**`original_rr` (the paper's loss) is dead with LoRA.** cos_sim=1.0000 in every run regardless of masking policy. The cosine similarity gradient vanishes exactly when model_rep ≈ frozen_rep (LoRA init). The paper used full fine-tuning where this problem doesn't exist.

#### Why Nothing Moves — Root Cause Analysis

**1. Fujitsu drowns AgentDojo (3:1 data imbalance)**
- 7661 Fujitsu harmful vs ~2500 AD harmful
- With batch=4, 200 steps, the model sees ~1600 total samples
- That's ~1200 Fujitsu, ~400 AD — the model barely touches AD data
- It finds the easy Fujitsu pattern (one tool swap) and "converges"

**2. The model sees only 16% of the dataset**
- Dataset: ~10,000 harmful + ~8,500 benign = ~18,500 samples
- 200 steps × batch 4 × grad_accum 2 = 1,600 samples
- Each AD sample seen ~0.16 times on average — not enough to learn anything

**3. Training vs eval data mismatch for AgentDojo**
- Training: truncated to injection window (`truncate_to_injection_window.py`)
- Eval: original full-length traces
- The model never sees full-length AD patterns during training

**4. AD baseline already resists 81%**
- The remaining 19% are the hardest cases
- These may require understanding injection CONTENT, not just structural patterns
- Representation rerouting at layers 10/20 might not be the right mechanism for these

**5. Separate frozen model vs disable_adapter()**
- Old working sweep (d086ec9): loaded a SEPARATE frozen model (2× memory)
- Current: reuses same model with `disable_adapter()`
- Old also used `MAX_SEQ_LENGTH=2048` (not 4096), COMBINED frozen forward (not split)
- These differences haven't been ruled out

---

### 1.4 Architecture & Data Summary

#### Data Pipeline
```
Raw traces:
  → fujitsu_b4_ds.jsonl (7661 harmful) — DS = dangerous set
  → fujitsu_b4_dr.jsonl (5652 benign)  — DR = retain set
  → agentdojo_augmented.jsonl (5882: 2513 harmful, 2894 benign, 250 resisted)
ETL_B (render + lossmask with policy):
  → renders/*.jsonl + lossmasks/*.jsonl
Split (AgentDojo only; Fujitsu is pre-split as DS/DR):
  → agentdojo_renders_harmful/benign.jsonl
Training (train_schema.py → trainer.py):
  → LoRA adapter
Eval (eval.py, next_tool_prediction mode):
  → ASR, correct_tool_rate, no_tool_rate
```

#### Masking Policies
| Policy | Harmful coverage | Benign coverage | Verdict |
|--------|-----------------|----------------|---------|
| `injection_aware` | ~3.5% (144 tokens) | 0% (no injection) | Too sparse; benign invisible |
| `cb_full_sequence` | 75% | 68-75% | Works but signal still weak |

#### Loss Modes — Final Verdict
| Mode | Status | Why |
|------|--------|-----|
| `original_rr` | **DEAD** | Cosine gradient vanishes at LoRA init. Irreparable. |
| `per_token_cb` + margins | **Only one that does anything** | dl2rc + margin_free=false = best config |
| `per_token_cb` margin_free | **Dead** | Converges too fast, no push signal |
| `simplified_pooled` | Untested / pending | — |
| `triplet_full` | Abandoned (Sweep 4) | Cluster centers noisy with small batch |
| `legacy_cb` | Abandoned (Sweep 3) | Same cosine gradient problem as original_rr |

---

### 1.5 All Bugs Found & Fixed (Sweeps 7–8)

| Bug | Impact | Fix |
|-----|--------|-----|
| Schedule backwards | Zero reroute at init | Unified `get_dual_coefficients` for all modes |
| LoRA restricted to 0–20 | Frozen layers block CB propagation | Removed --lora-target-layers |
| LR/alpha halved | 10× weaker signal | Restored to 5e-5 / 32 |
| Missing Fujitsu DR | 0 benign from Fujitsu | Use DS/DR files directly (not split-by-category) |
| DR lossmask all-zero | Benign invisible under injection_aware | Switch to cb_full_sequence |
| KL OOM | log_softmax [8,4096,128256] = 16 GiB | Chunk KL along seq dim (512) |
| ZeroDivisionError | Crash when benign_samples empty | Defensive guard + per-file logging |
| Race condition in split | Concurrent jobs read partial files | Check all 4 split files |
| margin_free config missing | ValueError on --margin-free flag | Added field to CircuitBreakerConfig |
| alpha_benign used cs not cr | Retain coefficient decayed instead of ramping | Changed to cr at 5 call sites |
| POLICY/POOLING_POLICY hardcoded | Couldn't test cb_full_sequence | Made configurable via env vars |

---

## Part 2: What Comes Next

### Hypotheses to Test

**H1: Fujitsu is drowning AD.** Train on AD-only (no Fujitsu). If AD eval changes, the problem is data imbalance. If it still doesn't change, the problem is deeper.

**H2: Not enough steps.** 200 steps = 16% dataset coverage. Try 500-1000 steps so the model actually sees all the data.

**H3: Seq length matters.** Old working sweep used 2048, not 4096. Shorter sequences = denser injection signal (7% vs 3.5%). Also matches what was validated.

**H4: Separate frozen model matters.** Old code loaded a second model copy. Current code uses `disable_adapter()`. Worth a direct comparison — numerical precision or hook behavior could differ subtly.

**H5: The CB mechanism is fundamentally too weak for tool-calling.** LoRA-based representation rerouting might not be sufficient when harmful and benign are structurally identical. The paper worked on clearly different distributions. We might need a different approach entirely (e.g., direct supervised fine-tuning to refuse, or full fine-tuning).

### Recommended Experiments (Priority Order)

1. **AD-only + more steps** — most informative single experiment. Remove Fujitsu, train on AD only, 500 steps. Tests H1 + H2 together.
2. **Balanced data** — downsample Fujitsu to ~2500 (matching AD), keep 500 steps.
3. **Seq length 2048** — match old working sweep exactly.
4. **Restore separate frozen model** — rule out disable_adapter() issues.

### Lessons Learned (So Far)

1. **Verify against known-good baselines before exploring.** Sweep 7 introduced 3 regressions simultaneously. Always A/B test against old working code.
2. **Data pipeline bugs are silent killers.** Missing DR file, all-zero lossmasks, race conditions — none produced errors, all produced zero signal.
3. **Cosine similarity + LoRA = dead.** The gradient vanishes at initialization. Use L2-based distances (dl2rc, dl2sq) instead.
4. **Margins matter.** margin_free=true causes premature convergence (−1% vs −21.4% with margins).
5. **Masking policy determines what the model can learn.** injection_aware makes benign data invisible; cb_full_sequence at least provides signal everywhere.
6. **"Training loss looks good" ≠ "model learned something useful."** Loss converges in all configs, but eval barely moves. Always evaluate, never trust training curves alone.
7. **16% dataset coverage is not training.** With 200 steps and 18k samples, the model sees each sample <0.2 times. Need more steps or less data.
