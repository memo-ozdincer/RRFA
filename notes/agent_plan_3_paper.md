# Agent Plan 3: Paper Strategy, SRMU Implementation & Figures

**FIRST**: Read `notes/shared_context.md` — it contains ALL raw data, the current paper state analysis, training results, probe results, and strategic context.

**ALSO READ**: `paper/_COLM_2026__AgentDefense/colm2026_conference.tex` — the current (stale) paper draft. It needs major updates. The shared_context.md §2 describes what needs to change.

**Role**: Research scientist. Implement the novel contribution (feature-selective rerouting), write the paper narrative, prepare publication figures, connect to literature.

**Worktree**: YES — use `isolation: "worktree"`. Changes touch:
- `src/training/losses.py` (add importance-weighted CB loss)
- `scripts/extract_probe_direction.py` (NEW — extract linear separator weights)
- `paper/_COLM_2026__AgentDefense/colm2026_conference.tex` (paper writing)
- `scripts/plot_publication_figures.py` (figures)
- `notes/experiment_log.tex` (update with SRMU section)

**Does NOT touch**: `slurm/pipeline/*.sbatch`, `scripts/amplify_injections.py` (Agent 2's territory)

---

## Context: The Scientific Story (Updated by Agent 1 Analysis)

**READ FIRST**: `notes/analysis_report.md` — comprehensive analysis with all data tables, training curve phase analysis, Pareto frontier, and cross-dataset comparison. This is the empirical foundation for everything below.

### What We've Found (Novel Contributions)

1. **The selectivity problem in agentic CB**: Circuit breaking for tool-calling agents faces a fundamental asymmetry — the push direction (comply→refuse) is strongly linearly separable (AUC 0.96 at last token) but the selectivity signal (injection-context vs clean-context) is weak (AUC 0.83). This predicts a Pareto tradeoff, confirmed empirically: ad_5000_v2 (Pareto-optimal) achieves 72% benign vs 83% theoretical ceiling.

2. **Three-phase training dynamic** (NEW from Sweep 10 analysis): CB training follows Push → Plateau → Collapse. Phase 1 saturates margin in ~30 steps. Phase 2 is benign-only training. Phase 3 onset at ~68-72% of training is a margin-induced oscillation not described in prior CB work. This explains why ad_5000_v2 (more Phase 2 time) outperforms ad_3000_v2 despite longer total training.

3. **Cross-dataset divergence**: Fujitsu (structural injection) gets clean tool redirects; AD (semantic injection) gets gibberish. This is a paper-worthy finding about injection structure dependence on CB output quality (Section 7 of analysis report).

4. **Probe-guided feature selection for CB**: Inspired by SRMU (Selective Representation Misdirection for Unlearning — Jin's paper), we can use the linear probe's weight vector as an importance mask, rerouting only along harm-discriminative dimensions. This directly addresses the entanglement problem.

5. **Comprehensive empirical characterization**: 10+ sweeps, ~30 configs, systematic identification of 4 compounding training inefficiencies, probe diagnostics across pooling strategies and all 32 layers.

### Key Numbers for the Paper (from analysis_report.md)

| Config | AD Trunc mal | Benign | Fujitsu Δ | Notes |
|--------|-------------|--------|-----------|-------|
| Baseline | 68% | 78-80% | 0% | No CB |
| Sweep 9 best | 63% (-5%) | 77% | -8% | Before training fixes |
| **ad_5000_v2** | **24% (-44%)** | **72%** | **-58%** | Pareto optimal, v2 fixes |
| Theoretical ceiling | ~3% | ~83% | — | From probe AUCs |

- **Selectivity ratio**: push_AUC / selectivity_AUC = 0.967 / 0.83 = 1.17
- **Benign gap to ceiling**: 72% actual vs 83% theoretical = 11% recoverable by SRMU
- **Phase 3 onset**: consistently at 68-72% of training across step counts
- **Gibberish rate (benign)**: 24% at 1000 steps → 10% at 3000 → 2% at 5000

### Paper Framing

> Title idea: "Representation Rerouting for Agentic Prompt Injection Defense: A Probe-Guided Approach"
>
> Core argument: Standard CB rerouting fails on tool-calling agents because harmful and benign tool calls share representation space (high entanglement). We characterize the selectivity problem via probing, discover a three-phase training dynamic that explains the benign destruction, and apply feature-selective rerouting (SRMU-inspired) to reroute only along discriminative dimensions. This achieves X% attack reduction while preserving Y% benign capability.

### Figures Available (Agent 1 data ready)
1. **Training curve (ad_3000_v2)**: Full step-by-step data for Triplet_H, Triplet_B, KL, Alpha. Three phases clearly visible. Data in `results_for_agent1/traininglogfor3000v2.txt`.
2. **Pareto plot**: 6 data points (baseline, sweep 9, ad_1000_v2, ad_3000_v2, ad_5000_v2, theoretical ceiling). Only ad_5000_v2 is on the frontier.
3. **Probe heatmap**: Full AUC tables for Tests 1, 2, 3 in `results_for_agent1/latestprobespresultsfull.txt`. The pooling flip between Test 1 and Test 2 is the key visual.
4. **Output examples**: Fujitsu clean redirects vs AD gibberish progression. Data in the three eval result files.

### Connection to Jin's SRMU Paper

SRMU solves "when forget and retain distributions are entangled, indiscriminate perturbation fails" by:
1. Computing a **Dynamic Importance Map** identifying harm-specific feature dimensions
2. Applying **directional misdirection** only along those dimensions

Our application: the probe's logistic regression weight vector IS the importance map. Dimensions with large |w_i| discriminate harmful from benign; dimensions with small |w_i| are shared. Rerouting only along large-|w_i| dimensions preserves benign capability on shared features.

The selectivity gap (72% → 83% theoretical ceiling = 11% recoverable) is the direct target for SRMU-style importance masking.

---

## Task 1: Extract Probe Direction Vector

**Create `scripts/extract_probe_direction.py`**:

This script:
1. Loads the base model (Llama-3.1-8B-Instruct)
2. Extracts hidden states at CB target layers (10, 20) for harmful and benign samples
3. Trains a logistic regression probe (same as probe_separability.py)
4. Saves the **weight vector** (4096-dim) as a torch tensor
5. Also saves the intercept, AUC, and metadata

```python
#!/usr/bin/env python3
"""
Extract the linear probe direction vector for importance-weighted CB loss.

The weight vector from a logistic regression probe identifies which feature
dimensions discriminate harmful from benign representations. This is used
as an importance mask in the CB loss (SRMU-style feature-selective rerouting).

Output: A .pt file containing:
  - direction: (4096,) tensor — the probe weight vector (normalized)
  - importance: (4096,) tensor — |w_i| / max(|w_i|) ∈ [0, 1]
  - metadata: dict with AUC, layer, pooling, etc.

Usage:
    python scripts/extract_probe_direction.py \
        --traces data/traces/agentdojo_truncated.jsonl \
        --model /path/to/llama-3.1-8b-instruct \
        --layer 10 \
        --pooling last_token \
        --output configs/probe_directions/layer10_last_token.pt \
        --max-samples 400
"""
```

**Key implementation details**:
- Reuse the hidden state extraction logic from `scripts/probe_separability.py`
- The weight vector is `probe.coef_[0]` from sklearn LogisticRegression
- Normalize to unit vector for the direction
- Compute importance as `|w_i| / max(|w_i|)` — values in [0, 1]
- Save for both layer 10 and layer 20

**Reference**: `scripts/probe_separability.py` lines ~1-200 for hidden state extraction

## Task 2: Implement Importance-Weighted CB Loss

**Modify `src/training/losses.py`**:

Add a new parameter `importance_mask` to the existing `per_token_cb_loss` function (or create a wrapper). The idea:

Current CB loss (simplified):
```python
# Harmful: push away
d = dl2rc(h_lora, h_frozen)  # scalar distance per token
loss_h = ReLU(margin - d)     # hinge loss

# Benign: keep close
loss_b = dl2rc(h_lora, h_frozen)  # minimize distance
```

Modified (importance-weighted):
```python
# Project the representation difference onto importance-weighted space
diff = h_lora - h_frozen  # (batch, seq, 4096)
weighted_diff = diff * importance_mask  # element-wise, importance_mask is (4096,)

# Now compute distance on weighted diff
d_weighted = weighted_diff.norm(dim=-1) + 10 * ReLU(1 - cos_sim(h_lora * importance_mask, h_frozen * importance_mask))
loss_h = ReLU(margin - d_weighted)
```

**The key insight**: By multiplying both `h_lora` and `h_frozen` by the importance mask before computing distance, we only measure and penalize divergence along dimensions that the probe identified as discriminative. Shared dimensions (small importance) contribute negligibly to the loss, so the optimizer doesn't push them.

**Implementation approach** (minimal changes):

1. Add `--importance-mask` CLI arg to `train_schema.py` (path to .pt file)
2. Load the mask in `CircuitBreakerTrainer.__init__`
3. In the loss computation (trainer.py ~line 1895), apply the mask to hidden states before passing to `per_token_cb_loss`
4. Make it optional — if no mask provided, behavior is unchanged (backwards compatible)

**Files to modify**:
- `src/training/losses.py` — add `importance_mask` parameter to distance computation
- `src/training/trainer.py` — load mask, pass to loss
- `src/training/train_schema.py` — add `--importance-mask` CLI arg
- `src/training/config.py` — add `importance_mask_path` field

**Critical**: Keep existing behavior when no mask is provided. Test that unmasked loss matches exactly.

## Task 3: Paper Writing — Results Section Draft

**File**: `paper/_COLM_2026__AgentDefense/colm2026_conference.tex`

Read the existing paper first to understand the structure. Then draft/update these sections:

### Section: Experimental Setup
- Datasets: Fujitsu B4 (tool-flip attacks, HTML injection in user messages, 8 tools) and AgentDojo (arbitrary-action attacks, semantic injection in tool outputs, 64 tools)
- Model: Llama-3.1-8B-Instruct with LoRA (all 32 layers, r=16, α=32)
- CB target layers: 10, 20
- Loss: per_token_cb with dl2rc distance, ReLU hinge margins (harmful=20.0, benign=3.0)
- Eval: next_tool_prediction (truncate to decision point, generate next tool call)
- Data: 2513 harmful AD, 3141 benign AD, 7661 harmful Fujitsu, 5652 benign Fujitsu

### Section: The Selectivity Problem (Novel Finding)
- Present probe results as the diagnostic
- **Table 1**: Test 1 vs Test 2 AUCs — the pooling flip is the key visual (mean best in Test 1, last_token best in Test 2)
- Explain the selectivity asymmetry (push AUC 0.967 at L10 vs selectivity AUC 0.83)
- **Table 2**: Full Sweep 10 results (1000/3000/5000 steps) showing Pareto frontier
- Theoretical ceiling: 83% benign (from selectivity AUC), 97% harmful reduction (from push AUC)
- Actual best: ad_5000_v2 at 72% benign, 76% resist — approaching but not reaching ceiling

### Section: Three-Phase Training Dynamic (Novel Finding)
- **Figure 1**: Training curve of ad_3000_v2 with phases annotated
- Phase 1 (Push): margin exceeded in 30 steps, zero harmful gradient thereafter
- Phase 2 (Plateau): benign-only training, LoRA drifts for benign preservation
- Phase 3 (Collapse): harmful reps drift back within margin, re-activate push, benign destroyed
- **Table 3**: Phase boundaries across step counts (all at 68-72%)
- Why ad_5000_v2 > ad_3000_v2: more Phase 2 anchoring epochs (11.1 vs 6.4)
- This is a novel training insight not described in prior CB work

### Section: Cross-Dataset Analysis
- **Table 4**: Fujitsu vs AD comparison (analysis_report.md Section 7)
- Fujitsu: clean correct tool call redirects (search_web → retrieve_multimodal_docs)
- AD: gibberish output modes (token repetition → character gibberish → garbled JSON)
- Explanation: structural injection + 1-to-1 tool mapping → clean redirect; semantic injection + 64 tools → degenerate output

### Section: Feature-Selective Rerouting
- Motivation from SRMU + the selectivity gap (72% → 83% = 11% recoverable)
- Method: probe direction → importance mask → weighted CB loss
- Results: (pending from Sweep 11 + importance-weighted runs)
- Comparison table: standard CB vs feature-selective CB

### Section: Training Inefficiency Analysis
- The 4 compounding bugs and their fixes (math from experiment_log.tex Section 9)
- Before/after comparison (Sweep 9 → Sweep 10): 9x improvement on AD
- This is a secondary contribution but shows careful engineering and validates the v2 training setup

## Task 4: Publication Figures

**Create/update `scripts/plot_publication_figures.py`**:

Figures needed:

1. **Probe AUC heatmap** (2 panels):
   - Panel A: Test 1 (harmful vs benign) — 6 layers × 3 poolings × 2 probes
   - Panel B: Test 2 (harmful vs refusal) — same grid
   - Highlight the pattern flip (mean best in A, last_token best in B)

2. **Training curve phases** (ad_3000_v2):
   - X: steps, Y: Triplet_H and Triplet_B (dual axis or normalized)
   - Annotate phase transitions (margin exceeded, benign drift, harmful spike)
   - Shade the "useful training" region vs "benign corruption" region

3. **Pareto frontier** (when Sweep 11 results are in):
   - X: benign correct rate, Y: harmful malicious rate
   - Each point is a config (step count × dataset × amplification)
   - Annotated with config name
   - Show the baseline point and the Pareto-optimal curve

4. **Importance mask visualization**:
   - Bar plot or heatmap of the top-100 most important dimensions
   - Show the distribution of importance values (most are near zero)

**Style requirements**:
- Use matplotlib with a clean publication style (seaborn-paper or similar)
- Font size appropriate for 2-column format
- Save as both PDF (for LaTeX) and PNG (for review)
- Color-blind friendly palette

## Task 5: Literature Integration

Read these files and extract relevant citations/connections:

1. **`PAPERS/RMU.tex`** (Jin's SRMU paper) — the importance map mechanism. Key equations to cite:
   - Dynamic Importance Map: `I_i = log(1 + v_f,i / v_r,i)` where v_f, v_r are feature variances
   - Our adaptation: use probe weights instead of variance ratios

2. **`PAPERS/circuitbreakers.tex`** — original CB. Note that their task had clearly separable harmful/benign (bomb-making vs chat). Our task has entangled distributions (both are tool calls).

3. **`PAPERS/Triplet.tex`** — triplet loss for CB. We use elements of this but with dl2rc distance instead of pure cosine.

4. **`PAPERS/promptinjection.tex`** — Instruction Hierarchy. Their "conditional instruction following" is conceptually similar to our selectivity problem.

5. **`PAPERS/latentunlearning.tex`** — forget/retain n-gram overlap. Directly relevant — our harmful/benign traces share identical JSON format.

6. **`PAPERS/microsoftBIPIA.tex`** — boundary awareness for injection defense.

**For each paper**: Write 1-2 sentences of how it connects to our work, to be used in Related Work.

## Task 6: Update Experiment Log

Add to `notes/experiment_log.tex`:
- Test 2 and Test 3 probe results (full tables — already partially added, verify completeness)
- The selectivity analysis section
- SRMU-inspired feature-selective rerouting method description
- Results from Sweep 11 (when available)

---

## File References

### Must Read Before Writing
- `paper/_COLM_2026__AgentDefense/colm2026_conference.tex` — current paper draft
- `scripts/probe_separability.py` — probe implementation (reuse for direction extraction)
- `src/training/losses.py` — current loss implementation (you're modifying this)
- `src/training/trainer.py:1880-1930` — where loss is called in training loop
- `PAPERS/RMU.tex` — SRMU importance map (the key inspiration)
- `PAPERS/circuitbreakers.tex` — original CB paper

### Read for Context
- `notes/experiment_log.tex` — full experiment history
- `PAPERS/Triplet.tex`, `PAPERS/latentunlearning.tex`, `PAPERS/promptinjection.tex`
- `src/training/config.py` — config dataclass (add importance_mask_path field)
- `src/training/train_schema.py` — CLI arg parser (add --importance-mask)

### DO NOT MODIFY (Agent 2's territory)
- `slurm/pipeline/sweep_ad_focused.sbatch`
- `scripts/amplify_injections.py`
- `scripts/generate_refusal_traces.py`

---

## Deliverables

1. `scripts/extract_probe_direction.py` — extracts importance mask from probe
2. Modified `src/training/losses.py` — importance-weighted CB loss (backwards compatible)
3. Modified `src/training/trainer.py` — loads and applies importance mask
4. Modified `src/training/train_schema.py` + `config.py` — CLI arg + config field
5. Paper draft sections in `paper/_COLM_2026__AgentDefense/colm2026_conference.tex`
6. `scripts/plot_publication_figures.py` — publication-ready figures
7. Updated `notes/experiment_log.tex` — SRMU section + complete probe results

---

## Priority Order (if time-limited)

1. **Extract probe direction** (Task 1) — needed for everything else
2. **Importance-weighted loss** (Task 2) — the novel contribution
3. **Paper selectivity section** (Task 3, "Selectivity Problem" subsection) — the main finding
4. **Figures** (Task 4, #1 probe heatmap and #2 training curve) — most impactful visuals
5. **Literature integration** (Task 5) — needed for Related Work
6. **Remaining paper sections** (Task 3, remaining) — fill in once results are in
