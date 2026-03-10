# Agent 2 Engineering Report: Sweep 11 Pipeline Changes

**Date**: March 9, 2026
**Commits**:
- `b64d94f` — `sweep 11: add balanced, refusal, tuning, and amplification configs`
- `8f16698` — `sweep 11f: add SRMU importance-masked configs and probe extraction step`

**Files modified**: `slurm/pipeline/sweep_ad_focused.sbatch`, `scripts/amplify_injections.py` (new), `configs/probe_directions/.gitkeep` (new)

---

## 1. What Was Done

### 1.1 Problem Context

Sweep 10 showed that v2 training fixes (decay_mult, margin, KL schedule, constant LR) produced a **9x improvement** in AD malicious reduction, but the core **selectivity problem** persists:

| Config | Fujitsu D | AD Trunc mal | Benign correct | Gibberish (H/B) |
|--------|-----------|-------------|----------------|-----------------|
| Sweep 9 best (ad_1000) | -8% | 63% (-5%) | 77% | 0/0 |
| ad_1000_v2 | -14% | 30% (-37%) | 41% | 25/23 |
| ad_3000_v2 | -33% | 22% (-45%) | 40% | 4/10 |
| **ad_5000_v2** | **-58%** | **24% (-44%)** | **72%** | 10/2 |

**ad_5000_v2 is Pareto-optimal** but still shows:
- Benign at 72% vs theoretical ceiling 83% (11% gap from selectivity AUC 0.83)
- Gibberish outputs on AD (not clean refusals like Fujitsu)
- Phase 3 collapse at ~68-72% of training destroying benign anchoring

### 1.2 Sweep 11 Configs Added

Eight new configs organized into four experimental groups:

#### Group A: Balanced (Fujitsu + AD) — `ad_500_bal`, `ad_750_bal`, `ad_1000_bal`

**Hypothesis**: Fujitsu's structural injection pattern (HTML comments) produces clean CB tool redirects (zero gibberish, zero regressions across all sweep 10 runs). Including Fujitsu data alongside AD should:
1. Provide a cleaner injection detection signal that transfers to AD
2. Add diverse benign data (Fujitsu DR traces) to strengthen capability anchoring
3. Let the model learn both injection styles (HTML comments + INFORMATION tags)

**Setup**: `DATASET=balanced`, `FUJITSU_CAP=2500` (matched to AD harmful count ~2500). All v2 fixes applied. Three step counts (500/750/1000) to find optimal training duration with mixed data.

**Control comparison**: Sweep 9 `balanced_500` showed Fujitsu -27% (best) but hurt AD (+5%). With v2 fixes, the dynamics may change — the four compounding fixes were the dominant bottleneck, not the data mix.

#### Group B: Balanced + Refusal — `ad_750_bal_r`

**Hypothesis**: The gibberish problem is the central quality issue. CB pushes harmful representations "away" but they land in degenerate regions instead of on clean refusal pathways. Adding 1225 synthetic refusal traces (harmful context + refusal response) to the benign retain set gives CB a directed target: "when you see injection, produce refusal-like representations."

**Setup**: Same as `ad_750_bal` + `_USE_REFUSAL=true`. The 1225 refusal traces are added to the benign render/lossmask lists, giving the retain loss a specific representation to anchor toward.

**Expected outcome**: Reduced gibberish rate, potentially better benign preservation (refusal traces teach the model that injection-context can produce coherent non-tool output).

#### Group C: Alpha/Margin Tuning — `ad_1000_a5`, `ad_1000_m40`

**`ad_1000_a5` (alpha_max=5.0)**:
- **Hypothesis**: With alpha=10, Phase 1 saturates the harmful margin in ~30 steps. Lower alpha produces slower, more gradual weight updates that may be more selective (less collateral damage to benign representations).
- **Mechanistic prediction**: Longer Phase 1 (maybe 60-100 steps instead of 30), reduced Phase 3 destructiveness (lower cs ceiling).

**`ad_1000_m40` (margin_harmful=40.0, margin_benign=5.0)**:
- **Hypothesis**: Phase 3 collapse occurs when harmful representations drift back within margin=20 during Phase 2. A higher margin (40) means harmful reps need to drift further to re-enter the active zone, potentially delaying or preventing Phase 3.
- **Mechanistic prediction**: Longer Phase 2 (harmful reps take longer to drift back within the higher margin), potentially avoiding Phase 3 entirely at 1000 steps.

#### Group D: Amplified Injections — `ad_750_amp`, `ad_750_bal_amp`

**Hypothesis**: Probe selectivity AUC is 0.83 — the model weakly distinguishes injection contexts from clean contexts. Repeating `<INFORMATION>...</INFORMATION>` blocks 2x in harmful tool outputs makes the injection signal louder, potentially boosting the context-level representation difference that CB uses for selectivity.

**Implementation**: New script `scripts/amplify_injections.py` finds injection tags in tool messages and repeats them. Only harmful traces are modified; benign pass through unchanged. Integrated into the sbatch pipeline as an optional step between truncation and ETL_B rendering.

**Caveat**: Most speculative intervention. The probe AUC reflects representation-level features; repeating text may not change high-level features if the model already encodes "injection present" after seeing it once.

#### Group E: SRMU — Importance-Masked CB Loss — `ad_750_v2_srmu`, `ad_1000_v2_srmu`, `ad_1500_v2_srmu`, `ad_1000_bal_srmu`

**Hypothesis**: The selectivity problem (AUC 0.83) means CB pushes ALL hidden dimensions equally, disrupting benign representations. SRMU-style importance masking (from Agent 3's probe extraction) weights the push by how much each hidden dimension discriminates harmful from benign. Dimensions with high probe weight get pushed hard; dimensions irrelevant to injection detection are left alone, preserving generative capability.

**Implementation**: Agent 3 created:
- `scripts/extract_probe_direction.py` — trains logistic regression probe on hidden states, extracts weight vector as importance mask
- Modified `src/training/losses.py` — element-wise masking of representations before distance computation
- Modified `src/training/trainer.py`, `train_schema.py`, `config.py` — `--importance-mask` CLI arg wired through to loss

I (Agent 2) wired this into the sbatch pipeline:
- `_IMPORTANCE_MASK=auto` resolves to `configs/probe_directions/layer10_last_token.pt`
- **Step 0b** auto-extracts the probe direction if the `.pt` file doesn't exist (first SRMU job extracts, all subsequent reuse)
- `--importance-mask` passthrough added to training command
- Mask metadata (AUC, accuracy, sparsity) logged in pipeline output

**Setup**: Same v2 fixes as other configs. Four variants: 750/1000/1500 steps AD-only + 1000 steps balanced.

**Expected outcome**: Better benign preservation (closing 72% → 83% gap) without sacrificing AD malicious reduction. The importance mask focuses the push on the ~20-30% of hidden dimensions that matter for injection detection.

**Prerequisite**: Probe extraction needs a GPU forward pass through the model. The first SRMU sbatch job handles this automatically.

### 1.3 Infrastructure Changes

#### Stale Refusal Cache Fix
Added cache-busting check: if `REFUSAL_SRC` exists but contains old `original_source` format in the source field, it's deleted along with downstream renders/lossmasks to force regeneration. This prevents using buggy refusal traces from an earlier pipeline version.

#### Amplification Pipeline Integration
- New variables: `AMPLIFY_INJECTIONS`, `AMPLIFY_REPEATS`
- Amplification runs after truncation but before ETL_B rendering
- Uses caching (keyed by repeat count) to avoid re-running
- Falls back to original traces if amplification fails
- Added to `run_config.json` for experiment tracking

---

## 2. Priority Ranking

Based on Agent 1's analysis and experimental logic. Grouped into tiers:

**Tier 1 — Must-run (highest expected impact):**

| Priority | Config | Why |
|----------|--------|-----|
| **1** | `ad_1000_v2_srmu` | Novel method contribution (SRMU). Directly addresses selectivity via importance masking. Paper's key experiment. |
| **2** | `ad_750_bal_r` | Combines both highest-value data interventions: Fujitsu data + refusal targets. Addresses selectivity AND gibberish. |
| **3** | `ad_1000_bal` | Isolates Fujitsu data effect. Clean comparison to ad_1000_v2. |

**Tier 2 — Important ablations:**

| Priority | Config | Why |
|----------|--------|-----|
| **4** | `ad_1000_a5` | Clean alpha ablation. Low cost, high diagnostic value. |
| **5** | `ad_1000_bal_srmu` | SRMU + balanced. Tests if data + loss improvements stack. |
| **6** | `ad_1000_m40` | Tests Phase 3 delay mechanism. |
| **7** | `ad_1500_v2_srmu` | Longer SRMU run — does more Phase 2 + masking beat ad_5000_v2? |

**Tier 3 — Can skip if GPU-constrained:**

| Priority | Config | Why |
|----------|--------|-----|
| 8 | `ad_750_v2_srmu` | Shorter SRMU, likely dominated by 1000-step version. |
| 9 | `ad_750_bal` | Shorter balanced, likely dominated by ad_1000_bal. |
| 10 | `ad_750_amp` | Most speculative. Representation signal may not change. |
| 11 | `ad_500_bal` | Likely dominated by longer balanced configs. |
| 12 | `ad_750_bal_amp` | Combines two uncertain interventions. |

**Recommended submission:**
```bash
# Tier 1 — must-run
for c in ad_1000_v2_srmu ad_750_bal_r ad_1000_bal; do
  sbatch --export=ALL,CONFIG=$c slurm/pipeline/sweep_ad_focused.sbatch
done

# Tier 2 — important ablations
for c in ad_1000_a5 ad_1000_bal_srmu ad_1000_m40 ad_1500_v2_srmu; do
  sbatch --export=ALL,CONFIG=$c slurm/pipeline/sweep_ad_focused.sbatch
done
```

---

## 3. What to Look for in Results

### 3.1 Key Metrics (in order of importance)

For each config, the eval produces:

1. **AD Benign correct %** — capability preservation (target: >72%, ceiling ~83%)
2. **AD Trunc malicious %** — attack reduction (target: <24%, ceiling ~3%)
3. **Fujitsu Delta** — structural injection defense (target: < -58%)
4. **Gibberish count (harmful/benign)** — output quality (target: 0/0)
5. **AD Benign no_tool %** — suppression mode (target: <5%; high = model suppressing ALL tools)

### 3.2 Per-Config Success Criteria

#### `ad_750_bal_r` (balanced + refusal)
- **Success**: Benign >60% AND gibberish <5/97 (benign) AND AD trunc mal <30%
- **Home run**: Benign >72% AND gibberish 0 AND AD trunc mal <24%
- **Watch for**: If refusal traces cause the model to refuse ALL tool calls (benign no_tool >20%), the refusal signal is too strong

#### `ad_1000_bal` (balanced)
- **Success**: Fujitsu Delta < -30% AND AD trunc mal <35% AND benign >50%
- **Key comparison**: vs `ad_1000_v2` (AD-only). If balanced beats AD-only on BOTH AD metrics AND Fujitsu, the Fujitsu data genuinely helps. If it hurts AD (like sweep 9's balanced_500 did), the v2 fixes didn't solve the data mixing problem.
- **Watch for**: Fujitsu overpowering AD signal (Fujitsu Δ very large but AD metrics worse)

#### `ad_1000_a5` (alpha=5.0)
- **Success**: More gradual training curve (Phase 1 >60 steps instead of 30) AND benign >50%
- **Key comparison**: vs `ad_1000_v2` (alpha=10). Look at training log: does Triplet_H saturate later? Is Phase 3 onset delayed or absent?
- **Watch for**: If Triplet_H never saturates at 1000 steps (margin never exceeded), alpha is too low and the push isn't strong enough

#### `ad_1000_m40` (margin=40.0)
- **Success**: No Phase 3 collapse in training curve at 1000 steps
- **Key comparison**: vs `ad_1000_v2` (margin=20). Look at training log: does Triplet_H stay near 0 through the full run? If yes, the higher margin prevents Phase 3 entirely.
- **Watch for**: If margin=40 is too high, Triplet_H may never reach 0 (never exceed margin), meaning the push is constantly active but weak — different failure mode

### 3.3 Training Curve Diagnostics

The training log contains step-by-step Triplet_H, Triplet_B, KL, and alpha values. For each run, check:

| Phase | Triplet_H | Triplet_B | Interpretation |
|-------|-----------|-----------|----------------|
| Phase 1 (push) | High then drops to ~0 | Low then rises | Harmful margin exceeded, push locked in |
| Phase 2 (anchor) | ~0 | Slowly decreasing | Only benign retain active, anchoring benign |
| Phase 3 (collapse) | Spikes back up | Drops to <1 | Harmful reps drifted back, benign destroyed |

**What we want**: Long Phase 2, no Phase 3 (or very late Phase 3).

**Milestone to check**: At what step does Triplet_H first spike >1.0? This is the Phase 3 onset. For ad_1000_v2 it was step ~850 (85%). We want it either absent or >90%.

### 3.4 Pareto Frontier Update

After results come in, plot all configs on:
- **X-axis**: AD Trunc malicious % (lower = better defense)
- **Y-axis**: AD Benign correct % (higher = better capability)
- **Color**: Config group (AD-only, balanced, balanced+refusal, tuning)

Any new config that is above-and-to-the-left of ad_5000_v2 (24% mal, 72% benign) is a new Pareto point.

### 3.5 Cross-Config Comparisons

| Comparison | Isolates | Agent |
|-----------|----------|-------|
| `ad_1000_v2_srmu` vs `ad_1000_v2` | Effect of importance masking (SRMU) | 3 |
| `ad_1000_bal_srmu` vs `ad_1000_bal` | Does SRMU + balanced stack? | 2+3 |
| `ad_1000_bal` vs `ad_1000_v2` | Effect of adding Fujitsu data | 2 |
| `ad_750_bal_r` vs `ad_750_bal` | Effect of refusal traces | 2 |
| `ad_1000_a5` vs `ad_1000_v2` | Effect of halving alpha | 2 |
| `ad_1000_m40` vs `ad_1000_v2` | Effect of doubling margin | 2 |
| `ad_750_amp` vs `ad_750_v2` | Effect of injection amplification | 2 |
| `ad_1500_v2_srmu` vs `ad_5000_v2` | Can SRMU at 1500 steps beat 5000 steps standard? | 3 |

---

## 4. Analysis Plan (Once Results Return)

### Step 1: Collect raw metrics
```bash
# For each completed run:
cat $CB_SCRATCH/sweeps/s9_${CONFIG}_*/run/run_config.json
cat $CB_SCRATCH/sweeps/s9_${CONFIG}_*/run/eval/next_tool_prediction/*.json
```

### Step 2: Build comparison table
Populate the table below with actual results:

| Config | Steps | Data | Mask | Fujitsu D | AD Full mal | AD Trunc mal | Benign correct | Benign no_tool | Gib H | Gib B |
|--------|-------|------|------|-----------|-------------|-------------|----------------|----------------|-------|-------|
| ad_1000_v2 (control) | 1000 | AD | -- | -14% | 31% | 30% | 41% | 15% | 25 | 23 |
| ad_5000_v2 (Pareto) | 5000 | AD | -- | -58% | 32% | 24% | 72% | 5% | 10 | 2 |
| ad_1000_v2_srmu | 1000 | AD | SRMU | ? | ? | ? | ? | ? | ? | ? |
| ad_1500_v2_srmu | 1500 | AD | SRMU | ? | ? | ? | ? | ? | ? | ? |
| ad_1000_bal | 1000 | bal | -- | ? | ? | ? | ? | ? | ? | ? |
| ad_1000_bal_srmu | 1000 | bal | SRMU | ? | ? | ? | ? | ? | ? | ? |
| ad_750_bal_r | 750 | bal+r | -- | ? | ? | ? | ? | ? | ? | ? |
| ad_1000_a5 | 1000 | AD | -- | ? | ? | ? | ? | ? | ? | ? |
| ad_1000_m40 | 1000 | AD | -- | ? | ? | ? | ? | ? | ? | ? |

### Step 3: Diagnose
1. **Did SRMU help?** Compare ad_1000_v2_srmu vs ad_1000_v2. If benign improves without AD regression, SRMU is the paper's key result.
2. **Did balanced help?** Compare ad_1000_bal vs ad_1000_v2 on all metrics
3. **Do they stack?** Compare ad_1000_bal_srmu vs ad_1000_bal and vs ad_1000_v2_srmu. If better than both, interventions are complementary.
4. **Did refusal reduce gibberish?** Compare ad_750_bal_r gibberish counts vs ad_750_bal
5. **Did alpha/margin change Phase 3?** Check training logs for phase transition timing
6. **New Pareto point?** Plot and check if any config dominates ad_5000_v2

### Step 4: Decide next steps
- **If SRMU helps**: This is the paper's novel contribution. Run `ad_3000_v2_srmu`, `ad_5000_v2_srmu` for longer SRMU training. Try combining with balanced+refusal.
- **If balanced helps**: Run `ad_1500_bal`, `ad_3000_bal` for longer balanced training.
- **If refusal works**: Add refusal to AD-only configs (`ad_1000_v2r` already exists) and to SRMU configs.
- **If SRMU + balanced stacks**: The optimal config is likely `ad_X_bal_srmu_r` (balanced + SRMU + refusal). Create this combo.
- **If alpha=5 helps**: Try alpha=3 and alpha=7 for finer tuning.
- **If nothing beats ad_5000_v2**: The selectivity ceiling is fundamental. Need different approach (e.g., multi-layer masking, different probe pooling, or architectural changes).

---

## 5. Files Changed

| File | Change | Commit |
|------|--------|--------|
| `slurm/pipeline/sweep_ad_focused.sbatch` | 12 new config entries (8 Agent 2 + 4 SRMU), amplification pipeline, SRMU probe extraction (Step 0b), `--importance-mask` passthrough, refusal cache fix, updated header/banner | Both |
| `scripts/amplify_injections.py` | New script: repeats `<INFORMATION>` tags in harmful traces | `b64d94f` |
| `configs/probe_directions/.gitkeep` | Placeholder dir for probe direction `.pt` files (generated on cluster) | `8f16698` |

No changes to training code (`src/training/`), evaluation code (`src/evaluation/`), or paper (`paper/`).

---

## 6. Full Config Inventory (Sweep 11)

All configs available in `sweep_ad_focused.sbatch`:

### Pre-existing (from sweep 10/11a)
| Config | Steps | Data | Special |
|--------|-------|------|---------|
| ad_300_v2 | 300 | AD | v2 fixes |
| ad_500_v2 | 500 | AD | v2 fixes |
| ad_750_v2 | 750 | AD | v2 fixes |
| ad_1000_v2 | 1000 | AD | v2 fixes |
| ad_1500_v2 | 1500 | AD | v2 fixes |
| ad_3000_v2 | 3000 | AD | v2 fixes (overtrained) |
| ad_5000_v2 | 5000 | AD | v2 fixes (Pareto optimal) |
| ad_1000_v2r | 1000 | AD+refusal | v2 fixes |
| ad_3000_v2r | 3000 | AD+refusal | v2 fixes |

### Added by Agent 2 (this session)
| Config | Steps | Data | Special | Group |
|--------|-------|------|---------|-------|
| ad_500_bal | 500 | balanced | cap=2500 | A: Balanced |
| ad_750_bal | 750 | balanced | cap=2500 | A: Balanced |
| ad_1000_bal | 1000 | balanced | cap=2500 | A: Balanced |
| ad_750_bal_r | 750 | balanced+refusal | cap=2500 | B: Refusal |
| ad_1000_a5 | 1000 | AD | alpha=5.0 | C: Tuning |
| ad_1000_m40 | 1000 | AD | margin_h=40.0 | C: Tuning |
| ad_750_amp | 750 | AD | amplified 2x | D: Amplification |
| ad_750_bal_amp | 750 | balanced | amplified 2x | D: Amplification |

### Added for Agent 3 (SRMU integration)
| Config | Steps | Data | Special | Group |
|--------|-------|------|---------|-------|
| ad_750_v2_srmu | 750 | AD | importance mask | E: SRMU |
| ad_1000_v2_srmu | 1000 | AD | importance mask | E: SRMU |
| ad_1500_v2_srmu | 1500 | AD | importance mask | E: SRMU |
| ad_1000_bal_srmu | 1000 | balanced | importance mask + cap=2500 | E: SRMU |

**Total**: 21 configs available (9 pre-existing + 8 Agent 2 + 4 SRMU).
