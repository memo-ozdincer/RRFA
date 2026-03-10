# Agent 2 Engineering Report: Sweep 11 Pipeline Changes

**Date**: March 9, 2026
**Commit**: `b64d94f` — `sweep 11: add balanced, refusal, tuning, and amplification configs`
**Files modified**: `slurm/pipeline/sweep_ad_focused.sbatch`, `scripts/amplify_injections.py` (new)

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

Based on Agent 1's analysis and the experimental logic:

| Priority | Config | Why |
|----------|--------|-----|
| **1 (must-run)** | `ad_750_bal_r` | Combines both highest-value interventions: Fujitsu data + refusal targets. Addresses selectivity AND gibberish. |
| **2 (must-run)** | `ad_1000_bal` | Isolates Fujitsu data effect. Clean comparison to ad_1000_v2. |
| **3 (must-run)** | `ad_1000_a5` | Clean alpha ablation. Low cost, high diagnostic value. |
| 4 (nice-to-have) | `ad_1000_m40` | Tests Phase 3 delay mechanism. Mechanistically interesting. |
| 5 (nice-to-have) | `ad_750_bal` | Shorter balanced for step-count comparison. |
| 6 (can skip) | `ad_750_amp` | Most speculative. Representation-level signal may not change. |
| 7 (can skip) | `ad_500_bal` | Likely dominated by longer balanced configs. |
| 8 (can skip) | `ad_750_bal_amp` | Combines two uncertain interventions. |

**Recommended submission:**
```bash
# Top 3 priority
for c in ad_750_bal_r ad_1000_bal ad_1000_a5; do
  sbatch --export=ALL,CONFIG=$c slurm/pipeline/sweep_ad_focused.sbatch
done

# If GPU hours available, add:
for c in ad_1000_m40 ad_750_bal; do
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

| Comparison | Isolates |
|-----------|----------|
| `ad_1000_bal` vs `ad_1000_v2` | Effect of adding Fujitsu data |
| `ad_750_bal_r` vs `ad_750_bal` | Effect of refusal traces |
| `ad_1000_a5` vs `ad_1000_v2` | Effect of halving alpha |
| `ad_1000_m40` vs `ad_1000_v2` | Effect of doubling margin |
| `ad_750_amp` vs `ad_750_v2` | Effect of injection amplification |

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

| Config | Steps | Data | Fujitsu D | AD Full mal | AD Trunc mal | Benign correct | Benign no_tool | Gib H | Gib B |
|--------|-------|------|-----------|-------------|-------------|----------------|----------------|-------|-------|
| ad_1000_v2 (control) | 1000 | AD | -14% | 31% | 30% | 41% | 15% | 25 | 23 |
| ad_5000_v2 (Pareto) | 5000 | AD | -58% | 32% | 24% | 72% | 5% | 10 | 2 |
| ad_1000_bal | 1000 | bal | ? | ? | ? | ? | ? | ? | ? |
| ad_750_bal_r | 750 | bal+r | ? | ? | ? | ? | ? | ? | ? |
| ad_1000_a5 | 1000 | AD | ? | ? | ? | ? | ? | ? | ? |
| ad_1000_m40 | 1000 | AD | ? | ? | ? | ? | ? | ? | ? |

### Step 3: Diagnose
1. **Did balanced help?** Compare ad_1000_bal vs ad_1000_v2 on all metrics
2. **Did refusal reduce gibberish?** Compare ad_750_bal_r gibberish counts vs ad_750_bal
3. **Did alpha/margin change Phase 3?** Check training logs for phase transition timing
4. **New Pareto point?** Plot and check if any config dominates ad_5000_v2

### Step 4: Decide next steps
- If balanced helps: run `ad_1500_bal`, `ad_3000_bal` for longer balanced training
- If refusal works: add refusal to AD-only configs (`ad_1000_v2r` already exists)
- If alpha=5 helps: try alpha=3 and alpha=7 for finer tuning
- If nothing beats ad_5000_v2: the intervention limit is the selectivity AUC, and only Agent 3's SRMU importance masking can close the gap

---

## 5. Files Changed

| File | Change | Lines |
|------|--------|-------|
| `slurm/pipeline/sweep_ad_focused.sbatch` | 8 new config entries, amplification pipeline, refusal cache fix, updated header/banner | +255 |
| `scripts/amplify_injections.py` | New script: repeats `<INFORMATION>` tags in harmful traces | +100 |

No changes to training code (`src/training/`), evaluation code (`src/evaluation/`), or paper (`paper/`).
