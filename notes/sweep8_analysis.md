# Sweep 8 Analysis: Why Sweep 7 Failed and What Changed

**Date**: March 8, 2026
**Context**: Sweep 7 (original_rr + per_token_cb) produced zero eval effect. This document explains why and what Sweep 8 changes.

---

## Sweep 7 Results (FAILED)

### original_rr: cos_sim stuck at 1.0 (Training Failure)
```
Step   0: cs=0.000 cr=10.000 cos_sim=1.0000 grad_norm=0.00
Step  50: cs=1.667 cr=8.333  cos_sim=1.0000 grad_norm=2.88
Step 100: cs=3.333 cr=6.667  cos_sim=1.0000 grad_norm=2.93
Step 150: cs=5.000 cr=5.000  cos_sim=1.0000 (end of training)
```
The reroute loss never decreased. The model learned nothing about rerouting.

### per_token_cb (dl2rc): Good Training, Zero Eval Effect
```
Training: d_h: 0.011 -> 4.98 (harmful reps moved far)
Training: L_h: 1.59 -> 0.0001 (loss converged)
Eval:     Fujitsu ASR=84% (baseline=83%) -- NO improvement
          CB responses literally identical to baseline
```
The representations changed at layers 10/20, but model output was unchanged.

---

## Root Cause: Three Compounding Regressions from Old Working Sweep

We compared Sweep 7 against the old working sweep at commit `d086ec9` (sweep_hparams_simple, ~Feb 2). That sweep actually changed outputs significantly and in the right direction.

### Regression 1: Coefficient Schedule Direction (CRITICAL)

**Old (worked):** `get_dual_coefficients` — cs: alpha_max -> 0 (reroute DECAYS), cr: 0 -> alpha_max (retain RAMPS)
```
Step 0:   cs=10.0 cr=0.0  -- FULL reroute, ZERO retain
Step 100: cs=5.0  cr=5.0  -- balanced
Step 200: cs=0.0  cr=10.0 -- ZERO reroute, FULL retain
```

**Sweep 7 (failed):** "Paper exact" schedule — cs: 0 -> alpha/2, cr: alpha -> alpha/2
```
Step 0:   cs=0.0  cr=10.0 -- ZERO reroute, FULL retain
Step 75:  cs=2.5  cr=7.5  -- retain still 3x stronger
Step 150: cs=5.0  cr=5.0  -- equal, but training is over
```

**Why the old schedule works better for our task:**
- The paper's harmful vs benign are *very different distributions* (bomb-making prompts vs general chat). Reroute signal is strong even at low cs.
- Our harmful vs benign are *similar distributions* (both tool-calling conversations, differing only by injection presence). Need aggressive initial rerouting.
- With only 144 harmful tokens (injection_aware mask) vs 2820 benign tokens, the retain gradient completely overwhelms reroute when cr >> cs.

### Regression 2: LoRA Layer Restriction (CRITICAL)

**Old (worked):** `train_schema.py` did NOT pass `layers_to_transform` to `LoraConfig`. Result: LoRA on ALL 32 layers.

**Sweep 7 (failed):** `--lora-target-layers "0-20"` passed, restricting LoRA to layers 0-20 only.

**Why this matters:**
- CB target layers are 10 and 20. The loss pushes representations at these layers away from frozen.
- With LoRA on layers 21-31, these later layers can *learn to amplify* the CB perturbation and translate it into output distribution changes.
- With LoRA only on 0-20, layers 21-31 are frozen. The CB perturbation at layer 20 passes through frozen layers that may dampen or absorb it without changing the output.
- This explains why per_token_cb showed great training metrics (d_h=4.98) but zero eval effect -- the representation shift at layer 20 never propagated to output behavior.

### Regression 3: Learning Rate and LoRA Alpha (Compounding)

| Parameter     | Old (worked) | Sweep 7 (failed) | Ratio |
|--------------|-------------|-------------------|-------|
| Learning rate | 5e-5        | 1e-5              | 5x    |
| LoRA alpha    | 32 (scale 2.0) | 16 (scale 1.0) | 2x    |
| Combined      |             |                   | 10x weaker |

The effective LoRA learning rate is `lr * (alpha/r)`. Old: `5e-5 * 2.0 = 1e-4`. Sweep 7: `1e-5 * 1.0 = 1e-5`. The LoRA weights updated 10x slower.

---

## Additional Differences (Minor)

| Factor | Old | Sweep 7 | Impact |
|--------|-----|---------|--------|
| Batch size | 8x1=8 | 4x4=16 | Different noise profile |
| Seq length | 2048 | 4096 | Denser masking per token in old |
| KL | beta_kl=0.5 under cr | gamma_kl=0.0 | Old had KL ramping with retain |
| Frozen model | Separate full copy | disable_adapter() | Semantically equivalent |
| Backward pass | Two sequential .backward() | One combined .backward() | Mathematically equivalent |
| Gradient checkpointing | No use_reentrant | use_reentrant=False | Fixes DDP deadlock |

---

## Sweep 8 Changes

### trainer.py
1. **Unified schedule for ALL loss modes**: All modes now use `get_dual_coefficients` (cs starts high, decays; cr starts low, ramps up). This replaces:
   - original_rr's "paper exact" schedule (cs:0->5, cr:10->5)
   - Triplet modes' warmup schedule (cs:0->1, cr=1 fixed)
   - legacy_cb already used get_dual_coefficients (unchanged)

2. **alpha_benign uses cr, beta_harmful uses cs**: In all loss modes (triplet_full, per_token_cb, simplified_pooled, cosine_simple, l2_simple), the benign (retain) coefficient ramps up while the harmful (reroute) coefficient decays. This matches the old working behavior: strong initial rerouting, gradual retain lock-in.

3. **original_rr**: `total = cs * L_rr + cr * L_ret + gamma_kl * L_kl`. At step 0: pure reroute + fixed KL. At step T: pure retain + fixed KL.

### sweep_completed_v1.sbatch
- BATCH_SIZE: 4 -> 8, GRAD_ACCUM: 4 -> 1 (effective 8, matches old)
- LEARNING_RATE: 1e-5 -> 5e-5
- LORA_ALPHA: 16 -> 32
- LORA_LAYERS: "0-20" -> removed (all 32 layers via preset default)
- TOTAL_STEPS: 150/300 -> 200 (overridable via TOTAL_STEPS env var)
- GAMMA_KL: unified to 0.3 for all modes

### config.py
- Added `margin_free: bool = False` field (was missing, caused ValueError crash)

---

## Expected Behavior After Fix

At step 0 with original_rr (alpha_max=10):
```
cs=10.0, cr=0.0
total_loss = 10.0 * ReLU(cos_sim) + 0.0 * L2_retain + 0.3 * KL
           = pure reroute + mild KL anchor
```
The model is free to push harmful representations away immediately. No competing retain gradient.

At step 100 (midpoint):
```
cs=5.0, cr=5.0
total_loss = 5.0 * ReLU(cos_sim) + 5.0 * L2_retain + 0.3 * KL
           = balanced reroute + retain
```

At step 200 (end):
```
cs=0.0, cr=10.0
total_loss = 0.0 * ReLU(cos_sim) + 10.0 * L2_retain + 0.3 * KL
           = pure retain + KL (locks in benign behavior)
```

With LoRA on all 32 layers, the perturbation at CB layers 10/20 can propagate through LoRA-equipped layers 21-31 to actually change the output distribution.

---

## Lessons Learned

1. **"Matching the paper" isn't always right.** The paper's schedule was optimized for their setting (very different harmful/benign distributions). Our task has fundamentally different characteristics.

2. **LoRA layer coverage matters for CB.** The CB loss only targets intermediate layers (10, 20). If downstream layers are frozen, they can absorb the perturbation without changing output. LoRA on downstream layers lets the model learn to translate representation changes into behavioral changes.

3. **Verify against known-good baselines.** The old sweep worked. Every change from that baseline should be justified. The "improvements" (paper schedule, restricted LoRA, lower LR) were all regressions in disguise.

4. **Training metrics != eval metrics.** per_token_cb showed perfect training convergence (d_h=4.98, L_h->0) but zero eval effect. Representation-level metrics don't guarantee behavior change, especially when downstream layers are frozen.
