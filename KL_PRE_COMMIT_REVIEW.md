# KL Divergence - Pre-Commit Review

**Date**: 2026-01-24
**Status**: ✅ READY TO COMMIT

---

## Questions Addressed

### 1. ✅ KL Direction - Correct

**Implementation:**
```python
student_log_probs = F.log_softmax(student_logits / T, dim=-1)  # log q(v)
teacher_probs = F.softmax(teacher_logits / T, dim=-1)          # p(v)
F.kl_div(student_log_probs, teacher_probs, log_target=False)   # D_KL(p || q)
```

**Verification:**
- ✅ Direction: `D_KL(teacher || student)` (standard distillation)
- ✅ PyTorch convention: `F.kl_div(input=log_probs, target=probs)`
- ✅ Computes: `sum_v p(v) * (log p(v) - log q(v))`

**Location:** `src/training/trainer.py:172-228`

---

### 2. ✅ Teacher Logits Detachment - Correct

**Forward Pass Protection:**
```python
# In both hooks and hidden_states paths:
with torch.no_grad():
    frozen_outputs = self.frozen_model(...)
    teacher_logits = frozen_outputs.logits
```

**Parameter Protection:**
```python
# In _load_models():
for param in self.frozen_model.parameters():
    param.requires_grad = False
```

**Verification:**
- ✅ Teacher forward wrapped in `torch.no_grad()` (lines 1432-1443, 1405-1417)
- ✅ Teacher params set to `requires_grad=False` (line 1144-1145)
- ✅ No risk of backprop into teacher graph

---

### 3. ✅ Mask Semantics - Correct

**Mask Computation:**
```python
# benign_loss_mask is computed via _compute_completion_mask():
# 1. Finds assistant header using find_assistant_start_position()
# 2. Masks out prompt tokens (before header)
# 3. Keeps ONLY benign completion tokens
```

**KL Application:**
```python
# KL is ONLY applied on benign tokens:
loss_kl = kl_divergence_loss(
    student_logits=student_logits_benign,  # Only benign samples
    teacher_logits=teacher_logits_benign,  # Only benign samples
    loss_mask=benign_loss_mask,            # Only completion tokens
)
```

**Verification:**
- ✅ KL applied **only** on `benign_loss_mask` (truly benign segments)
- ✅ Never applied on harmful samples (harmful samples use reroute loss)
- ✅ Excludes prompt tokens (only completion tokens)
- ✅ Tool-call tokens within benign completions ARE included (preserves capability)

**What's Excluded:**
- ❌ Prompt tokens (masked out)
- ❌ Harmful samples (use reroute_loss instead)
- ❌ Padding tokens (masked by attention_mask)

**What's Included:**
- ✅ Benign assistant completion tokens
- ✅ Tool-call tokens in benign responses (we WANT to preserve this!)

**Location:** `src/training/dataset.py:809-863`, `src/training/trainer.py:1519-1528`

---

### 4. ✅ Default beta_kl - ADJUSTED

**Original:** `beta_kl = 0.5` (too aggressive, not empirically validated)

**Updated:** `beta_kl = 0.3` (conservative, less risk of suppressing CB effect)

**Rationale:**
- KL divergence is a strong "stay close to teacher" force in vocab space
- Higher beta_kl could suppress the circuit-breaker rerouting effect
- Better to start conservative and increase if needed
- 0.3 provides meaningful benign preservation without dominating the loss

**Changed Files:**
- `src/training/config.py`: All preset configs now use `beta_kl=0.3`
  - `CircuitBreakerConfig` (base)
  - `CircuitBreakerConfigLlama3_8B`
  - `CircuitBreakerConfigLlama3_1_8B_Instruct`
  - `CircuitBreakerConfigMistral_7B`
  - `CircuitBreakerConfigLlama4Scout`

**SLURM Scripts:** Still use explicit values (can be adjusted per experiment)
- `slurm/05_train.sbatch`: `--beta-kl 0.5` (for testing)
- `slurm/HPO/*.sbatch`: `--beta-kl 0.5` (HPO experiments)

---

## New Test Script

**Created:** `slurm/tests/train_kl.sbatch`

**What it does:**
- Runs 5 ablation experiments (50 steps each)
- Tests `beta_kl ∈ {0.0, 0.1, 0.3, 0.5}` and `kl_temperature=2.0`
- Monitors gradient flow and loss curves
- Generates comparison report

**Run with:**
```bash
sbatch slurm/tests/train_kl.sbatch
```

**What to check:**
1. `loss_reroute` should be **similar** across all beta_kl values (CB effect preserved)
2. `loss_kl` should scale with beta_kl (0.0 → 0.0, 0.3 → ~0.2, 0.5 → ~0.3)
3. `grad_norm_total` should be similar (no gradient suppression)

---

## Summary of Changes

### Code Files (3)
1. ✅ `src/training/config.py` - Added `beta_kl`, `kl_temperature`, updated defaults to 0.3
2. ✅ `src/training/trainer.py` - Added `kl_divergence_loss()`, integrated into `train_step()`
3. ✅ `src/training/train.py` - Added CLI args `--beta-kl`, `--kl-temperature`

### SLURM Scripts (4)
4. ✅ `slurm/05_train.sbatch` - Added `--beta-kl 0.5 --kl-temperature 1.0`
5. ✅ `slurm/HPO/05_train_mid1.sbatch` - Added KL params
6. ✅ `slurm/HPO/05_train_mid2.sbatch` - Added KL params
7. ✅ `slurm/HPO/05_train_mid3.sbatch` - Added KL params

### Test Script (1)
8. ✅ `slurm/tests/train_kl.sbatch` - Automated ablation tests

### Documentation (3)
9. ✅ `KL_DIVERGENCE_IMPLEMENTATION.md` - Full technical documentation
10. ✅ `PIPELINE_UPDATES.md` - SLURM script update summary
11. ✅ `KL_PRE_COMMIT_REVIEW.md` - This file

---

## Verification Checklist

- [x] KL direction: D_KL(teacher || student) ✅
- [x] Teacher detachment: `torch.no_grad()` + `requires_grad=False` ✅
- [x] Mask semantics: Only benign completion tokens ✅
- [x] Conservative default: `beta_kl=0.3` ✅
- [x] Test script: Automated ablation tests ✅
- [x] Documentation: Complete ✅
- [x] SLURM pipeline: Updated ✅

---

## Recommended Workflow

### Before Committing
1. ✅ Review this document
2. ✅ Verify all files modified correctly
3. ✅ Commit changes

### After Committing
1. Run test script: `sbatch slurm/tests/train_kl.sbatch`
2. Monitor logs: Check `loss_kl` magnitude and `loss_reroute` consistency
3. Verify CB effect: Ensure reroute loss doesn't increase with KL
4. Adjust if needed: Lower `beta_kl` if CB effect is suppressed

---

## Final Recommendation

**✅ ALL GOOD - READY TO COMMIT**

All concerns addressed:
- ✅ KL direction is correct
- ✅ Teacher is properly detached
- ✅ Masks are correctly scoped to benign segments only
- ✅ Default `beta_kl=0.3` is conservative and safe
- ✅ Test script ready for empirical validation

**Next step:** Commit and run `slurm/tests/train_kl.sbatch` to validate!

---

**Commit message suggestion:**
```
Add KL divergence for benign token preservation

- Implement D_KL(teacher || student) on benign completion tokens
- Conservative default: beta_kl=0.3 (tunable via --beta-kl)
- Teacher properly detached (no_grad + requires_grad=False)
- Masked to benign segments only (preserves CB reroute effect)
- Test script: slurm/tests/train_kl.sbatch for ablation studies
- Docs: KL_DIVERGENCE_IMPLEMENTATION.md for full details
```
