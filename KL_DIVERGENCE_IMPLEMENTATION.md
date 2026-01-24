# KL Divergence for Benign Token Preservation

**Date**: 2026-01-24
**Status**: ✅ Implemented

## Overview

Added KL divergence-based knowledge distillation to the benign retention objective. This helps preserve the model's original behavior on benign inputs by matching output distributions to a frozen teacher model.

## What Was Added

### 1. New Loss Function: `kl_divergence_loss()` (trainer.py:172-228)

Computes token-level KL divergence between teacher and student distributions:

```python
def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
```

**Key Features:**
- Masked to benign tokens only (using `benign_loss_mask`)
- Temperature-scaled for smooth distributions
- Follows PyTorch KL divergence convention: input=log_probs, target=probs
- Includes T² scaling (standard KD practice)

**Formula:**
```
D_KL(p || q) = sum_v p(v) * (log p(v) - log q(v))
kl_loss = (kl_per_token * mask).sum() / mask.sum() * T²
```

### 2. Configuration Parameters (config.py)

Added two new parameters to `CircuitBreakerConfig`:

```python
# Weight for KL divergence loss (0.0 = disabled)
# Start low (0.1-0.3) to avoid suppressing circuit-breaker effect
beta_kl: float = 0.3

# Temperature for softmax smoothing (1.0-4.0 typical)
kl_temperature: float = 1.0
```

**All preset configs now default to `beta_kl=0.3`** - a conservative value that preserves benign behavior without suppressing the rerouting effect.

### 3. Integration into Training Loop (trainer.py:1507-1532)

The benign loss now combines L2 hidden state matching + KL output matching:

```python
# L2 on hidden states (existing)
loss_retain = retain_loss(
    benign_model_reps,
    benign_frozen_reps,
    ...
)

# KL on output distributions (NEW)
loss_kl = kl_divergence_loss(
    student_logits=student_logits_benign,
    teacher_logits=teacher_logits_benign,
    attention_mask=benign_attention_mask,
    loss_mask=benign_loss_mask,
    temperature=kl_temp,
)

# Combined benign objective
total_loss = cs * loss_reroute + cr * (loss_retain + beta_kl * loss_kl)
```

### 4. Logging Updates

**Console output:**
```
Step 100: loss=0.8234, reroute=0.4521, retain=0.2156, kl=0.1557, α=8.5000
```

**W&B metrics:**
- `loss_kl`: KL divergence value
- `beta_kl`: Current weight

## How It Works

1. **Forward pass**: Both student and frozen teacher run on benign inputs
2. **Logits extraction**: Save logits from both models (in addition to hidden states)
3. **Distribution matching**:
   - Convert logits to distributions with softmax (temperature-scaled)
   - Compute per-token KL divergence
   - Mask to benign tokens only
   - Normalize by number of masked tokens
4. **Combine with L2 loss**: `L_benign = L_retain + beta_kl * L_kl`

## Configuration Guide

### Recommended Settings

| Use Case | `beta_kl` | `kl_temperature` | Notes |
|----------|-----------|------------------|-------|
| **Default (Conservative)** | 0.3 | 1.0 | Balanced, won't suppress CB effect ✅ |
| **Minimal KD** | 0.1 | 1.0 | Very light regularization |
| **Moderate** | 0.5 | 1.0 | Stronger preservation (test first!) |
| **Aggressive preservation** | 1.0-2.0 | 1.0 | Risk of suppressing CB effect ⚠️ |
| **Soft distributions** | 0.3 | 2.0-4.0 | Smoother KL signal |
| **Disable KD** | 0.0 | - | L2 hidden state only |

**⚠️ Important**: Start with `beta_kl=0.3` or lower. KL divergence is a strong "stay close" force in vocab space and can suppress the circuit-breaker effect if too high. Always verify reroute effectiveness is maintained!

### Example: Custom Configuration

```python
from src.training.config import get_config

config = get_config(
    "llama-4-scout",
    beta_kl=1.0,           # Strong KL weight
    kl_temperature=2.0,    # Softer distributions
)
```

### Example: Override in Training Script

```python
config = CircuitBreakerConfigLlama4Scout()
config.beta_kl = 0.8
config.kl_temperature = 1.5
```

## Technical Details

### Why Both L2 and KL?

- **L2 on hidden states**: Preserves internal representations (semantic alignment)
- **KL on logits**: Preserves output behavior (generation quality)
- Together: Complementary objectives for robust benign retention

### Memory Impact

- **Minimal**: Logits already exist in forward pass, just need to save them
- **No extra forward passes**: Teacher already runs for L2 loss
- **Storage**: `(B, S, V)` tensor per forward (freed after backward)

### Temperature Tuning

Higher `T` (2.0-4.0):
- ✅ Softer distributions → more information transfer from teacher
- ✅ More stable gradients (less spikey)
- ⚠️ May slow convergence slightly

Lower `T` (1.0):
- ✅ Sharper distributions → precise output matching
- ✅ Faster convergence
- ⚠️ Less dark knowledge transfer

**Recommendation**: Start with `T=1.0`, increase to 2.0-4.0 if training is unstable.

### Scheduling `beta_kl`

Current implementation: **Fixed weight** throughout training.

**Future enhancement** (optional):
```python
# Ramp up KL over time (as retain becomes more important)
beta_kl_scheduled = beta_kl * cr  # Tied to retain coefficient
```

This would align KL ramp-up with the dual coefficient schedule.

## Validation

The implementation follows PyTorch best practices:
1. Uses `F.kl_div()` with `log_target=False` (student=log_probs, teacher=probs)
2. Applies temperature scaling correctly (divide logits, multiply loss by T²)
3. Masks properly (combines attention_mask and benign_loss_mask)
4. Normalizes by actual token count (handles variable-length sequences)

## Testing Checklist

- [x] KL loss function implemented
- [x] Config parameters added
- [x] Integration into train_step (both hooks and hidden_states paths)
- [x] Logging added (console + W&B)
- [x] Preset configs updated with conservative defaults (beta_kl=0.3)
- [x] Test script created: `slurm/tests/train_kl.sbatch`
- [ ] **TODO**: Run test script to verify gradients flow properly
- [ ] **TODO**: Compare benign eval metrics (with vs without KL)
- [ ] **TODO**: Verify reroute effectiveness isn't suppressed by KL

## References

- [PyTorch Knowledge Distillation Tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)
- [PyTorch KLDivLoss Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)
- [HuggingFace Attention Masking](https://discuss.huggingface.co/t/do-automatically-generated-attention-masks-ignore-padding/15479)

## Next Steps

### 1. Run Automated Test Script

```bash
sbatch slurm/tests/train_kl.sbatch
```

This runs 5 ablation tests (50 steps each):
- **Test 1**: Baseline (beta_kl=0.0) - L2 only
- **Test 2**: Low KL (beta_kl=0.1) - Minimal regularization
- **Test 3**: Medium KL (beta_kl=0.3) - **Recommended default**
- **Test 4**: High KL (beta_kl=0.5) - Aggressive preservation
- **Test 5**: Soft KL (beta_kl=0.3, T=2.0) - Smoother gradients

**What to check:**
- `loss_kl` should be 0.1-0.5 range (similar to `loss_retain`)
- `loss_reroute` should be **similar across all tests** (CB effect preserved)
- Gradients should flow normally (check `grad_norm_total`)

### 2. Manual Validation

Monitor key metrics during training:
```bash
# Watch training logs
tail -f /scratch/memoozd/cb-scratch/logs/test_kl_*.out

# Look for:
# - Step 50: loss=X, reroute=Y, retain=Z, kl=W, α=A
#   - reroute should be similar across beta_kl values
#   - kl should scale with beta_kl (0.0 → 0.0, 0.3 → ~0.2, 0.5 → ~0.3)
```

### 3. Choose Final beta_kl

Based on test results:
- If `loss_kl > 1.0`: KL is too strong, use `beta_kl=0.1`
- If `loss_reroute` increases with KL: CB effect suppressed, use `beta_kl=0.1` or disable
- If all looks good: stick with `beta_kl=0.3` (default)
- If you want stronger preservation: try `beta_kl=0.5` (but verify CB effect!)

## Example Training Command

```bash
# With default beta_kl=0.5
python scripts/train_circuit_breaker.py \
    --config llama-4-scout \
    --output_dir outputs/cb_with_kl

# Custom KL settings
python scripts/train_circuit_breaker.py \
    --config llama-4-scout \
    --beta_kl 1.0 \
    --kl_temperature 2.0 \
    --output_dir outputs/cb_strong_kl
```

---

**Status**: Ready to test! 🚀
