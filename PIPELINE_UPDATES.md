# SLURM Pipeline Updates for KL Divergence

**Date**: 2026-01-24
**Status**: ✅ Complete

## Summary

Updated all SLURM training scripts to include the new KL divergence parameters for benign token preservation.

## Files Modified

### 1. Training Script (src/training/train.py)

**Added CLI arguments:**
```bash
--beta-kl BETA_KL               # Weight for KL loss (0.0 = disabled, default: 0.5)
--kl-temperature KL_TEMPERATURE # Temperature for KL softmax (default: 1.0)
```

**Added config printout:**
- Now displays `KL Beta: {config.beta_kl}`
- Now displays `KL Temperature: {config.kl_temperature}`

### 2. Main Training Script (slurm/05_train.sbatch)

**Updated command:**
```bash
accelerate launch --num_processes ${SLURM_GPUS_ON_NODE:-1} \
    src/training/train.py \
    --preset llama-3.1-8b-instruct \
    ...
    --beta-kl 0.5 \                    # NEW
    --kl-temperature 1.0 \             # NEW
    --wandb-project "circuit-breakers"
```

### 3. HPO Scripts (slurm/HPO/)

**All three HPO variants updated:**
- `slurm/HPO/05_train_mid1.sbatch` (alpha_max=2.5, 350 steps)
- `slurm/HPO/05_train_mid2.sbatch` (alpha_max=5.0, 400 steps)
- `slurm/HPO/05_train_mid3.sbatch` (alpha_max=7.5, 450 steps)

**Each now includes:**
```bash
--beta-kl 0.5 \
--kl-temperature 1.0 \
```

## Scripts NOT Modified

The following scripts were **not modified** as they don't involve training:

- ✅ `slurm/01_generate_ds.sbatch` - Data generation (Ds harmful set)
- ✅ `slurm/02_generate_dr.sbatch` - Data generation (Dr benign set)
- ✅ `slurm/03_create_eval.sbatch` - Evaluation set creation
- ✅ `slurm/04_rebuild_data.sbatch` - Data preprocessing
- ✅ `slurm/06_eval.sbatch` - Evaluation (inference only)
- ✅ `slurm/full_pipeline.sbatch` - Pipeline orchestrator (calls other scripts)

## Default Values

All training scripts now use:
- **`beta_kl = 0.5`** - Balanced weight for KL divergence loss
- **`kl_temperature = 1.0`** - Standard temperature (sharp distributions)

These match the defaults in `CircuitBreakerConfig` presets.

## Usage Examples

### Run main training with defaults
```bash
sbatch slurm/05_train.sbatch
# Uses beta_kl=0.5, kl_temperature=1.0
```

### Run with custom KL settings
```bash
# Edit the .sbatch file or override via train.py:
accelerate launch --num_processes 4 \
    src/training/train.py \
    --preset llama-3.1-8b-instruct \
    --beta-kl 1.0 \           # Stronger KL weight
    --kl-temperature 2.0      # Softer distributions
```

### Disable KL divergence
```bash
# Edit .sbatch and change:
--beta-kl 0.0 \  # Disables KL, uses L2 only
```

### Run HPO experiments
```bash
# All three HPO configs now include KL
sbatch slurm/HPO/05_train_mid1.sbatch  # Lower alpha
sbatch slurm/HPO/05_train_mid2.sbatch  # Medium alpha
sbatch slurm/HPO/05_train_mid3.sbatch  # Higher alpha
```

## Verification Checklist

- [x] `src/training/train.py` - Added `--beta-kl` and `--kl-temperature` args
- [x] `src/training/train.py` - Added config overrides for KL parameters
- [x] `src/training/train.py` - Added KL parameters to config printout
- [x] `slurm/05_train.sbatch` - Added `--beta-kl 0.5 --kl-temperature 1.0`
- [x] `slurm/HPO/05_train_mid1.sbatch` - Added KL parameters
- [x] `slurm/HPO/05_train_mid2.sbatch` - Added KL parameters
- [x] `slurm/HPO/05_train_mid3.sbatch` - Added KL parameters
- [x] All preset configs default to `beta_kl=0.5`

## Testing

**Before running on cluster:**
1. Test locally with quick run:
   ```bash
   python src/training/train.py \
       --preset llama-3.1-8b-instruct \
       --total-steps 10 \
       --beta-kl 0.5 \
       --no-wandb
   ```

2. Verify KL loss appears in output:
   ```
   Step 10: loss=0.8234, reroute=0.4521, retain=0.2156, kl=0.1557, α=8.5000
   ```

3. Check config printout includes:
   ```
   KL Beta: 0.5
   KL Temperature: 1.0
   ```

**On cluster:**
```bash
sbatch slurm/05_train.sbatch
# Monitor logs for KL loss values
tail -f /scratch/memoozd/cb-scratch/logs/mvp_train_*.out
```

## Rollback Instructions

If you need to revert to L2-only retention:

**Option 1: Edit SLURM scripts**
```bash
# Change in all .sbatch files:
--beta-kl 0.0 \  # Disables KL
```

**Option 2: Revert to previous commit**
```bash
git checkout <commit_before_kl_changes>
```

## Related Documentation

- `KL_DIVERGENCE_IMPLEMENTATION.md` - Technical details of KL implementation
- `IMPLEMENTATION_AUDIT.md` - Overall training implementation audit
- `QUICK_FIXES.md` - Quick reference guide

---

**Status**: All pipeline scripts updated and ready to use! 🚀
