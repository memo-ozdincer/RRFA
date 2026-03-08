# Sweep 7: Paper-Exact Loss + Bug Fixes — Implementation Status

**Date:** March 7, 2026
**Key Insight:** Our coefficient schedule was backwards. Paper's ReLU(cos_sim) + L2 loss works with LoRA — the schedule ramps rerouting UP, not down.

---

## What Changed (Mar 7)

### Critical Discovery: Backwards Coefficient Schedule
- **Our schedule**: cs: 1→0 (reroute decays), cr: 0→1 (retain ramps)
- **Paper's schedule**: c_s: 0→α/2 (reroute ramps UP), c_r: α→α/2 (retain ramps DOWN)
- This caused all previous "legacy_cb FAILED" and "coefficient schedule FAILED" conclusions
- The paper DOES use LoRA (LoRRA). Same setup as ours. No loss/LoRA incompatibility.

### New Loss Mode: `original_rr`
- Paper's exact Algorithm 1: `L = c_s * ReLU(cos_sim) + c_r * ||frozen - cb||_2`
- Optional KL: when γ_kl > 0, adds `+ γ_kl * KL(cb || frozen)` on benign
- c_s = α·step/(2T), c_r = α·(1-step/(2T)), α=10
- **No** margins, triplet, cluster centers, dl2rc, contrastive pairs
- Harmful mask: injection_aware (loss_mask). Benign mask: attention_mask.
- 150 steps (paper default)

### Bug Fixes (also applied)
| Fix | File | Change |
|-----|------|--------|
| KL on benign = ZERO | `losses.py` | `loss_mask=None` for benign KL |
| Fujitsu template garbage | `augment_llm.py` | 3-message traces (was 5) |
| Benign data silently filtered | `train_schema.py` | Removed zero-mask filter |
| augment_llm.py needs --dry-run | `sweep_completed_v1.sbatch` | Added --dry-run (no vLLM on compute) |
| Fujitsu data gen fallback | `sweep_completed_v1.sbatch` | Uses fujitsu_b4_ds.jsonl directly if available |

---

## How to Run

```bash
# Paper exact (default, recommended):
sbatch slurm/pipeline/sweep_completed_v1.sbatch

# Paper + KL preservation:
GAMMA_KL=0.3 sbatch slurm/pipeline/sweep_completed_v1.sbatch

# Other modes (with dl2rc bug fixes):
LOSS_MODE=triplet_full sbatch slurm/pipeline/sweep_completed_v1.sbatch
LOSS_MODE=per_token_cb sbatch slurm/pipeline/sweep_completed_v1.sbatch
```

---

## What to Watch in Logs

### Step-0 Diagnostic
```
original_rr: c_s=0.000 c_r=10.000 alpha_max=10.0 (no KL)
harmful mask active: X/Y | benign mask active: X/Y
```
- c_s=0 at step 0 is EXPECTED (paper schedule)
- Benign mask should be NON-ZERO (attention_mask)

### Every 50 Steps
```
[Step N] mode=original_rr cs=X.XXX cr=Y.YYY | cos_sim=0.XXXX | L_rr=0.XXXX L_ret=0.XXXX
  grad_norm=X.XXXX
```
- **cos_sim**: Should start at 1.0, MUST decrease by step 50. If stuck at 1.0000 → problem.
- **L_rr**: Should decrease from ~1.0 toward 0 (harmful becoming orthogonal)
- **L_ret**: Should start ~0, increase slightly (benign drift from rerouting)
- **grad_norm**: MUST be non-zero by step 5-10 (weight decay breaks symmetry)
- **cs/cr**: cs ramps UP (0→5), cr ramps DOWN (10→5)

---

## Hypotheses

1. **H1**: original_rr (γ_kl=0) matches or beats Sweep 5 best (33.7% ASR)
2. **H2**: If it works → confirms backwards schedule was the root cause
3. **H3**: γ_kl=0.3 improves benign preservation without hurting ASR
4. **H4**: If it underperforms → difference is data/task-specific, not loss

---

## Files Modified

```
src/training/losses.py          — original_rr_loss (with optional KL), LOSS_MODE_ORIGINAL_RR constant
src/training/trainer.py         — Paper's coefficient schedule, dispatch block, diagnostics
src/training/train_schema.py    — Auto-picks up via SUPPORTED_LOSS_MODES
scripts/augment_llm.py          — Fujitsu 3-message truncation
slurm/pipeline/sweep_completed_v1.sbatch — original_rr default, GAMMA_KL env var, data gen fixes
```

## Detailed Analysis
See `notes/experiment_log_mar7.tex` for the full retrospective, paper analysis, and complexity progression.
