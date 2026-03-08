# Sweep 7: Bug Fixes + New Loss Modes — Implementation Status

**Date:** March 8, 2026 (updated)
**Reference:** `notes/RRFA_TECHNICAL_PLAN.md` (authoritative plan)

---

## What Was Done

### Bug Fixes (Phase 1)

| Fix | File | Change |
|-----|------|--------|
| KL on benign = ZERO | `losses.py` | `loss_mask=None` in `triplet_full_loss` benign KL |
| Template garbage | `augment_llm.py` | `_build_completed_trace` now 3 msgs (was 5) |
| Distance = zero grad | `losses.py` | Added `dl2rc` = `L2 + 10*ReLU(1-cos_sim)` |
| Benign data silently filtered | `train_schema.py` | Removed zero-mask filter |
| Fujitsu data gen | `sweep_completed_v1.sbatch` | Added `--dry-run`, 3-path fallback |
| Heredoc arg passing | `sweep_completed_v1.sbatch` | `python3 - $ARGS << 'EOF'` (was broken) |

### New Loss Modes (Phase 2)

| Mode | File | Description |
|------|------|-------------|
| `per_token_cb` | `losses.py` | Per-token distance with margin. No pooling, no cluster center. Harmful=injection_aware, benign=attention_mask |
| `simplified_pooled` | `losses.py` | Pooled distance with margin. No cluster center, no cross-terms |
| `original_rr` | `losses.py` | Paper exact (Zou 2024). **FAILED** — zero gradient at LoRA init. Do not use. |

### Distance Functions

| Distance | Code | Description |
|----------|------|-------------|
| `dl2rc` | `L2 + 10*ReLU(1-cos_sim)` | Default. Non-zero gradient at LoRA scale |
| `dl2sq` | `(a-b).pow(2).sum(-1)` | Squared L2. Simpler, always has gradient |
| All others | `d2`, `dcos`, `dmix`, `d0` | Legacy, mostly for comparison |

### Margin-Free Option (Option C)

| Feature | Description |
|---------|-------------|
| `--margin-free` | Eliminates margin hyperparameters entirely |
| Harmful | `exp(-d/scale)` — bounded [0,1], always has gradient, auto-scaled from frozen norms |
| Benign | `d.mean()` — direct penalty on drift |
| Usage | `MARGIN_FREE=true LOSS_MODE=per_token_cb sbatch ...` |

### Infrastructure

| Change | File | Description |
|--------|------|-------------|
| Trainer wiring | `trainer.py` | All modes wired with step-0 diagnostics |
| Distance configurable | `trainer.py`, `losses.py` | `DISTANCE` env var passes through to per_token_cb/simplified_pooled |
| Step-0 distance printing | `trainer.py` | Prints initial distances per layer for margin calibration |
| Training history summary | `sweep_completed_v1.sbatch` | TRAINDIAG heredoc parses train.log |
| run_config.json | `sweep_completed_v1.sbatch` | Saves all hyperparameters |
| Results summary | `sweep_completed_v1.sbatch` | Formatted eval results at end |

---

## How to Run

```bash
# Primary runs (per_token_cb is default):
LOSS_MODE=per_token_cb      sbatch slurm/pipeline/sweep_completed_v1.sbatch
LOSS_MODE=simplified_pooled sbatch slurm/pipeline/sweep_completed_v1.sbatch

# With squared L2 distance:
LOSS_MODE=per_token_cb DISTANCE=dl2sq sbatch slurm/pipeline/sweep_completed_v1.sbatch

# Margin-free (Option C — no margin tuning):
LOSS_MODE=per_token_cb MARGIN_FREE=true sbatch slurm/pipeline/sweep_completed_v1.sbatch

# With KL preservation:
LOSS_MODE=per_token_cb GAMMA_KL=0.3 sbatch slurm/pipeline/sweep_completed_v1.sbatch

# NOT recommended (FAILED):
# LOSS_MODE=original_rr sbatch ...  # zero gradient at LoRA init
```

---

## What to Watch in Logs

### Step-0 Diagnostics
- `per_token_cb: distance=dl2rc | harmful mask active X/Y | benign mask active X/Y`
- Step-0 layer distances: `harmful_dist=0.XXXX benign_dist=0.XXXX (margins: h=1.6 b=1.2)`
- If margin_free: `per_token_cb: distance=dl2rc | MARGIN-FREE | ...`

### Every 50 Steps
```
[Step N] mode=per_token_cb | L_b=0.xxxx L_h=0.xxxx L_kl=0.xxxx d_h=0.xxx d_b=0.xxx
```
- **L_kl MUST be non-zero** (was zero before fix)
- **d_h should increase** (harmful reps pushed away)
- **d_b should stay small** (benign reps close to frozen)
- **grad_norm MUST be non-zero**

---

## Known Risks / Open Items

1. **Margin calibration**: 1.2/1.6 were tuned for cosine [0,2]. With dl2rc scale is different. Use step-0 distances to recalibrate. Or use `MARGIN_FREE=true` to skip margins entirely.
2. **original_rr FAILED**: Paper's ReLU(cos_sim) has zero gradient when a=b (LoRA init). c_s starts at 0. Model never moves. Kept in code but DO NOT USE.
3. **Benign data was being filtered in ALL previous injection_aware sweeps**: Now fixed. Results may change significantly with 16K+ benign samples.

---

## Files Modified

```
src/training/losses.py          — dl2rc, dl2sq, per_token_cb (margin-free opt), simplified_pooled, original_rr
src/training/trainer.py         — Wire all modes, step-0 diagnostics, margin_free flag
src/training/train_schema.py    — --margin-free CLI arg, config mapping
scripts/augment_llm.py          — Fujitsu 3-message truncation
slurm/pipeline/sweep_completed_v1.sbatch — DISTANCE/MARGIN_FREE env vars, training summary, results summary
```