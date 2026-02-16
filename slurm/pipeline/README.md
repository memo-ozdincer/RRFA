# SLURM Pipeline (Minimal Surface Area)

This directory intentionally uses two canonical entrypoints:

1. `slurm/pipeline/unified_pipeline.sbatch` (regular run)
2. `slurm/pipeline/sweep_hparams.sbatch` (grid sweep)

Convenience wrapper:
- `slurm/pipeline/sweep_hparams_simple.sbatch` (reduced sweep preset; forwards into `sweep_hparams.sbatch`)

## 1) Regular training run (default)

Use:

```bash
sbatch slurm/pipeline/unified_pipeline.sbatch
```

Defaults:
- Uses all configured datasets (`agentdojo,fujitsu_b4,llmail_inject`)
- Runs full data path (ETL_A -> generate/judge optional -> ETL_B -> split -> train -> eval)
- Uses `LOSS_MODE=triplet_full`
- Uses detailed logging and writes train/eval artifacts under `/scratch/memoozd/cb-scratch`

High-value toggles:

```bash
# Skip data generation/processing, only train+eval
SKIP_ETL_A=true SKIP_GENERATE=true SKIP_ETL_B=true SKIP_SPLIT=true sbatch slurm/pipeline/unified_pipeline.sbatch

# Disable training/eval and only build data artifacts
SKIP_TRAIN=true SKIP_EVAL=true sbatch slurm/pipeline/unified_pipeline.sbatch

# Remove one dataset
DATASETS=agentdojo sbatch slurm/pipeline/unified_pipeline.sbatch

# Quick smoke run
TOTAL_STEPS=30 EVAL_LIMIT=20 DATASETS=agentdojo sbatch slurm/pipeline/unified_pipeline.sbatch

# Skip LLMail if needed
DATASETS=agentdojo,fujitsu_b4 sbatch slurm/pipeline/unified_pipeline.sbatch
```

## 2) Hyperparameter sweep

Use:

```bash
sbatch slurm/pipeline/sweep_hparams.sbatch
```

Typical overrides:

```bash
QUICK_TEST=true sbatch slurm/pipeline/sweep_hparams.sbatch
ALPHAS=8.0,10.0 LAYER_CONFIGS=10,20 POLICIES=assistant_only,cb_full_sequence sbatch slurm/pipeline/sweep_hparams.sbatch

# Reduced preset wrapper
sbatch slurm/pipeline/sweep_hparams_simple.sbatch
```

## Loss configuration

Both canonical scripts pass through the same training loss settings into `src/training/train_schema.py`.
`src/training/train_schema.py` now delegates into the shared `src/training/trainer.py` core,
so both training entrypoints use the same loss implementation and step logic.

Supported loss modes:
- `triplet_full` (default): full triplet paper objective (benign triplet + harmful triplet + benign KL)
- `legacy_cb`: CB-style reroute ReLU + benign L2 (+ optional KL)
- `legacy_schema`: historical schema baseline (random reroute + benign CE)

Key env vars:
- `LOSS_MODE`
- `LOSS_WEIGHTING` (legacy modes only)
- `TRIPLET_ALPHA_BENIGN`, `TRIPLET_BETA_HARMFUL`, `TRIPLET_GAMMA_KL`
- `TRIPLET_MARGIN_BENIGN`, `TRIPLET_MARGIN_HARMFUL`
- `TRIPLET_BENIGN_POS_DISTANCE`, `TRIPLET_BENIGN_NEG_DISTANCE`
- `TRIPLET_HARMFUL_POS_DISTANCE`, `TRIPLET_HARMFUL_NEG_DISTANCE`
- `TRIPLET_MIX_L2_WEIGHT`, `TRIPLET_MIX_COS_WEIGHT`
