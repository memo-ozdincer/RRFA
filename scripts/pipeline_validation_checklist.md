# Pipeline Validation Checklist (Lossmask + MWCS + Tool Context)

This checklist is designed for cluster runs launched from:
- `slurm/pipeline/unified_pipeline.sbatch`
- `slurm/pipeline/sweep_hparams.sbatch`
- `slurm/pipeline/sweep_hparams_simple.sbatch`

## What Was Changed
- Added explicit MWCS control in pipeline entrypoints:
  - `MWCS_SCHEDULE` (`none`, `balanced_cb`, `attack_focus`, `capability_focus`, `staged_introduction`)
  - `MWCS_REGISTRY` (default: `configs/mwcs_registry_v1.json`)
  - `MWCS_STEP` (for schedule phase selection)
- Added AgentDojo eval-split control:
  - `AGENTDOJO_EVAL_SPLIT` in `{complete,harmful,benign}` (default: `complete`)
- Added run metadata per sweep run:
  - `run_config.json` in each run directory
- Added MWCS registry defaults:
  - `configs/mwcs_registry_v1.json`
- Updated evaluator behavior for sample-context runs:
  - Generation comparison now supports per-sample tool inference when embedded tools are missing
  - Evaluation context source reports `per_sample_inferred`
- Added external pre-tokenized lossmask sanity checks at train start.
- Updated triplet defaults to keep triplet terms active longer:
  - `TRIPLET_GAMMA_KL=0.8`
  - `TRIPLET_MARGIN_BENIGN=80.0`
  - `TRIPLET_MARGIN_HARMFUL=220.0`
- Added intermediate-checkpoint evaluation controls:
  - `EVAL_INTERMEDIATE_CHECKPOINTS=true`
  - `CHECKPOINT_EVAL_STEPS=80,100,110,150`
  - `CHECKPOINT_EVAL_LIMIT=<limit>`
  - `SAVE_STEPS=10` (so the above checkpoints exist)
- Visualization summary now prioritizes concrete metrics:
  - Fujitsu ASR deltas
  - AgentDojo diff/tool-call/resistance
  - verification status

## Pre-Run Sanity
Run once before launching jobs:

```bash
python scripts/verify_pipeline_run.py \
  --tool-schema configs/tool_schemas/b4_standard_v1.json \
  --trace-jsonl /scratch/memoozd/cb-scratch/data/traces/fujitsu_b4_ds.jsonl \
  --trace-jsonl /scratch/memoozd/cb-scratch/data/traces/agentdojo_complete.jsonl
```

## Recommended 3x3 Ablation (Lossmask x MWCS)
Use 3 lossmask policies and run 3 MWCS schedules as separate sweeps.

```bash
for S in balanced_cb attack_focus capability_focus; do
  sbatch slurm/pipeline/sweep_hparams.sbatch \
    ALPHAS=10.0 \
    LAYER_CONFIGS=10,20 \
    POLICIES=assistant_only,cb_full_sequence,tool_calls_only \
    MWCS_SCHEDULE=$S \
    MWCS_STEP=0 \
    AGENTDOJO_EVAL_SPLIT=complete \
    FULL_RENDER_AUDIT=true \
    LIMIT_EVAL=100
 done
```

Default sweep lane (if you do not override):
- `MWCS_SCHEDULE=capability_focus`
- `POLICIES=assistant_only,cb_full_sequence,tool_calls_only`

## Intermediate Checkpoint Eval Mechanics
Expected behavior for sweep/unified runs:
- Checkpoints are created every `SAVE_STEPS` (default `10`).
- If `EVAL_INTERMEDIATE_CHECKPOINTS=true`, pipeline evaluates each step in
  `CHECKPOINT_EVAL_STEPS` (default `80,100,110,150`).
- Missing checkpoints are skipped with a log line; run does not fail just for a missing step.
- For each present checkpoint, expected files:
  - `eval/checkpoints/checkpoint-<step>/fujitsu_eval.json`
  - `eval/checkpoints/checkpoint-<step>/agentdojo_eval.json` (if AgentDojo eval file exists)

## Per-Run Required Checks
For each run directory `<RUN_DIR>`:

1. Verification status
```bash
python scripts/verify_pipeline_run.py \
  --run-dir <RUN_DIR> \
  --tool-schema configs/tool_schemas/b4_standard_v1.json \
  --fail-on-error
```

2. Training loss behavior
- Confirm train log exists: `<RUN_DIR>/train.log`
- Confirm no NaN/Inf.
- Confirm triplet benign/harmful losses do not collapse to zero too early.

3. Eval context integrity
- In `<RUN_DIR>/eval/agentdojo_eval.json` check:
  - `cb_model.evaluation_context.tool_source` is **not** `schema_fallback`
  - tool-call rates are present in `generation_comparison`

4. Tool-call collapse guard
- Check:
  - `baseline.generation_comparison.tool_call_rate`
  - `cb_model.generation_comparison.tool_call_rate`
- Flag regression when CB is near 0 while baseline is substantially higher.

5. Intermediate checkpoint behavior
- Inspect:
  - `<RUN_DIR>/eval/checkpoints/checkpoint-80/`
  - `<RUN_DIR>/eval/checkpoints/checkpoint-100/`
  - `<RUN_DIR>/eval/checkpoints/checkpoint-110/`
  - `<RUN_DIR>/eval/checkpoints/checkpoint-150/`
- Compare final vs early checkpoints to detect over-suppression later in training.
- If final checkpoint has near-zero CB tool-call rate but checkpoint-80/100 does not,
  treat final as over-suppressed and consider earlier checkpoint for selection.

6. Metric validity guard (concrete metrics only)
- Prefer decisions based on:
  - `tool_flip_asr.attack_success_rate`
  - `generation_comparison.tool_call_rate`
  - `generation_comparison.harmful_resistance_rate`
  - `output_comparison.difference_rate`
  - verifier summary (`errors`, `warnings`, `passed`)
- Do not use deprecated/hallucinated metrics as selection criteria.

## Sweep-Level Comparisons
After sweep completion:

```bash
python scripts/visualize_sweep_results.py <SWEEP_DIR> --no-color --csv <SWEEP_DIR>/eval_summary.csv
```

Review in the summary:
- `Fuj Δ` (baseline ASR - CB ASR) should be positive.
- `AD TCR(B)` vs `AD TCR(CB)` should not show collapse.
- `AD Resist` should increase without killing tool-call capability.
- `Verif` should be `pass`.

Suggested run selection order:
1. Filter out runs/checkpoints where verifier fails or `diagnostics.tools_broken=true`.
2. Keep runs/checkpoints with non-collapsed CB tool-call rate.
3. Rank by lower Fujitsu ASR and higher AgentDojo resistance.

## Interpreting “Skipping completion-mask validation ...”
This now means:
- Data is external pre-tokenized (`SchemaDataset` path)
- Trainer runs lightweight lossmask sanity checks instead of internal completion-boundary checks.
- This is expected for ETL_B-generated render/lossmask datasets.
