# Circuit Breaker MVP Pipeline - Quick Start

## Submit Complete Pipeline (Overnight Run)

```bash
# 1. SSH to cluster
ssh trillium

# 2. Navigate to scratch workspace
cd /scratch/memoozd/harmful-agents-meta-dataset

# 3. Pull latest changes
git pull

# 4. Run the complete pipeline with dependencies
bash slurm/Trillium/run_mvp_pipeline.sh
```

This will submit 7 jobs with automatic dependencies:

1. **Ds generation** (~2h) - Generate harmful tool-flip samples
2. **Dr generation** (~2h) - Generate benign twin samples  
3. **Validation** (~10min) - Validate data quality
4. **Eval set creation** (~5min) - Split train/eval sets
5. **Training** (~12-24h) - Train Circuit Breaker adapter
6. **Sanity check** (~5min) - Quick model verification
7. **Evaluation** (~1-2h) - Full evaluation on test set

**Total estimated time: ~16-30 hours**

## Monitor Progress

```bash
# Check job queue
squeue -u $USER

# Watch specific job
watch -n 10 squeue -j <JOB_ID>

# Check logs
tail -f /scratch/memoozd/logs/<job_name>_<job_id>.out
tail -f /scratch/memoozd/logs/<job_name>_<job_id>.err

# Cancel all jobs (if needed)
scancel <DS_JOB> <DR_JOB> <VALIDATE_JOB> <EVAL_SET_JOB> <TRAIN_JOB> <SANITY_JOB> <EVAL_JOB>
```

## Check Results (Morning After)

```bash
# Check if all jobs completed
squeue -u $USER  # Should be empty

# Check final outputs
ls -lh /scratch/memoozd/cb_mvp_data/
ls -lh /scratch/memoozd/cb_mvp_model/
ls -lh /scratch/memoozd/cb_mvp_eval/

# View evaluation results
cat /scratch/memoozd/cb_mvp_eval/eval_results.json
```

## Key Files

- **Data**: `/scratch/memoozd/cb_mvp_data/`
  - `ds_stage1.jsonl` - Harmful samples
  - `dr_stage1.jsonl` - Benign twins
  - `ds_train.jsonl`, `dr_train.jsonl` - Training sets
  - `ds_eval.jsonl`, `dr_eval.jsonl` - Evaluation sets

- **Model**: `/scratch/memoozd/cb_mvp_model/`
  - Circuit Breaker LoRA adapter

- **Results**: `/scratch/memoozd/cb_mvp_eval/`
  - `eval_results.json` - Performance metrics
  - `predictions.jsonl` - Detailed predictions

- **Logs**: `/scratch/memoozd/logs/`
  - `mvp_*_<job_id>.out` - Standard output
  - `mvp_*_<job_id>.err` - Error output

## Troubleshooting

If a job fails:

```bash
# Find the job ID from the logs
cat /scratch/memoozd/logs/mvp_ds_1h100_<JOB_ID>.err

# Rerun just that step (example: rerun Ds generation)
sbatch slurm/Trillium/trillium_mvp_generate_ds_1h100.sbatch

# Or run diagnostic first
sbatch slurm/Trillium/trillium_debug_toolcall.sbatch
```

## What Changed from Previous Run

- **Switched from abliterated to standard Llama-3.1-8B-Instruct**
  - Previous run: 0% yield (model didn't generate tool calls)
  - New approach: Standard model has proper tool-calling support
  - Trade-off: May refuse some harmful requests, but we only need 10%+ yield

- **Complete dependency chain**
  - Jobs automatically wait for prerequisites
  - No manual intervention needed
  - Safe to run overnight
