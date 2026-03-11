# Morning Checklist — Mar 12, 2026

Everything is coded and ready. Here's what to do in order.

## What Was Done Overnight

Claude Code implemented all 5 phases. I verified the code — everything is correct with one note below.

### Files Created/Modified:
- `src/training/losses.py` — Added `kl_scaling` param to `original_rr_loss` and `per_token_cb_loss`
- `src/training/config.py` — Added `kl_scaling: str = "constant"` field
- `src/training/train_schema.py` — Added `--kl-scaling` CLI arg
- `src/training/trainer.py` — Wired `kl_scaling`, `alpha_max`, `c_r` to both loss calls
- `slurm/pipeline/sweep_kl_restore.sbatch` — 5 KL verification runs
- `scripts/train_sft_refusal.py` — Standalone SFT trainer for refusal baseline
- `slurm/pipeline/train_sft_baseline.sbatch` — Full SFT pipeline
- `scripts/eval_pipeline_defenses.py` — Regex injection detector + metrics
- `scripts/generate_paper_table.py` — Auto-generates LaTeX results table

### One Important Note (Breaking Change):
`per_token_cb_loss` previously ALWAYS cr-ramped KL internally. Now it defaults to `kl_scaling='constant'`. This means if you re-run any old Sweep 10/11 configs without adding `--kl-scaling cr`, the KL behavior will silently change. Old completed runs are NOT affected — only new runs of old configs.

## Step 1: Commit the code changes
```bash
cd /path/to/rrfa
git add src/training/losses.py src/training/config.py src/training/train_schema.py src/training/trainer.py
git add slurm/pipeline/sweep_kl_restore.sbatch slurm/pipeline/train_sft_baseline.sbatch
git add scripts/train_sft_refusal.py scripts/eval_pipeline_defenses.py scripts/generate_paper_table.py
git commit -m "Add kl_scaling parameter + KL verification sweep + SFT baseline + pipeline defense eval + paper table generator"
git push
```

## Step 2: Submit KL verification sweep (HIGHEST PRIORITY)
```bash
sbatch slurm/pipeline/sweep_kl_restore.sbatch
```
This runs 5 configs testing whether stronger KL restores coherent AD generation:
1. `orig_rr_kl5_const` — original_rr loss, KL=5.0 constant
2. `orig_rr_kl05_cr` — original_rr loss, KL=0.5 cr-ramped (EXACT old formula)
3. `ptcb_kl5_mb0` — per_token_cb, KL=5.0, margin_benign=0 (no dead zone)
4. `ptcb_kl05_cr_mb0` — per_token_cb, KL=0.5 cr-ramped, margin_benign=0
5. `ptcb_kl5_mb0_mh1` — per_token_cb, KL=5.0, margin_harmful=1.0 (bounded push)

**What you're looking for:** Run 2 should most closely reproduce the old code's behavior. If ANY run produces coherent (non-gibberish) tool calls on AgentDojo harmful, the KL hypothesis is confirmed. Check the AD harmful full_trace eval outputs for actual JSON quality.

## Step 3: Submit SFT baseline
```bash
sbatch slurm/pipeline/train_sft_baseline.sbatch
```
**Before submitting:** Quick-check that the data paths in the sbatch match your cluster layout. The script expects:
- `$REPO_DIR/data/traces/agentdojo_augmented.jsonl`
- `$REPO_DIR/data/traces/fujitsu_b4_completed.jsonl` (or similar)
- `scripts/truncate_to_injection_window.py`
- `scripts/generate_refusal_traces.py` (if it exists — check this)

If `generate_refusal_traces.py` doesn't exist, the sbatch has an inline refusal generation step (Step 0). Check lines ~120-160 of the sbatch.

## Step 4: Run pipeline defense baselines (no GPU needed)
```bash
python scripts/eval_pipeline_defenses.py \
  --traces data/traces/agentdojo_augmented.jsonl \
  --output results/pipeline_defense_results.json
```
Also run on Fujitsu traces. This gives you the regex detection rate for the paper table.

## Step 5: While waiting — review paper outline
See `notes/paper_outline_colm2026.md`. Key decisions:
- **Abstract due Mar 26** (15 days)
- **Full paper due Mar 31** (20 days)
- 9 pages max
- Is the KL-coherence finding strong enough as the headline contribution? Or do you want "train on Fujitsu, generalize to AD"?
- Do you want to run a second model (Qwen-2.5)? Timeline is tight.

## Step 6: When KL sweep results come back

If coherent AD outputs are restored:
- This is the paper's headline: "KL strength, not push magnitude, determines whether CB preserves tool-calling"
- Run the winning config at 1000, 3000, 5000 steps for the step-count ablation
- Generate paper table: `python scripts/generate_paper_table.py --sweep-dir results/kl_restore/`

If still gibberish:
- The loss math isn't the only issue — likely need to also fix the train/eval tool schema mismatch
- Ask Claude Code to implement the tool schema fix (inject schemas into training data to match eval format)
- Re-run with both KL fix AND schema fix

## What Still Needs Doing (not yet coded)

1. **Figures**: Three-phase dynamics plot, probe AUC bar chart, Pareto frontier, KL vs gibberish rate
2. **Second model** (optional): Qwen-2.5-7B-Instruct — needs new ETL for its chat template
3. **AgentDojo-Structural experiment**: Reformat AD into Fujitsu-like structure to test structural hypothesis causally
4. **Paper writing**: Actual LaTeX draft

## Priority Order
1. Submit KL sweep ← DO THIS FIRST
2. Submit SFT baseline
3. Run pipeline defenses
4. Start writing Sections 3-4 of the paper while waiting
5. Review results when they come back (~3-6 hours depending on cluster queue)
