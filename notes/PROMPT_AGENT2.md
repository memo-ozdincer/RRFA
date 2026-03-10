# Agent 2: Engineering

You are implementing training pipeline changes for the RRFA project (representation rerouting for agentic prompt injection defense).

## Read First
1. `notes/shared_context.md` — project context
2. `notes/analysis_report.md` — Agent 1's analysis with Pareto frontier and recommendations
3. `notes/agent_plan_2_engineering.md` — your task list with code snippets

## Your Files (ONLY modify these)
- `slurm/pipeline/sweep_ad_focused.sbatch` — add new configs
- `scripts/amplify_injections.py` — NEW: injection amplification script

## Do NOT modify
- `src/training/` (Agent 3's territory)
- `paper/` (Agent 3's territory)
- `scripts/extract_probe_direction.py`, `scripts/plot_publication_figures.py` (Agent 3 creates these)

## Tasks
1. Add balanced configs (AD + Fujitsu) to sbatch: `ad_500_bal`, `ad_750_bal`, `ad_1000_bal`
2. Add refusal config: `ad_750_bal_r` (with `_USE_REFUSAL=true`)
3. Add alpha/margin tuning configs: `ad_1000_a5` (α=5.0), `ad_1000_m40` (margin_h=40.0)
4. Create `scripts/amplify_injections.py` and add amplified configs: `ad_750_amp`
5. Integrate amplification as optional sbatch step
6. Fix stale refusal cache check in sbatch

## Key Numbers (from Agent 1)
- ad_5000_v2 is Pareto-optimal: AD trunc 24%, benign 72%, Fujitsu -58%
- Phase 3 collapse at ~68-72% of training destroys benign
- More Phase 2 anchoring = better benign (5000 steps > 3000 steps)
- Fujitsu outputs are clean; AD outputs are gibberish
- FUJITSU_CAP=2500 for balanced configs

Read the sbatch file first, then implement. Commit when done.
