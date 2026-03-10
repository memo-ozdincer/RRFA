# Agent 3: Paper + SRMU Implementation

You are implementing the novel contribution (feature-selective rerouting) and writing the paper for the RRFA project (representation rerouting for agentic prompt injection defense). Target: COLM 2026.

## Read First
1. `notes/shared_context.md` — project context
2. `notes/analysis_report.md` — Agent 1's analysis (your empirical foundation)
3. `notes/agent_plan_3_paper.md` — your task list
4. `paper/_COLM_2026__AgentDefense/colm2026_conference.tex` — current (stale) paper
5. `PAPERS/RMU.tex` — Jin's SRMU paper (importance map you're adapting)

## Your Files (ONLY modify these)
- `src/training/losses.py` — add importance-weighted CB loss
- `src/training/trainer.py` — load importance mask
- `src/training/train_schema.py` + `config.py` — new --importance-mask CLI arg
- `scripts/extract_probe_direction.py` — NEW: extract probe weights as importance mask
- `scripts/plot_publication_figures.py` — NEW: publication figures
- `paper/_COLM_2026__AgentDefense/colm2026_conference.tex` — paper rewrite

## Do NOT modify
- `slurm/pipeline/` (Agent 2's territory)
- `scripts/amplify_injections.py` (Agent 2 creates this)

## Priority Order
1. **Extract probe direction** — `scripts/extract_probe_direction.py` (reuse logic from `scripts/probe_separability.py`)
2. **Importance-weighted loss** — mask h_lora and h_frozen by probe |w_i| before computing dl2rc distance. Must be backwards compatible (no mask = unchanged behavior).
3. **Paper: Selectivity Problem section** — the central finding. Push AUC 0.967 vs selectivity AUC 0.83. Pooling flip between Test 1 and Test 2.
4. **Paper: Three-Phase Training Dynamic** — novel finding. Phase 1 (push, 30 steps) → Phase 2 (plateau, benign anchoring) → Phase 3 (collapse at ~70%). Training curve data in `results_for_agent1/traininglogfor3000v2.txt`.
5. **Paper: Results tables** — Sweep 10 full comparison, Pareto frontier, cross-dataset (Fujitsu clean vs AD gibberish)
6. **Figures** — training curve with phases, probe heatmap, Pareto plot
7. **Literature integration** — read PAPERS/*.tex, write Related Work connections

## Key Numbers
- Pareto-optimal: ad_5000_v2 (AD trunc 24%, benign 72%, Fujitsu -58%)
- Theoretical ceiling: 83% benign, 97% harmful reduction (from probe AUCs)
- Selectivity gap: 72% actual → 83% ceiling = 11% recoverable by SRMU masking
- Sweep 9 → 10: 9x improvement from training fixes alone

Read the plan file, then probe_separability.py, then losses.py, then implement. Commit when done.
