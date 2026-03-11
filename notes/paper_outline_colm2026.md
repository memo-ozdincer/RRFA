# COLM 2026 Paper Outline
## Working Title: "Circuit Breakers for Agentic Tool-Calling: When Representation Rerouting Works, When It Fails, and Why"

**Deadlines:** Abstract Mar 26, Full paper Mar 31

---

## Abstract (150 words)

Representation rerouting ("circuit breakers") can defend LLMs against harmful text generation, but their application to tool-calling agents—where the harm is a wrong tool call, not wrong text—remains unexplored. We adapt circuit breakers to agentic prompt injection defense and conduct a systematic study across two benchmarks (Fujitsu B4, AgentDojo), multiple loss formulations, and 50+ training configurations. We find that (1) CB produces clean, correct defense on structural injections (tool A→B redirection: 85% correct, 0% gibberish) but degenerates on semantic injections (diverse goals, 64 tools); (2) a previously undocumented three-phase training dynamic (Push→Plateau→Collapse) governs all configurations; (3) linear probes reveal a selectivity ceiling (AUC 0.83) that fundamentally bounds the harmful/benign tradeoff; and (4) KL divergence strength—not push magnitude—is the primary determinant of output coherence. We provide corrected training recipes, comprehensive baselines, and actionable guidance for practitioners.

---

## 1. Introduction (1 page)

- Agentic LLMs call tools based on context → prompt injection in tool outputs can cause harmful actions (send_money, leak data)
- Existing defenses: pipeline (input filters, output validators) vs. internal (model-level)
- Circuit breakers (Zou et al. 2024) work for text-only LLMs but haven't been studied for tool-calling agents
- The challenge: harmful and benign tool calls share identical JSON format. "Send_money to attacker" and "send_money to friend" are representationally similar.
- **Our contributions** (see below)

## 2. Background & Related Work (1 page)

- Circuit Breakers / Representation Rerouting (Zou et al. 2024)
- Prompt injection attacks: AgentDojo (Debenedetti et al.), Fujitsu B4, InjectAgent
- Defense approaches: Instruction Hierarchy (OpenAI), BIPIA, StruQ, pipeline defenses
- Representation-level unlearning: Latent Unlearning, SRMU
- Agentic fine-tuning: AgentFlux DualTune, Hammer, TinyAgent, ToolACE

## 3. Method: Agentic Circuit Breakers (1.5 pages)

### 3.1 Problem Formulation
- Agent receives multi-turn trace: system → user → assistant[tool_call] → tool[+injection] → assistant
- Attacker injects instructions in tool output → model calls wrong tool
- Goal: reroute representations at decision point so model refuses/redirects

### 3.2 Training Format
- Native Llama 3.1 multi-turn chat template with <|python_tag|> tool calls
- Injection-aware loss masking: 0.0 (system/tool), 0.5 (injection span), 1.0 (post-injection assistant)
- Contrastive pairs: each trace yields (injected → wrong tool) + (clean → correct tool)
- The <|python_tag|> token as natural CB target

### 3.3 Loss Formulations
- Original RR: ReLU(cos_sim) push + L2 retain + KL (Zou et al. recipe)
- Per-token CB: dl2rc distance with margins (our extension)
- Key finding: KL coefficient must be 10x+ standard to preserve generation coherence for tool-calling
- The cr-ramped KL schedule (old code's implicit advantage)

### 3.4 Corrected Training Recipe
- Four compounding bugs fixed: alpha decay, margin saturation, KL fighting push, LR decay stacking
- Layer selection: 10+20 (31%/62% depth) validated
- 300-5000 step range with constant LR

## 4. Experimental Setup (1 page)

### 4.1 Datasets
- **Fujitsu B4**: 13,246 traces, 4 tools, structural injection (search_web ↔ retrieve_multimodal_docs)
- **AgentDojo**: 4,369 traces, 64 tools, semantic injection (diverse attack goals)
- Completed traces via LLM-based skeleton completion (same model, in-distribution)
- Benign retain set: 16,140 traces with all-zero CB masks

### 4.2 Baselines
- **Base model** (no defense): Fujitsu ASR ~85%, AD malicious ~75%
- **SFT refusal**: Same data, standard NTP on refusal responses
- **Pipeline: regex detector**: Pattern matching for injection markers
- **Pipeline: tool allowlist**: Block unexpected tool calls

### 4.3 Evaluation
- Fujitsu: attack success rate (ASR), correct tool rate
- AgentDojo: malicious tool rate, resistance rate, gibberish count, real defense rate
- Benign: correct tool rate (NTP mode), no-tool rate
- Full-trace free generation for output quality assessment

## 5. Results (2 pages)

### 5.1 Main Results Table
| Method | Fuj ASR↓ | Fuj Correct↑ | AD Mal↓ | AD Gibberish↓ | Benign Correct↑ |
|--------|----------|--------------|---------|---------------|-----------------|
| Baseline | 85% | 15% | 75% | 0 | 21% |
| Pipeline (regex) | TBD | TBD | TBD | 0 | TBD |
| SFT refusal | TBD | TBD | TBD | TBD | TBD |
| CB (original_rr, KL=5.0) | TBD | TBD | TBD | TBD | TBD |
| CB (per_token_cb, KL=5.0) | TBD | TBD | TBD | TBD | TBD |
| CB (best config) | ~15-28% | ~72-85% | TBD | TBD | TBD |

### 5.2 Structural vs Semantic Injections
- Fujitsu: clean redirects, 0% gibberish across all configs
- AgentDojo: gibberish/suppression unless KL is strong → then coherent-but-redirected
- Key insight: CB works when the model "has somewhere safe to go" (small tool set, clear alternative)

### 5.3 The KL-Coherence Finding
- Old code (KL ramping to 5.0): coherent outputs even if wrong tool
- New code (KL = 0.3): gibberish across all configs
- The KL coefficient is the primary determinant of output quality, not push magnitude
- Verification: 5 runs spanning KL=0.3 to KL=5.0

## 6. Analysis (2 pages)

### 6.1 Three-Phase Training Dynamics
- Phase 1 (Push, steps 0-40): Margin exceeded, harmful reps displaced
- Phase 2 (Plateau, steps 40-~70%T): Only retain/KL active, benign anchoring
- Phase 3 (Collapse, ~70%T onwards): Harmful re-enters margin, benign destroyed
- Universal across all configs: onset at ~70% regardless of alpha, margin, or SRMU

### 6.2 The Selectivity Ceiling
- Probe Test 1 (harmful vs benign, different context): AUC 0.97 mean-pool — largely trivial (detecting injection text)
- Probe Test 2 (comply vs refuse, same context): AUC 0.96 last-token — the direction CB needs to push
- Probe Test 3 (all layers, last-token): AUC 0.83 — the selectivity signal for WHEN to push
- Theoretical implication: ~17% inherent benign destruction from the 0.83/0.96 gap
- Confirmed empirically: every config achieving <30% AD malicious also destroys >25% benign
- Pooling pattern flip between Test 1 and Test 2: injection detection is distributed, response selection is concentrated

### 6.3 What Didn't Work and Why
- SRMU importance masking: improves push efficiency 5x, but selectivity unchanged → global suppression
- Balanced + refusal data: eliminates gibberish, but model learns redirect pattern globally → wrong-tool hallucination
- Higher margins: more destruction, same Phase 3 timing
- Combining interventions: orthogonality assumption is wrong, they interfere destructively
- Each "fix" trades one failure mode for another at the selectivity ceiling

### 6.4 Failure Mode Taxonomy
1. Gibberish (token repetition, character noise) — from aggressive push + weak KL
2. Suppression (no tool calls at all) — from SRMU concentrated push
3. Wrong-tool hallucination (valid JSON, wrong tool) — from balanced data overfitting redirect pattern
4. Code generation (Python instead of tool call) — from insufficient KL

## 7. Discussion & Practical Guidance (0.5 pages)

- When to use CB for agents: structural injections, constrained tool sets, clear safe alternatives
- When to use pipeline defenses: semantic injections, large tool sets, diverse attack goals
- The KL recipe: always use KL coefficient ≥5.0 (cr-ramped) for tool-calling agents
- Training recipe: layers 10+20, 300-1000 steps, constant LR, margin_benign=0
- Complementary approach: CB catches structural injections the pipeline can't pattern-match; pipeline catches semantic injections CB can't selectively reroute

## 8. Conclusion (0.25 pages)

---

## Key Contributions (for abstract/intro)

1. **First systematic study** of representation rerouting for agentic prompt injection defense
2. **KL-coherence finding**: KL divergence strength (not push magnitude) determines whether CB preserves tool-calling ability — practitioners need 10x+ standard KL for agents
3. **Three-phase training dynamics**: Push→Plateau→Collapse is universal across configs; Phase 3 onset at ~70% is independent of hyperparameters
4. **Selectivity ceiling**: Linear probes predict a hard Pareto bound (AUC 0.83) on the harmful/benign tradeoff, confirmed empirically across 50+ configs
5. **Corrected training recipe**: Four compounding training bugs took defense from -5% to -59% improvement; immediately applicable to any CB deployment
6. **Structural vs semantic dichotomy**: CB cleanly defends structural injections (85% correct, 0% gibberish) but degrades on semantic injections — actionable guidance for practitioners

---

## Figures Needed

1. **Main results table** (auto-generated by generate_paper_table.py)
2. **Three-phase training curve**: Triplet_H and Triplet_B over steps for ad_3000_v2
3. **Probe AUC comparison**: Bar chart of Test 1/2/3 AUC across pooling strategies
4. **Pareto frontier**: AD malicious vs benign correct, all configs as points
5. **KL sweep**: gibberish rate vs KL coefficient across verification runs
6. **Example outputs**: side-by-side baseline/CB for Fujitsu (clean redirect) and AD (gibberish vs coherent)

---

## Timeline

- **Mar 11-14**: Run KL verification sweep (5 runs), SFT baseline, pipeline baselines
- **Mar 14-17**: Analyze results, generate tables/figures, draft Sections 3-5
- **Mar 17-20**: Draft Sections 1-2, 6-8; second model (Qwen optional)
- **Mar 20-24**: Internal review, polish, supplementary
- **Mar 26**: Submit abstract
- **Mar 31**: Submit full paper
