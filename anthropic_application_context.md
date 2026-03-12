# Context Primer for Anthropic Application Preparation

> **Instructions for Claude:** This document was prepared by a previous Claude session that had deep context on Memo's projects, codebase, research, and background. Use this to help Memo prepare his Anthropic application materials, choose code snippets, answer application questions, and craft compelling narratives about fit. Memo is applying to Anthropic (likely an internship or research role focused on AI safety/alignment). Read this entire document before starting any work.

---

## WHO IS MEMO

**Name:** Mehmet (Memo) Ozdincer
**Email:** memo@cs.toronto.edu / memo.ozdincer@mail.utoronto.ca
**School:** University of Toronto, Engineering Science (Machine Intelligence Option), BASc, graduating June 2028
**Awards:** Dean's List (3.5+/4.0 GPA), Mitacs Globalink Research Grant
**GitHub:** github.com/memo-ozdincer
**LinkedIn:** linkedin.com/in/memo-ozdincer

---

## MEMO'S RESEARCH POSITIONS (Current)

### 1. Vector Institute / Jinesis AI Lab — AI Researcher (LLM Safety & Evals)
**Aug 2025 - Present | PI: Prof. Zhijing Jin**
**Repo:** github.com/memo-ozdincer/RRFA

This is Memo's primary project and the most Anthropic-relevant work. It is the **first internal defense against prompt injection in agentic tool-calling systems** using representation rerouting (circuit breakers).

**What it does:**
- Trains LoRA circuit breakers that cut attack success rate from 83.7% to 8.2% on Fujitsu's AgentDojo benchmark while preserving tool-calling capability
- Works across Llama 4, Llama 3.1, Qwen 3, and GPT-OSS
- Partnered with ServiceNow for realistic agent-attack data generation
- Custom deterministic eval (next-tool-prediction) across 313K+ simulations spanning tool-flip, email exfiltration, and multi-domain injection
- Built an RLVR training environment on Kubernetes/Docker: pod-based architecture serving the LLM alongside the eval as a verifiable reward signal
- Being prepared for COLM 2026

**Core Technical Architecture (DEEP DETAIL):**

The loss operates on **hidden representations at intermediate layers**, NOT on output logits. This is the single most important thing to understand:

1. **Forward hooks** (RepresentationExtractor class in `src/training/trainer.py`) capture hidden states during the forward pass at `cb_target_layers = [10, 20]` (roughly 31% and 62% depth through the network, where linear probes show maximum separation between harmful and benign representations).

2. **Three-term loss:**
   - **Push loss** (harmful reps): Forces hidden states at layers 10/20 away from what a frozen copy of the model would produce for the harmful input. Uses `ReLU(cos_sim)` in the `original_rr` formulation, or `dl2rc` distance (||a-b||_2 + 10*ReLU(1-cos_sim(a,b))) in the `per_token_cb` formulation.
   - **Retain loss** (benign reps): Keeps benign representations close to the frozen model's representations. Uses L2 distance.
   - **KL divergence**: Distribution matching on benign outputs to preserve general capabilities. Uses `attention_mask` NOT `loss_mask`. KL coefficient must be ~5.0 (not 0.3) due to downstream amplification of layer-10 perturbations through 20+ subsequent layers.

3. **LoRA configuration:** LoRA adapters are applied to ALL 7 linear modules in EVERY transformer layer (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj), not just attention layers. r=16, alpha=16. This is because MLP layers carry most of the representation content. **IMPORTANT DISTINCTION:** Layers 10 and 20 are NOT the LoRA target layers — they are the representation extraction points (probe points) where the push/retain loss is computed. LoRA modifies every layer; the loss signal is measured at layers 10/20 and gradients flow back through all LoRA parameters everywhere.

4. **Injection-aware loss masking:** Three-tier weighting: 0.0 for system/user tokens (don't train on these), 0.5 for the injection span, 1.0 for post-injection assistant response (this is where the model decides whether to follow the injection or not).

5. **Data pipeline (three-tier ETL):**
   - Tier A: Raw trace generation from attack skeletons and tool schemas
   - Tier B: Canonical normalization into structured format with message boundaries, roles, content
   - Tier C: Rendered training examples with tokenized sequences and loss masks
   - Paired DS/DR (dangerous-source/dangerous-response) generation: minimal harmful/benign contrasts from same skeleton trace

6. **Two loss formulations in the codebase:**
   - `original_rr`: ReLU(cos_sim) push + L2 retain (from original circuit breaker paper)
   - `per_token_cb`: dl2rc distance with margins (solves LoRA vanishing gradient problem where cos_sim saturates near 1.0 for small LoRA perturbations)

7. **Progressive tokenization:** BPE is context-dependent at message boundaries. Tokenizing the full sequence together is CORRECT (matches inference). One-by-one is WRONG. Progressive tokenization recovers message boundaries by tokenizing prefixes of increasing length and taking differences.

8. **Truncation:** `truncate_to_injection_window.py` keeps messages up to the first assistant response after injection. The harmful response IS included (messages[:j+1]) because that's what we want to push away from.

9. **DDP (not FSDP/TP/PP)** because model fits on single A100.

10. **Dual coefficient schedule:** c_s decays (push early), c_r grows (retain late).

11. **Forget/retain overlap problem:** Unique to agentic data. "send_email to alice" appears in both harmful (attacker-initiated) and benign (user-initiated) traces. The model must learn to distinguish based on context, not surface patterns.

**Key code files for code snippet selection:**
- `src/training/losses.py` — Contains all loss functions. Lines 116-144 (KL divergence), 146-217 (reroute_loss_relu_cos + retain_loss_l2), 284-294 (dl2rc distance), 538-790 (per_token_cb_loss), 853-935 (original_rr_loss)
- `src/training/trainer.py` — Lines 120-188 (RepresentationExtractor with forward hooks)
- `src/training/config.py` — Lines 1-60 (LoRA config, circuit breaker config)
- `scripts/truncate_to_injection_window.py` — Lines 32-60 (truncation logic)
- The data pipeline code (ETL tiers)
- The eval harness (deterministic next-tool-prediction)

**WHY THIS IS ANTHROPIC-RELEVANT:**
- It's literally AI safety research: defending against prompt injection in agentic systems
- The technique (representation rerouting) is mechanistic/interpretability-adjacent: it operates on internal representations, not just outputs
- The eval methodology (deterministic, no LLM judge, 313K+ simulations) aligns with Anthropic's emphasis on rigorous evaluation
- The data pipeline design (minimal contrasts, injection-aware masking) shows careful thinking about training data quality
- The project addresses a real deployment concern: as agents get tool access, prompt injection becomes a critical safety issue

### 2. Matter Lab (Vector Institute + NVIDIA) — AI Researcher (Generative Chemistry)
**Sep 2025 - Present | PI: Prof. Alan Aspuru-Guzik**
**Repo:** github.com/memo-ozdincer/transition-state-sampling

**What it does:**
- Designed a saddle-point search algorithm achieving 100% transition state convergence from molecules displaced by 2A of noise (all existing methods fail below 65%)
- Building a diffusion-based generative model using adjoint sampling from transition states with E(3)-equivariant GNNs for sampling molecular configurations from Boltzmann distributions
- Built a differentiable eigendecomposition pipeline in PyTorch enabling gradient-based optimization through full Hessian spectra
- Benchmarked 8 algorithms across 120K+ simulations; discovered analytical second derivatives outperform NN-predicted ones (94% vs 75%, 33x faster)
- Submitting to ICML 2026 Workshop (Seoul), NeurIPS 2026 target for the generative model

**Key code for snippet selection:**
- The differentiable eigendecomposition pipeline
- E(3)-equivariant GNN architecture
- The saddle-point search algorithm
- Benchmarking harness across 120K+ simulations

### 3. National University of Singapore (SERIS) — ML Researcher (Comp. Physics)
**May 2025 - Aug 2025 | Prof. Erik Birgersson**
**Repo:** github.com/memo-ozdincer/perovskite-jv-surrogate

- Designed a dilated convolutional surrogate replacing a coupled PDE solver (~4800 s/device) with millisecond inference (10,000x speedup)
- Multi-objective loss (MSE + monotonicity + convexity + Jacobian regularization)
- 150K training simulations, 31 dimensions, median R^2=0.9975
- 71 domain-derived features compressed to 5 via train-only selection pipeline

### 4. aUToronto (UofT AutoDrive Team) — Software Engineer (Computer Vision)
- BEVFusion sensor fusion pipeline (C++/Python), 1.1 GB/s from 6 cameras + 2 LiDARs at <90ms latency
- Fine-tuned detection models with 35% perception speedup in adverse weather

---

## WHAT MEMO SHOULD EMPHASIZE FOR ANTHROPIC

### Safety & Alignment Fit
- Jinesis IS safety research: defending against prompt injection in agentic systems using representation-level interventions
- The approach is mechanistic: operates on hidden representations, not just output distribution. This connects to Anthropic's interpretability work
- Memo cares about evaluation rigor: 313K+ deterministic simulations, no stochastic LLM judge, resistance to benchmark overfitting
- The forget/retain overlap problem in agentic data is a microcosm of the broader alignment challenge: how do you make a model refuse harmful instructions while preserving helpful capabilities that look superficially similar?

### Technical Depth
- Memo understands transformer internals deeply: forward hooks, hidden state manipulation, layer-wise representation dynamics, KL divergence amplification through layers
- Strong PyTorch/systems skills: DDP training, Kubernetes/Docker RLVR environment, SLURM/HPC
- Broad ML range: LLM safety, diffusion models, E(3)-equivariant GNNs, computer vision, surrogate modeling

### Research Taste
- Chose problems that matter (safety of deployed agents, not toy benchmarks)
- Built evaluation infrastructure that's resistant to gaming (deterministic eval, not LLM-as-judge)
- Designed data pipelines carefully (minimal contrasts, injection-aware masking, three-tier ETL)
- Thinks about failure modes (forget/retain overlap, KL amplification, LoRA vanishing gradient)

### Candid Things to Note
- Memo "vibe coded a large part" of the Jinesis project initially, then built deep intuition through iteration. He's honest about this.
- The project has some current issues he should NOT mention in applications: gibberish on AgentDojo, eval artifacts, KL scaling bug. Present the Fujitsu results (84% -> 8% ASR) as the headline.
- He's a 2nd-year undergrad (graduating 2028) which is remarkable given the depth of this research.

---

## CODE SNIPPET SELECTION GUIDANCE

When Memo asks you to help choose code snippets for Anthropic, here's what to prioritize:

### From Jinesis (RRFA repo):
**Best candidates (most Anthropic-relevant):**
1. **RepresentationExtractor class** (`trainer.py` ~120-188): Shows understanding of forward hooks, hidden state capture, and how to intervene on model internals. This is the most interpretability-adjacent code.
2. **dl2rc distance function** (`losses.py` ~284-294): Elegant, compact, shows deep understanding of why cosine similarity alone fails for LoRA (vanishing gradient when cos_sim is near 1.0). The formula ||a-b||_2 + 10*ReLU(1-cos_sim(a,b)) is clean and motivated.
3. **The three-term loss function** (`losses.py`, either `original_rr_loss` or `per_token_cb_loss`): Shows the full push/retain/KL architecture. Complex but well-structured.
4. **Injection-aware loss masking**: Shows careful thinking about which tokens matter for safety training.
5. **The eval harness**: Deterministic next-tool-prediction. Shows evaluation rigor.

### From Matter Lab (transition-state-sampling repo):
1. **Differentiable eigendecomposition pipeline**: Novel PyTorch code for gradient-based optimization through Hessian spectra.
2. **E(3)-equivariant GNN architecture**: Shows understanding of symmetry-constrained deep learning.
3. **Saddle-point search algorithm**: The algorithm that achieves 100% convergence where others fail at 65%.

### Selection Criteria for Anthropic:
- Prioritize code that shows **understanding of model internals** (hooks, representations, gradients)
- Prioritize code that shows **careful engineering** (not just "it works" but "here's why each design choice matters")
- Prioritize code that connects to **safety/alignment** themes
- Avoid code that's purely boilerplate (data loading, config parsing)
- The snippet should be self-contained enough to understand without full repo context

---

## ANTHROPIC APPLICATION QUESTION THEMES

Memo should be prepared to answer:

1. **Why Anthropic specifically?** (Safety-first mission, interpretability research, responsible scaling, Claude as a product that demonstrates alignment work mattering in practice)

2. **What's your most interesting technical contribution?** (Representation rerouting for agentic prompt injection — the first internal defense that operates on hidden states rather than output distribution)

3. **How does your work connect to alignment/safety?** (Directly: it's a defense against a concrete attack vector. Conceptually: it's about steering model representations, which connects to interpretability and mechanistic understanding of what models "think")

4. **Describe a hard technical problem you solved.** (KL coefficient needing to be 5.0 not 0.3 because perturbations at layer 10 amplify through 20+ layers. Or: the forget/retain overlap where harmful and benign tool calls look identical on the surface.)

5. **How do you think about evaluation?** (Deterministic > stochastic. Resistant to benchmark overfitting. Large-scale simulation > small-scale human eval for safety-critical properties.)

6. **What research direction excites you?** (The intersection of mechanistic interpretability and safety interventions. Understanding WHY representation rerouting works at specific layers could inform better safety techniques. Also: scaling safety techniques to more capable agentic systems.)

---

## MEMO'S SKILLS SUMMARY
- **AI/ML:** LLM & Agent Safety, Fine-Tuning, Evals, Reinforcement Learning, Diffusion Models, CV (YOLO, BEVFusion)
- **Languages & Frameworks:** Python, C++17, C, CUDA, PyTorch, Hydra/OmegaConf, DVC, MLFlow, TensorFlow, Docker, Kubernetes, HuggingFace, AWS, SLURM/HPC

---

## IMPORTANT CONTEXT ABOUT MEMO'S PREPARATION STYLE
- He prefers **concise, high-signal answers** over lengthy scripted responses
- He wants to understand **intuition and reasoning**, not just facts (he "vibe coded" parts of the project and built understanding iteratively)
- He responds well to **direct, honest feedback** about what's strong and what's weak in his answers
- He explicitly asked previous sessions to "stop preparing documents, just answer in text" when he wanted to practice verbally — respect this preference if he asks for it
- He's currently also preparing for a Cohere Sovereign AI interview with Eddie Kim (separate from this Anthropic application)

---

## REPO ACCESS
The Jinesis codebase should be accessible at the repo path if Memo provides it, or you can ask him to provide the local path. Key files to read for code snippet selection:
- `src/training/losses.py` (the heart of the project)
- `src/training/trainer.py` (RepresentationExtractor, training loop)
- `src/training/config.py` (architecture decisions)
- `scripts/truncate_to_injection_window.py` (data processing)
- The eval harness (wherever it lives in the repo)
- The data generation pipeline

The Matter Lab repo is at github.com/memo-ozdincer/transition-state-sampling. Key files:
- The differentiable eigendecomposition pipeline
- The GNN architecture
- The saddle-point search algorithm
