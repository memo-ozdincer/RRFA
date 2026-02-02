# Circuit Breakers for Agentic Safety

This repository contains the pipeline for training and evaluating Circuit Breakers - a technique for making language models resist tool-flip attacks in agentic settings.

## Repository Structure

```
├── configs/
│   └── tool_schemas/
│       └── b4_standard_v1.json     # Tool definitions for B4 attacks
├── data/
│   └── fujitsu/
│       └── orchestrator_attacks_combined_deduplicated.jsonl  # Source attack data
├── src/
│   ├── data_generation/            # Data pipeline scripts
│   │   ├── generate_ds.py      # Generate harmful set (Ds)
│   │   ├── generate_dr.py      # Generate retain set (Dr)
│   │   ├── create_eval_set.py      # Create held-out eval set
│   │   ├── validate_format.py      # Validate Llama 3.1 format
│   │   └── rebuild_training_data.py # Rebuild with proper format
│   ├── training/                   # Training module
│   │   ├── train.py                # Main training entry point
│   │   ├── config.py               # Training configurations
│   │   ├── trainer.py              # CircuitBreakerTrainer class
│   │   └── hf_utils.py             # HuggingFace utilities
│   ├── evaluation/                 # Evaluation scripts
│   │   ├── eval.py             # MVP evaluation (ASR, capability)
│   │   └── sanity_check.py         # Adapter sanity check
│   └── utils/
│       └── wandb_logging.py        # W&B logging utilities
└── slurm/                          # SLURM batch scripts
    ├── 01_generate_ds.sbatch       # Phase 1: Generate Ds
    ├── 02_generate_dr.sbatch       # Phase 1: Generate Dr
    ├── 03_create_eval.sbatch       # Phase 1: Create eval set
    ├── 04_validate.sbatch          # Phase 2: Validate format
    ├── 05_train.sbatch             # Phase 3: Train CB adapter
    └── 06_eval.sbatch              # Phase 4: Evaluate
```

## Pipeline

The pipeline consists of 4 phases:

### Phase 1: Data Generation
```bash
sbatch slurm/01_generate_ds.sbatch   # Generate harmful set (Ds)
# Wait for completion, then:
sbatch slurm/02_generate_dr.sbatch   # Generate retain set (Dr)
sbatch slurm/03_create_eval.sbatch   # Create eval set (can run in parallel with 02)
```

### Phase 2: Validation
```bash
sbatch slurm/04_validate.sbatch      # Validate Llama 3.1 format
```

### Phase 3: Training
```bash
sbatch slurm/05_train.sbatch         # Train Circuit Breaker adapter
```

### Phase 4: Evaluation
```bash
sbatch slurm/06_eval.sbatch          # Evaluate on held-out set
```

## Requirements

See `requirements.txt` for Python dependencies. Key requirements:
- PyTorch with CUDA support
- Transformers
- PEFT (for LoRA)
- vLLM (for data generation)
- Accelerate (for distributed training)
- Streamlit (for sweep visualization UI)

## Sweep Visualization UI

Use the Streamlit viewer to browse sweep metrics, filter by lossmask policy and hparams, and inspect sample traces.

```bash
streamlit run scripts/visualize_sweep_streamlit.py -- --sweep-dir /path/to/sweep_dir
```

Optional trace context:

```bash
streamlit run scripts/visualize_sweep_streamlit.py -- --sweep-dir /path/to/sweep_dir --traces-dir /path/to/traces
```

## Stage 1 Gates

The evaluation checks these criteria:
- **ASR Relative Reduction ≥ 50%**: CB model reduces attack success rate
- **Capability Retention > 85%**: Benign tool-calling preserved
- **Output Difference > 10%**: CB model produces different outputs
