#!/usr/bin/env python3
"""
Circuit Breakers Training Script

Train a model with Representation Rerouting (RR) to refuse harmful requests
while preserving benign capabilities.

Usage:
    # Default: Llama-4-Scout-17B-16E-Instruct
    python scripts/train_circuit_breaker.py

    # With preset
    python scripts/train_circuit_breaker.py --preset llama-3-8b

    # With overrides
    python scripts/train_circuit_breaker.py --alpha-max 8.0 --total-steps 200

    # Multi-GPU with accelerate
    accelerate launch --num_processes 8 scripts/train_circuit_breaker.py

Examples:
    # Quick test run
    python scripts/train_circuit_breaker.py --total-steps 10 --no-wandb

    # Full training on 8 x H100
    accelerate launch --num_processes 8 scripts/train_circuit_breaker.py \\
        --preset llama-4-scout \\
        --output-dir outputs/cb_llama4_scout \\
        --wandb-run-name cb-scout-v1
"""

import argparse
import sys
import os
from pathlib import Path

# Add repo root to path for imports
# train.py is at <repo>/src/training/train.py, so parents[2] == <repo>
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# CRITICAL FIX: Ensure HOME is redirected to cache directory
# This protects against any libraries defaulting to ~/.cache
cache_root = "/scratch/memoozd/cb-scratch/cache"

# Only override if cache directory exists (i.e., running on cluster)
if os.path.exists(cache_root):
    os.environ["HOME"] = cache_root
    os.makedirs(os.path.join(cache_root, "xdg_cache"), exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(cache_root, "xdg_cache"))


from src.training.config import get_config, CONFIG_PRESETS
from src.training.trainer import CircuitBreakerTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Circuit Breakers (Representation Rerouting)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Preset selection
    parser.add_argument(
        "--preset",
        type=str,
        default="llama-4-scout",
        choices=list(CONFIG_PRESETS.keys()),
        help="Configuration preset to use",
    )
    
    # Model overrides
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model path/name",
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=None,
        help="Maximum alpha for rerouting loss (default: from preset)",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Total training steps (default: from preset)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: from preset)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-GPU batch size (default: from preset)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps",
    )
    
    # Data
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to cb_training_batches.jsonl",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Maximum sequence length for tokenization",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints",
    )

    # Loss weighting
    parser.add_argument(
        "--loss-weighting",
        type=str,
        default=None,
        choices=["single_alpha", "dual"],
        help="Loss weighting strategy: 'single_alpha' (original) or 'dual' (paper-style)",
    )
    
    # Logging
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team (optional)",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Weights & Biases group (e.g., slurm cluster + run family)",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help="Comma-separated Weights & Biases tags",
    )
    parser.add_argument(
        "--wandb-notes",
        type=str,
        default=None,
        help="Weights & Biases notes (optional)",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode override (also respects WANDB_MODE env var)",
    )
    parser.add_argument(
        "--wandb-log-artifacts",
        type=str,
        default=None,
        choices=["none", "final"],
        help="Log artifacts to W&B: 'none' or 'final' (final checkpoint only)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    
    # Advanced
    parser.add_argument(
        "--cb-target-layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to extract representations from (e.g., --cb-target-layers 12 24 36)",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )

    parser.add_argument(
        "--representation-extraction",
        type=str,
        default=None,
        choices=["hidden_states", "hooks"],
        help="How to extract representations: 'hidden_states' (preferred) or 'hooks'",
    )

    parser.add_argument(
        "--alpha-decay-multiplier",
        type=float,
        default=None,
        help="Alpha decays to 0 over (multiplier * total_steps). Set 1.0 to hit 0 by end.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build config overrides from CLI args
    overrides = {}
    
    if args.base_model:
        overrides['base_model'] = args.base_model
    if args.alpha_max is not None:
        overrides['alpha_max'] = args.alpha_max
    if args.total_steps is not None:
        overrides['total_steps'] = args.total_steps
    if args.learning_rate is not None:
        overrides['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.gradient_accumulation_steps is not None:
        overrides['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if args.data_path:
        overrides['data_path'] = args.data_path
    if args.max_seq_length is not None:
        overrides['max_seq_length'] = args.max_seq_length
    if args.output_dir:
        overrides['output_dir'] = args.output_dir
    if args.loss_weighting:
        overrides['loss_weighting'] = args.loss_weighting
    if args.wandb_project:
        overrides['wandb_project'] = args.wandb_project
    if args.wandb_entity:
        overrides['wandb_entity'] = args.wandb_entity
    if args.wandb_group:
        overrides['wandb_group'] = args.wandb_group
    if args.wandb_tags:
        overrides['wandb_tags'] = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    if args.wandb_notes:
        overrides['wandb_notes'] = args.wandb_notes
    if args.wandb_mode:
        overrides['wandb_mode'] = args.wandb_mode
    if args.wandb_log_artifacts:
        overrides['wandb_log_artifacts'] = args.wandb_log_artifacts
    if args.wandb_run_name:
        overrides['wandb_run_name'] = args.wandb_run_name
    if args.no_wandb:
        overrides['use_wandb'] = False
    if args.cb_target_layers:
        overrides['cb_target_layers'] = args.cb_target_layers
    if args.no_gradient_checkpointing:
        overrides['gradient_checkpointing'] = False
    if args.representation_extraction:
        overrides['representation_extraction'] = args.representation_extraction
    if args.alpha_decay_multiplier is not None:
        overrides['alpha_decay_multiplier'] = args.alpha_decay_multiplier
    
    # Get config with preset and overrides
    config = get_config(args.preset, **overrides)
    
    # Handle LoRA overrides (nested config)
    if args.lora_r is not None:
        config.lora.r = args.lora_r
    if args.lora_alpha is not None:
        config.lora.alpha = args.lora_alpha
    
    # Print configuration
    print("=" * 60)
    print("Circuit Breaker Training Configuration")
    print("=" * 60)
    print(f"  Preset: {args.preset}")
    print(f"  Base Model: {config.base_model}")
    print(f"  Alpha Max: {config.alpha_max}")
    print(f"  Alpha Decay Multiplier: {config.alpha_decay_multiplier}")
    print(f"  Total Steps: {config.total_steps}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"  CB Target Layers: {config.cb_target_layers}")
    print(f"  Representation Extraction: {config.representation_extraction}")
    print(f"  LoRA Rank: {config.lora.r}")
    print(f"  LoRA Alpha: {config.lora.alpha}")
    print(f"  Data Path: {config.data_path}")
    print(f"  Output Dir: {config.output_dir}")
    print(f"  WandB: {config.use_wandb}")
    print("=" * 60)
    
    # Create trainer and run
    trainer = CircuitBreakerTrainer(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        trainer.save_checkpoint()
    finally:
        trainer.cleanup()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
