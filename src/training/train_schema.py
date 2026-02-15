#!/usr/bin/env python3
"""
Circuit Breaker training entrypoint for schema v1 datasets.

This script keeps schema-specific data loading (render/lossmask JSONL), but
uses the shared `CircuitBreakerTrainer` core so training logic is not duplicated.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import CircuitBreakerConfig, get_config
from src.training.hf_utils import resolve_hf_token, resolve_local_model_path
from src.training.losses import SUPPORTED_DISTANCES, SUPPORTED_LOSS_MODES
from src.training.trainer import CircuitBreakerTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Iterate over JSONL file."""
    with open(path, "r") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_renders_and_masks(render_path: Path, lossmask_path: Path) -> List[Dict[str, Any]]:
    """Load and join render_v1 and lossmask_v1 rows by render_id."""
    renders = {row["render_id"]: row for row in _iter_jsonl(render_path)}

    samples: List[Dict[str, Any]] = []
    for mask_row in _iter_jsonl(lossmask_path):
        render_id = mask_row.get("render_id")
        render = renders.get(render_id)
        if render is None:
            logger.warning("No render found for lossmask %s", render_id)
            continue

        input_ids = render["input_ids"]
        samples.append(
            {
                "input_ids": input_ids,
                "attention_mask": render.get("attention_mask", [1] * len(input_ids)),
                "loss_mask": mask_row["loss_mask"],
                "sample_weight": mask_row.get("sample_weight", 1.0),
                "trace_id": mask_row.get("trace_id"),
                "render_id": render_id,
                "policy_id": mask_row.get("policy_id"),
            }
        )

    return samples


def load_ds_dr_data(
    ds_render_path: Path,
    ds_lossmask_path: Path,
    dr_render_path: Path,
    dr_lossmask_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load DS as harmful and DR as benign."""
    logger.info("Loading DS (harmful) from %s", ds_render_path)
    harmful = load_renders_and_masks(ds_render_path, ds_lossmask_path)
    logger.info("Loaded %d harmful DS samples", len(harmful))

    logger.info("Loading DR (benign) from %s", dr_render_path)
    benign = load_renders_and_masks(dr_render_path, dr_lossmask_path)
    logger.info("Loaded %d benign DR samples", len(benign))

    return harmful, benign


def load_labeled_data(
    render_path: Path,
    lossmask_path: Path,
    traces_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load data and split by labels.is_harmful from traces."""
    all_samples = load_renders_and_masks(render_path, lossmask_path)

    trace_labels: Dict[str, bool] = {}
    if traces_path and traces_path.exists():
        for row in _iter_jsonl(traces_path):
            trace_id = row.get("id")
            labels = row.get("labels", {})
            trace_labels[trace_id] = bool(labels.get("is_harmful", False))

    harmful: List[Dict[str, Any]] = []
    benign: List[Dict[str, Any]] = []

    for sample in all_samples:
        if trace_labels.get(sample.get("trace_id"), False):
            harmful.append(sample)
        else:
            benign.append(sample)

    logger.info(
        "Split %d samples -> harmful=%d benign=%d",
        len(all_samples),
        len(harmful),
        len(benign),
    )
    return harmful, benign


class SchemaDataset(Dataset):
    """Dataset of harmful/benign pairs from pre-tokenized schema rows."""

    def __init__(
        self,
        harmful_samples: List[Dict[str, Any]],
        benign_samples: List[Dict[str, Any]],
        max_length: int = 2048,
        pad_token_id: int = 0,
    ):
        self.harmful_samples = harmful_samples
        self.benign_samples = benign_samples
        self.max_length = max_length
        self.pad_token_id = pad_token_id

        self.num_pairs = min(len(harmful_samples), len(benign_samples))
        if self.num_pairs == 0:
            raise ValueError("No paired data available (need both harmful and benign samples)")

        logger.info(
            "SchemaDataset pairs=%d (harmful=%d benign=%d)",
            self.num_pairs,
            len(harmful_samples),
            len(benign_samples),
        )

    def __len__(self) -> int:
        return self.num_pairs

    def _prepare_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        input_ids = sample["input_ids"][: self.max_length]
        attention_mask = sample["attention_mask"][: self.max_length]
        loss_mask = sample["loss_mask"][: self.max_length]

        def _pad(values: List[Any], value: Any) -> List[Any]:
            if len(values) < self.max_length:
                return values + [value] * (self.max_length - len(values))
            return values

        input_ids = _pad(input_ids, self.pad_token_id)
        attention_mask = _pad(attention_mask, 0)
        loss_mask = _pad(loss_mask, 0.0)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
            "sample_weight": torch.tensor(sample.get("sample_weight", 1.0), dtype=torch.float),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        harmful = self._prepare_sample(self.harmful_samples[idx % len(self.harmful_samples)])
        benign = self._prepare_sample(self.benign_samples[idx % len(self.benign_samples)])

        return {
            "harmful_input_ids": harmful["input_ids"],
            "harmful_attention_mask": harmful["attention_mask"],
            "harmful_loss_mask": harmful["loss_mask"],
            "harmful_sample_weight": harmful["sample_weight"],
            "benign_input_ids": benign["input_ids"],
            "benign_attention_mask": benign["attention_mask"],
            "benign_loss_mask": benign["loss_mask"],
            "benign_sample_weight": benign["sample_weight"],
        }


def _resolve_training_tokenizer(config: CircuitBreakerConfig) -> AutoTokenizer:
    hf_token = resolve_hf_token()
    offline_mode = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

    model_path = config.base_model
    if offline_mode:
        model_path = resolve_local_model_path(config.base_model, hf_token)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=hf_token,
        trust_remote_code=True,
        local_files_only=offline_mode,
    )
    if getattr(tokenizer, "padding_side", None) != "right":
        tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def run_training(
    config: CircuitBreakerConfig,
    harmful_samples: List[Dict[str, Any]],
    benign_samples: List[Dict[str, Any]],
) -> None:
    """Build schema dataloader and run the shared trainer core."""
    tokenizer = _resolve_training_tokenizer(config)

    dataset = SchemaDataset(
        harmful_samples=harmful_samples,
        benign_samples=benign_samples,
        max_length=config.max_seq_length,
        pad_token_id=tokenizer.pad_token_id,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    trainer = CircuitBreakerTrainer(config=config, dataloader=dataloader, tokenizer=tokenizer)
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user; saving checkpoint")
        trainer.save_checkpoint()
    finally:
        trainer.cleanup()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Circuit Breaker training with schema v1 data format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_argument_group("Data Loading")
    mode_group.add_argument(
        "--mode",
        type=str,
        default="ds_dr",
        choices=["ds_dr", "labeled", "mixed"],
        help="Data loading mode",
    )

    mode_group.add_argument("--ds-renders", type=Path, help="DS renders JSONL (harmful)")
    mode_group.add_argument("--ds-lossmasks", type=Path, help="DS lossmasks JSONL (harmful)")
    mode_group.add_argument("--dr-renders", type=Path, help="DR renders JSONL (benign)")
    mode_group.add_argument("--dr-lossmasks", type=Path, help="DR lossmasks JSONL (benign)")

    mode_group.add_argument("--renders", type=Path, help="Renders JSONL for labeled mode")
    mode_group.add_argument("--lossmasks", type=Path, help="Lossmasks JSONL for labeled mode")
    mode_group.add_argument("--traces", type=Path, help="Traces JSONL for labels")

    mode_group.add_argument("--harmful-renders", type=Path, nargs="+", help="Harmful renders JSONL")
    mode_group.add_argument("--harmful-lossmasks", type=Path, nargs="+", help="Harmful lossmasks JSONL")
    mode_group.add_argument("--benign-renders", type=Path, nargs="+", help="Benign renders JSONL")
    mode_group.add_argument("--benign-lossmasks", type=Path, nargs="+", help="Benign lossmasks JSONL")

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--preset", type=str, default="llama-3.1-8b-instruct", help="Config preset")
    model_group.add_argument("--model", type=str, help="Override base model")
    model_group.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    model_group.add_argument("--resume-from", type=Path, help="Reserved for future resume support")

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--total-steps", type=int, help="Total training steps")
    train_group.add_argument("--batch-size", type=int, help="Batch size per GPU")
    train_group.add_argument("--learning-rate", type=float, help="Learning rate")
    train_group.add_argument("--warmup-steps", type=int, help="Warmup steps")
    train_group.add_argument("--alpha-max", type=float, help="Initial alpha")
    train_group.add_argument("--max-seq-length", type=int, help="Max sequence length")
    train_group.add_argument("--gradient-accumulation-steps", type=int, help="Gradient accumulation")

    cb_group = parser.add_argument_group("Circuit Breaker")
    cb_group.add_argument("--cb-target-layers", type=int, nargs="+", help="Target layers")
    cb_group.add_argument(
        "--loss-mode",
        type=str,
        choices=list(SUPPORTED_LOSS_MODES),
        help="Core loss objective",
    )
    cb_group.add_argument(
        "--loss-weighting",
        type=str,
        choices=["single_alpha", "dual"],
        help="Legacy weighting strategy",
    )
    cb_group.add_argument("--triplet-alpha-benign", type=float, help="Triplet alpha")
    cb_group.add_argument("--triplet-beta-harmful", type=float, help="Triplet beta")
    cb_group.add_argument("--triplet-gamma-kl", type=float, help="Triplet gamma")
    cb_group.add_argument("--triplet-margin-benign", type=float, help="Triplet benign margin")
    cb_group.add_argument("--triplet-margin-harmful", type=float, help="Triplet harmful margin")
    cb_group.add_argument(
        "--triplet-benign-positive-distance",
        type=str,
        choices=list(SUPPORTED_DISTANCES),
        help="Distance for benign positive pair",
    )
    cb_group.add_argument(
        "--triplet-benign-negative-distance",
        type=str,
        choices=list(SUPPORTED_DISTANCES),
        help="Distance for benign negative pair",
    )
    cb_group.add_argument(
        "--triplet-harmful-positive-distance",
        type=str,
        choices=list(SUPPORTED_DISTANCES),
        help="Distance for harmful positive pair",
    )
    cb_group.add_argument(
        "--triplet-harmful-negative-distance",
        type=str,
        choices=list(SUPPORTED_DISTANCES),
        help="Distance for harmful negative pair",
    )
    cb_group.add_argument("--triplet-mix-l2-weight", type=float, help="dmix L2 coefficient")
    cb_group.add_argument("--triplet-mix-cos-weight", type=float, help="dmix cosine coefficient")

    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument("--lora-r", type=int, help="LoRA rank")
    lora_group.add_argument("--lora-alpha", type=int, help="LoRA alpha")
    lora_group.add_argument("--lora-dropout", type=float, help="LoRA dropout")

    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--wandb-project", type=str, help="W&B project")
    log_group.add_argument("--wandb-run-name", type=str, help="W&B run name")
    log_group.add_argument("--no-wandb", action="store_true", help="Disable W&B")
    log_group.add_argument("--logging-steps", type=int, help="Log every N steps")
    log_group.add_argument("--save-steps", type=int, help="Save every N steps")

    return parser.parse_args()


def _load_samples(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if args.mode == "ds_dr":
        if not all([args.ds_renders, args.ds_lossmasks, args.dr_renders, args.dr_lossmasks]):
            raise ValueError("ds_dr mode requires --ds-renders --ds-lossmasks --dr-renders --dr-lossmasks")
        return load_ds_dr_data(
            args.ds_renders,
            args.ds_lossmasks,
            args.dr_renders,
            args.dr_lossmasks,
        )

    if args.mode == "labeled":
        if not all([args.renders, args.lossmasks]):
            raise ValueError("labeled mode requires --renders --lossmasks")
        return load_labeled_data(args.renders, args.lossmasks, args.traces)

    if not all([args.harmful_renders, args.harmful_lossmasks, args.benign_renders, args.benign_lossmasks]):
        raise ValueError("mixed mode requires harmful/benign renders+lossmasks arguments")

    if len(args.harmful_renders) != len(args.harmful_lossmasks):
        raise ValueError("--harmful-renders and --harmful-lossmasks must have same length")
    if len(args.benign_renders) != len(args.benign_lossmasks):
        raise ValueError("--benign-renders and --benign-lossmasks must have same length")

    harmful: List[Dict[str, Any]] = []
    for render_path, mask_path in zip(args.harmful_renders, args.harmful_lossmasks):
        harmful.extend(load_renders_and_masks(render_path, mask_path))

    benign: List[Dict[str, Any]] = []
    for render_path, mask_path in zip(args.benign_renders, args.benign_lossmasks):
        benign.extend(load_renders_and_masks(render_path, mask_path))

    logger.info("Mixed mode loaded harmful=%d benign=%d", len(harmful), len(benign))
    return harmful, benign


def _build_config(args: argparse.Namespace) -> CircuitBreakerConfig:
    overrides: Dict[str, Any] = {
        "output_dir": str(args.output_dir),
    }

    mapping = {
        "model": "base_model",
        "total_steps": "total_steps",
        "batch_size": "batch_size",
        "learning_rate": "learning_rate",
        "warmup_steps": "warmup_steps",
        "alpha_max": "alpha_max",
        "max_seq_length": "max_seq_length",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "cb_target_layers": "cb_target_layers",
        "loss_mode": "loss_mode",
        "loss_weighting": "loss_weighting",
        "triplet_alpha_benign": "triplet_alpha_benign",
        "triplet_beta_harmful": "triplet_beta_harmful",
        "triplet_gamma_kl": "triplet_gamma_kl",
        "triplet_margin_benign": "triplet_margin_benign",
        "triplet_margin_harmful": "triplet_margin_harmful",
        "triplet_benign_positive_distance": "triplet_benign_positive_distance",
        "triplet_benign_negative_distance": "triplet_benign_negative_distance",
        "triplet_harmful_positive_distance": "triplet_harmful_positive_distance",
        "triplet_harmful_negative_distance": "triplet_harmful_negative_distance",
        "triplet_mix_l2_weight": "triplet_mix_l2_weight",
        "triplet_mix_cos_weight": "triplet_mix_cos_weight",
        "wandb_project": "wandb_project",
        "wandb_run_name": "wandb_run_name",
        "logging_steps": "logging_steps",
        "save_steps": "save_steps",
    }

    for arg_name, config_name in mapping.items():
        value = getattr(args, arg_name)
        if value is not None:
            overrides[config_name] = value

    if args.no_wandb:
        overrides["use_wandb"] = False

    config = get_config(args.preset, **overrides)

    if args.lora_r is not None:
        config.lora.r = args.lora_r
    if args.lora_alpha is not None:
        config.lora.alpha = args.lora_alpha
    if args.lora_dropout is not None:
        config.lora.dropout = args.lora_dropout

    return config


def main() -> None:
    args = _parse_args()

    if args.resume_from is not None:
        logger.warning("--resume-from is currently ignored by shared trainer path")

    try:
        harmful_samples, benign_samples = _load_samples(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    config = _build_config(args)

    logger.info("Starting schema training via shared trainer core")
    logger.info("  loss_mode=%s", config.loss_mode)
    logger.info("  output_dir=%s", config.output_dir)
    logger.info("  harmful=%d benign=%d", len(harmful_samples), len(benign_samples))

    run_training(config, harmful_samples, benign_samples)


if __name__ == "__main__":
    main()
