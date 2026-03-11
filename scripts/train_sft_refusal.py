#!/usr/bin/env python3
"""
SFT refusal baseline trainer.

Trains a LoRA adapter with standard causal LM loss on refusal response tokens.
Uses the same render/lossmask format as the CB pipeline (ETL_B output), but
instead of representation rerouting, applies next-token prediction loss on
tokens where loss_mask > 0 (assistant/refusal tokens).

This serves as a baseline: "what if we just fine-tuned the model to produce
refusal text instead of rerouting representations?"

Usage:
    python scripts/train_sft_refusal.py \
        --model /path/to/llama-3.1-8b \
        --renders renders.jsonl \
        --lossmasks lossmasks.jsonl \
        --output-dir ./sft_output \
        --total-steps 300 \
        --batch-size 4 \
        --lr 5e-5
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.hf_utils import resolve_hf_token, resolve_local_model_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading (reuses render/lossmask JSONL format from ETL_B)
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_renders_and_masks(
    render_paths: List[Path],
    lossmask_paths: List[Path],
) -> List[Dict[str, Any]]:
    """Load and join render_v1 + lossmask_v1 rows by render_id."""
    samples: List[Dict[str, Any]] = []

    for render_path, lossmask_path in zip(render_paths, lossmask_paths):
        renders = {row["render_id"]: row for row in _iter_jsonl(render_path)}

        for mask_row in _iter_jsonl(lossmask_path):
            render_id = mask_row.get("render_id")
            render = renders.get(render_id)
            if render is None:
                continue

            loss_mask = mask_row["loss_mask"]
            # Skip samples with no trainable tokens
            if not any(m > 0 for m in loss_mask):
                continue

            input_ids = render["input_ids"]
            samples.append({
                "input_ids": input_ids,
                "attention_mask": render.get("attention_mask", [1] * len(input_ids)),
                "loss_mask": loss_mask,
            })

    logger.info("Loaded %d samples with non-zero loss masks", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """Simple SFT dataset from pre-tokenized renders + loss masks."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        max_length: int = 4096,
        pad_token_id: int = 0,
    ):
        self.samples = samples
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.samples)

    def _pad(self, values: list, fill: Any) -> list:
        if len(values) < self.max_length:
            return values + [fill] * (self.max_length - len(values))
        return values[: self.max_length]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        input_ids = self._pad(sample["input_ids"][: self.max_length], self.pad_token_id)
        attention_mask = self._pad(sample["attention_mask"][: self.max_length], 0)
        loss_mask = self._pad(sample["loss_mask"][: self.max_length], 0.0)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_sft_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Standard causal LM cross-entropy on tokens where loss_mask > 0.

    logits:    (B, T, V)
    input_ids: (B, T)
    loss_mask: (B, T)  -- 1.0 on assistant/refusal tokens, 0.0 elsewhere
    """
    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
    shift_labels = input_ids[:, 1:].contiguous()     # (B, T-1)
    shift_mask = loss_mask[:, 1:].contiguous()       # (B, T-1)

    # Flatten
    B, T, V = shift_logits.shape
    flat_logits = shift_logits.view(-1, V)
    flat_labels = shift_labels.view(-1)
    flat_mask = shift_mask.view(-1)

    # Per-token CE
    per_token_loss = F.cross_entropy(flat_logits, flat_labels, reduction="none")

    # Mask and average over active tokens
    masked_loss = per_token_loss * flat_mask
    num_active = flat_mask.sum().clamp(min=1.0)

    return masked_loss.sum() / num_active


def train(
    model_path: str,
    render_paths: List[Path],
    lossmask_paths: List[Path],
    output_dir: Path,
    total_steps: int = 300,
    batch_size: int = 4,
    grad_accum: int = 2,
    lr: float = 5e-5,
    warmup_steps: int = 20,
    max_seq_length: int = 4096,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    logging_steps: int = 10,
) -> None:
    """Run SFT training loop."""

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision="bf16",
    )

    # ---- Resolve model path ----
    hf_token = resolve_hf_token()
    offline_mode = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    resolved_path = model_path
    if offline_mode:
        resolved_path = resolve_local_model_path(model_path, hf_token)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_path,
        token=hf_token,
        trust_remote_code=True,
        local_files_only=offline_mode,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Model ----
    logger.info("Loading model from %s", resolved_path)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=offline_mode,
    )

    # ---- LoRA ----
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA applied: trainable=%d (%.2f%% of %d)",
        trainable, 100.0 * trainable / total_params, total_params,
    )

    # ---- Data ----
    samples = load_renders_and_masks(render_paths, lossmask_paths)
    if not samples:
        raise ValueError("No training samples loaded. Check render/lossmask paths.")

    dataset = SFTDataset(
        samples=samples,
        max_length=max_seq_length,
        pad_token_id=tokenizer.pad_token_id,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ---- Accelerate ----
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # ---- Training loop ----
    output_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    global_step = 0
    data_iter = iter(dataloader)
    running_loss = 0.0

    logger.info("Starting SFT training: %d steps, batch=%d, grad_accum=%d, lr=%s",
                total_steps, batch_size, grad_accum, lr)

    while global_step < total_steps:
        # Get batch (cycle through data)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = compute_sft_loss(
                logits=outputs.logits,
                input_ids=batch["input_ids"],
                loss_mask=batch["loss_mask"],
            )

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        global_step += 1

        if global_step % logging_steps == 0:
            avg_loss = running_loss / logging_steps
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                "[Step %4d/%d] loss=%.4f lr=%.2e",
                global_step, total_steps, avg_loss, current_lr,
            )
            running_loss = 0.0

    # ---- Save ----
    logger.info("Training complete. Saving adapter to %s", output_dir / "final")
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SFT refusal baseline trainer")
    parser.add_argument("--model", type=str, required=True, help="Base model path or HF ID")
    parser.add_argument("--renders", type=Path, nargs="+", required=True, help="Render JSONL files")
    parser.add_argument("--lossmasks", type=Path, nargs="+", required=True, help="Lossmask JSONL files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for adapter")
    parser.add_argument("--total-steps", type=int, default=300, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2, help="Gradient accumulation")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=20, help="Warmup steps")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    args = parser.parse_args()

    if len(args.renders) != len(args.lossmasks):
        raise ValueError("--renders and --lossmasks must have the same number of files")

    train(
        model_path=args.model,
        render_paths=args.renders,
        lossmask_paths=args.lossmasks,
        output_dir=args.output_dir,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        grad_accum=args.gradient_accumulation_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        logging_steps=args.logging_steps,
    )


if __name__ == "__main__":
    main()
