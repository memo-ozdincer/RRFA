"""
Shared loss primitives for circuit-breaker training.

This module centralizes the math used by both:
- src/training/train_schema.py
- src/training/trainer.py
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


LOSS_MODE_TRIPLET_FULL = "triplet_full"
LOSS_MODE_LEGACY_SCHEMA = "legacy_schema"
LOSS_MODE_LEGACY_CB = "legacy_cb"

SUPPORTED_LOSS_MODES = (
    LOSS_MODE_TRIPLET_FULL,
    LOSS_MODE_LEGACY_SCHEMA,
    LOSS_MODE_LEGACY_CB,
)

DISTANCE_L2 = "d2"
DISTANCE_COSINE = "dcos"
DISTANCE_MIX = "dmix"
DISTANCE_NULL = "d0"

SUPPORTED_DISTANCES = (
    DISTANCE_L2,
    DISTANCE_COSINE,
    DISTANCE_MIX,
    DISTANCE_NULL,
)


def _combine_masks(
    attention_mask: Optional[torch.Tensor],
    loss_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return loss_mask
    if loss_mask is None:
        return attention_mask
    return attention_mask * loss_mask


def _masked_mean(
    values: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1e-8)
    return (values * mask_f).sum() / denom


def _masked_mean_per_sample(
    values: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=1)
    mask_f = mask.float()
    # Unsqueeze mask for broadcasting with 3D hidden states [B, T, H]
    if values.dim() == 3 and mask_f.dim() == 2:
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [B, 1]
        return (values * mask_f.unsqueeze(-1)).sum(dim=1) / denom  # [B, H]
    denom = mask_f.sum(dim=1).clamp_min(1e-8)
    return (values * mask_f).sum(dim=1) / denom


def select_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    target_layers: Iterable[int],
) -> Dict[int, torch.Tensor]:
    selected: Dict[int, torch.Tensor] = {}
    for layer_idx in target_layers:
        if 0 <= layer_idx < len(hidden_states):
            selected[layer_idx] = hidden_states[layer_idx]
    return selected


def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    temp = max(float(temperature), 1e-6)
    student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temp, dim=-1)

    kl_per_token = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
        log_target=False,
    ).sum(dim=-1)

    combined_mask = _combine_masks(attention_mask, loss_mask)
    loss = _masked_mean(kl_per_token, combined_mask)
    return loss * (temp * temp)


def reroute_loss_relu_cos(
    model_reps: Dict[int, torch.Tensor],
    frozen_reps: Dict[int, torch.Tensor],
    target_layers: Sequence[int],
    attention_mask: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    return_metrics: bool = False,
) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
    combined_mask = _combine_masks(attention_mask, loss_mask)
    total_loss = 0.0
    num_layers = 0
    metric_values = []

    for layer_idx in target_layers:
        if layer_idx not in model_reps or layer_idx not in frozen_reps:
            continue

        h_model = model_reps[layer_idx]
        h_frozen = frozen_reps[layer_idx]
        cos_sim = F.cosine_similarity(h_model, h_frozen, dim=-1, eps=1e-8)

        if return_metrics:
            if combined_mask is None:
                metric_values.append(cos_sim.reshape(-1).detach())
            else:
                metric_values.append(cos_sim[combined_mask.bool()].detach())

        relu_sim = F.relu(cos_sim)
        total_loss = total_loss + _masked_mean(relu_sim, combined_mask)
        num_layers += 1

    final_loss = total_loss / max(num_layers, 1)
    if not return_metrics:
        return final_loss, None

    if metric_values:
        all_values = torch.cat(metric_values)
        metrics = {
            "cos_sim_mean": all_values.mean().item(),
            "cos_sim_std": all_values.std().item(),
            "cos_sim_positive_frac": (all_values > 0).float().mean().item(),
            "target_type": "frozen_baseline",
        }
    else:
        metrics = {
            "cos_sim_mean": 0.0,
            "cos_sim_std": 0.0,
            "cos_sim_positive_frac": 0.0,
            "target_type": "frozen_baseline",
        }
    return final_loss, metrics


def retain_loss_l2(
    model_reps: Dict[int, torch.Tensor],
    frozen_reps: Dict[int, torch.Tensor],
    target_layers: Sequence[int],
    attention_mask: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    combined_mask = _combine_masks(attention_mask, loss_mask)
    total_loss = 0.0
    num_layers = 0

    for layer_idx in target_layers:
        if layer_idx not in model_reps or layer_idx not in frozen_reps:
            continue
        distances = torch.norm(model_reps[layer_idx] - frozen_reps[layer_idx], p=2, dim=-1)
        total_loss = total_loss + _masked_mean(distances, combined_mask)
        num_layers += 1

    return total_loss / max(num_layers, 1)


def random_reroute_loss(
    model_reps: Dict[int, torch.Tensor],
    target_layers: Sequence[int],
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    layer_values = []
    for layer_idx in target_layers:
        if layer_idx not in model_reps:
            continue
        h_model = model_reps[layer_idx]
        h_random = torch.randn_like(h_model)
        h_random = h_random / (h_random.norm(dim=-1, keepdim=True) + 1e-8)
        h_model = h_model / (h_model.norm(dim=-1, keepdim=True) + 1e-8)
        cos_sim = (h_model * h_random).sum(dim=-1).abs()
        layer_values.append(-torch.log(1.0 - cos_sim + 1e-8))

    if not layer_values:
        device = next(iter(model_reps.values())).device if model_reps else "cpu"
        return torch.tensor(0.0, device=device)

    per_token = torch.stack(layer_values, dim=0).mean(dim=0)
    return _masked_mean(per_token, loss_mask)


def retain_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = None if loss_mask is None else loss_mask[..., 1:].contiguous()

    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.size())
    return _masked_mean(loss_per_token, shift_mask)


def pooled_representations(
    reps_by_layer: Dict[int, torch.Tensor],
    target_layers: Sequence[int],
    token_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    pooled_layers = []
    for layer_idx in target_layers:
        if layer_idx not in reps_by_layer:
            continue
        pooled_layers.append(_masked_mean_per_sample(reps_by_layer[layer_idx], token_mask))

    if not pooled_layers:
        if reps_by_layer:
            any_rep = next(iter(reps_by_layer.values()))
            return torch.zeros(any_rep.size(0), any_rep.size(-1), device=any_rep.device)
        raise ValueError("No representations available for pooling")

    return torch.stack(pooled_layers, dim=0).mean(dim=0)


def pair_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    distance: str,
    mix_l2_weight: float = 0.5,
    mix_cos_weight: float = 0.5,
) -> torch.Tensor:
    if distance == DISTANCE_L2:
        return torch.norm(left - right, p=2, dim=-1)
    if distance == DISTANCE_COSINE:
        return 1.0 - F.cosine_similarity(left, right, dim=-1, eps=1e-8)
    if distance == DISTANCE_MIX:
        d2 = torch.norm(left - right, p=2, dim=-1)
        dcos = 1.0 - F.cosine_similarity(left, right, dim=-1, eps=1e-8)
        return mix_l2_weight * d2 + mix_cos_weight * dcos
    if distance == DISTANCE_NULL:
        return torch.zeros(left.size(0), device=left.device, dtype=left.dtype)
    raise ValueError(f"Unknown distance: {distance}")


def triplet_full_loss(
    harmful_new: torch.Tensor,
    harmful_old: torch.Tensor,
    benign_new: torch.Tensor,
    benign_old: torch.Tensor,
    benign_student_logits: torch.Tensor,
    benign_teacher_logits: torch.Tensor,
    benign_attention_mask: Optional[torch.Tensor],
    benign_loss_mask: Optional[torch.Tensor],
    *,
    alpha_benign: float,
    beta_harmful: float,
    gamma_kl: float,
    margin_benign: float,
    margin_harmful: float,
    benign_positive_distance: str,
    benign_negative_distance: str,
    harmful_positive_distance: str,
    harmful_negative_distance: str,
    mix_l2_weight: float,
    mix_cos_weight: float,
    kl_temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    harmful_mean = harmful_new.mean(dim=0, keepdim=True)
    harmful_mean_expanded = harmful_mean.expand_as(harmful_new)

    benign_triplet = F.relu(
        pair_distance(
            benign_old,
            benign_new,
            benign_positive_distance,
            mix_l2_weight,
            mix_cos_weight,
        )
        - pair_distance(
            benign_new,
            harmful_mean_expanded,
            benign_negative_distance,
            mix_l2_weight,
            mix_cos_weight,
        )
        + margin_benign
    ).mean()

    harmful_triplet = F.relu(
        pair_distance(
            harmful_new,
            harmful_mean_expanded,
            harmful_positive_distance,
            mix_l2_weight,
            mix_cos_weight,
        )
        - pair_distance(
            harmful_new,
            harmful_old,
            harmful_negative_distance,
            mix_l2_weight,
            mix_cos_weight,
        )
        + margin_harmful
    ).mean()

    benign_kl = kl_divergence_loss(
        student_logits=benign_student_logits,
        teacher_logits=benign_teacher_logits,
        attention_mask=benign_attention_mask,
        loss_mask=benign_loss_mask,
        temperature=kl_temperature,
    )

    total = alpha_benign * benign_triplet + beta_harmful * harmful_triplet + gamma_kl * benign_kl
    metrics = {
        "triplet_benign_loss": benign_triplet.item(),
        "triplet_harmful_loss": harmful_triplet.item(),
        "triplet_kl_loss": benign_kl.item(),
        "triplet_alpha": float(alpha_benign),
        "triplet_beta": float(beta_harmful),
        "triplet_gamma": float(gamma_kl),
        "total_loss": total.item(),
    }
    return total, metrics
