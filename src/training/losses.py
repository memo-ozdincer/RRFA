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
LOSS_MODE_COSINE_SIMPLE = "cosine_simple"
LOSS_MODE_L2_SIMPLE = "l2_simple"
LOSS_MODE_LEGACY_SCHEMA = "legacy_schema"
LOSS_MODE_LEGACY_CB = "legacy_cb"
LOSS_MODE_PER_TOKEN_CB = "per_token_cb"
LOSS_MODE_SIMPLIFIED_POOLED = "simplified_pooled"
LOSS_MODE_ORIGINAL_RR = "original_rr"

SUPPORTED_LOSS_MODES = (
    LOSS_MODE_TRIPLET_FULL,
    LOSS_MODE_COSINE_SIMPLE,
    LOSS_MODE_L2_SIMPLE,
    LOSS_MODE_LEGACY_SCHEMA,
    LOSS_MODE_LEGACY_CB,
    LOSS_MODE_PER_TOKEN_CB,
    LOSS_MODE_SIMPLIFIED_POOLED,
    LOSS_MODE_ORIGINAL_RR,
)

DISTANCE_L2 = "d2"
DISTANCE_COSINE = "dcos"
DISTANCE_MIX = "dmix"
DISTANCE_NULL = "d0"
DISTANCE_L2_RELU_COS = "dl2rc"
DISTANCE_L2_SQUARED = "dl2sq"

SUPPORTED_DISTANCES = (
    DISTANCE_L2,
    DISTANCE_COSINE,
    DISTANCE_MIX,
    DISTANCE_NULL,
    DISTANCE_L2_RELU_COS,
    DISTANCE_L2_SQUARED,
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
    pooling_mode: str = "legacy",
) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=1)
    mask_f = mask.float()

    if values.dim() == 3 and mask_f.dim() == 2:
        if pooling_mode == "correct":
            # Proper token masking: [B, T] -> [B, T, 1], broadcast along H
            mask_f_expanded = mask_f.unsqueeze(-1)
            denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1e-8)
            return (values * mask_f_expanded).sum(dim=1) / denom
        else:
            # Legacy: when T==H, mask broadcasts as [B, 1, H] — masks hidden
            # dims instead of tokens. Margins 500/1500 tuned for this behavior.
            H = values.size(-1)
            T_mask = mask_f.size(-1)
            if T_mask < H:
                mask_f = F.pad(mask_f, (0, H - T_mask), value=0.0)
            elif T_mask > H:
                mask_f = mask_f[:, :H]

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
    pooling_mode: str = "legacy",
) -> torch.Tensor:
    pooled_layers = []
    for layer_idx in target_layers:
        if layer_idx not in reps_by_layer:
            continue
        pooled_layers.append(
            _masked_mean_per_sample(reps_by_layer[layer_idx], token_mask, pooling_mode=pooling_mode)
        )

    if not pooled_layers:
        if reps_by_layer:
            any_rep = next(iter(reps_by_layer.values()))
            return torch.zeros(any_rep.size(0), any_rep.size(-1), device=any_rep.device)
        raise ValueError("No representations available for pooling")

    return torch.stack(pooled_layers, dim=0).mean(dim=0)


def _l2_relu_cos_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """L2 + 10*ReLU(cos_dist) — non-zero gradient at LoRA scale.

    Cosine distance alone has near-zero gradient when representations are
    nearly identical (cos_sim ≈ 1.0 at LoRA scale).  L2 provides non-zero
    gradient from the start; 10*ReLU(cos) amplifies angular differences
    once L2 has moved representations apart.  Validated by Sam's code.
    """
    l2 = torch.norm(a - b, dim=-1)
    cos_dist = 1.0 - F.cosine_similarity(a, b, dim=-1, eps=1e-8)
    return l2 + 10.0 * F.relu(cos_dist)


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
    if distance == DISTANCE_L2_RELU_COS:
        return _l2_relu_cos_distance(left, right)
    if distance == DISTANCE_L2_SQUARED:
        return (left - right).pow(2).sum(dim=-1)
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

    # FIX: benign KL must NOT use loss_mask (injection_aware = all-zeros
    # for benign traces → KL was always zero).  Use attention_mask only.
    benign_kl = kl_divergence_loss(
        student_logits=benign_student_logits,
        teacher_logits=benign_teacher_logits,
        attention_mask=benign_attention_mask,
        loss_mask=None,
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


def simple_cosine_loss(
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
    kl_temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Simple cosine-based circuit breaker loss with bounded terms.

    Harmful: ReLU(cos_sim(new, old)) -- pushes reps away from frozen until
        perpendicular (cos_sim <= 0), then gradient stops.  In [0, 1].
    Benign: 1 - cos_sim(new, old) -- keeps reps close to frozen.  In [0, 2].
    KL: standard KL divergence on benign logits.

    Total = alpha_benign * L_benign + beta_harmful * L_harmful + gamma_kl * L_kl

    All cosine terms are naturally bounded, so gamma_kl=1-10 is typically
    sufficient (vs 50-200 for triplet_full with unbounded margins).
    """
    # Harmful: minimize cosine similarity (push away from frozen)
    harmful_cos = F.cosine_similarity(harmful_new, harmful_old, dim=-1, eps=1e-8)
    harmful_loss = F.relu(harmful_cos).mean()

    # Benign: keep representations close to frozen
    benign_cos = F.cosine_similarity(benign_new, benign_old, dim=-1, eps=1e-8)
    benign_loss = (1.0 - benign_cos).mean()

    # KL divergence: preserve language modeling capability
    kl_loss = kl_divergence_loss(
        student_logits=benign_student_logits,
        teacher_logits=benign_teacher_logits,
        attention_mask=benign_attention_mask,
        loss_mask=benign_loss_mask,
        temperature=kl_temperature,
    )

    total = alpha_benign * benign_loss + beta_harmful * harmful_loss + gamma_kl * kl_loss

    metrics = {
        "triplet_benign_loss": benign_loss.item(),
        "triplet_harmful_loss": harmful_loss.item(),
        "triplet_kl_loss": kl_loss.item(),
        "triplet_alpha": float(alpha_benign),
        "triplet_beta": float(beta_harmful),
        "triplet_gamma": float(gamma_kl),
        "mean_harmful_cos": harmful_cos.mean().item(),
        "mean_benign_cos": benign_cos.mean().item(),
        "total_loss": total.item(),
    }
    return total, metrics


def simple_l2_loss(
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
    kl_temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """L2-based circuit breaker loss with bounded terms.

    Uses L2 distance instead of cosine similarity.  Cosine has vanishing
    gradients when vectors are nearly identical (cos ~ 1), which makes it
    unable to learn with LoRA (rank-16 changes in 4096-dim space barely
    move the cosine angle).  L2 distance provides a gradient of -1/scale
    at d=0, so learning starts immediately.

    Harmful: exp(-||new - old||₂ / scale) -- bounded [0, 1].
        1.0 when identical, decreases as reps diverge.
    Benign:  1 - exp(-||new - old||₂² / scale²) -- bounded [0, 1].
        0.0 when identical, increases as reps diverge.
    KL: standard KL divergence on benign logits.

    The scale is auto-computed from the frozen model's representation norm
    so the loss is well-calibrated regardless of layer / model size.
    """
    # Auto-scale: use the frozen model's norm as reference.
    # Detach so it doesn't affect gradients.
    scale_h = harmful_old.detach().norm(dim=-1).mean().clamp(min=1.0)
    scale_b = benign_old.detach().norm(dim=-1).mean().clamp(min=1.0)

    # Harmful: push apart (maximize L2 distance from frozen)
    harmful_l2 = (harmful_new - harmful_old).norm(dim=-1)  # per-token or per-sample
    harmful_loss = torch.exp(-harmful_l2 / scale_h).mean()

    # Benign: keep close (minimize L2 distance from frozen)
    benign_l2 = (benign_new - benign_old).norm(dim=-1)
    benign_loss = (1.0 - torch.exp(-(benign_l2 / scale_b).square())).mean()

    # KL divergence: preserve language modeling capability
    kl_loss = kl_divergence_loss(
        student_logits=benign_student_logits,
        teacher_logits=benign_teacher_logits,
        attention_mask=benign_attention_mask,
        loss_mask=benign_loss_mask,
        temperature=kl_temperature,
    )

    total = alpha_benign * benign_loss + beta_harmful * harmful_loss + gamma_kl * kl_loss

    metrics = {
        "triplet_benign_loss": benign_loss.item(),
        "triplet_harmful_loss": harmful_loss.item(),
        "triplet_kl_loss": kl_loss.item(),
        "triplet_alpha": float(alpha_benign),
        "triplet_beta": float(beta_harmful),
        "triplet_gamma": float(gamma_kl),
        "mean_harmful_l2": harmful_l2.mean().item(),
        "mean_benign_l2": benign_l2.mean().item(),
        "scale_h": scale_h.item(),
        "scale_b": scale_b.item(),
        "total_loss": total.item(),
    }
    return total, metrics


def _per_token_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    distance: str = "dl2rc",
) -> torch.Tensor:
    """Per-token distance between representations. Returns [B, T]."""
    if distance == DISTANCE_L2_SQUARED:
        return (a - b).pow(2).sum(dim=-1)
    elif distance == DISTANCE_L2_RELU_COS:
        return _l2_relu_cos_distance(a, b)
    elif distance == DISTANCE_L2:
        return torch.norm(a - b, p=2, dim=-1)
    elif distance == DISTANCE_COSINE:
        return 1.0 - F.cosine_similarity(a, b, dim=-1, eps=1e-8)
    else:
        return _l2_relu_cos_distance(a, b)


def _per_token_directional_loss(
    model_reps: Dict[int, torch.Tensor],
    frozen_reps: Dict[int, torch.Tensor],
    target_layers: Sequence[int],
    mask: torch.Tensor,
    margin: float,
    direction: str,
    distance: str = "dl2rc",
) -> Tuple[torch.Tensor, float]:
    """Per-token distance with margin.

    direction="push": harmful — loss when dist < margin (push apart)
    direction="pull": benign  — loss when dist > margin (keep close)

    Returns (loss, mean_masked_distance) for diagnostics.
    """
    total = torch.tensor(0.0, device=mask.device)
    dist_sum = 0.0
    num_layers = 0
    mask_f = mask.float()
    mask_denom = mask_f.sum().clamp_min(1e-8)

    for layer_idx in target_layers:
        if layer_idx not in model_reps or layer_idx not in frozen_reps:
            continue

        # Per-token distance: [B, T]
        d = _per_token_distance(model_reps[layer_idx], frozen_reps[layer_idx], distance)

        if direction == "push":
            per_token_loss = F.relu(margin - d)
        else:
            per_token_loss = F.relu(d - margin)

        weighted = per_token_loss * mask_f
        total = total + weighted.sum() / mask_denom
        dist_sum += (d.detach() * mask_f).sum().item() / mask_denom.item()
        num_layers += 1

    n = max(num_layers, 1)
    return total / n, dist_sum / n


def per_token_cb_loss(
    harmful_model_reps: Dict[int, torch.Tensor],
    harmful_frozen_reps: Dict[int, torch.Tensor],
    benign_model_reps: Dict[int, torch.Tensor],
    benign_frozen_reps: Dict[int, torch.Tensor],
    benign_student_logits: torch.Tensor,
    benign_teacher_logits: torch.Tensor,
    target_layers: Sequence[int],
    harmful_mask: torch.Tensor,
    benign_mask: torch.Tensor,
    *,
    alpha_benign: float,
    beta_harmful: float,
    gamma_kl: float,
    margin_benign: float,
    margin_harmful: float,
    benign_attention_mask: Optional[torch.Tensor] = None,
    kl_temperature: float = 1.0,
    distance: str = "dl2rc",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Per-token circuit breaker loss (Option A from technical plan).

    No pooling, no cluster center, no decoupled masks.
    Per-token distance weighted by masks.

    Harmful: push each token beyond margin (injection_aware mask)
    Benign:  keep each token within margin (attention_mask)
    KL:      preserve benign output distribution (attention_mask, no loss_mask)
    """
    harmful_loss, mean_harmful_dist = _per_token_directional_loss(
        harmful_model_reps, harmful_frozen_reps, target_layers,
        harmful_mask, margin_harmful, "push", distance=distance,
    )

    benign_loss, mean_benign_dist = _per_token_directional_loss(
        benign_model_reps, benign_frozen_reps, target_layers,
        benign_mask, margin_benign, "pull", distance=distance,
    )

    kl_loss = kl_divergence_loss(
        student_logits=benign_student_logits,
        teacher_logits=benign_teacher_logits,
        attention_mask=benign_attention_mask if benign_attention_mask is not None else benign_mask,
        loss_mask=None,
        temperature=kl_temperature,
    )

    total = alpha_benign * benign_loss + beta_harmful * harmful_loss + gamma_kl * kl_loss

    metrics = {
        "triplet_benign_loss": benign_loss.item(),
        "triplet_harmful_loss": harmful_loss.item(),
        "triplet_kl_loss": kl_loss.item(),
        "triplet_alpha": float(alpha_benign),
        "triplet_beta": float(beta_harmful),
        "triplet_gamma": float(gamma_kl),
        "mean_harmful_dist": mean_harmful_dist,
        "mean_benign_dist": mean_benign_dist,
        "total_loss": total.item(),
    }
    return total, metrics


def simplified_pooled_loss(
    harmful_new: torch.Tensor,
    harmful_old: torch.Tensor,
    benign_new: torch.Tensor,
    benign_old: torch.Tensor,
    benign_student_logits: torch.Tensor,
    benign_teacher_logits: torch.Tensor,
    benign_attention_mask: Optional[torch.Tensor],
    *,
    alpha_benign: float,
    beta_harmful: float,
    gamma_kl: float,
    margin_benign: float,
    margin_harmful: float,
    kl_temperature: float = 1.0,
    distance: str = "dl2rc",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Simplified pooled loss (Option B from technical plan).

    No cluster center, no cross-terms.  Pooled vectors with margin losses.

    Harmful pool: injection_aware mask (focused on injection+decision tokens)
    Benign pool:  attention_mask (all real tokens)
    KL:           attention_mask, no loss_mask
    """
    # Distance on pooled vectors
    harmful_dist = pair_distance(harmful_new, harmful_old, distance)
    benign_dist = pair_distance(benign_new, benign_old, distance)

    # Harmful: push apart (loss when dist < margin)
    harmful_loss = F.relu(margin_harmful - harmful_dist).mean()
    # Benign: keep close (loss when dist > margin)
    benign_loss = F.relu(benign_dist - margin_benign).mean()

    kl_loss = kl_divergence_loss(
        student_logits=benign_student_logits,
        teacher_logits=benign_teacher_logits,
        attention_mask=benign_attention_mask,
        loss_mask=None,
        temperature=kl_temperature,
    )

    total = alpha_benign * benign_loss + beta_harmful * harmful_loss + gamma_kl * kl_loss

    metrics = {
        "triplet_benign_loss": benign_loss.item(),
        "triplet_harmful_loss": harmful_loss.item(),
        "triplet_kl_loss": kl_loss.item(),
        "triplet_alpha": float(alpha_benign),
        "triplet_beta": float(beta_harmful),
        "triplet_gamma": float(gamma_kl),
        "mean_harmful_dist": harmful_dist.mean().item(),
        "mean_benign_dist": benign_dist.mean().item(),
        "total_loss": total.item(),
    }
    return total, metrics


def original_rr_loss(
    harmful_model_reps: Dict[int, torch.Tensor],
    harmful_frozen_reps: Dict[int, torch.Tensor],
    benign_model_reps: Dict[int, torch.Tensor],
    benign_frozen_reps: Dict[int, torch.Tensor],
    target_layers: Sequence[int],
    *,
    c_s: float,
    c_r: float,
    harmful_mask: Optional[torch.Tensor] = None,
    benign_mask: Optional[torch.Tensor] = None,
    gamma_kl: float = 0.0,
    benign_student_logits: Optional[torch.Tensor] = None,
    benign_teacher_logits: Optional[torch.Tensor] = None,
    benign_attention_mask: Optional[torch.Tensor] = None,
    kl_temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Original RR loss from Zou et al. (2024), Algorithm 1, with optional KL.

    L = c_s * ReLU(cos_sim(frozen, cb)) + c_r * ||frozen - cb||_2
      + gamma_kl * KL(cb || frozen)   [only when gamma_kl > 0]

    Paper schedule:
        c_s = alpha * step / (2 * total_steps)    # 0 -> alpha/2
        c_r = alpha * (1 - step / (2 * total_steps))  # alpha -> alpha/2

    When gamma_kl=0 (default): paper exact. No margins, no triplet.
    When gamma_kl>0: adds KL on benign logits (attention_mask, NOT loss_mask).
    """
    # RR loss: push harmful cos_sim toward 0 (orthogonal to frozen)
    loss_rr, rr_metrics = reroute_loss_relu_cos(
        model_reps=harmful_model_reps,
        frozen_reps=harmful_frozen_reps,
        target_layers=target_layers,
        loss_mask=harmful_mask,
        return_metrics=True,
    )

    # Retain loss: keep benign reps close to frozen
    loss_retain = retain_loss_l2(
        model_reps=benign_model_reps,
        frozen_reps=benign_frozen_reps,
        target_layers=target_layers,
        loss_mask=benign_mask,
    )

    total = c_s * loss_rr + c_r * loss_retain

    # Optional KL term
    loss_kl_val = 0.0
    if gamma_kl > 0 and benign_student_logits is not None and benign_teacher_logits is not None:
        loss_kl = kl_divergence_loss(
            student_logits=benign_student_logits,
            teacher_logits=benign_teacher_logits,
            attention_mask=benign_attention_mask,
            loss_mask=None,  # NOT loss_mask — benign loss_mask is all-zero with injection_aware
            temperature=kl_temperature,
        )
        total = total + gamma_kl * loss_kl
        loss_kl_val = loss_kl.item()

    metrics = {
        "loss_rr": loss_rr.item(),
        "loss_retain": loss_retain.item(),
        "loss_kl": loss_kl_val,
        "c_s": c_s,
        "c_r": c_r,
        "gamma_kl": gamma_kl,
        "cos_sim_mean": rr_metrics["cos_sim_mean"] if rr_metrics else 0.0,
        "cos_sim_positive_frac": rr_metrics["cos_sim_positive_frac"] if rr_metrics else 0.0,
    }
    return total, metrics
