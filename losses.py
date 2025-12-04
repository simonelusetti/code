import math
from typing import Callable, Optional

import torch
from torch import Tensor, nn, functional as F


def entropy_loss(
    assignments: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    attn = attention_mask.float()
    tokens = attn.sum()
    # Avoid log(0) -> -inf -> NaN by clamping probabilities.
    clamped = assignments.clamp_min(1e-8)
    entropy = -(clamped * clamped.log()).sum(dim=-1)  # [B, T]
    entropy = (entropy * attn).sum() / tokens
    return entropy  # minimize to encourage confident assignments


def balance_loss(
    global_mass: torch.Tensor,
    token_count: torch.Tensor,
    num_buckets: int,
) -> torch.Tensor:
    eps = 1e-8
    p = (global_mass / token_count).clamp_min(eps)
    log_uniform = math.log(1.0 / num_buckets)
    return (p * (p.log() - log_uniform)).sum()


def recon_loss(
    gates: torch.Tensor,
    attn: torch.Tensor,
    token_embeddings: torch.Tensor,
    sent_repr: torch.Tensor,
    aggregate_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    agg = aggregate_fn or (lambda x: x.sum(dim=1))
    weights = gates * attn.unsqueeze(-1)  # [B, T, K]
    num = torch.einsum("btk,btd->bkd", weights, token_embeddings)
    denom = weights.sum(dim=1).clamp_min(1e-6)
    subsent_repr = num / denom.unsqueeze(-1)
    target_repr = agg(subsent_repr)

    sent_norm = sent_repr.norm(dim=-1).clamp_min(1e-8)
    target_norm = target_repr.norm(dim=-1).clamp_min(1e-8)
    cos_sim = (sent_repr * target_repr).sum(dim=-1) / (sent_norm * target_norm)
    return 1.0 - cos_sim.mean()
