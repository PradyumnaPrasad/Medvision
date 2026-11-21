"""Multi-positive InfoNCE (PLAN.md T18).

Plain InfoNCE assumes exactly one positive per row. Chest X-rays break that: in a batch many
images share the same finding set (most are simply "No Finding"), so treating the other
copies as negatives would push apart images that mean the same thing. Here every column
whose label set equals the row's is a positive, and the loss is -log of the positives' share
of the softmax mass.
"""

from __future__ import annotations

import torch


def multi_positive_nce(logits: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    """logits (B, C); positives (B, C) bool with at least one True per row."""
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_mass = torch.logsumexp(log_prob.masked_fill(~positives, float("-inf")), dim=1)
    return -positive_log_mass.mean()


def same_label_set(masks: torch.Tensor) -> torch.Tensor:
    """(B, B) bool: True where two rows have identical label sets (diagonal is always True)."""
    return (masks[:, None, :] == masks[None, :, :]).all(dim=-1)


def symmetric_set_loss(
    image_emb: torch.Tensor, set_emb: torch.Tensor, masks: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Image -> label-set and label-set -> image, with duplicate sets counted as positives."""
    logits = scale * image_emb @ set_emb.T
    positives = same_label_set(masks)
    return 0.5 * (multi_positive_nce(logits, positives) + multi_positive_nce(logits.T, positives))


def node_loss(
    image_emb: torch.Tensor,
    finding_emb: torch.Tensor,
    masks: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Image -> KG finding nodes: each image must rank its true nodes above the others."""
    logits = scale * image_emb @ finding_emb.T
    return multi_positive_nce(logits, masks > 0)
