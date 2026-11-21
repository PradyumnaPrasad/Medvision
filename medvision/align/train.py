"""Trains the image <-> knowledge-graph alignment on cached features (PLAN.md T18).

Loss = symmetric multi-positive InfoNCE between each image and the mean embedding of its true
finding nodes, plus a node-level InfoNCE term (image -> individual finding nodes) so retrieval
of single KG nodes works too. The best epoch on validation loss is kept (early stopping).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from medvision.align.data import FeatureSet
from medvision.align.graph_encoder import GraphTensors
from medvision.align.loss import node_loss, symmetric_set_loss
from medvision.align.model import AlignmentModel
from medvision.config import AlignConfig, get_settings

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    model: AlignmentModel
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0


def _losses(
    model: AlignmentModel, feats: torch.Tensor, masks: torch.Tensor, cfg: AlignConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    image_emb = model.image_embeddings(feats)
    set_loss = symmetric_set_loss(image_emb, model.label_set_embeddings(masks), masks, model.scale)
    nodes_loss = node_loss(image_emb, model.finding_embeddings(), masks, model.scale)
    return set_loss, nodes_loss


def train_alignment(
    tensors: GraphTensors,
    train: FeatureSet,
    val: FeatureSet,
    cfg: AlignConfig | None = None,
    device: str = "cpu",
) -> TrainResult:
    cfg = cfg or get_settings().align
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = AlignmentModel(tensors, cfg).to(device)
    model.set_findings(train.finding_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    x_train = torch.tensor(train.feats, device=device)
    m_train = torch.tensor(train.masks, device=device)
    x_val = torch.tensor(val.feats, device=device)
    m_val = torch.tensor(val.masks, device=device)

    best_state, best_val, best_epoch, stale = None, float("inf"), 0, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = rng.permutation(len(x_train))
        running, batches = 0.0, 0
        for start in range(0, len(order), cfg.batch_size):
            rows = torch.tensor(order[start : start + cfg.batch_size], device=device)
            if len(rows) < 2:
                continue
            set_loss, nodes_loss = _losses(model, x_train[rows], m_train[rows], cfg)
            loss = set_loss + cfg.node_loss_weight * nodes_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.detach().item()
            batches += 1

        model.eval()
        with torch.no_grad():
            val_set, val_nodes = _losses(model, x_val, m_val, cfg)
            val_total = float(val_set + cfg.node_loss_weight * val_nodes)
        history.append(
            {"epoch": epoch, "train_loss": running / max(batches, 1), "val_loss": val_total}
        )
        if val_total < best_val - 1e-5:
            best_val, best_epoch, stale = val_total, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg.patience:
                logger.info("Early stop at epoch %d (best %d)", epoch, best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model=model, history=history, best_epoch=best_epoch)
