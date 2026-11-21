"""Evaluation of the alignment against honest baselines (PLAN.md section 6).

Two tasks, each compared with simpler alternatives that use the same frozen features:
  * Image -> KG node: per-finding AUROC and hit@k of the aligned model vs (a) the
    classifier's own outputs and (b) a logistic-regression probe trained on the same split.
  * Image -> similar cases: precision@k of retrieval in the aligned space vs (a) cosine on the
    raw features and (b) a random ranking. A retrieved case is relevant when it shares at
    least one finding with the query (a normal image's relevant cases are other normals).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from medvision.align.data import NO_FINDING_NODE, FeatureSet
from medvision.align.model import AlignmentModel


@torch.no_grad()
def embed_images(model: AlignmentModel, feats: np.ndarray) -> np.ndarray:
    device = model.logit_scale.device
    return model.image_embeddings(torch.tensor(feats, device=device)).cpu().numpy()


@torch.no_grad()
def node_scores(model: AlignmentModel, feats: np.ndarray) -> np.ndarray:
    """(N, F) cosine similarity of each image to each KG finding node."""
    finding = model.finding_embeddings().cpu().numpy()
    return embed_images(model, feats) @ finding.T


def per_finding_auroc(
    scores: np.ndarray, masks: np.ndarray, finding_ids: list[str], min_positives: int
) -> dict[str, float]:
    out: dict[str, float] = {}
    for j, node in enumerate(finding_ids):
        positives = int(masks[:, j].sum())
        if positives < min_positives or len(masks) - positives < min_positives:
            continue
        out[node] = float(roc_auc_score(masks[:, j], scores[:, j]))
    return out


def hit_at_k(scores: np.ndarray, masks: np.ndarray, ks: tuple[int, ...]) -> dict[int, float]:
    """Share of images whose top-k nodes contain at least one true finding node."""
    ranked = np.argsort(-scores, axis=1)
    out = {}
    for k in ks:
        top = ranked[:, :k]
        out[k] = float(np.take_along_axis(masks, top, axis=1).max(axis=1).mean())
    return out


def classifier_scores(data: FeatureSet, labels_map: dict[str, str]) -> np.ndarray:
    """The pretrained classifier's own outputs, laid out over the finding columns (NaN when the
    classifier has no output for that finding, e.g. F_NO_FINDING)."""
    column = {node: i for i, node in enumerate(data.finding_ids)}
    scores = np.full((len(data), len(data.finding_ids)), np.nan, dtype=np.float32)
    for p, pathology in enumerate(data.xrv_pathologies):
        node = labels_map.get(pathology)
        if node in column:
            scores[:, column[node]] = data.xrv_outputs[:, p]
    return scores


def probe_scores(train: FeatureSet, test: FeatureSet) -> np.ndarray:
    """One logistic regression per finding on the raw features: the usual linear-probe baseline."""
    mean, std = train.feats.mean(0), train.feats.std(0) + 1e-6
    x_train, x_test = (train.feats - mean) / std, (test.feats - mean) / std
    scores = np.full((len(test), len(train.finding_ids)), np.nan, dtype=np.float32)
    for j in range(len(train.finding_ids)):
        y = train.masks[:, j]
        if y.min() == y.max():
            continue
        clf = LogisticRegression(C=0.05, max_iter=300).fit(x_train, y)
        scores[:, j] = clf.decision_function(x_test)
    return scores


def _precision_at_k(
    sim: np.ndarray, query_masks: np.ndarray, index_masks: np.ndarray, ks: tuple[int, ...]
) -> dict[int, float]:
    ranked = np.argsort(-sim, axis=1)
    shares = (query_masks @ index_masks.T) > 0  # (Q, I): shares at least one finding
    out = {}
    for k in ks:
        top = np.take_along_axis(shares, ranked[:, :k], axis=1)
        out[k] = float(top.mean())
    return out


def case_retrieval(
    query_emb: np.ndarray,
    query: FeatureSet,
    index_emb: np.ndarray,
    index: FeatureSet,
    ks: tuple[int, ...],
) -> dict[str, dict[int, float]]:
    """Precision@k for one embedding space, over all queries and over abnormal queries only
    (queries with at least one finding other than No Finding)."""
    no_finding = query.finding_ids.index(NO_FINDING_NODE)
    abnormal_q = query.masks.copy()
    abnormal_q[:, no_finding] = 0
    abnormal_i = index.masks.copy()
    abnormal_i[:, no_finding] = 0
    has_abnormal = abnormal_q.sum(1) > 0

    sim = query_emb @ index_emb.T
    result = {"all": _precision_at_k(sim, query.masks, index.masks, ks)}
    result["abnormal"] = _precision_at_k(
        sim[has_abnormal], abnormal_q[has_abnormal], abnormal_i, ks
    )
    return result


def random_retrieval(
    query: FeatureSet, index: FeatureSet, ks: tuple[int, ...], seed: int
) -> dict[str, dict[int, float]]:
    rng = np.random.default_rng(seed)
    noise = rng.random((len(query), len(index)))
    no_finding = query.finding_ids.index(NO_FINDING_NODE)
    abnormal_q = query.masks.copy()
    abnormal_q[:, no_finding] = 0
    abnormal_i = index.masks.copy()
    abnormal_i[:, no_finding] = 0
    has_abnormal = abnormal_q.sum(1) > 0
    return {
        "all": _precision_at_k(noise, query.masks, index.masks, ks),
        "abnormal": _precision_at_k(noise[has_abnormal], abnormal_q[has_abnormal], abnormal_i, ks),
    }


def l2_normalise(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
