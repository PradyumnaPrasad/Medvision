"""Loading cached image features + labels for alignment training and evaluation.

`scripts/extract_features.py` runs the frozen DenseNet once over the NIH sample and caches
the pooled features, so training and evaluation afterwards take seconds. Labels are NIH
ChestX-ray14 names; they are mapped to KG Finding node ids through configs/labels.yaml
(plus "No Finding" -> F_NO_FINDING) so every training target is a graph node.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

NO_FINDING_LABEL = "No Finding"
NO_FINDING_NODE = "F_NO_FINDING"


@dataclass(frozen=True)
class FeatureSet:
    feats: np.ndarray  # (N, D) float32 pooled DenseNet features
    masks: np.ndarray  # (N, F) float32 over the KG finding nodes, in `finding_ids` order
    finding_ids: list[str]
    case_ids: list[str]
    label_names: list[list[str]]  # the original NIH label names per image, for display
    xrv_outputs: np.ndarray  # (N, P) raw classifier outputs, for baselines
    xrv_pathologies: list[str]

    def __len__(self) -> int:
        return len(self.feats)

    def subset(self, rows: np.ndarray) -> FeatureSet:
        return FeatureSet(
            feats=self.feats[rows],
            masks=self.masks[rows],
            finding_ids=self.finding_ids,
            case_ids=[self.case_ids[i] for i in rows],
            label_names=[self.label_names[i] for i in rows],
            xrv_outputs=self.xrv_outputs[rows],
            xrv_pathologies=self.xrv_pathologies,
        )


def label_to_node(labels_map: dict[str, str]) -> dict[str, str]:
    return {**labels_map, NO_FINDING_LABEL: NO_FINDING_NODE}


def load_feature_sets(
    path: Path, finding_ids: list[str], labels_map: dict[str, str]
) -> dict[str, FeatureSet]:
    """Return {"train": ..., "validation": ..., "test": ...}. Rows with no mappable label
    are dropped (they would have no positive node to align to)."""
    raw = np.load(path, allow_pickle=False)
    names = [str(n) for n in raw["label_names"]]
    mapping = label_to_node(labels_map)
    column = {node: i for i, node in enumerate(finding_ids)}
    label_columns = [
        (i, column[mapping[n]]) for i, n in enumerate(names) if mapping.get(n) in column
    ]

    multi_hot = raw["labels"]
    masks = np.zeros((len(multi_hot), len(finding_ids)), dtype=np.float32)
    for src, dst in label_columns:
        masks[:, dst] = np.maximum(masks[:, dst], multi_hot[:, src])

    splits = raw["split"].astype(str)
    case_ids = [str(c) for c in raw["case_ids"]]
    out: dict[str, FeatureSet] = {}
    for split in ("train", "validation", "test"):
        rows = np.where((splits == split) & (masks.sum(axis=1) > 0))[0]
        if len(rows) == 0:
            continue
        out[split] = FeatureSet(
            feats=raw["feats"][rows].astype(np.float32),
            masks=masks[rows],
            finding_ids=finding_ids,
            case_ids=[case_ids[i] for i in rows],
            label_names=[[names[j] for j in np.where(multi_hot[i])[0]] for i in rows],
            xrv_outputs=raw["xrv_outputs"][rows].astype(np.float32),
            xrv_pathologies=[str(p) for p in raw["xrv_pathologies"]],
        )
    return out
