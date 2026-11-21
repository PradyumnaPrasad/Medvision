"""Chest X-ray pathology classifier.

Wraps the pretrained torchxrayvision DenseNet (18 pathologies) as MEDVISION's Stage-A
vision module (see PLAN.md section 4.4). Every output probability is keyed by a
knowledge-graph Finding node id (configs/labels.yaml), not the raw model label, so
downstream modules (Grad-CAM, KG reasoning, retrieval) never need to know about
torchxrayvision's naming.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
from PIL import Image

from medvision.config import get_device, get_labels_map, get_settings


@dataclass(frozen=True)
class ClassifierPrediction:
    """One pathology's raw model output, already mapped onto a KG node id."""

    node_id: str
    label: str
    prob: float


def _load_model_uncached(weights: str, device: str) -> xrv.models.DenseNet:
    model = xrv.models.DenseNet(weights=weights)
    model.eval()
    model.to(device)
    return model


@lru_cache(maxsize=1)
def _cached_model(weights: str, device: str) -> xrv.models.DenseNet:
    return _load_model_uncached(weights, device)


class ChestXrayClassifier:
    """Loads the torchxrayvision model once and exposes a simple predict(image) API.

    Instantiate once (e.g. at FastAPI startup) and reuse — model + weights are cached
    process-wide via an lru_cache so repeated construction is cheap.
    """

    def __init__(self, weights: str | None = None, device: str | None = None) -> None:
        settings = get_settings()
        self.weights = weights or settings.vision.weights
        self.device = device or get_device()
        self.resolution = settings.vision.input_resolution
        self.model = _cached_model(self.weights, self.device)
        self.labels_map = get_labels_map()
        self._transform = torchvision.transforms.Compose(
            [
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(self.resolution),
            ]
        )

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL image -> normalized single-channel tensor of shape (1, 1, H, W) on self.device."""
        arr = np.array(image.convert("L")).astype(np.float32)
        arr = xrv.datasets.normalize(arr, 255)  # 8-bit -> torchxrayvision's [-1024, 1024] range
        arr = arr[None, ...]  # add channel dim -> (1, H, W)
        arr = self._transform(arr)
        tensor = torch.from_numpy(arr).float().unsqueeze(0)  # (1, 1, H, W)
        return tensor.to(self.device)

    @torch.no_grad()
    def _forward(self, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """One forward pass -> (raw outputs (B, n_pathologies), pooled features (B, 1024)).

        The pooled features are the exact input of the model's linear head, captured with a
        pre-hook, so the embedding is by construction what the classifier itself "saw".
        """
        captured: list[torch.Tensor] = []
        handle = self.model.classifier.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach())
        )
        try:
            outputs = self.model(batch).detach().cpu().numpy()
        finally:
            handle.remove()
        return outputs, captured[0].cpu().numpy()

    def _to_predictions(self, outputs: np.ndarray) -> list[ClassifierPrediction]:
        predictions: list[ClassifierPrediction] = []
        for label, prob in zip(self.model.pathologies, outputs, strict=True):
            node_id = self.labels_map.get(label)
            if node_id is None:
                continue  # unmapped / unsupported-by-this-checkpoint pathology
            if np.isnan(prob):
                continue  # this checkpoint doesn't supervise this label
            predictions.append(ClassifierPrediction(node_id=node_id, label=label, prob=float(prob)))
        return predictions

    def predict(self, image: Image.Image) -> list[ClassifierPrediction]:
        """Run inference and return one ClassifierPrediction per known (mapped) pathology."""
        return self.predict_with_embedding(image)[0]

    def predict_with_embedding(
        self, image: Image.Image
    ) -> tuple[list[ClassifierPrediction], np.ndarray]:
        """Like predict(), plus the (1024,) pooled feature vector used for retrieval."""
        outputs, features = self._forward(self.preprocess(image))
        return self._to_predictions(outputs[0]), features[0]

    def embed_batch(self, images: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        """Batch version for dataset preprocessing: (features (B, 1024), raw outputs (B, P))."""
        batch = torch.cat([self.preprocess(image) for image in images])
        outputs, features = self._forward(batch)
        return features, outputs
