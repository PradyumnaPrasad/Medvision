"""Grad-CAM explanations for the chest X-ray classifier (PLAN.md section 4.4).

For a given finding (KG node id) that the classifier considers present, this module
produces a heatmap over the input image showing which pixels drove that prediction,
and an RGB overlay PNG suitable for the UI. This is the visual half of "explainable" —
the KG path (medvision/kg/reasoning.py) is the textual half.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from medvision.vision.classifier import ChestXrayClassifier


def _resolve_layer(model: torch.nn.Module, dotted_path: str) -> torch.nn.Module:
    """Resolve e.g. 'features.norm5' into model.features.norm5."""
    layer = model
    for part in dotted_path.split("."):
        layer = getattr(layer, part)
    return layer


class GradCamExplainer:
    """Wraps pytorch-grad-cam around a ChestXrayClassifier's DenseNet backbone.

    Grad-CAM needs gradients, and MPS backward support for DenseNet's grouped conv
    layers is inconsistent across torch versions, so this explainer always runs on
    CPU regardless of which device the classifier uses for plain inference. A single
    Grad-CAM pass on one image is fast enough on CPU that this costs nothing visible.
    """

    def __init__(self, classifier: ChestXrayClassifier, target_layer: str | None = None) -> None:
        from medvision.config import get_settings

        settings = get_settings()
        self.classifier = classifier
        self.pathologies = list(classifier.model.pathologies)
        self.node_to_index = {
            classifier.labels_map[label]: idx
            for idx, label in enumerate(self.pathologies)
            if label in classifier.labels_map
        }

        # Run Grad-CAM on a CPU copy of the model to sidestep MPS backward-pass gaps.
        self._cam_model = classifier.model.to("cpu")
        layer_path = target_layer or settings.gradcam.target_layer
        target_layers = [_resolve_layer(self._cam_model, layer_path)]
        self._cam = GradCAM(model=self._cam_model, target_layers=target_layers)

    def heatmap(self, image: Image.Image, node_id: str) -> np.ndarray:
        """Return a (H, W) float array in [0, 1], H/W matching the classifier's input resolution."""
        if node_id not in self.node_to_index:
            raise KeyError(f"No classifier output is mapped to KG node '{node_id}'")
        index = self.node_to_index[node_id]

        tensor = self.classifier.preprocess(image).to("cpu")
        targets = [ClassifierOutputTarget(index)]
        grayscale_cam = self._cam(input_tensor=tensor, targets=targets)
        return grayscale_cam[0]  # drop batch dim -> (H, W)

    def overlay(self, image: Image.Image, node_id: str, alpha: float = 0.5) -> Image.Image:
        """Return an RGB PIL image: the (resized) input with a jet-colormap CAM overlay."""
        cam = self.heatmap(image, node_id)
        resolution = cam.shape[0]

        base = image.convert("L").resize((resolution, resolution))
        base_rgb = np.stack([np.array(base)] * 3, axis=-1).astype(np.float32) / 255.0

        blended = show_cam_on_image(base_rgb, cam, use_rgb=True, image_weight=1 - alpha)
        return Image.fromarray(blended)
