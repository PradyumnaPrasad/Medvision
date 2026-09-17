"""Central configuration loader for MEDVISION.

All tunables (paths, thresholds, model names, k-values) live in configs/default.yaml so that
no module hardcodes a magic number. Environment variables prefixed MEDVISION__ override nested
keys using double-underscore separators, e.g. MEDVISION__VISION__WEIGHTS=... overrides
vision.weights. This keeps local dev, CI and Docker consistent without editing YAML.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class PathsConfig(BaseModel):
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    kg_dir: str = "data/kg"
    chroma_dir: str = "data/chroma"
    models_dir: str = "data/models"
    uploads_dir: str = "data/uploads"
    kg_ontology_dir: str = "kg/ontology"
    labels_config: str = "configs/labels.yaml"

    def resolve(self, key: str) -> Path:
        return REPO_ROOT / getattr(self, key)


class DeviceConfig(BaseModel):
    prefer: list[str] = ["mps", "cuda", "cpu"]


class VisionConfig(BaseModel):
    classifier: str = "torchxrayvision"
    weights: str = "densenet121-res224-all"
    input_resolution: int = 224
    default_pos_threshold: float = 0.5
    default_uncertain_threshold: float = 0.3


class GradCamConfig(BaseModel):
    target_layer: str = "features.denseblock4"


class QcConfig(BaseModel):
    ood_similarity_threshold: float = 0.22
    min_resolution: int = 128


class KgReasoningConfig(BaseModel):
    top_k_differentials: int = 3
    contradiction_penalty_weight: float = 1.0
    activation_threshold: float = 0.3


class KgConfig(BaseModel):
    reasoning: KgReasoningConfig = KgReasoningConfig()


class RetrievalConfig(BaseModel):
    top_k_nodes: int = 5
    top_k_snippets: int = 5
    top_k_cases: int = 5


class LlmConfig(BaseModel):
    provider: str = "ollama"
    model: str = "llama3.1:8b"
    temperature: float = 0.1
    timeout_s: int = 60
    grounding_regenerate_threshold: float = 0.8


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 20
    allowed_extensions: list[str] = [".png", ".jpg", ".jpeg", ".dcm"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEDVISION__", env_nested_delimiter="__")

    paths: PathsConfig = PathsConfig()
    device: DeviceConfig = DeviceConfig()
    vision: VisionConfig = VisionConfig()
    gradcam: GradCamConfig = GradCamConfig()
    qc: QcConfig = QcConfig()
    kg: KgConfig = KgConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LlmConfig = LlmConfig()
    api: ApiConfig = ApiConfig()
    disclaimer: str = "Research prototype. Not for clinical use."


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings(config_path: str | Path = REPO_ROOT / "configs" / "default.yaml") -> Settings:
    """Load settings from YAML, then let environment variables override individual fields."""
    raw = _load_yaml(Path(config_path))
    return Settings(**raw)


@lru_cache(maxsize=1)
def get_labels_map() -> dict[str, str]:
    """Load the torchxrayvision-label -> KG-node-id mapping from configs/labels.yaml."""
    settings = get_settings()
    path = settings.paths.resolve("labels_config")
    raw = _load_yaml(path)
    return raw.get("label_to_node", {})


def get_device(prefer: list[str] | None = None) -> Literal["mps", "cuda", "cpu"]:
    """Pick the best available torch device, trying `prefer` order (default: mps, cuda, cpu)."""
    import torch

    order = prefer or get_settings().device.prefer
    for name in order:
        mps_backend = getattr(torch.backends, "mps", None)
        if name == "mps" and mps_backend is not None and mps_backend.is_available():
            return "mps"
        if name == "cuda" and torch.cuda.is_available():
            return "cuda"
        if name == "cpu":
            return "cpu"
    return "cpu"
