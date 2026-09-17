from fastapi import APIRouter

from backend.app.schemas import HealthOut
from medvision.config import get_device, get_settings
from medvision.pipeline import get_models

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    settings = get_settings()
    models = get_models()
    return HealthOut(
        status="ok",
        device=get_device(),
        kg_nodes=len(models.graph),
        kg_edges=len(models.graph.edges),
        disclaimer=settings.disclaimer,
    )
