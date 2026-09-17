"""MEDVISION FastAPI application entrypoint.

uv run uvicorn backend.app.main:app --reload
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1 import analyze, health, kg
from backend.app.db import get_engine
from medvision.pipeline import get_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medvision")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading models (classifier, Grad-CAM, knowledge graph)...")
    get_models()  # warm the process-wide cache so the first request isn't slow
    get_engine()  # create the SQLite tables if they don't exist yet
    logger.info("MEDVISION backend ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="MEDVISION API",
    description="Knowledge-graph anchored vision-reasoning framework for explainable "
    "clinical diagnosis. Research prototype — not for clinical use.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(analyze.router, prefix="/api/v1")
app.include_router(kg.router, prefix="/api/v1")
