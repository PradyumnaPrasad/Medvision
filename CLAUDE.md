# MEDVISION — coding guide

Explainable chest X-ray diagnosis assistant: vision + knowledge graph + retrieval + local LLM. **Research prototype, not for clinical use.**
The full design and ticket backlog are in `PLAN.md`. Implement **one ticket at a time** and meet its acceptance criteria.

## Layout
- `medvision/` holds the ML core (pure Python, no FastAPI imports). `backend/app/` is the API. `frontend/` is React + Vite + TS + Tailwind.
- `kg/ontology/*.yaml` is the source of truth for the knowledge graph. Never hand-edit the generated `data/kg/*`.
- `configs/default.yaml` holds all thresholds, model names, paths and k values. **No magic numbers in code.**
- `data/` is gitignored. Tests use only `tests/fixtures/`.

## Commands
- Setup: `uv sync`; frontend: `cd frontend && npm install`
- Tests: `uv run pytest -q`; lint: `uv run ruff check . && uv run ruff format --check .`
- API: `uv run uvicorn backend.app.main:app --reload`; UI: `cd frontend && npm run dev`
- KG: `uv run python scripts/build_kg.py` (runs the validator first)

## Conventions
- Python 3.11, full type hints, Pydantic v2 models at module boundaries, `pathlib`, `logging` (no prints).
- Pick the device via `medvision.config.get_device()` (mps → cuda → cpu). The GNN runs on CPU.
- Load models once (FastAPI lifespan). CPU/GPU work runs in a threadpool, never inside the event loop.
- Every classifier output, finding and differential is keyed by a **KG node id** (e.g. `F_EFFUSION`, `D_PNEUMONIA`, `A_LEFT_LOWER_ZONE`).
- Diagnosis ranking is deterministic (`medvision/kg/reasoning.py`). The LLM only narrates the evidence and must cite evidence IDs (`F*`, `D*`, `K*`, `S*`, `C*`).
- The LLM is local (Ollama) and set in config. The code must still work when the LLM is down (template fallback).
- Tests: mock the LLM and heavy model downloads. Keep unit tests fast (under 30 s total). Mark slow tests `@pytest.mark.slow`.
- De-identify DICOM before persisting it. Never log image bytes or PHI.
- Don't add dependencies without a reason. Don't commit data, weights or `.env`.
- Keep the UI disclaimer banner on every page.
