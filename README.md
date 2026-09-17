# MEDVISION

Knowledge-graph anchored vision-reasoning framework for explainable clinical diagnosis (chest X-ray).
**Research prototype for a college major project. Not a certified medical device. Not for clinical use.**

See [PLAN.md](PLAN.md) for the full design and ticket backlog, and [CLAUDE.md](CLAUDE.md) for coding conventions.

## Setup

```bash
uv sync
uv run pytest -q
```

## Run

```bash
uv run uvicorn backend.app.main:app --reload
```
