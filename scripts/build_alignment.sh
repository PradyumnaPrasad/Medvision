#!/usr/bin/env bash
# Rebuilds everything the image <-> knowledge-graph alignment needs, from nothing:
#   1. download the NIH ChestX-ray14 sample (~450 MB, resumable)
#   2. cache frozen DenseNet features for it
#   3. train + evaluate the alignment (writes data/models/alignment.pt and reports/alignment_eval.*)
#   4. build the ChromaDB index used for similar-case retrieval
# Without these artifacts MEDVISION still works; it just shows no similar-case panel.
#
# Usage: bash scripts/build_alignment.sh
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python scripts/download_nih_sample.py
uv run python scripts/extract_features.py
uv run python scripts/train_alignment.py
uv run python scripts/index_chroma.py
echo "Alignment artifacts built. Restart the backend to load them."
