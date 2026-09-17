#!/usr/bin/env bash
# Downloads a small, real, labeled (Normal vs Pneumonia) chest X-ray set used to fit
# and evaluate temperature-scaling calibration for the classifier (PLAN.md T14).
#
# Source: keremberke/chest-xray-classification on Hugging Face — a parquet mirror of
# the well-known Kermany/Mooney "Chest X-Ray Images (Pneumonia)" dataset. We use two
# splits: `full/test` (582 images) to FIT the temperature, and `full/validation`
# (1165 images) to EVALUATE it, so we never report calibration improvement on the
# same data it was fit on.
#
# Usage: bash scripts/download_calibration_set.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="data/raw/calibration"
mkdir -p "$OUT_DIR"

BASE="https://huggingface.co/api/datasets/keremberke/chest-xray-classification/parquet/full"

echo "Downloading fit split (test, ~20MB)..."
curl -sL "$BASE/test/0.parquet" -o "$OUT_DIR/pneumonia_fit.parquet"

echo "Downloading eval split (validation, ~41MB)..."
curl -sL "$BASE/validation/0.parquet" -o "$OUT_DIR/pneumonia_eval.parquet"

echo "Done. Files in $OUT_DIR:"
ls -la "$OUT_DIR"
