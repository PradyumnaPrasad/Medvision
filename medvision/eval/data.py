"""Loading the small real labeled chest X-ray set used for both calibration (T14)
and evaluation (T31) — see scripts/download_calibration_set.sh for where it comes
from. Shared here so scripts/fit_calibration.py and scripts/run_eval.py don't each
reimplement parquet decoding.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def load_pneumonia_split(path: Path) -> tuple[list[Image.Image], np.ndarray]:
    """Returns (images, labels) where labels are 0.0=Normal, 1.0=Pneumonia."""
    table = pq.read_table(path, columns=["image", "labels"])
    df = table.to_pandas()
    images = [Image.open(io.BytesIO(row["bytes"])).convert("RGB") for row in df["image"]]
    labels = df["labels"].to_numpy().astype(np.float64)
    return images, labels
