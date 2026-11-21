"""One-off preprocessing: run the frozen DenseNet over the NIH sample and cache the pooled
features, labels and raw classifier outputs (PLAN.md T16). Alignment training and
evaluation then run on the cache in seconds.

Needs the images from `uv run python scripts/download_nih_sample.py`.
Usage: uv run python scripts/extract_features.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from medvision.config import REPO_ROOT, get_settings
from medvision.vision.classifier import ChestXrayClassifier

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("extract_features")

SPLITS = ("train", "validation", "test")
BATCH = 32


def main(nih_dir: Path | None = None, out_path: Path | None = None) -> None:
    settings = get_settings()
    nih_dir = nih_dir or REPO_ROOT / settings.align.nih_dir
    out_path = out_path or settings.paths.resolve("processed_dir") / settings.align.features_file
    names_path = nih_dir / "label_names.json"
    if not names_path.exists():
        raise SystemExit(f"Missing {names_path}; run scripts/download_nih_sample.py first.")
    label_names = json.loads(names_path.read_text())
    classifier = ChestXrayClassifier()

    feats, outputs, labels, splits, case_ids = [], [], [], [], []
    for split in SPLITS:
        records = json.loads((nih_dir / f"{split}.json").read_text())
        missing = [
            r["idx"] for r in records if not (nih_dir / split / f"{r['idx']:05d}.jpg").exists()
        ]
        if missing:
            raise SystemExit(f"{len(missing)} {split} images are missing; re-run the download.")
        for start in range(0, len(records), BATCH):
            chunk = records[start : start + BATCH]
            images = [
                Image.open(nih_dir / split / f"{r['idx']:05d}.jpg").convert("L") for r in chunk
            ]
            f, o = classifier.embed_batch(images)
            feats.append(f)
            outputs.append(o)
            for r in chunk:
                hot = np.zeros(len(label_names), dtype=np.uint8)
                hot[list(r["labels"])] = 1
                labels.append(hot)
                splits.append(split)
                case_ids.append(f"nih-{split}-{r['idx']}")
            if (start // BATCH) % 20 == 0:
                logger.info("%s: %d/%d", split, start + len(chunk), len(records))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        feats=np.concatenate(feats).astype(np.float32),
        xrv_outputs=np.concatenate(outputs).astype(np.float32),
        labels=np.stack(labels),
        label_names=np.array(label_names),
        xrv_pathologies=np.array(classifier.model.pathologies),
        split=np.array(splits),
        case_ids=np.array(case_ids),
    )
    logger.info("Wrote %d images to %s", len(splits), out_path)


if __name__ == "__main__":
    main()
