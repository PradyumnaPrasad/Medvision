"""Downloads a multi-label NIH ChestX-ray14 sample (7,335 images, 15 labels) used to train and
evaluate the image <-> knowledge-graph alignment (PLAN.md T16-T18).

Source: Sohaibsoussi/NIH-Chest-X-ray-dataset-small on Hugging Face (public, ungated), read
through the datasets-server rows API. That serves each image as a ~60 KB JPEG, about six
times smaller than the dataset's parquet shards, so the whole sample is ~450 MB. The JPEG
re-encoding is a small distribution difference from the original PNGs; the evaluation report
notes it.

Idempotent and resumable: existing images are skipped. Standard library only.
Usage: uv run python scripts/download_nih_sample.py [split ...]
(Giving splits lets several copies run at once, e.g. one per split.)
"""

from __future__ import annotations

import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASET = "Sohaibsoussi/NIH-Chest-X-ray-dataset-small"
API = "https://datasets-server.huggingface.co"
SPLITS = ("train", "validation", "test")
PAGE = 100
WORKERS = 12
RETRIES = 6
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "nih"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("download_nih_sample")


def _get(url: str, timeout: int = 60) -> bytes:
    delay = 2.0
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == RETRIES:
                raise RuntimeError(f"Giving up on {url}: {exc}") from exc
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def _rows(split: str, offset: int) -> dict:
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": split, "offset": offset, "length": PAGE}
    )
    return json.loads(_get(f"{API}/rows?{query}"))


def _fetch_image(args: tuple[str, Path]) -> None:
    url, dest = args
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(_get(url))
    tmp.rename(dest)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names: list[str] | None = None
    for split in sys.argv[1:] or SPLITS:
        split_dir = OUT_DIR / split
        split_dir.mkdir(exist_ok=True)
        first = _rows(split, 0)
        total = first["num_rows_total"]
        if names is None:
            label_feature = next(f for f in first["features"] if f["name"] == "labels")
            names = label_feature["type"]["feature"]["names"]
            (OUT_DIR / "label_names.json").write_text(json.dumps(names))

        records, jobs = [], []
        offset = 0
        page = first
        while offset < total:
            if offset:
                page = _rows(split, offset)
            for item in page["rows"]:
                idx = item["row_idx"]
                dest = split_dir / f"{idx:05d}.jpg"
                records.append({"idx": idx, "labels": item["row"]["labels"]})
                jobs.append((item["row"]["image"]["src"], dest))
            offset += PAGE
        (OUT_DIR / f"{split}.json").write_text(json.dumps(records))

        done = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for _ in pool.map(_fetch_image, jobs):
                done += 1
                if done % 250 == 0 or done == len(jobs):
                    logger.info("%s: %d/%d images", split, done, len(jobs))
    logger.info("Done. Files in %s", OUT_DIR)


if __name__ == "__main__":
    main()
