"""DICOM loading with allow-list de-identification (PLAN.md 4.1 / T07).

De-identification works by *allow-listing*: a new dataset is built containing only the
technical elements needed to render the image, so every patient/study/institution/
device tag (including private tags) is dropped by default rather than needing to be
enumerated. Everything downstream (modality checks, pixel decoding, metadata) reads
from the scrubbed dataset, and the original dataset is never kept or persisted.
Only the rendered pixel data leaves this module. Tag values are never logged.

Limitation: text burned into the pixels cannot be removed; it is flagged instead.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.errors import InvalidDicomError
from pydicom.pixels import apply_modality_lut, apply_voi_lut

from medvision.config import get_settings

logger = logging.getLogger(__name__)

_ALLOWED_KEYWORDS = frozenset(
    {
        "SamplesPerPixel",
        "PhotometricInterpretation",
        "PlanarConfiguration",
        "Rows",
        "Columns",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
        "NumberOfFrames",
        "PixelData",
        "RescaleSlope",
        "RescaleIntercept",
        "WindowCenter",
        "WindowWidth",
        "VOILUTFunction",
        "Modality",
        "BodyPartExamined",
        "ViewPosition",
        "BurnedInAnnotation",
        "PresentationLUTShape",
    }
)
_LUT_SEQUENCES = ("VOILUTSequence", "ModalityLUTSequence")
_LUT_ITEM_KEYWORDS = ("LUTDescriptor", "LUTExplanation", "LUTData", "ModalityLUTType")


class DicomError(ValueError):
    """The upload is not a DICOM we can (or should) turn into a chest-X-ray image."""


@dataclass(frozen=True)
class DicomImage:
    image: Image.Image  # 8-bit grayscale, ready for the pipeline
    modality: str | None
    view_position: str | None
    body_part: str | None
    elements_removed: int  # count of top-level DICOM elements dropped by de-identification
    warnings: list[str] = field(default_factory=list)


def deidentify(ds: Dataset) -> tuple[Dataset, int]:
    """Return (scrubbed copy, number of top-level elements dropped). Allow-list based."""
    clean = Dataset()
    kept = 0
    for elem in ds:
        if elem.keyword in _ALLOWED_KEYWORDS:
            clean[elem.tag] = elem
            kept += 1
    for seq_name in _LUT_SEQUENCES:
        if seq_name not in ds:
            continue
        items = []
        for item in ds[seq_name].value:
            new_item = Dataset()
            for keyword in _LUT_ITEM_KEYWORDS:
                if keyword in item:
                    new_item[keyword] = item[keyword]
            items.append(new_item)
        clean[seq_name] = pydicom.DataElement(ds[seq_name].tag, "SQ", items)
        kept += 1

    clean.file_meta = FileMetaDataset()
    transfer_syntax = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
    if transfer_syntax is not None:
        clean.file_meta.TransferSyntaxUID = transfer_syntax
    return clean, len(ds) - kept


def _text(ds: Dataset, keyword: str) -> str | None:
    value = ds.get(keyword)
    return str(value).strip() if value not in (None, "") else None


def _to_uint8(pixels: np.ndarray, ds: Dataset) -> np.ndarray:
    cfg = get_settings().io.dicom
    arr = apply_modality_lut(pixels, ds).astype(np.float64)
    if "WindowCenter" in ds or "VOILUTSequence" in ds:
        arr = apply_voi_lut(arr, ds).astype(np.float64)
        lo, hi = float(arr.min()), float(arr.max())
    else:
        lo, hi = (float(v) for v in np.percentile(arr, cfg.fallback_percentiles))
    if hi <= lo:
        raise DicomError("The DICOM image is blank (no intensity variation).")
    return (np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0).round().astype(np.uint8)


def load_dicom(raw: bytes) -> DicomImage:
    """Decode DICOM bytes into a de-identified, display-ready grayscale image."""
    cfg = get_settings().io.dicom
    try:
        original = pydicom.dcmread(io.BytesIO(raw))
    except (InvalidDicomError, EOFError, ValueError) as exc:
        raise DicomError("Could not read this file as DICOM.") from exc

    ds, removed = deidentify(original)
    del original
    logger.info("DICOM de-identified: %d element(s) dropped", removed)

    modality = _text(ds, "Modality")
    view = _text(ds, "ViewPosition")
    body_part = _text(ds, "BodyPartExamined")
    warnings: list[str] = []

    if modality is None:
        warnings.append("DICOM has no Modality tag; could not confirm this is a radiograph.")
    elif modality.upper() not in cfg.allowed_modalities:
        raise DicomError(
            f"DICOM modality '{modality}' is not a plain radiograph "
            f"(accepted: {', '.join(cfg.allowed_modalities)})."
        )
    if view is not None and view.upper() in cfg.rejected_view_positions:
        raise DicomError(f"Lateral view ('{view}') is not supported; upload a frontal view.")
    if body_part is not None and not any(k in body_part.upper() for k in cfg.chest_body_parts):
        warnings.append(f"DICOM body part is '{body_part}', not chest; results may be meaningless.")
    if (_text(ds, "BurnedInAnnotation") or "").upper() == "YES":
        warnings.append(
            "DICOM flags burned-in annotations: identifying text may be visible in the pixels."
        )

    photometric = _text(ds, "PhotometricInterpretation")
    if photometric not in ("MONOCHROME1", "MONOCHROME2"):
        raise DicomError(f"Unsupported photometric interpretation '{photometric}'.")

    try:
        pixels = ds.pixel_array
    except Exception as exc:  # pydicom raises many types (missing handler, bad transfer syntax)
        raise DicomError(
            "Could not decode DICOM pixel data (unsupported compression or corrupt file)."
        ) from exc
    if pixels.ndim == 3:
        warnings.append(f"Multi-frame DICOM: analysed frame 1 of {pixels.shape[0]}.")
        pixels = pixels[0]

    gray = _to_uint8(pixels, ds)
    if photometric == "MONOCHROME1":
        gray = 255 - gray

    return DicomImage(
        image=Image.fromarray(gray, mode="L"),
        modality=modality,
        view_position=view,
        body_part=body_part,
        elements_removed=removed,
        warnings=warnings,
    )
