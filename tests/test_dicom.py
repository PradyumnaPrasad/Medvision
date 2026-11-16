import io
import logging

import numpy as np
import pydicom
import pytest
from pydicom.data import get_testdata_file
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from medvision.io import DicomError, deidentify, load_dicom

PHI = {
    "PatientName": "Doe^Jane",
    "PatientID": "MRN-99887766",
    "PatientBirthDate": "19700101",
    "InstitutionName": "Springfield General Hospital",
    "ReferringPhysicianName": "Smith^Greg",
    "AccessionNumber": "ACC554433",
    "StudyDescription": "Chest PA pt Doe",
    "OtherPatientIDs": "ALT-12345",
}


def make_dicom(
    pixels: np.ndarray | None = None,
    *,
    modality: str = "DX",
    photometric: str = "MONOCHROME2",
    view: str | None = "PA",
    body_part: str | None = "CHEST",
    with_phi: bool = True,
    **extra,
) -> bytes:
    """Build a synthetic uncompressed DICOM (no real patient data) and return its bytes."""
    if pixels is None:
        pixels = np.tile(np.linspace(100, 3000, 64, dtype=np.uint16), (64, 1))
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Modality = modality
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    if view:
        ds.ViewPosition = view
    if body_part:
        ds.BodyPartExamined = body_part
    if with_phi:
        for key, value in PHI.items():
            setattr(ds, key, value)
        ds.add_new((0x0009, 0x0010), "LO", "PRIVATE-CREATOR-Springfield")
        ds.add_new((0x0009, 0x1001), "LO", "private-secret-tag")
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.Rows, ds.Columns = pixels.shape[-2:]
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
    ds.PixelRepresentation = 0
    if pixels.ndim == 3:
        ds.NumberOfFrames = pixels.shape[0]
    for key, value in extra.items():
        setattr(ds, key, value)
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf)
    return buf.getvalue()


def _read(raw: bytes):
    return pydicom.dcmread(io.BytesIO(raw))


def test_deidentify_drops_all_phi_including_private_tags():
    scrubbed, removed = deidentify(_read(make_dicom()))
    for keyword in PHI:
        assert keyword not in scrubbed
    assert (0x0009, 0x1001) not in scrubbed
    assert (0x0009, 0x0010) not in scrubbed
    assert "StudyInstanceUID" not in scrubbed
    assert "SOPInstanceUID" not in scrubbed
    assert removed >= len(PHI) + 2
    assert scrubbed.Modality == "DX"
    assert "PixelData" in scrubbed


def test_scrubbed_dataset_bytes_contain_no_phi_strings():
    raw = make_dicom()
    assert all(value.encode() in raw for value in PHI.values())  # the input really has PHI
    scrubbed, _ = deidentify(_read(raw))
    buf = io.BytesIO()
    scrubbed.save_as(buf, enforce_file_format=False)
    blob = buf.getvalue()
    for value in [*PHI.values(), "private-secret-tag", "Springfield"]:
        assert value.encode() not in blob


def test_deidentify_real_pydicom_sample_removes_patient_name():
    path = get_testdata_file("CT_small.dcm")
    original = pydicom.dcmread(path)
    assert str(original.PatientName)  # sanity: the sample really has a name
    scrubbed, _ = deidentify(original)
    assert "PatientName" not in scrubbed
    assert "PatientID" not in scrubbed


def test_deidentify_keeps_only_lut_data_inside_sequences():
    ds = _read(make_dicom())
    item = pydicom.Dataset()
    item.LUTDescriptor = [4, 0, 16]
    item.LUTData = [0, 1, 2, 3]
    item.PatientName = "Leak^Inside^Sequence"
    ds.VOILUTSequence = [item]
    scrubbed, _ = deidentify(ds)
    kept = scrubbed.VOILUTSequence[0]
    assert "LUTData" in kept
    assert "PatientName" not in kept


def test_load_monochrome2_preserves_intensity_order():
    result = load_dicom(make_dicom())
    arr = np.asarray(result.image)
    assert result.image.mode == "L"
    assert arr.shape == (64, 64)
    assert arr[:, 0].mean() < arr[:, -1].mean()
    assert arr.min() == 0 and arr.max() == 255


def test_load_monochrome1_is_inverted():
    mono2 = np.asarray(load_dicom(make_dicom()).image).astype(int)
    mono1 = np.asarray(load_dicom(make_dicom(photometric="MONOCHROME1")).image).astype(int)
    assert np.abs((255 - mono2) - mono1).max() <= 1
    assert mono1[:, 0].mean() > mono1[:, -1].mean()


def test_voi_window_changes_contrast():
    base = make_dicom()
    windowed = make_dicom(WindowCenter=1000, WindowWidth=400)
    a = np.asarray(load_dicom(base).image).astype(int)
    b = np.asarray(load_dicom(windowed).image).astype(int)
    assert (b == 0).mean() > (a == 0).mean()
    assert (b == 255).mean() > (a == 255).mean()


def test_rescale_slope_and_intercept_are_applied():
    pixels = np.full((16, 16), 500, dtype=np.uint16)
    pixels[:, 8:] = 1500
    result = load_dicom(make_dicom(pixels, RescaleSlope=2, RescaleIntercept=-1000))
    arr = np.asarray(result.image)
    assert arr[:, :8].max() < arr[:, 8:].min()


def test_result_exposes_no_phi():
    result = load_dicom(make_dicom())
    exposed = repr(result.__dict__)
    for value in PHI.values():
        assert value not in exposed
    assert result.modality == "DX" and result.view_position == "PA"
    assert result.elements_removed > 0


def test_loading_does_not_log_phi(caplog):
    with caplog.at_level(logging.DEBUG):
        load_dicom(make_dicom())
    assert caplog.text
    for value in PHI.values():
        assert value not in caplog.text


@pytest.mark.parametrize("modality", ["CT", "MR", "US"])
def test_rejects_non_radiograph_modality(modality):
    with pytest.raises(DicomError, match="modality"):
        load_dicom(make_dicom(modality=modality))


def test_rejects_real_ct_sample():
    with open(get_testdata_file("CT_small.dcm"), "rb") as f:
        raw = f.read()
    with pytest.raises(DicomError, match="modality"):
        load_dicom(raw)


@pytest.mark.parametrize("view", ["LL", "LAT", "RL"])
def test_rejects_lateral_view(view):
    with pytest.raises(DicomError, match="Lateral"):
        load_dicom(make_dicom(view=view))


def test_rejects_color_dicom():
    with open(get_testdata_file("SC_rgb_dcmtk_+eb+cr.dcm"), "rb") as f:
        raw = f.read()
    with pytest.raises(DicomError):
        load_dicom(raw)


def test_rejects_non_dicom_bytes():
    with pytest.raises(DicomError, match="read"):
        load_dicom(b"this is definitely not a dicom file" * 20)


def test_rejects_blank_image():
    with pytest.raises(DicomError, match="blank"):
        load_dicom(make_dicom(np.full((32, 32), 700, dtype=np.uint16)))


def test_warnings_for_missing_modality_body_part_burned_in_and_multiframe():
    frames = np.stack([np.tile(np.linspace(0, 4000, 32, dtype=np.uint16), (32, 1))] * 3)
    raw = make_dicom(frames, body_part="KNEE", BurnedInAnnotation="YES")
    text = " | ".join(load_dicom(raw).warnings)
    assert "KNEE" in text and "burned-in" in text and "frame 1 of 3" in text

    ds = _read(make_dicom())
    del ds.Modality
    buf = io.BytesIO()
    ds.save_as(buf)
    assert any("Modality" in w for w in load_dicom(buf.getvalue()).warnings)
