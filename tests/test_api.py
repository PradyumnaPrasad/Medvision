import io
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from medvision.config import get_settings
from tests.test_dicom import PHI, make_dicom

FIXTURE = Path(__file__).parent / "fixtures" / "sample_cxr.png"


@pytest.fixture(scope="module")
def client():
    from backend.app.main import app
    from medvision.pipeline import get_models

    with TestClient(app) as c:
        # Force the deterministic template report path for API tests: fast and
        # reproducible, independent of whether Ollama happens to be running on the
        # machine these tests execute on. The real LLM path has its own coverage in
        # tests/test_pipeline.py::test_analyze_with_real_llm_when_available.
        models = get_models()
        original_is_available = models.llm_client.is_available
        models.llm_client.is_available = lambda: False
        try:
            yield c
        finally:
            models.llm_client.is_available = original_is_available


@pytest.fixture
def uploaded_analysis(client):
    """Upload the sample fixture once and clean up its files afterward."""
    with FIXTURE.open("rb") as f:
        files = {"file": ("sample_cxr.png", f, "image/png")}
        response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 200
    data = response.json()
    yield data
    upload_dir = get_settings().paths.resolve("uploads_dir") / data["id"]
    shutil.rmtree(upload_dir, ignore_errors=True)


@pytest.mark.slow
def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["kg_nodes"] > 0
    assert body["kg_edges"] > 0
    assert "not for clinical use" in body["disclaimer"].lower()


@pytest.mark.slow
def test_analyze_rejects_unsupported_extension(client):
    response = client.post("/api/v1/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def _cxr_dicom(**kwargs) -> bytes:
    gray = np.asarray(Image.open(FIXTURE).convert("L"), dtype=np.uint16) * 16  # 12-bit range
    if kwargs.get("photometric") == "MONOCHROME1":
        gray = 4095 - gray
    return make_dicom(gray, **kwargs)


def _post_dicom(client, raw: bytes, name: str = "scan.dcm"):
    return client.post("/api/v1/analyze", files={"file": (name, raw, "application/dicom")})


@pytest.mark.slow
def test_analyze_rejects_garbage_dicom(client):
    response = _post_dicom(client, b"fake")
    assert response.status_code == 400
    assert "DICOM" in response.json()["detail"]


@pytest.mark.slow
def test_analyze_rejects_non_radiograph_dicom(client):
    response = _post_dicom(client, _cxr_dicom(modality="CT"))
    assert response.status_code == 400
    assert "modality" in response.json()["detail"]


@pytest.mark.slow
def test_analyze_dicom_end_to_end_stores_no_phi(client):
    response = _post_dicom(client, _cxr_dicom())
    assert response.status_code == 200
    data = response.json()
    upload_dir = get_settings().paths.resolve("uploads_dir") / data["id"]
    try:
        assert data["findings"] and data["differentials"]

        stored = [p for p in upload_dir.rglob("*") if p.is_file()]
        assert stored and all(p.suffix == ".png" for p in stored)  # no DICOM persisted
        blobs = [p.read_bytes() for p in stored]
        blobs.append(response.content)
        blobs.append(client.get(f"/api/v1/analyses/{data['id']}").content)
        for value in PHI.values():
            assert not any(value.encode() in blob for blob in blobs)
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


@pytest.mark.slow
def test_analyze_monochrome1_dicom_matches_monochrome2(client):
    responses = [
        _post_dicom(client, _cxr_dicom(photometric="MONOCHROME2")),
        _post_dicom(client, _cxr_dicom(photometric="MONOCHROME1")),
    ]
    try:
        assert all(r.status_code == 200 for r in responses)
        a, b = (r.json() for r in responses)
        probs_a = {f["node_id"]: f["prob"] for f in a["findings"]}
        probs_b = {f["node_id"]: f["prob"] for f in b["findings"]}
        assert probs_a.keys() == probs_b.keys()
        assert max(abs(probs_a[k] - probs_b[k]) for k in probs_a) < 0.02
    finally:
        for r in responses:
            if r.status_code == 200:
                shutil.rmtree(
                    get_settings().paths.resolve("uploads_dir") / r.json()["id"],
                    ignore_errors=True,
                )


@pytest.mark.slow
def test_analyze_dicom_warnings_reach_the_response(client):
    response = _post_dicom(client, _cxr_dicom(body_part="KNEE"))
    assert response.status_code == 200
    data = response.json()
    shutil.rmtree(get_settings().paths.resolve("uploads_dir") / data["id"], ignore_errors=True)
    assert any("KNEE" in w for w in data["quality_warnings"])


@pytest.mark.slow
def test_analyze_rejects_a_color_photo_via_the_quality_gate(client):
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(300, 300, 3), dtype=np.uint8)
    fake_photo = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    fake_photo.save(buf, format="PNG")
    buf.seek(0)

    response = client.post("/api/v1/analyze", files={"file": ("photo.png", buf, "image/png")})
    assert response.status_code == 400
    assert "doesn't look like a usable chest X-ray" in response.json()["detail"]


@pytest.mark.slow
def test_analyze_returns_findings_and_differentials(uploaded_analysis):
    data = uploaded_analysis
    assert data["findings"]
    assert data["differentials"]
    assert "FINDINGS:" in data["report_text"]
    assert data["disclaimer"]

    # LLM narration is forced off (see the `client` fixture), so this should be the
    # deterministic, fully-grounded template report
    assert data["report_source"] == "template"
    assert data["grounding"]["score"] == 1.0
    assert data["evidence"]["findings"]
    assert data["report"]["findings"]


@pytest.mark.slow
def test_analysis_is_persisted_and_listed(client, uploaded_analysis):
    analysis_id = uploaded_analysis["id"]

    get_response = client.get(f"/api/v1/analyses/{analysis_id}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == analysis_id

    list_response = client.get("/api/v1/analyses?limit=50")
    assert list_response.status_code == 200
    ids = [a["id"] for a in list_response.json()]
    assert analysis_id in ids


@pytest.mark.slow
def test_heatmap_is_served_for_present_finding(client, uploaded_analysis):
    shown = [f for f in uploaded_analysis["findings"] if f["status"] != "absent"]
    assert shown
    node_id = shown[0]["node_id"]

    response = client.get(f"/api/v1/analyses/{uploaded_analysis['id']}/heatmap/{node_id}.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0


@pytest.mark.slow
def test_feedback_roundtrip(client, uploaded_analysis):
    node_id = uploaded_analysis["findings"][0]["node_id"]
    response = client.post(
        f"/api/v1/analyses/{uploaded_analysis['id']}/feedback",
        json={"node_id": node_id, "verdict": "agree", "comment": "looks right"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "agree"
    assert body["node_id"] == node_id


@pytest.mark.slow
def test_feedback_on_unknown_analysis_404s(client):
    response = client.post(
        "/api/v1/analyses/does-not-exist/feedback",
        json={"node_id": "F_EFFUSION", "verdict": "agree"},
    )
    assert response.status_code == 404


@pytest.mark.slow
def test_kg_node_lookup(client):
    response = client.get("/api/v1/kg/nodes/F_EFFUSION")
    assert response.status_code == 200
    body = response.json()
    assert body["node"]["name"] == "Pleural Effusion"
    assert body["outgoing"]  # F_EFFUSION has suggestive_of edges


@pytest.mark.slow
def test_kg_node_not_found(client):
    response = client.get("/api/v1/kg/nodes/NOT_REAL")
    assert response.status_code == 404
