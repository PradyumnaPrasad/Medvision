import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from medvision.config import get_settings

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


@pytest.mark.slow
def test_analyze_rejects_dicom_for_now(client):
    response = client.post(
        "/api/v1/analyze", files={"file": ("scan.dcm", b"fake", "application/dicom")}
    )
    assert response.status_code == 400


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
