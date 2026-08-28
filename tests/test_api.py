"""Tests for the FastAPI prediction service (Step 14).

Run:
    pytest tests/test_api.py -v

Requires the artefacts produced by Steps 6, 9, 10, 12 and 13 to already
exist (see BUILD_LOG.md) — this suite exercises the live API end-to-end, not mocks,
including cross-checking the API's on-the-fly prediction against the offline
batch pipeline's precomputed value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def known_customer():
    """A real customer_id with its precomputed churn probability, so the live
    API can be checked against the offline Step 12 batch run rather than only
    checked for "some" plausible-looking response.
    """
    df = pd.read_csv("reports/retention_priority_list.csv")
    row = df.iloc[0]
    return int(row["customer_id"]), float(row["churn_probability"])


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True
    assert body["n_customers"] > 4000


def test_predict_matches_offline_pipeline(client, known_customer):
    customer_id, expected_probability = known_customer
    resp = client.post("/predict", json={"customer_id": customer_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["customer_id"] == customer_id
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert body["estimated_customer_value"] > 0
    assert body["retention_priority"] is not None
    assert isinstance(body["segment"], str) and len(body["segment"]) > 0
    # The live API must apply the exact same calibrated model to the exact
    # same engineered features as the offline Step 12 batch run.
    assert body["churn_probability"] == pytest.approx(expected_probability, abs=1e-3)


def test_predict_unknown_customer_returns_404(client):
    resp = client.post("/predict", json={"customer_id": 1})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


@pytest.mark.parametrize(
    "bad_body",
    [
        {"customer_id": 0},  # violates gt=0
        {"customer_id": -5},  # negative
        {"customer_id": "abc"},  # wrong type
        {},  # missing required field
        {"customer_id": 5_000_000},  # out of realistic range (custom validator)
    ],
)
def test_predict_rejects_invalid_requests(client, bad_body):
    resp = client.post("/predict", json=bad_body)
    assert resp.status_code == 422


def test_predict_explain_includes_shap_factors(client, known_customer):
    customer_id, _ = known_customer
    resp = client.post("/predict/explain", json={"customer_id": customer_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["customer_id"] == customer_id
    assert isinstance(body["top_risk_factors"], list)
    assert isinstance(body["top_protective_factors"], list)
    assert len(body["top_risk_factors"]) + len(body["top_protective_factors"]) > 0
    for factor in body["top_risk_factors"] + body["top_protective_factors"]:
        assert {"feature", "value", "shap_value"} <= factor.keys()
    assert isinstance(body["narrative"], str) and len(body["narrative"]) > 0
    assert str(customer_id) in body["narrative"]


def test_predict_explain_unknown_customer_returns_404(client):
    resp = client.post("/predict/explain", json={"customer_id": 1})
    assert resp.status_code == 404


def test_openapi_schema_lists_all_endpoints(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert {"/health", "/predict", "/predict/explain", "/analyst/ask"} <= paths.keys()


def test_docs_page_loads(client):
    resp = client.get("/docs")
    assert resp.status_code == 200
