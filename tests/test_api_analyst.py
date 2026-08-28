"""Tests for POST /analyst/ask (Step 21).

No real LLM API call here — `ask_analyst` is monkeypatched at the point the
router imports it, so these tests check the ENDPOINT's contract (request
validation, response shape, error mapping) without needing a real API key
or spending real money. The underlying tool-calling loop is covered for
real in `tests/test_llm_agent.py` (mocked SDK clients) and the grounding
functions in `tests/test_llm_tools.py` (real project data).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api.routers.analyst as analyst_router  # noqa: E402
from api.main import app  # noqa: E402
from src.llm.providers import AgentResult, AnalystNotConfiguredError, ToolCallRecord  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_ask_returns_answer_and_tool_calls_when_configured(client, monkeypatch):
    fake_result = AgentResult(
        answer="Customer 12346 has a 37.4% churn probability.",
        tool_calls=[
            ToolCallRecord(
                name="get_customer_summary",
                arguments={"customer_id": 12346},
                result={"customer_id": 12346, "churn_probability": 0.3736},
            )
        ],
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
    )
    monkeypatch.setattr(analyst_router, "ask_analyst", lambda question: fake_result)

    resp = client.post("/analyst/ask", json={"question": "What's customer 12346's churn risk?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == fake_result.answer
    assert body["provider"] == "anthropic"
    assert len(body["tool_calls"]) == 1
    assert body["tool_calls"][0]["name"] == "get_customer_summary"
    assert body["tool_calls"][0]["result"]["churn_probability"] == 0.3736


def test_ask_returns_503_when_no_provider_configured(client, monkeypatch):
    def raise_not_configured(question):
        raise AnalystNotConfiguredError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env.")

    monkeypatch.setattr(analyst_router, "ask_analyst", raise_not_configured)

    resp = client.post("/analyst/ask", json={"question": "Anything"})

    assert resp.status_code == 503
    assert "API_KEY" in resp.json()["detail"]


def test_ask_returns_500_on_unexpected_provider_error(client, monkeypatch):
    def raise_unexpected(question):
        raise RuntimeError("the provider's SDK blew up")

    monkeypatch.setattr(analyst_router, "ask_analyst", raise_unexpected)

    resp = client.post("/analyst/ask", json={"question": "Anything"})

    assert resp.status_code == 500
    # The real exception message must never leak to the client.
    assert "blew up" not in resp.json()["detail"]


def test_ask_rejects_empty_question(client):
    resp = client.post("/analyst/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ask_rejects_overly_long_question(client):
    resp = client.post("/analyst/ask", json={"question": "x" * 5000})
    assert resp.status_code == 422


def test_ask_rejects_missing_question(client):
    resp = client.post("/analyst/ask", json={})
    assert resp.status_code == 422


def test_openapi_schema_lists_the_analyst_endpoint(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert "/analyst/ask" in resp.json()["paths"]
