"""P1.11: FastAPI integration tests (offline fake provider, in-memory DB)."""

import os

import pytest

# Configure an in-memory DB + offline provider BEFORE the app/state is built.
os.environ["STYLIST_DB_URL"] = "sqlite://"
os.environ["STYLIST_LLM"] = "fake"

from core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.state import reset_state  # noqa: E402


@pytest.fixture(scope="module")
def client():
    reset_state()
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    assert client.get("/healthz/").json()["status"] == "ok"


def test_profile_parse(client):
    r = client.post("/api/profile/parse", json={"text": "vintage platinum ring under 1,50,000"})
    prof = r.json()["profile"]
    assert prof["budget_max"] == 150000
    assert "vintage" in prof["styles"]
    assert "embedding" not in prof  # large vector excluded from the API response


def test_recommend_returns_grounded_claims_audit(client):
    r = client.post(
        "/api/recommend",
        json={"text": "vintage white gold diamond ring up to 2,00,000", "top_k": 4},
    )
    body = r.json()
    assert body["recommendations"], "expected recommendations"
    assert body["blocked_any"] is False
    for rec in body["recommendations"]:
        assert rec["grounding_decision"] == "passed"
        assert rec["claims_audit"], "audit must be exposed (the differentiator)"
        for claim in rec["claims_audit"]:
            if claim["claim_type"] == "factual":
                assert claim["grounded"] and not claim["blocked"]
        assert rec["product"]["metal"] in rec["rationale"]  # factual fields verbatim


def test_recommend_respects_hard_constraints(client):
    r = client.post("/api/recommend", json={"text": "platinum only, under 90,000", "top_k": 5})
    for rec in r.json()["recommendations"]:
        assert rec["product"]["metal"] == "platinum"
        assert rec["product"]["price"] <= 90000


def test_recommend_demo_hallucination_is_blocked(client):
    r = client.post(
        "/api/recommend",
        json={"text": "a nice ring under 2,00,000", "top_k": 2, "demo_hallucination": True},
    )
    body = r.json()
    assert body["blocked_any"] is True
    for rec in body["recommendations"]:
        assert "4-carat" not in rec["rationale"]  # fabricated claim never reaches the user
        assert any(c["blocked"] for c in rec["claims_audit"])


def test_products_separate_factual_and_inferred(client):
    r = client.get("/api/products/syn-0000")
    body = r.json()
    assert "factual" in body and "inferred" in body
    assert body["factual"]["source"] == "synthetic"
    assert body["inferred"] is None or "styles" in body["inferred"]
    assert client.get("/api/products/does-not-exist").status_code == 404


def test_eval_run_and_fetch(client):
    r = client.post("/api/eval/run", json={"n": 8, "top_k": 3})
    body = r.json()
    eval_id = body["eval"]["id"]
    assert body["eval"]["metrics"]["groundedness_rate"] == 1.0
    # Honest framing: structural guarantees + offline subjective suppressed.
    assert "Structural guarantees" in body["report"]
    assert body["eval"]["metrics"]["subjective_relevance"]["offline_placeholder"] is True
    assert body["eval"]["metrics"]["subjective_relevance"]["mean_score_1to5"] is None
    # Fetch it back.
    got = client.get(f"/api/eval/{eval_id}")
    assert got.status_code == 200
    assert got.json()["eval"]["id"] == eval_id
    assert client.get("/api/eval/missing").status_code == 404


def test_tagging_accuracy_endpoint_suppressed_offline(client):
    # The fake tagger echoes the hint it is scored against, so the endpoint must NOT serve
    # a tautological number on the offline provider.
    body = client.get("/api/tagging-accuracy").json()
    assert body["offline_placeholder"] is True
    assert body["tagging_accuracy"] is None
    assert "tautology" in body["caveat"].lower()


def test_index_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Faithful" in r.text and "claims audit" in r.text
