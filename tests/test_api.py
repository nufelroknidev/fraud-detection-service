"""
Endpoint integration tests for the CNP fraud detection API.

Uses FastAPI TestClient with the real lifespan (model loaded from models/pipeline.pkl).
Requires models/pipeline.pkl to exist — run src/model/train.py first if missing.
"""

import math
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, _state


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _valid_payload(**overrides) -> dict:
    base = {
        "amount_gbp": 45.50,
        "hour_of_day": 14,
        "day_of_week": 2,
        "card_avg_amount_30d": 60.0,
        "card_txn_count_1h": 1,
        "card_amount_sum_1h": 45.50,
        "card_txn_count_24h": 3,
        "card_amount_sum_24h": 120.0,
        "merch_txn_count_1h": 50,
        "time_since_last_card_txn_sec": 14400.0,
        "amount_to_card_avg_ratio": 0.758,
        "log_amount": math.log(45.50),
        "hour_sin": math.sin(2 * math.pi * 14 / 24),
        "hour_cos": math.cos(2 * math.pi * 14 / 24),
        "dow_sin": math.sin(2 * math.pi * 2 / 7),
        "dow_cos": math.cos(2 * math.pi * 2 / 7),
        "merchant_category": "clothing",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

def test_metrics_returns_thresholds(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "threshold_f1_opt" in body
    assert "threshold_recall80" in body
    assert body["model_loaded"] is True
    assert 0.0 < body["threshold_f1_opt"] < 1.0
    assert 0.0 < body["threshold_recall80"] < 1.0


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

def test_predict_returns_valid_response(client):
    r = client.post("/predict", json=_valid_payload())
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["f1_opt_decision"] in ("BLOCK", "PASS")
    assert body["recall80_decision"] in ("REVIEW", "PASS")
    assert "f1_opt_threshold" in body
    assert "recall80_threshold" in body


def test_predict_missing_field_returns_422(client):
    payload = _valid_payload()
    del payload["amount_gbp"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_invalid_field_returns_422(client):
    r = client.post("/predict", json=_valid_payload(amount_gbp=-99.0))
    assert r.status_code == 422


def test_predict_high_risk_transaction(client):
    """A transaction with a very high amount ratio should score high."""
    r = client.post("/predict", json=_valid_payload(
        amount_gbp=5000.0,
        card_avg_amount_30d=50.0,
        amount_to_card_avg_ratio=100.0,
        log_amount=math.log(5000.0),
        card_txn_count_1h=5,
        card_amount_sum_1h=5000.0,
        hour_of_day=2,   # night
        hour_sin=math.sin(2 * math.pi * 2 / 24),
        hour_cos=math.cos(2 * math.pi * 2 / 24),
    ))
    assert r.status_code == 200
    assert r.json()["fraud_probability"] > 0.0
