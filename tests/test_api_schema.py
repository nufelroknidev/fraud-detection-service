"""Unit tests for API request/response schema validation."""

import math
import pytest
from pydantic import ValidationError
from src.api.schema import PredictRequest, PredictResponse


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


def test_valid_request_parses():
    req = PredictRequest(**_valid_payload())
    assert req.amount_gbp == 45.50
    assert req.merchant_category == "clothing"


def test_first_transaction_sentinel():
    req = PredictRequest(**_valid_payload(time_since_last_card_txn_sec=-1))
    assert req.time_since_last_card_txn_sec == -1


def test_negative_amount_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(**_valid_payload(amount_gbp=-10.0))


def test_hour_out_of_range_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(**_valid_payload(hour_of_day=24))


def test_invalid_hour_sin_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(**_valid_payload(hour_sin=1.5))


def test_response_model_block():
    resp = PredictResponse(
        fraud_probability=0.97,
        f1_opt_decision="BLOCK",
        recall80_decision="REVIEW",
        f1_opt_threshold=0.9487,
        recall80_threshold=0.2529,
    )
    assert resp.f1_opt_decision == "BLOCK"
    assert resp.recall80_decision == "REVIEW"


def test_response_model_pass():
    resp = PredictResponse(
        fraud_probability=0.01,
        f1_opt_decision="PASS",
        recall80_decision="PASS",
        f1_opt_threshold=0.9487,
        recall80_threshold=0.2529,
    )
    assert resp.f1_opt_decision == "PASS"
