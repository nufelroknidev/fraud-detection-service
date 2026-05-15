from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # Raw amount and time signals
    amount_gbp: float = Field(..., gt=0)
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    card_avg_amount_30d: float = Field(..., ge=0)

    # Card velocity (pre-computed by caller's streaming layer)
    card_txn_count_1h: int = Field(..., ge=0)
    card_amount_sum_1h: float = Field(..., ge=0)
    card_txn_count_24h: int = Field(..., ge=0)
    card_amount_sum_24h: float = Field(..., ge=0)

    # Merchant velocity
    merch_txn_count_1h: int = Field(..., ge=0)

    # Time since last transaction (-1 sentinel = first ever transaction on this card)
    time_since_last_card_txn_sec: float = Field(..., ge=-1)

    # Derived amount signals (caller computes alongside velocity)
    amount_to_card_avg_ratio: float = Field(..., ge=0)
    log_amount: float

    # Cyclic time encodings
    hour_sin: float = Field(..., ge=-1, le=1)
    hour_cos: float = Field(..., ge=-1, le=1)
    dow_sin: float = Field(..., ge=-1, le=1)
    dow_cos: float = Field(..., ge=-1, le=1)

    # Categorical
    merchant_category: str


class PredictResponse(BaseModel):
    fraud_probability: float
    f1_opt_decision: str       # "BLOCK" or "PASS"   — high-precision operating point
    recall80_decision: str     # "REVIEW" or "PASS"  — high-recall operating point
    f1_opt_threshold: float
    recall80_threshold: float
