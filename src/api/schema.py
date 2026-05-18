import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float  # log-odds; positive = pushes toward fraud
    direction: Literal["increases_risk", "decreases_risk"]

    @model_validator(mode="after")
    def direction_matches_sign(self) -> "FeatureContribution":
        expected = "increases_risk" if self.shap_value > 0 else "decreases_risk"
        if self.direction != expected:
            raise ValueError(
                f"direction '{self.direction}' contradicts shap_value={self.shap_value:.6f}"
            )
        return self


class PredictRequest(BaseModel):
    # Raw amount and time signals
    amount_gbp: float = Field(..., gt=0, le=25000)
    # hour_of_day must be derived from a UTC timestamp — do not send local time.
    # UK operates in UTC+0 (winter) / UTC+1 BST (summer); send the UTC hour so
    # cyclic features and is_night are consistent with how the model was trained.
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    card_avg_amount_30d: float = Field(..., ge=0)

    # Card velocity (pre-computed by caller's streaming layer)
    card_txn_count_1h: int = Field(..., ge=0)
    card_amount_sum_1h: float = Field(..., ge=0)
    card_txn_count_6h: int = Field(..., ge=0)
    card_amount_sum_6h: float = Field(..., ge=0)
    card_txn_count_24h: int = Field(..., ge=0)
    card_amount_sum_24h: float = Field(..., ge=0)
    card_txn_count_7d: int = Field(..., ge=0)
    card_amount_sum_7d: float = Field(..., ge=0)

    # Card×merchant velocity: how many times this card hit this merchant in the last 1h.
    # Low for legitimate one-off purchases; elevated for card-testing or repeat fraud.
    card_merch_txn_count_1h: int = Field(..., ge=0)

    # Time since last transaction (-1 sentinel = first ever transaction on this card)
    time_since_last_card_txn_sec: float = Field(..., ge=-1)

    # Derived amount signals (caller computes alongside velocity)
    # Model trained with clip=0.01; card_avg_amount_30d from generator is clipped [5, 2000],
    # so training max ratio ≈ 5000/5 = 1000. Upper bound 2000 leaves headroom for real data.
    amount_to_card_avg_ratio: float = Field(..., ge=0, le=2000)
    log_amount: float

    # Cyclic time encodings
    hour_sin: float = Field(..., ge=-1, le=1)
    hour_cos: float = Field(..., ge=-1, le=1)
    dow_sin: float = Field(..., ge=-1, le=1)
    dow_cos: float = Field(..., ge=-1, le=1)

    # Categorical
    merchant_category: str

    @model_validator(mode="after")
    def derived_fields_coherent(self) -> "PredictRequest":
        """Guard against unit bugs: log_amount and amount_to_card_avg_ratio must be
        consistent with the raw inputs they are derived from."""
        expected_log = math.log1p(self.amount_gbp)
        if abs(self.log_amount - expected_log) > 0.01:
            raise ValueError(
                f"log_amount={self.log_amount:.4f} is inconsistent with "
                f"amount_gbp={self.amount_gbp} (expected {expected_log:.4f} ± 0.01). "
                "Compute as math.log1p(amount_gbp)."
            )
        # Ratio coherence: caller must compute amount / max(card_avg_amount_30d, 0.01).
        # Tolerance is 10% to accommodate rounding in the caller's streaming layer.
        if self.card_avg_amount_30d > 0:
            expected_ratio = self.amount_gbp / max(self.card_avg_amount_30d, 0.01)
            if abs(self.amount_to_card_avg_ratio - expected_ratio) / (expected_ratio + 1e-9) > 0.10:
                raise ValueError(
                    f"amount_to_card_avg_ratio={self.amount_to_card_avg_ratio:.4f} is inconsistent "
                    f"with amount_gbp={self.amount_gbp} / card_avg_amount_30d={self.card_avg_amount_30d} "
                    f"(expected {expected_ratio:.4f} ± 10%). "
                    "Compute as amount_gbp / max(card_avg_amount_30d, 0.01)."
                )
        return self


class PredictResponse(BaseModel):
    fraud_probability: float
    f1_opt_decision: str       # "BLOCK" or "PASS"   — high-precision operating point
    recall80_decision: str     # "REVIEW" or "PASS"  — high-recall operating point
    f1_opt_threshold: float
    recall80_threshold: float
    top_features: list[FeatureContribution]  # top 3 by |shap_value|
    oot_features: list[str] = []  # features with out-of-training-distribution values
