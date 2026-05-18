"""
Locust load test for the CNP Fraud Detection API.

Usage (Docker container must be running on port 8000):
    locust -f tests/locustfile.py --host http://localhost:8000

Headless / CI:
    locust -f tests/locustfile.py --host http://localhost:8000 \
           --headless -u 50 -r 10 --run-time 60s \
           --csv results/locust --html results/locust_report.html

Targets:
    p99 latency < 200 ms at 50 concurrent users
    error rate  < 1%
"""

import math
import random

from locust import HttpUser, between, task


# Merchant categories that appear in the training data (src/data/generate.py).
# Do NOT add categories not in this list — unknown categories are OOD and
# will trigger the oot_features flag on every request, polluting load test metrics.
_MERCHANT_CATEGORIES = [
    "electronics",
    "travel",
    "fashion",
    "grocery",
    "gaming",
    "jewellery",
    "food_delivery",
    "subscriptions",
]


def _normal_txn() -> dict:
    """Typical low-risk CNP transaction."""
    amount = round(random.uniform(5.0, 150.0), 2)
    hour = random.randint(8, 22)
    dow = random.randint(0, 6)
    avg_30d = round(random.uniform(30.0, 120.0), 2)
    return {
        "amount_gbp": amount,
        "hour_of_day": hour,
        "day_of_week": dow,
        "card_avg_amount_30d": avg_30d,
        "card_txn_count_1h": random.randint(0, 2),
        "card_amount_sum_1h": round(random.uniform(0.0, 60.0), 2),
        "card_txn_count_6h": random.randint(0, 5),
        "card_amount_sum_6h": round(random.uniform(0.0, 150.0), 2),
        "card_txn_count_24h": random.randint(1, 8),
        "card_amount_sum_24h": round(random.uniform(20.0, 300.0), 2),
        "card_txn_count_7d": random.randint(5, 25),
        "card_amount_sum_7d": round(random.uniform(100.0, 800.0), 2),
        "card_merch_txn_count_1h": random.randint(0, 2),
        "time_since_last_card_txn_sec": round(random.uniform(3600.0, 86400.0), 1),
        "amount_to_card_avg_ratio": round(amount / max(avg_30d, 0.01), 4),
        "log_amount": round(math.log1p(amount), 6),
        "hour_sin": round(math.sin(2 * math.pi * hour / 24), 6),
        "hour_cos": round(math.cos(2 * math.pi * hour / 24), 6),
        "dow_sin": round(math.sin(2 * math.pi * dow / 7), 6),
        "dow_cos": round(math.cos(2 * math.pi * dow / 7), 6),
        "merchant_category": random.choice(_MERCHANT_CATEGORIES),
    }


def _suspicious_txn() -> dict:
    """High-value night-time transaction — likely to hit REVIEW/BLOCK thresholds."""
    amount = round(random.uniform(800.0, 2500.0), 2)
    hour = random.randint(1, 4)   # is_night window
    dow = random.randint(0, 6)
    avg_30d = round(random.uniform(30.0, 80.0), 2)  # low baseline → high ratio
    return {
        "amount_gbp": amount,
        "hour_of_day": hour,
        "day_of_week": dow,
        "card_avg_amount_30d": avg_30d,
        "card_txn_count_1h": random.randint(3, 8),   # velocity burst
        "card_amount_sum_1h": round(random.uniform(500.0, 2000.0), 2),
        "card_txn_count_6h": random.randint(5, 15),
        "card_amount_sum_6h": round(random.uniform(800.0, 3000.0), 2),
        "card_txn_count_24h": random.randint(8, 20),
        "card_amount_sum_24h": round(random.uniform(1000.0, 5000.0), 2),
        "card_txn_count_7d": random.randint(15, 40),
        "card_amount_sum_7d": round(random.uniform(3000.0, 12000.0), 2),
        "card_merch_txn_count_1h": random.randint(1, 5),
        "time_since_last_card_txn_sec": round(random.uniform(30.0, 300.0), 1),
        "amount_to_card_avg_ratio": round(amount / max(avg_30d, 0.01), 4),
        "log_amount": round(math.log1p(amount), 6),
        "hour_sin": round(math.sin(2 * math.pi * hour / 24), 6),
        "hour_cos": round(math.cos(2 * math.pi * hour / 24), 6),
        "dow_sin": round(math.sin(2 * math.pi * dow / 7), 6),
        "dow_cos": round(math.cos(2 * math.pi * dow / 7), 6),
        "merchant_category": random.choice(["electronics", "gaming", "travel"]),
    }


class FraudApiUser(HttpUser):
    """Simulates a downstream service calling the scoring API."""

    wait_time = between(0.05, 0.3)  # 50–300 ms think time between requests

    @task(10)
    def predict_normal(self):
        self.client.post("/predict", json=_normal_txn(), name="/predict [normal]")

    @task(3)
    def predict_suspicious(self):
        self.client.post("/predict", json=_suspicious_txn(), name="/predict [suspicious]")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")
