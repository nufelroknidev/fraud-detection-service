# Model Card — UK CNP Fraud Detection (XGBoost)

## Model Details

| Field | Value |
|-------|-------|
| Model type | XGBoost binary classifier (sklearn Pipeline with OrdinalEncoder) |
| Version | post-retrain-2026-05-18 (clip=0.01, card_merch_txn_count_1h) |
| Training date | 2026-05-18 |
| Framework | XGBoost 2.x, scikit-learn 1.4+ |
| Tracking | DagsHub MLflow — nufel.rokni.dev/fraud-detection-service |
| Owner | Nawfal Rokni |

---

## Intended Use

**Primary use case:** Real-time scoring of card-not-present (CNP) transactions
for a UK BNPL / e-commerce payment processor. The model returns a fraud probability
and two binary decisions at different operating points.

**Out-of-scope uses:**
- Card-present (POS) fraud — different feature set required
- Account takeover detection — requires session/behavioural signals not in this model
- Batch retroactive scoring of historical transactions

---

## Training Data

| Property | Value |
|----------|-------|
| Source | Synthetic (domain-aware generator — `src/data/generate.py`) |
| Size | 600,000 transactions |
| Fraud rate | 0.30% (≈ UK CNP industry rate) |
| Time span | Simulated 365-day window |
| Split | Temporal 80/20 (no shuffling — prevents leakage) |

**Synthetic data note:** The generator applies domain-aware fraud rules
(night-time bias, amount spikes, high-velocity bursts) but does not capture
all real-world distribution shifts. Velocity features are correct in structure
but underweight compared to live data.

---

## Features

| Feature | Type | Description |
|---------|------|-------------|
| `amount_gbp` | Numeric | Transaction amount in GBP |
| `card_avg_amount_30d` | Numeric | Card's 30-day rolling average spend |
| `amount_to_card_avg_ratio` | Numeric | Amount / 30d avg (strongest signal) |
| `log_amount` | Numeric | log(amount_gbp) |
| `card_txn_count_1h` | Numeric | Transactions on card in last 1 hour |
| `card_amount_sum_1h` | Numeric | Spend on card in last 1 hour |
| `card_txn_count_24h` | Numeric | Transactions on card in last 24 hours |
| `card_amount_sum_24h` | Numeric | Spend on card in last 24 hours |
| `card_txn_count_6h` | Numeric | Transactions on card in last 6 hours |
| `card_amount_sum_6h` | Numeric | Spend on card in last 6 hours |
| `card_txn_count_7d` | Numeric | Transactions on card in last 7 days |
| `card_amount_sum_7d` | Numeric | Spend on card in last 7 days |
| `card_merch_txn_count_1h` | Numeric | Times this card hit this merchant in the last 1 hour |
| `time_since_last_card_txn_sec` | Numeric | Seconds since card's last transaction (-1 = first ever) |
| `hour_sin`, `hour_cos` | Numeric | Cyclic encoding of hour of day |
| `dow_sin`, `dow_cos` | Numeric | Cyclic encoding of day of week |
| `hour_of_day` | Integer | Raw hour (0–23) |
| `day_of_week` | Integer | Raw day (0=Mon, 6=Sun) |
| `merchant_category` | Categorical | Merchant type (OrdinalEncoded) |

---

## Performance

Evaluated on temporal holdout (last 20% of data by timestamp).

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9483 |
| PR-AUC | 0.2291 |
| Gini coefficient | 0.8966 |
| KS statistic | 0.7668 |
| Best XGBoost round | 774 / 1000 |
| `scale_pos_weight` | 334.8 (class imbalance correction) |

### Operating Points

| Threshold | Purpose | Precision | Recall | Use case |
|-----------|---------|-----------|--------|----------|
| 0.8820 | F1-optimal (BLOCK) | 36.4% | 23.5% | High-confidence auto-block |
| 0.1013 | 80%-recall (REVIEW) | 2.4% | 80% | Human review queue |

---

## Limitations

1. **Synthetic data bias:** Precision at high recall is low (2.4%) because the
   synthetic generator does not produce velocity bursts. Live data would improve
   this significantly.

2. **`card_merch_txn_count_1h` has zero feature importance in this model.** With
   50k cards and 2000 merchants distributed uniformly, the same card almost never
   hits the same merchant twice within an hour by chance — the feature is always 0
   in training. The 3.5% PR-AUC gain vs. the previous model came entirely from the
   `clip=0.01` fix on `amount_to_card_avg_ratio`, not from the new feature. On live
   payments data this feature would activate immediately for card-testing attacks
   (same card probing the same merchant repeatedly in seconds).

2. **Static merchant categories:** The OrdinalEncoder assigns arbitrary ordinal
   values to categories. A learned embedding would better capture category risk.

3. **No cross-card signals:** The model scores cards independently. Coordinated
   fraud rings operating across multiple cards are not detectable.

4. **Threshold stability:** The F1-optimal threshold (0.9487) is high because
   synthetic fraud patterns are concentrated. Production recalibration is required
   before deployment.

---

## Fairness Considerations

The model does not use protected characteristics (age, gender, nationality,
postcode). However, merchant category and spend patterns may correlate with
demographic groups. A fairness audit against protected attributes should be
conducted before production deployment using real customer data.

---

## Monitoring & Retraining

| Trigger | Implemented | Action |
|---------|-------------|--------|
| PSI > 0.20 on any feature | Yes — `src/monitoring/drift.py` | Initiate retraining pipeline |
| Score distribution PSI > 0.10 | Yes — `src/monitoring/drift.py` | Alert + review |
| PR-AUC drops > 5% on rolling 7-day eval | **No** — requires chargeback label pipeline | Alert + review |
| False positive rate > 2x baseline | **No** — requires ground truth labels | Threshold recalibration |

**Monitoring gap:** Feature PSI and score distribution are monitored in
`src/monitoring/drift.py`. Model performance metrics (PR-AUC, recall, false
positive rate) require ground truth labels which arrive 30–90 days after
transaction via chargeback / dispute data. In production, a label ingestion
pipeline would join chargeback events to scored transactions and compute
rolling PR-AUC on confirmed-labelled windows. This is not implemented in
the synthetic data environment.

Reports are written to `results/drift/` on each run.

---

## Ethical Use Statement

This model is intended to protect customers from financial fraud. Incorrect
blocking decisions (false positives) cause customer harm through declined
legitimate transactions. The system provides two operating thresholds to allow
operators to choose an appropriate trade-off between fraud prevention and
customer experience based on their specific risk appetite.
