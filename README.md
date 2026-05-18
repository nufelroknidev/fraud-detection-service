# Fraud Scoring Service

**A real-time card-not-present (CNP) fraud scoring service for a UK BNPL / e-commerce payment processor.**

---

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9497 |
| PR-AUC | 0.2358 |
| Gini coefficient | 0.8995 (target > 0.60) |
| KS statistic | 0.7711 |
| F1-optimal threshold | 0.9298 → 43% precision / 22% recall |
| High-recall threshold | 0.2496 → 2.5% precision / 80% recall |
| p50 latency (50 concurrent users) | 110 ms |
| p95 latency (50 concurrent users) | 200 ms |
| p99 latency (50 concurrent users) | 270 ms — model pre-loaded at startup; overhead is Locust + local Docker, not inference |
| Throughput | 166 RPS, 0% errors |
| PSI retrain threshold | > 0.2 per feature |
| Cost-optimal threshold | 0.5813 → £91k saving vs no-model baseline |
| Expected £ saving vs F1-threshold | £38k per 120k transactions |

Experiment tracking: [DagsHub — nufel.rokni.dev/fraud-detection-service](https://dagshub.com/nufel.rokni.dev/fraud-detection-service)

---

## Explainability

Each `/predict` response includes the top 3 SHAP feature contributions for the transaction, enabling compliance officers to audit every automated decision.

The beeswarm below shows global feature impact across a 2,000-transaction test-set sample. Each dot is one transaction; x-position is the SHAP value (log-odds contribution to fraud probability); colour is the raw feature value (red = high, blue = low).

![SHAP Beeswarm — Global Feature Impact](docs/images/shap_beeswarm.png)

`amount_to_card_avg_ratio` is the dominant signal: transactions where the amount greatly exceeds the card's 30-day average are pushed strongly toward fraud. `card_avg_amount_30d` and velocity features (`card_txn_count_1h`, `card_amount_sum_24h`) provide secondary lift.

---

## Architecture

```
[Synthetic Transaction Generator]
           │
           ▼
[Feature Engineering (velocity + static)]
           │
           ▼
[XGBoost Classifier] ──► [MLflow / DagsHub]
           │
           ▼
[FastAPI Scoring Service]
  /predict  /health  /metrics
           │
           ▼
[Evidently Drift Monitor (PSI per feature)]
```

---

## Run Locally (3 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data + train model
python -m src.model.train

# 3. Start the API
uvicorn src.api.main:app --reload
```

---

## Domain Context

Card-not-present fraud is the highest-volume fraud type in UK e-commerce (~0.03–0.1% of transactions). This service scores each transaction at submission time, returning a fraud probability and SHAP-based explanation for compliance use.

### Fraud Signals Used

- Velocity: transactions per card in last 1h, 6h, 24h, and 7d
- Amount deviation from card's 30-day rolling average
- Merchant category risk score (CNP-weighted)
- Hour-of-day and day-of-week encoding

---

## Monitoring

PSI (Population Stability Index) is tracked per feature using Evidently. Run with `python -m src.monitoring.drift`.

![Evidently PSI Drift Report](docs/images/evidently_drift.png)

All 11 features are **stable** on synthetic data (PSI < 0.10). `time_since_last_card_txn_sec` is the highest at 0.0925 — a natural consequence of temporal drift in inter-transaction gaps as the simulated dataset progresses.

| PSI Range | Status | Action |
|-----------|--------|--------|
| < 0.1 | Stable | None |
| 0.1 – 0.2 | Investigate | Review feature source |
| > 0.2 | Retrain trigger | Initiate retraining pipeline |

---

## Champion / Challenger Deployment

See `docs/champion_challenger.md` for the full rollout design.

Rollback: single MLflow model alias change — no redeployment required.

---

## Project Status

- [x] Phase 1: Model trains, API serves, Docker runs
- [x] Phase 2: p99 published, SHAP notebook, Evidently PSI, business README complete
- [x] Phase 3: Cost matrix, model card, CI/CD
