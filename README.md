# Fraud Scoring Service

**A real-time card-not-present (CNP) fraud scoring service for a UK BNPL / e-commerce payment processor.**

---

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9476 |
| PR-AUC | 0.2214 |
| Gini coefficient | 0.895 |
| F1-optimal threshold | 0.9487 → 51% precision / 19% recall |
| High-recall threshold | 0.2529 → 2.4% precision / 80% recall |
| p50 latency (50 concurrent users) | 110 ms |
| p95 latency (50 concurrent users) | 200 ms |
| p99 latency (50 concurrent users) | 270 ms (single-replica Docker, local hardware) |
| Throughput | 166 RPS, 0% errors |
| PSI retrain threshold | > 0.2 per feature |
| Cost-optimal threshold | 0.5813 → £91k saving vs no-model baseline |
| Expected £ saving vs F1-threshold | £38k per 120k transactions |

Experiment tracking: [DagsHub — nufel.rokni.dev/fraud-detection-service](https://dagshub.com/nufel.rokni.dev/fraud-detection-service)

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

- Velocity: transactions per card in last 1h / 6h / 24h / 7d
- Amount deviation from card's 30-day rolling average
- Merchant category risk score (CNP-weighted)
- Hour-of-day and day-of-week encoding

---

## Monitoring

PSI (Population Stability Index) is tracked per feature:

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
