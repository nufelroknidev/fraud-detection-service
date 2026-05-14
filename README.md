# Fraud Scoring Service

**A real-time card-not-present (CNP) fraud scoring service for a UK BNPL / e-commerce payment processor.**

---

## Results

| Metric | Value |
|--------|-------|
| Gini coefficient | _TBD — Phase 2_ |
| KS statistic | _TBD — Phase 2_ |
| PR-AUC | _TBD — Phase 2_ |
| p99 latency (50 concurrent users) | _TBD — Phase 2_ |
| PSI retrain threshold | > 0.2 per feature |

Experiment tracking: _[DagsHub URL — configure in Phase 1]_

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

- [ ] Phase 1: Model trains, API serves, Docker runs
- [ ] Phase 2: p99 published, SHAP notebook, Evidently PSI, business README complete
- [ ] Phase 3: Cost matrix, model card, CI/CD
