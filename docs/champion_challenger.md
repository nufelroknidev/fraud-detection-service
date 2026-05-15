# Champion / Challenger Deployment Design

## Overview

A champion/challenger setup runs two model versions in parallel production traffic,
routing a small percentage of requests to the challenger while the champion handles
the remainder. This allows safe evaluation of new models on live data without a
full cutover.

---

## Traffic Split

| Role | Model | Traffic Share |
|------|-------|---------------|
| Champion | Current production model (MLflow alias: `champion`) | 95% |
| Challenger | Candidate model under evaluation (MLflow alias: `challenger`) | 5% |

The 5% challenger slice is sufficient to reach statistical significance on ~10k
daily transactions within 48 hours.

---

## Decision Logic

```
Request → Router (hash of card_id % 100)
              ├─ 0–94  → Champion model → response + log(champion)
              └─ 95–99 → Challenger model → response + log(challenger)
```

Routing by `card_id` hash (not random) ensures the same card always hits the
same model within a routing window, preventing incoherent decision histories
for a single customer.

---

## Promotion Criteria

A challenger is promoted to champion only when **all** gates pass over a 48-hour
evaluation window:

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| PR-AUC delta | >= -0.005 (challenger within 0.5pp of champion) | Model quality floor |
| p99 latency | < 200 ms at current traffic | Latency SLA |
| Error rate | < 0.1% | Stability |
| PSI (all features) | < 0.10 | Same data distribution as champion |
| Cost delta | Expected £ loss <= champion's | Business outcome |

Promotion is a **single MLflow alias change** — no redeployment required.

```bash
# Promote challenger to champion
mlflow models set-alias --model-name cnp-fraud-xgboost \
    --alias champion --run-id <challenger_run_id>
```

---

## Rollback

Rollback is the reverse alias change:

```bash
mlflow models set-alias --model-name cnp-fraud-xgboost \
    --alias champion --run-id <previous_champion_run_id>
```

Time-to-rollback target: < 2 minutes (alias change + pod restart).

---

## Shadow Mode (Pre-challenger)

Before entering live A/B traffic, a new model runs in **shadow mode**:
- Receives all production requests
- Makes predictions but **does not serve them** to callers
- Logs predictions alongside champion decisions for offline comparison

Shadow mode runs for 24 hours minimum before challenger promotion.

---

## Monitoring During Evaluation

The following metrics are tracked per model alias in real time:

- Fraud detection rate (recall at operational threshold)
- False positive rate
- p50 / p95 / p99 latency
- PSI per feature (challenger vs champion training reference)
- Decision distribution (BLOCK% / REVIEW% / PASS%)

Alerts fire if any champion metric degrades by > 10% relative during a
challenger evaluation window — this triggers automatic traffic reversion to
100% champion while the degradation is investigated.

---

## Implementation Notes

- Model aliases are managed in DagsHub MLflow (`nufel.rokni.dev/fraud-detection-service`)
- The FastAPI `lifespan` loader reads `MLFLOW_RUN_ID` from `.env` — in production
  this is replaced by an alias lookup: `models:/cnp-fraud-xgboost@champion`
- Traffic routing is implemented as a thin middleware layer in `src/api/main.py`
  (Phase 3 implementation task)
