# Remaining Work

Items identified against the ML_Project_Build_Plan. Tackle one per session.

---

## High Priority

- [x] **SHAP in `/predict` response**
  Plan requires: *"SHAP local explanations are required for every prediction. A compliance officer must be able to receive a plain-English breakdown of why a transaction was flagged."*
  Done: `FeatureContribution` model added to schema; `TreeExplainer` loaded at startup; top 3 features by |shap_value| returned with `direction` field. All 6 tests pass.

- [x] **SHAP summary plot in README**
  Plan requires: *"Include a SHAP summary plot in the README."*
  Done: beeswarm generated via script (correct feature order from `get_feature_names_out()`), saved to `docs/images/shap_beeswarm.png`, embedded in README under new Explainability section.

- [x] **Evidently HTML report screenshot in README**
  Plan requires: *"Linked artefacts: DagsHub experiment tracking URL in the README. Evidently HTML report screenshot."*
  Done: PSI chart generated from drift_summary.json, saved to docs/images/evidently_drift.png, embedded in README Monitoring section. All 11 features stable; worst PSI = 0.0925 (time_since_last_card_txn_sec).

---

## Lower Priority

- [x] **Prediction score distribution monitoring in `drift.py`**
  Done: `prediction_score` column added to both reference/current windows via `pipeline.predict_proba()`; `ValueDrift` metric included in Evidently report. Score PSI = 0.0009 (stable). Graceful fallback if model not available.

- [x] **6h and 7d velocity windows**
  Done: `_card_velocity` loop extended to ("1h","6h","24h","7d"); 4 new columns in FEATURE_COLS across velocity.py, train.py, main.py, schema.py, test_api.py, drift.py. Retrained: ROC-AUC 0.9497, PR-AUC 0.2358, Gini 0.8995. README, beeswarm, and drift chart all regenerated.

---

## Done

- [x] Gini coefficient computed, logged to MLflow, shown in README
- [x] KS statistic computed, logged to MLflow, shown in README
- [x] `/metrics` endpoint added to FastAPI
- [x] Endpoint integration tests (`tests/test_api.py`)
- [x] Schema unit tests (`tests/test_api_schema.py`)
- [x] `scipy` added to `requirements.txt`
- [x] p99 latency context note added to README
