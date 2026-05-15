# Remaining Work

Items identified against the ML_Project_Build_Plan. Tackle one per session.

---

## High Priority

- [ ] **SHAP in `/predict` response**
  Plan requires: *"SHAP local explanations are required for every prediction. A compliance officer must be able to receive a plain-English breakdown of why a transaction was flagged."*
  The SHAP notebook covers global explainability but the API returns probability + decision only — no per-prediction explanation.
  Work: add `shap_values` + top-3 feature contributions to `PredictResponse`; load a `TreeExplainer` at startup alongside the pipeline.

- [ ] **SHAP summary plot in README**
  Plan requires: *"Include a SHAP summary plot in the README."*
  The beeswarm plot exists in the notebook output but is not embedded in the README.
  Work: export the beeswarm PNG from the notebook, save to `docs/images/`, link in README.

- [ ] **Evidently HTML report screenshot in README**
  Plan requires: *"Linked artefacts: DagsHub experiment tracking URL in the README. Evidently HTML report screenshot."*
  DagsHub URL is present; the Evidently screenshot is not.
  Work: run `python -m src.monitoring.drift`, screenshot or export the HTML report, embed in README.

---

## Lower Priority

- [ ] **Prediction score distribution monitoring in `drift.py`**
  Plan says: *"Also monitor: feature means, missing rate, and prediction score distribution."*
  Verify that `drift.py` monitors the model output score distribution (not just input features). Add if missing.

- [ ] **6h and 7d velocity windows**
  Plan lists: *"Transactions per card in last 6h / 7d."*
  Currently only 1h and 24h are built. Legitimate drift — 24h bookends the signal — but adding 6h at minimum would be a stronger demo.
  Work: extend `src/features/velocity.py`; retrain and update README metrics.

---

## Done

- [x] Gini coefficient computed, logged to MLflow, shown in README
- [x] KS statistic computed, logged to MLflow, shown in README
- [x] `/metrics` endpoint added to FastAPI
- [x] Endpoint integration tests (`tests/test_api.py`)
- [x] Schema unit tests (`tests/test_api_schema.py`)
- [x] `scipy` added to `requirements.txt`
- [x] p99 latency context note added to README
