# Remaining Work

Items identified against the ML_Project_Build_Plan. Tackle one per session.

---

## CRITICAL — Round 3 Battle Failures

- [x] **[R3-R6] Retrain model — DONE 2026-05-18**
  Both bundled changes applied: clip=0.01 + card_merch_txn_count_1h feature.
  New metrics: PR-AUC 0.2291 (+3.5%), ROC-AUC 0.9483, Gini 0.8966, KS 0.7668.
  New thresholds: F1-opt 0.8820, Recall80 0.1013. Fallbacks in main.py updated.
  Note: card_merch_txn_count_1h has zero importance on synthetic data (expected — card-merchant pairs rarely repeat within 1h in uniform synthetic distribution). Feature is correctly designed for production card-testing signal.
  MLflow remote logging failed (INVALID_PARAMETER_VALUE); local pipeline.pkl saved and confirmed working — all 16 tests pass.

- [x] **[R3-R6-BUG] Fix SHAP direction rounding edge case**
  Bug: `main.py:173` — `direction` derived from unrounded `val`; `shap_value` stores `round(float(val), 6)`.
  A raw value of +0.0000003 produces `shap_value=0.0` but `direction='increases_risk'` — Pydantic validator raises 500.
  Fix: `direction='increases_risk' if round(float(val), 6) > 0 else 'decreases_risk'` — done.

---

## CRITICAL — Interview Failures (Must Fix Before Interview)

- [x] **[R1] Fix cost_matrix.py test split — uses wrong population**
  Bug: `load_test_set()` in `cost_matrix.py:50-51` uses `df.iloc[split:]` (last 20% by row count).
  `train.py` uses temporal split: test = days 75–90 = last ~16.7% by time.
  The 3-day overlap means ~20k validation-era rows enter the cost analysis, making the £91k saving partially in-sample.
  Fix: import and call `temporal_split()` from `train.py` in `cost_matrix.py` instead of redefining the split.

- [x] **[R2] Implement champion/challenger routing OR remove the claim**
  Bug: `docs/champion_challenger.md` line 106 marks routing (`hash(card_id) % 100`) as a "Phase 3 implementation task" — it does NOT exist in `main.py`. Phase 3 is marked COMPLETE in README.
  Fix option A: Add `x-routing-key` header and traffic-split logic to `main.py`.
  Fix option B: Rename Phase 3 entry in README to "Champion/Challenger *design* complete" and add routing implementation as a new tracked item.

- [x] **[R3] Sync CI smoke test payload with test suite**
  Bug: `.github/workflows/ci.yml` lines 75–93 Docker smoke test payload is missing `card_txn_count_6h`, `card_amount_sum_6h`, `card_txn_count_7d`, `card_amount_sum_7d`. `tests/test_api.py` `_valid_payload()` includes them.
  If schema requires them, CI Docker stage returns 422 and fails silently.
  Fix: Update CI workflow payload to include all 4 fields, matching `_valid_payload()` exactly.

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

## Medium Priority — Interview Partials (Defensible, But Should Fix)

- [x] **[R4] Fix `amount_to_card_avg_ratio` clip — biases against low-spending high-risk cards**
  Bug: `velocity.py:65` clips `card_avg_amount_30d` at £1.0. Gift cards, stolen-card test transactions average well below £1 — the clip suppresses the ratio for the highest-risk segment by up to 3×.
  Fix: Replace `clip(lower=1.0)` with `clip(lower=card_avg_amount_30d.quantile(0.01))` or use log-ratio: `log1p(amount_gbp / clip(avg, lower=0.01))`.

- [x] **[R6] Add runtime invariant for SHAP `direction` field**
  Bug: `main.py:150-155` sets `direction` from `shap_val > 0` but no test verifies the string matches the numeric sign. A sign inversion in future code would silently mislead fraud analysts.
  Fix A: Add test assertion: for each `FeatureContribution`, `shap_value > 0 ↔ direction == 'increases_risk'`.
  Fix B: Add Pydantic `model_validator` on `FeatureContribution` that raises `ValueError` if sign and direction disagree.

- [x] **[R7] Make MLflow fallback visible to monitoring**
  Bug: `main.py:88-90` swallows all MLflow exceptions silently; container starts healthy with stale hardcoded thresholds and no observable signal.
  Fix: Add `thresholds_from_mlflow: bool` to `/health` response. Log at ERROR level before falling back, including exception type and message.

- [x] **[R8] Add OOD flag for unknown merchant categories**
  Bug: `schema.py:46` accepts any string for `merchant_category`. Unknown categories silently map to OrdinalEncoder value -1. Fraud analyst sees a score with no indication the merchant type was out-of-distribution.
  Fix: At startup, load known categories from `feature_cols.json` (already loaded). In `/predict`, check if `merchant_category` is in known set; if not, add `oot_features: ['merchant_category']` to response.

- [x] **[R3-R2] Add log_amount cross-field coherence validator**
  Bug: `log_amount` is sent by caller with no validation that it matches `log1p(amount_gbp)`. A unit bug (pence vs pounds) silently sends contradictory signals.
  Fix: Pydantic `model_validator` on `PredictRequest` asserts `abs(log_amount - log1p(amount_gbp)) < 0.01`. Done in `schema.py`.

- [x] **[R3-R2] Add amount_to_card_avg_ratio cross-field coherence validator**
  Done: `schema.py` `derived_fields_coherent` validator extended to check ratio coherence. Rejects requests where `|supplied_ratio - amount_gbp / max(card_avg_amount_30d, 0.01)|` exceeds 10% relative tolerance. Only applied when `card_avg_amount_30d > 0` to avoid divide-by-zero. All 16 tests pass.

- [x] **[R3-R3] Add synthetic-data caveat to README drift section**
  Bug: README presents PSI=0.0009 as production stability evidence. Both reference and current windows are from the same synthetic dataset — the result is a tautology.
  Fix: Added scope note to README clarifying these are synthetic-data results and describing the production monitoring setup.

- [x] **[R3-R4] Fix MONITORED_FEATURES in drift.py — third incomplete feature list**
  Bug: `drift.py` had a third separate feature list missing `hour_of_day`, `day_of_week`, and the 6h/7d velocity columns.
  Fix: Import `FEATURE_COLS` from `train.py`; derive `MONITORED_FEATURES` by excluding `_SKIP_PSI = {hour_sin, hour_cos, dow_sin, dow_cos}` with explanation.

---

## Medium Priority — Round 4 Battle Partials (New)

- [x] **[R4-R1] Replace `merch_txn_count_1h` with card×merchant interaction feature**
  Done: `velocity.py` — `_merchant_velocity()` replaced with `_card_merchant_velocity()` (groups by `[card_id, merchant_id]`, output `card_merch_txn_count_1h`). FEATURE_COLS updated in `train.py`, `main.py`, `schema.py`, `test_api.py`, `locustfile.py`. Model card updated.
  ⚠️ Requires retrain — existing `models/pipeline.pkl` was trained with `merch_txn_count_1h`. Bundle with R3-R6 retrain.

- [x] **[R4-R2] Correct model card performance monitoring claim**
  Done: `docs/model_card.md` monitoring table now has an "Implemented" column. PR-AUC and FPR rows explicitly marked "No — requires chargeback label pipeline". Monitoring gap section added explaining that performance metrics require 30–90 day label latency and are not implemented in the synthetic data environment.

- [x] **[R4-R3] Add explicit shadow mode promotion gates to champion/challenger design**
  Done: `docs/champion_challenger.md` shadow mode section now includes a four-gate promotion table: ≥10,000 observations, PR-AUC ≥ 0.85 of champion, KL-divergence < 0.10, p99 within 20% of champion. Failure policy (24h extension, two-strike retirement) also documented.

- [x] **[R4-R4] Document and enforce UTC timezone contract on `timestamp_sec`**
  Done: `schema.py` — `hour_of_day` field now has an explicit UTC comment. `amount_gbp` got upper bound `le=25000`. `amount_to_card_avg_ratio` got upper bound `le=2000` (matches training clip=1.0 max; comment flags it needs revision after retrain with clip=0.01).

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
