"""
FastAPI scoring service for UK CNP fraud detection.

Run:
    uvicorn src.api.main:app --reload

Requires MLFLOW_RUN_ID in .env (copy from DagsHub → Experiments → run page).
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import mlflow.sklearn
import pandas as pd
import shap
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response

from src.api.schema import FeatureContribution, PredictRequest, PredictResponse
from src.tracking import init_mlflow

load_dotenv()

PROJECT_ROOT = Path(__file__).parents[2]
LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "pipeline.pkl"

# Mirrors FEATURE_COLS in src/model/train.py — must stay in sync with training schema.
FEATURE_COLS = [
    "amount_gbp",
    "hour_of_day",
    "day_of_week",
    "card_avg_amount_30d",
    "card_txn_count_1h",
    "card_amount_sum_1h",
    "card_txn_count_6h",
    "card_amount_sum_6h",
    "card_txn_count_24h",
    "card_amount_sum_24h",
    "card_txn_count_7d",
    "card_amount_sum_7d",
    "card_merch_txn_count_1h",
    "time_since_last_card_txn_sec",
    "amount_to_card_avg_ratio",
    "log_amount",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "merchant_category",
]

_state: dict = {}


# Thresholds from the post-retrain run (clip=0.01, card_merch_txn_count_1h feature).
# Update these whenever a new training run completes and MLflow is unavailable.
_FALLBACK_THRESHOLD_F1_OPT   = 0.8820
_FALLBACK_THRESHOLD_RECALL80 = 0.1013


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_id = os.environ.get("MLFLOW_RUN_ID")

    init_mlflow("cnp-fraud-xgboost")

    if LOCAL_MODEL_PATH.exists():
        _state["pipeline"] = joblib.load(LOCAL_MODEL_PATH)
    elif run_id:
        _state["pipeline"] = mlflow.sklearn.load_model(f"runs:/{run_id}/pipeline")
    else:
        raise RuntimeError("No model found: set MLFLOW_RUN_ID or provide models/pipeline.pkl")

    pipeline = _state["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    _state["explainer"] = shap.TreeExplainer(pipeline.named_steps["classifier"])
    _state["feature_names"] = list(preprocessor.get_feature_names_out())
    # Known merchant categories from training — used to flag OOD requests.
    _state["known_merchant_categories"] = set(
        preprocessor.named_transformers_["cat"].categories_[0]
    )

    # MODEL_ROLE identifies this container in champion/challenger traffic splits.
    # Set via env var: MODEL_ROLE=champion (default) or MODEL_ROLE=challenger.
    _state["model_role"] = os.environ.get("MODEL_ROLE", "champion")

    # Fetch thresholds from MLflow if a run ID is available; fall back to
    # the committed constants from the training run otherwise.
    if run_id:
        try:
            client = mlflow.tracking.MlflowClient()
            params = client.get_run(run_id).data.params
            _state["threshold_f1_opt"]   = float(params["threshold_f1_opt"])
            _state["threshold_recall80"] = float(params["threshold_recall80"])
            _state["thresholds_from_mlflow"] = True
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error(
                "MLflow threshold fetch failed (%s: %s) — using hardcoded fallbacks.",
                type(exc).__name__, exc,
            )
            _state["threshold_f1_opt"]   = _FALLBACK_THRESHOLD_F1_OPT
            _state["threshold_recall80"] = _FALLBACK_THRESHOLD_RECALL80
            _state["thresholds_from_mlflow"] = False
    else:
        _state["threshold_f1_opt"]   = _FALLBACK_THRESHOLD_F1_OPT
        _state["threshold_recall80"] = _FALLBACK_THRESHOLD_RECALL80
        _state["thresholds_from_mlflow"] = False

    yield
    _state.clear()


app = FastAPI(
    title="CNP Fraud Detection API",
    description="Scores card-not-present transactions for fraud probability.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "pipeline" in _state,
        "model_role": _state.get("model_role", "unknown"),
        "thresholds_from_mlflow": _state.get("thresholds_from_mlflow", False),
    }


@app.get("/metrics")
def metrics():
    """Exposes the active decision thresholds for observability / dashboards."""
    return {
        "threshold_f1_opt":   _state.get("threshold_f1_opt"),
        "threshold_recall80": _state.get("threshold_recall80"),
        "model_loaded":       "pipeline" in _state,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, response: Response):
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([request.model_dump()])[FEATURE_COLS]

    # Run preprocessor once; feed encoded array to both classifier and explainer.
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier   = pipeline.named_steps["classifier"]
    X_enc = preprocessor.transform(df)
    prob  = float(classifier.predict_proba(X_enc)[0, 1])

    sv = _state["explainer"].shap_values(X_enc)
    if isinstance(sv, list):  # older shap returns [class0, class1]
        sv = sv[1]
    shap_row = sv[0]

    ranked = sorted(
        zip(_state["feature_names"], shap_row.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:3]

    top_features = [
        FeatureContribution(
            feature=name,
            shap_value=round(float(val), 6),
            direction="increases_risk" if round(float(val), 6) > 0 else "decreases_risk",
        )
        for name, val in ranked
    ]

    oot_features = []
    if request.merchant_category not in _state.get("known_merchant_categories", set()):
        oot_features.append("merchant_category")

    thr_f1  = _state["threshold_f1_opt"]
    thr_80r = _state["threshold_recall80"]

    response.headers["x-model-role"] = _state.get("model_role", "champion")

    return PredictResponse(
        fraud_probability=round(prob, 6),
        f1_opt_decision="BLOCK"  if prob >= thr_f1  else "PASS",
        recall80_decision="REVIEW" if prob >= thr_80r else "PASS",
        f1_opt_threshold=thr_f1,
        recall80_threshold=thr_80r,
        top_features=top_features,
        oot_features=oot_features,
    )
