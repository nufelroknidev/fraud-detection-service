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
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from src.api.schema import PredictRequest, PredictResponse
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
    "card_txn_count_24h",
    "card_amount_sum_24h",
    "merch_txn_count_1h",
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


# Thresholds from training run blushing-stag-185 — used as fallback when
# MLflow tracking is unavailable (CI, offline development).
_FALLBACK_THRESHOLD_F1_OPT   = 0.9487
_FALLBACK_THRESHOLD_RECALL80 = 0.2529


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

    # Fetch thresholds from MLflow if a run ID is available; fall back to
    # the committed values from the training run otherwise.
    if run_id:
        try:
            client = mlflow.tracking.MlflowClient()
            params = client.get_run(run_id).data.params
            _state["threshold_f1_opt"]   = float(params["threshold_f1_opt"])
            _state["threshold_recall80"] = float(params["threshold_recall80"])
        except Exception:
            _state["threshold_f1_opt"]   = _FALLBACK_THRESHOLD_F1_OPT
            _state["threshold_recall80"] = _FALLBACK_THRESHOLD_RECALL80
    else:
        _state["threshold_f1_opt"]   = _FALLBACK_THRESHOLD_F1_OPT
        _state["threshold_recall80"] = _FALLBACK_THRESHOLD_RECALL80

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
def predict(request: PredictRequest):
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([request.model_dump()])[FEATURE_COLS]
    prob = float(pipeline.predict_proba(df)[0, 1])

    thr_f1  = _state["threshold_f1_opt"]
    thr_80r = _state["threshold_recall80"]

    return PredictResponse(
        fraud_probability=round(prob, 6),
        f1_opt_decision="BLOCK"  if prob >= thr_f1  else "PASS",
        recall80_decision="REVIEW" if prob >= thr_80r else "PASS",
        f1_opt_threshold=thr_f1,
        recall80_threshold=thr_80r,
    )
