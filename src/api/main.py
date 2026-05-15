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


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        raise RuntimeError("MLFLOW_RUN_ID is not set in .env")

    init_mlflow("cnp-fraud-xgboost")

    if LOCAL_MODEL_PATH.exists():
        _state["pipeline"] = joblib.load(LOCAL_MODEL_PATH)
    else:
        _state["pipeline"] = mlflow.sklearn.load_model(f"runs:/{run_id}/pipeline")

    client = mlflow.tracking.MlflowClient()
    params = client.get_run(run_id).data.params
    _state["threshold_f1_opt"]   = float(params["threshold_f1_opt"])
    _state["threshold_recall80"] = float(params["threshold_recall80"])

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
