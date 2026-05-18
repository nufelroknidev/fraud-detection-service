"""
Feature drift monitor for the UK CNP fraud scoring service.

Uses Evidently 0.7+ to compute PSI (Population Stability Index) for each
model feature, comparing a reference window (training data) against a
current window (recent production traffic).

PSI thresholds (industry standard for financial services):
    < 0.1   — stable, no action
    0.1–0.2 — investigate feature source
    > 0.2   — trigger retraining pipeline

Usage:
    python -m src.monitoring.drift

    Or import and call run_drift_report() programmatically from a scheduler.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from evidently import Dataset, Report
from evidently.metrics import DriftedColumnsCount, ValueDrift

from src.model.train import FEATURE_COLS

PROJECT_ROOT = Path(__file__).parents[2]
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "transactions_featured.csv"
REPORT_DIR   = PROJECT_ROOT / "results" / "drift"
MODEL_PATH   = PROJECT_ROOT / "models" / "pipeline.pkl"

# FEATURE_COLS (from train.py) used for prediction score computation.
_SCORE_FEATURE_COLS = FEATURE_COLS

PSI_INVESTIGATE = 0.10
PSI_RETRAIN     = 0.20

# PSI is a weak signal for bounded periodic features — their distributions always
# span [-1, 1] regardless of fraud pattern shift. Monitor raw time fields instead.
_SKIP_PSI = {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}
MONITORED_FEATURES = [c for c in FEATURE_COLS if c not in _SKIP_PSI]


def _load_windows(
    df: pd.DataFrame,
    ref_frac: float = 0.6,
    cur_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split: first ref_frac of data = reference (training distribution),
    last cur_frac = current window (simulates recent production traffic).
    """
    df = df.sort_values("timestamp_sec").reset_index(drop=True)
    n = len(df)
    ref_end = int(n * ref_frac)
    cur_start = int(n * (1 - cur_frac))
    return df.iloc[:ref_end], df.iloc[cur_start:]


def run_drift_report(
    data_path: Path = DATA_PATH,
    report_dir: Path = REPORT_DIR,
    save_html: bool = True,
    save_json: bool = True,
) -> dict:
    """
    Compute per-feature PSI between reference and current windows.

    Returns a dict with:
        - per_feature: {feature: {"psi": float, "status": str}}
        - drifted_count: int
        - retrain_triggered: bool
        - worst_feature: str
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    ref_df, cur_df = _load_windows(df)

    print(f"Reference window : {len(ref_df):,} rows")
    print(f"Current window   : {len(cur_df):,} rows")

    # Optionally add model output score as a monitored column
    score_available = False
    if MODEL_PATH.exists():
        try:
            pipeline = joblib.load(MODEL_PATH)
            ref_df = ref_df.copy()
            cur_df = cur_df.copy()
            ref_df["prediction_score"] = pipeline.predict_proba(ref_df[_SCORE_FEATURE_COLS])[:, 1]
            cur_df["prediction_score"] = pipeline.predict_proba(cur_df[_SCORE_FEATURE_COLS])[:, 1]
            score_available = True
            print("Score monitoring : enabled (prediction_score added)")
        except Exception as e:
            print(f"Score monitoring : skipped — {e}")

    monitor_cols = MONITORED_FEATURES + (["prediction_score"] if score_available else [])
    ref_ds = Dataset.from_pandas(ref_df[monitor_cols])
    cur_ds = Dataset.from_pandas(cur_df[monitor_cols])

    metrics = [ValueDrift(column=col, method="psi") for col in monitor_cols]
    metrics.append(DriftedColumnsCount(method="psi", threshold=PSI_INVESTIGATE))

    report = Report(metrics=metrics)
    result = report.run(reference_data=ref_ds, current_data=cur_ds)

    # ── Parse results ──────────────────────────────────────────────────────
    raw = result.dict()["metrics"]

    per_feature: dict[str, dict] = {}
    drifted_count = 0

    for entry in raw:
        metric_name = entry.get("metric_name", "")
        value       = entry.get("value")

        # ValueDrift entries: metric_name = "ValueDrift(column=X,method=psi,...)"
        if "ValueDrift" in metric_name and isinstance(value, (int, float)):
            # Extract column name from metric_name string
            col = None
            for part in metric_name.split(","):
                if part.strip().startswith("column=") or "column=" in part:
                    col = part.split("column=")[-1].split(",")[0].strip().rstrip(")")
                    break
            if not col:
                continue
            psi = float(value)
            if psi >= PSI_RETRAIN:
                status = "RETRAIN"
                drifted_count += 1
            elif psi >= PSI_INVESTIGATE:
                status = "INVESTIGATE"
                drifted_count += 1
            else:
                status = "STABLE"
            per_feature[col] = {"psi": round(psi, 4), "status": status}

    worst = max(per_feature, key=lambda c: per_feature[c]["psi"]) if per_feature else None
    retrain_triggered = any(v["status"] == "RETRAIN" for v in per_feature.values())

    summary = {
        "per_feature": per_feature,
        "drifted_count": drifted_count,
        "retrain_triggered": retrain_triggered,
        "worst_feature": worst,
        "worst_psi": per_feature[worst]["psi"] if worst else None,
    }

    # ── Console output ─────────────────────────────────────────────────────
    print("\n-- PSI Drift Report ---------------------------------------------")
    print(f"{'Feature':<35} {'PSI':>7}  Status")
    print("-" * 55)
    for col, info in sorted(per_feature.items(), key=lambda x: -x[1]["psi"]):
        flag = "!" if info["status"] == "RETRAIN" else ("~" if info["status"] == "INVESTIGATE" else " ")
        print(f"{flag} {col:<33} {info['psi']:>7.4f}  {info['status']}")
    print("-" * 55)
    print(f"Features with drift : {drifted_count}/{len(per_feature)}")
    print(f"Retrain triggered   : {'YES' if retrain_triggered else 'no'}")
    print(f"Worst feature       : {worst} (PSI={summary['worst_psi']})")
    print("-" * 55 + "\n")

    # ── Persist outputs ────────────────────────────────────────────────────
    if save_json:
        out_json = report_dir / "drift_summary.json"
        out_json.write_text(json.dumps(summary, indent=2))
        print(f"Saved: {out_json}")

    if save_html:
        out_html = report_dir / "drift_report.html"
        result.save_html(str(out_html))
        print(f"Saved: {out_html}")

    return summary


if __name__ == "__main__":
    run_drift_report()
