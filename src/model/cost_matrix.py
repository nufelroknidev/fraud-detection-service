"""
Cost-matrix analysis for UK CNP fraud scoring.

Converts model performance into business outcomes by assigning £ costs to
each decision outcome (TP, FP, FN, TN) and finding the threshold that
minimises total expected £ loss rather than maximising F1.

Typical CNP cost assumptions (configurable via CLI or import):
    FN cost  = full transaction amount (fraud not caught)
    FP cost  = £5  (manual review + customer friction / churn risk)
    TP cost  = £2  (review overhead on correctly caught fraud)
    TN cost  = £0  (correct pass-through, no cost)

Run:
    python -m src.model.cost_matrix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).parents[2]
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "transactions_featured.csv"
MODEL_PATH   = PROJECT_ROOT / "models" / "pipeline.pkl"
RESULTS_DIR  = PROJECT_ROOT / "results"

FEATURE_COLS = [
    "amount_gbp", "hour_of_day", "day_of_week", "card_avg_amount_30d",
    "card_txn_count_1h", "card_amount_sum_1h", "card_txn_count_24h",
    "card_amount_sum_24h", "merch_txn_count_1h", "time_since_last_card_txn_sec",
    "amount_to_card_avg_ratio", "log_amount",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "merchant_category",
]
TARGET_COL = "is_fraud"


def load_test_set() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH).sort_values("timestamp_sec").reset_index(drop=True)
    split = int(len(df) * 0.8)
    test = df.iloc[split:]
    return test[FEATURE_COLS], test[TARGET_COL]


def compute_cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    fp_cost: float = 5.0,
    tp_cost: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sweep thresholds and compute total £ cost at each operating point.

    FN cost is variable (= actual transaction amount).
    FP cost is fixed (review + friction).
    TP cost is fixed (review overhead on caught fraud).
    TN cost = 0.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    costs = []

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # FN cost: mean amount of missed frauds * count
        fn_mask  = (y_true == 1) & (y_pred == 0)
        fn_total = amounts[fn_mask].sum()

        fp_total = fp * fp_cost
        tp_total = tp * tp_cost

        costs.append(fn_total + fp_total + tp_total)

    return thresholds, np.array(costs)


def run_cost_analysis(
    fp_cost: float = 5.0,
    tp_cost: float = 2.0,
) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = joblib.load(MODEL_PATH)
    X_test, y_test = load_test_set()

    y_prob   = pipeline.predict_proba(X_test)[:, 1]
    amounts  = X_test["amount_gbp"].values
    y_true   = y_test.values

    thresholds, costs = compute_cost_curve(y_true, y_prob, amounts, fp_cost, tp_cost)

    opt_idx      = np.argmin(costs)
    opt_threshold = float(thresholds[opt_idx])
    opt_cost      = float(costs[opt_idx])

    # Baseline cost: block nothing (all FNs)
    baseline_cost = float(amounts[y_true == 1].sum())

    # Cost at F1-optimal threshold (loaded from model params via simple heuristic)
    f1_thr_approx = 0.9487  # from training run blushing-stag-185
    f1_cost = float(costs[np.argmin(np.abs(thresholds - f1_thr_approx))])

    saving_vs_baseline = baseline_cost - opt_cost
    saving_vs_f1       = f1_cost - opt_cost

    print("\n-- Cost Matrix Analysis -----------------------------------------")
    print(f"Assumptions: FP=£{fp_cost:.0f}/txn, TP=£{tp_cost:.0f}/txn, FN=full amount")
    print(f"Test set    : {len(y_true):,} transactions, {y_true.sum():,} frauds")
    print(f"Fraud total : £{amounts[y_true==1].sum():,.0f}")
    print("-" * 55)
    print(f"Baseline (block nothing) cost : £{baseline_cost:>12,.0f}")
    print(f"F1-optimal threshold cost     : £{f1_cost:>12,.0f}")
    print(f"Cost-optimal threshold        : {opt_threshold:.4f}")
    print(f"Cost-optimal total cost       : £{opt_cost:>12,.0f}")
    print(f"Saving vs baseline            : £{saving_vs_baseline:>12,.0f}")
    print(f"Saving vs F1-threshold        : £{saving_vs_f1:>12,.0f}")
    print("-" * 55 + "\n")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, costs / 1000, color="#d65f5f", linewidth=2, label="Total cost (£k)")
    ax.axvline(opt_threshold, color="#2196F3", linestyle="--", label=f"Cost-optimal thr={opt_threshold:.3f}")
    ax.axvline(f1_thr_approx, color="#FF9800", linestyle="--", label=f"F1-optimal thr={f1_thr_approx:.3f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Total expected cost (£k)")
    ax.set_title("Cost Matrix: Total Expected £ Loss by Decision Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png = RESULTS_DIR / "cost_curve.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")

    summary = {
        "fp_cost_gbp": fp_cost,
        "tp_cost_gbp": tp_cost,
        "baseline_cost_gbp": round(baseline_cost, 2),
        "f1_threshold": f1_thr_approx,
        "f1_cost_gbp": round(f1_cost, 2),
        "cost_optimal_threshold": round(opt_threshold, 4),
        "cost_optimal_total_gbp": round(opt_cost, 2),
        "saving_vs_baseline_gbp": round(saving_vs_baseline, 2),
        "saving_vs_f1_gbp": round(saving_vs_f1, 2),
    }

    out_json = RESULTS_DIR / "cost_matrix_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out_json}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cost-matrix threshold analysis")
    parser.add_argument("--fp-cost", type=float, default=5.0, help="£ cost per false positive")
    parser.add_argument("--tp-cost", type=float, default=2.0, help="£ cost per true positive (review overhead)")
    args = parser.parse_args()
    run_cost_analysis(fp_cost=args.fp_cost, tp_cost=args.tp_cost)
