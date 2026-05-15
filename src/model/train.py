"""
XGBoost training script for UK CNP fraud detection.

Run:  python src/model/train.py

Produces a fitted sklearn Pipeline (OrdinalEncoder + XGBClassifier) logged to
DagsHub MLflow with params, metrics, and artefacts.  Four adaptive checkpoints
guard against unreliable splits, overfit, and unacceptable precision.
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless — no display server required on Windows/CI
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from src.tracking import init_mlflow

# ---------------------------------------------------------------------------
# Constants — single source of truth referenced by every downstream step
# ---------------------------------------------------------------------------

PROJECT_ROOT  = Path(__file__).parents[2]
DATA_PATH     = PROJECT_ROOT / "data" / "processed" / "transactions_featured.csv"
MODEL_DIR     = PROJECT_ROOT / "models"
MODEL_PATH    = MODEL_DIR / "pipeline.pkl"
EXPERIMENT    = "cnp-fraud-xgboost"

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
TARGET_COL = "is_fraud"
CAT_COLS   = ["merchant_category"]


# ---------------------------------------------------------------------------
# A. Data loading
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {path.name}: {missing}\n"
            "Re-run src/features/velocity.py to regenerate the processed CSV."
        )
    return df


# ---------------------------------------------------------------------------
# B. Temporal split
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """3-way time-based split using proportional day boundaries.

    Proportional fractions (not hardcoded seconds) so the split survives
    dataset regeneration with a different n_days parameter.
    """
    max_ts    = df["timestamp_sec"].max()
    train_end = max_ts * (67 / 90)
    val_end   = max_ts * (75 / 90)

    train = df[df["timestamp_sec"] < train_end].copy()
    val   = df[(df["timestamp_sec"] >= train_end) & (df["timestamp_sec"] < val_end)].copy()
    test  = df[df["timestamp_sec"] >= val_end].copy()
    return train, val, test


def _assert_fraud_counts(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict[str, int]:
    counts = {
        "train": int(train[TARGET_COL].sum()),
        "val":   int(val[TARGET_COL].sum()),
        "test":  int(test[TARGET_COL].sum()),
    }
    if counts["val"] < 100:
        raise RuntimeError(
            f"CHECKPOINT 1 FAIL: Validation set has only {counts['val']} fraud cases "
            f"(need ≥100 for reliable early stopping).\n"
            "Fix: shift train_end to day 60 — change 67/90 → 60/90 in temporal_split()."
        )
    if counts["test"] < 200:
        raise RuntimeError(
            f"CHECKPOINT 1 FAIL: Test set has only {counts['test']} fraud cases "
            f"(need ≥200 for reliable threshold calibration).\n"
            "Fix: shift val_end to day 70 — change 75/90 → 70/90 in temporal_split()."
        )
    return counts


# ---------------------------------------------------------------------------
# C. Feature matrices
# ---------------------------------------------------------------------------

def build_matrices(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple:
    def _xy(df: pd.DataFrame):
        return df[FEATURE_COLS].copy(), df[TARGET_COL].values

    X_train, y_train = _xy(train)
    X_val,   y_val   = _xy(val)
    X_test,  y_test  = _xy(test)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# D. Class imbalance
# ---------------------------------------------------------------------------

def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    if n_pos == 0:
        raise RuntimeError("Training set contains zero fraud cases.")
    spw = n_neg / n_pos
    print(f"  scale_pos_weight = {spw:.1f}  (n_neg={n_neg:,}, n_pos={n_pos:,})")
    return float(spw)


# ---------------------------------------------------------------------------
# E. Pipeline
# ---------------------------------------------------------------------------

def build_pipeline(scale_pos_weight: float) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,   # unseen merchant category at inference → -1, no crash
            ), CAT_COLS),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,  # keeps names clean: no cat__/remainder__ prefixes
    )

    classifier = XGBClassifier(
        objective             = "binary:logistic",
        eval_metric           = "aucpr",   # PR-AUC for early stopping, not log-loss
        tree_method           = "hist",
        scale_pos_weight      = scale_pos_weight,
        n_estimators          = 1000,      # high ceiling; early stopping controls actual count
        learning_rate         = 0.05,
        max_depth             = 6,
        min_child_weight      = 10,        # raised from default 1: prevents micro-cluster overfit
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        gamma                 = 1.0,
        early_stopping_rounds = 50,
        random_state          = 42,
        n_jobs                = -1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   classifier),
    ])


# ---------------------------------------------------------------------------
# F. Training
# ---------------------------------------------------------------------------

def train_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val:   pd.DataFrame, y_val:   np.ndarray,
) -> Pipeline:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier   = pipeline.named_steps["classifier"]

    # XGBoost only sees post-encoder data; eval_set must also be pre-transformed.
    # Fitting preprocessor separately here keeps the Pipeline object fully fitted
    # (both named_steps share references, so pipeline.predict_proba works after this).
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc   = preprocessor.transform(X_val)

    classifier.fit(
        X_train_enc, y_train,
        eval_set=[(X_val_enc, y_val)],
        verbose=100,
    )
    return pipeline


def _inspect_learning_curve(classifier: XGBClassifier) -> None:
    best  = classifier.best_iteration
    score = classifier.best_score
    print(f"  Best round : {best} / 1000  |  Best val PR-AUC : {score:.4f}")

    if best < 50:
        print("  WARNING (CP2): Stopped at round <50. Val set may be too small/noisy.")
    elif best >= 950:
        print("  WARNING (CP2): Hit n_estimators ceiling. Raise to 2000 and retrain.")


# ---------------------------------------------------------------------------
# G. Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict:
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    pr_auc  = float(average_precision_score(y_test, y_proba))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    gini    = float(2 * roc_auc - 1)

    fraud_scores = y_proba[y_test == 1]
    legit_scores = y_proba[y_test == 0]
    ks_stat, _   = ks_2samp(fraud_scores, legit_scores)

    p_curve, r_curve, thresholds = precision_recall_curve(y_test, y_proba)
    # precision_recall_curve returns arrays of length n+1; thresholds has length n.
    # Trim the last point (precision=1, recall=0, no threshold) before indexing.
    p_use = p_curve[:-1]
    r_use = r_curve[:-1]

    f1_vals  = 2 * p_use * r_use / (p_use + r_use + 1e-9)
    best_idx = int(f1_vals.argmax())
    threshold_f1 = float(thresholds[best_idx])

    # Highest threshold where recall is still >= 80% (most selective operating point)
    candidates = np.where(r_use >= 0.80)[0]
    if len(candidates) == 0:
        raise RuntimeError(
            "CHECKPOINT 3 FAIL: Model never achieves 80% recall on the test set.\n"
            "Check: scale_pos_weight from train-only? timestamp_sec absent from features?"
        )
    threshold_80r = float(thresholds[candidates[-1]])

    def _metrics_at(thr: float, label: str) -> dict:
        y_pred = (y_proba >= thr).astype(int)
        return {
            f"{label}_threshold": thr,
            f"{label}_precision":  float(precision_score(y_test, y_pred, zero_division=0)),
            f"{label}_recall":     float(recall_score(y_test, y_pred)),
            f"{label}_f1":         float(f1_score(y_test, y_pred, zero_division=0)),
            f"{label}_conf_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    return {
        "pr_auc":     pr_auc,
        "roc_auc":    roc_auc,
        "gini":       gini,
        "ks_stat":    float(ks_stat),
        "pr_curve":   (p_curve, r_curve, thresholds),
        **_metrics_at(threshold_f1,  "f1_opt"),
        **_metrics_at(threshold_80r, "recall80"),
    }


def _assert_minimum_precision(results: dict, n_fraud_test: int, n_test: int) -> None:
    base_rate = n_fraud_test / n_test

    # F1-optimal is the balanced operating point — hard minimum of 10%.
    p_f1 = results["f1_opt_precision"]
    if p_f1 < 0.10:
        raise RuntimeError(
            f"CHECKPOINT 3 FAIL: F1-opt precision = {p_f1:.3f} < 0.10 minimum.\n"
            "Model is not production-usable. Diagnose:\n"
            "  1. Is timestamp_sec in FEATURE_COLS? (it must not be)\n"
            "  2. Was scale_pos_weight computed from the full dataset? (use train-only)\n"
            "  3. Check feature importances — are velocity features non-zero?"
        )

    # 80%-recall is a high-sensitivity operating point; low precision is expected.
    # Minimum: 3× base rate (i.e., the model is at least 3× better than random at this recall).
    p_80r    = results["recall80_precision"]
    min_80r  = 3 * base_rate
    if p_80r < min_80r:
        raise RuntimeError(
            f"CHECKPOINT 3 FAIL: Recall80 precision = {p_80r:.4f} < 3× base rate "
            f"({min_80r:.4f}). Model is not lifting above random at 80% recall.\n"
            "This is a strong signal of model failure — check feature matrix."
        )
    if p_80r < 0.05:
        print(f"  NOTE: Recall80 precision = {p_80r:.3f} — low but above 3× base rate "
              f"({min_80r:.4f}). Expected for synthetic data with limited velocity signal.")


# ---------------------------------------------------------------------------
# H. Feature importance
# ---------------------------------------------------------------------------

def compute_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    classifier   = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    names        = list(preprocessor.get_feature_names_out())

    df = pd.DataFrame({"feature": names, "importance": classifier.feature_importances_})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    zero_imp = df[df["importance"] == 0]["feature"].tolist()
    if zero_imp:
        print(f"  WARNING (CP4): Zero-importance features: {zero_imp}")
        print("  Safe to keep for now — defer removal to Phase 3 permutation analysis.")
    return df


# ---------------------------------------------------------------------------
# I. MLflow logging
# ---------------------------------------------------------------------------

def log_to_mlflow(
    pipeline:       Pipeline,
    results:        dict,
    importance_df:  pd.DataFrame,
    split_counts:   dict[str, int],
    fraud_counts:   dict[str, int],
    X_train_sample: pd.DataFrame,
) -> None:
    import sklearn

    classifier = pipeline.named_steps["classifier"]
    xgb_params = classifier.get_params()

    mlflow.log_params({
        "n_estimators_ceiling":  1000,
        "best_iteration":        classifier.best_iteration,
        "learning_rate":         xgb_params["learning_rate"],
        "max_depth":             xgb_params["max_depth"],
        "min_child_weight":      xgb_params["min_child_weight"],
        "subsample":             xgb_params["subsample"],
        "colsample_bytree":      xgb_params["colsample_bytree"],
        "gamma":                 xgb_params["gamma"],
        "scale_pos_weight":      round(xgb_params["scale_pos_weight"], 2),
        "early_stopping_rounds": xgb_params["early_stopping_rounds"],
        "n_features":            len(FEATURE_COLS),
        "n_train":               split_counts["train"],
        "n_val":                 split_counts["val"],
        "n_test":                split_counts["test"],
        "n_fraud_train":         fraud_counts["train"],
        "n_fraud_val":           fraud_counts["val"],
        "n_fraud_test":          fraud_counts["test"],
        "threshold_f1_opt":      round(results["f1_opt_threshold"], 6),
        "threshold_recall80":    round(results["recall80_threshold"], 6),
        "sklearn_version":       sklearn.__version__,
    })

    mlflow.log_metrics({
        "pr_auc":             results["pr_auc"],
        "roc_auc":            results["roc_auc"],
        "gini":               results["gini"],
        "ks_stat":            results["ks_stat"],
        "f1_opt_precision":   results["f1_opt_precision"],
        "f1_opt_recall":      results["f1_opt_recall"],
        "f1_opt_f1":          results["f1_opt_f1"],
        "recall80_precision": results["recall80_precision"],
        "recall80_recall":    results["recall80_recall"],
        "recall80_f1":        results["recall80_f1"],
    })

    # Artifact: feature column list (schema contract for the inference API)
    mlflow.log_text(json.dumps(FEATURE_COLS, indent=2), "feature_cols.json")

    # Artifact: PR curve with both threshold markers
    p_curve, r_curve, _ = results["pr_curve"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r_curve, p_curve, lw=2, label=f"PR-AUC = {results['pr_auc']:.4f}")
    ax.axvline(
        results["f1_opt_recall"], color="steelblue", linestyle="--",
        label=f"F1-opt  (recall={results['f1_opt_recall']:.2f}, "
              f"prec={results['f1_opt_precision']:.2f})",
    )
    ax.axvline(
        results["recall80_recall"], color="tomato", linestyle="--",
        label=f"80%-recall (prec={results['recall80_precision']:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Test Set")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    mlflow.log_figure(fig, "pr_curve.png")
    plt.close(fig)

    # Artifact: top-20 feature importances
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    top = importance_df.head(20)
    ax2.barh(top["feature"][::-1], top["importance"][::-1], color="steelblue")
    ax2.set_xlabel("Gain Importance")
    ax2.set_title("Top-20 Feature Importances")
    ax2.grid(axis="x", alpha=0.3)
    mlflow.log_figure(fig2, "feature_importance.png")
    plt.close(fig2)

    # Artifact: fitted Pipeline (encoder state + booster in one object)
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="pipeline",
        input_example=X_train_sample,
    )

    # Local copy — used by the API (DagsHub free tier drops large pickle uploads)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    mlflow.log_artifact(str(MODEL_PATH), artifact_path="local")


# ---------------------------------------------------------------------------
# J. Stdout summary
# ---------------------------------------------------------------------------

def _print_summary(
    results:      dict,
    split_counts: dict[str, int],
    fraud_counts: dict[str, int],
    best_iter:    int,
) -> None:
    w   = 62
    sep = "=" * w
    div = "-" * w
    print(f"\n{sep}")
    print("CNP FRAUD DETECTION — TRAINING COMPLETE".center(w))
    print(sep)
    print(f"  Train / Val / Test     : "
          f"{split_counts['train']:,} / {split_counts['val']:,} / {split_counts['test']:,}")
    print(f"  Fraud (train/val/test) : "
          f"{fraud_counts['train']:,} / {fraud_counts['val']:,} / {fraud_counts['test']:,}")
    print(f"  Best XGBoost round     : {best_iter} / 1000")
    print(div)
    print(f"  PR-AUC  (test)         : {results['pr_auc']:.4f}")
    print(f"  ROC-AUC (test)         : {results['roc_auc']:.4f}")
    print(f"  Gini    (test)         : {results['gini']:.4f}  (target > 0.60)")
    print(f"  KS stat (test)         : {results['ks_stat']:.4f}")
    print(div)
    for label, name in (("f1_opt", "F1-optimal"), ("recall80", "80%-recall")):
        print(f"  {name} threshold  : {results[f'{label}_threshold']:.4f}")
        print(f"    Precision          : {results[f'{label}_precision']:.4f}")
        print(f"    Recall             : {results[f'{label}_recall']:.4f}")
        print(f"    F1                 : {results[f'{label}_f1']:.4f}")
    print(div)
    print("  MLflow run logged to DagsHub.")
    print("  Artifacts: pipeline/, feature_cols.json,")
    print("             pr_curve.png, feature_importance.png")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# K. Main
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # prevent MLflow emoji crash on CP1252

    init_mlflow(EXPERIMENT)

    with mlflow.start_run():
        print("\n[1/8] Loading data...")
        df = load_data(DATA_PATH)

        print("[2/8] Splitting (temporal: days 0-67 / 67-75 / 75-90)...")
        train_df, val_df, test_df = temporal_split(df)
        split_counts = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
        fraud_counts = _assert_fraud_counts(train_df, val_df, test_df)
        print(f"  Rows  — train: {split_counts['train']:,}  "
              f"val: {split_counts['val']:,}  test: {split_counts['test']:,}")
        print(f"  Fraud — train: {fraud_counts['train']:,}  "
              f"val: {fraud_counts['val']:,}  test: {fraud_counts['test']:,}")

        print("[3/8] Building feature matrices...")
        X_train, y_train, X_val, y_val, X_test, y_test = build_matrices(
            train_df, val_df, test_df
        )

        print("[4/8] Computing class weight...")
        spw = compute_scale_pos_weight(y_train)

        print("[5/8] Building pipeline...")
        pipeline = build_pipeline(spw)

        print("[6/8] Training with early stopping on val PR-AUC...")
        pipeline = train_pipeline(pipeline, X_train, y_train, X_val, y_val)
        _inspect_learning_curve(pipeline.named_steps["classifier"])

        print("[7/8] Evaluating on held-out test set...")
        results = evaluate(pipeline, X_test, y_test)
        _assert_minimum_precision(results, fraud_counts["test"], split_counts["test"])

        print("[8/8] Logging to MLflow / DagsHub...")
        importance_df = compute_feature_importance(pipeline)
        log_to_mlflow(
            pipeline        = pipeline,
            results         = results,
            importance_df   = importance_df,
            split_counts    = split_counts,
            fraud_counts    = fraud_counts,
            X_train_sample  = X_train.iloc[:5],
        )

        _print_summary(
            results, split_counts, fraud_counts,
            pipeline.named_steps["classifier"].best_iteration,
        )


if __name__ == "__main__":
    main()
