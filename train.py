"""Train and evaluate models for the BankMind term-deposit prediction task.

Steps (Track B):
  1. Load data and inspect class distribution / missing values.
  2. Encode categoricals.
  3. Train a Logistic Regression baseline and a Random Forest main model.
  4. Evaluate both (accuracy, precision, recall, F1, classification_report).
  5. Report Random Forest feature importance.
  6. Print 5 readable sample predictions from the test set.
  7. Save the Random Forest (plus encoders/defaults/metrics) to model/model.pkl.

Run:  python train.py
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bankmind.preprocessing import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    MODEL_PATH,
    TARGET,
    build_category_maps,
    build_defaults,
    encode_frame,
    load_raw,
)

RANDOM_STATE = 42


def evaluate(name: str, y_true, y_pred) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")
    print(classification_report(y_true, y_pred, target_names=["no", "yes"]))
    return metrics


def main() -> None:
    # ---- 1. Load + inspect -------------------------------------------------
    df = load_raw()
    print(f"Shape: {df.shape}")
    print("\nMissing values per column:")
    print(df.isnull().sum())

    counts = df[TARGET].value_counts()
    yes_pct = 100 * counts.get("yes", 0) / len(df)
    print("\nClass distribution (target = y):")
    print(counts)
    print(f"'yes' = {yes_pct:.2f}% of customers (class imbalance)")

    # ---- 2. Encode ---------------------------------------------------------
    category_maps = build_category_maps(df)
    defaults = build_defaults(df)
    X = encode_frame(df, category_maps)
    y = (df[TARGET] == "yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ---- 3. Models ---------------------------------------------------------
    # Logistic Regression benefits from scaled inputs; tree models do not.
    scaler = StandardScaler().fit(X_train)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(scaler.transform(X_train), y_train)
    y_pred_lr = lr.predict(scaler.transform(X_test))

    # max_depth/min_samples_leaf keep the serialized model small (and curb
    # overfitting); the custom class weight up-weights the rare "yes" class
    # enough to lift recall without collapsing precision (plain "balanced" over-
    # corrects and tanks precision on this dataset).
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=4,
        class_weight={0: 1, 1: 3},
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # ---- 4. Evaluate -------------------------------------------------------
    metrics_lr = evaluate("Logistic Regression (baseline)", y_test, y_pred_lr)
    metrics_rf = evaluate("Random Forest (main model)", y_test, y_pred_rf)

    # ---- 5. Feature importance --------------------------------------------
    importances = (
        pd.Series(rf.feature_importances_, index=FEATURE_COLS)
        .sort_values(ascending=False)
    )
    print("\nRandom Forest feature importance:")
    print(importances.round(4))

    # ---- 6. Sample predictions (>=2 yes, >=2 no) --------------------------
    proba = rf.predict_proba(X_test)[:, 1]
    results = pd.DataFrame(
        {"proba": proba, "pred": y_pred_rf, "actual": y_test.values},
        index=X_test.index,
    )
    yes_idx = results[results["pred"] == 1].head(2).index
    no_idx = results[results["pred"] == 0].head(2).index
    extra_idx = results.drop(list(yes_idx) + list(no_idx)).head(1).index
    sample_idx = list(yes_idx) + list(no_idx) + list(extra_idx)

    print("\n===== 5 sample predictions =====")
    raw = df.loc[sample_idx]
    show_cols = ["age", "job", "marital", "balance", "housing", "loan", "duration"]
    for i, idx in enumerate(sample_idx, 1):
        p = results.loc[idx, "proba"]
        pred = "YES" if results.loc[idx, "pred"] == 1 else "NO"
        actual = "yes" if results.loc[idx, "actual"] == 1 else "no"
        feats = ", ".join(f"{c}={raw.loc[idx, c]}" for c in show_cols)
        print(f"\nCustomer {i}  [{feats}]")
        print(f"   Prediction = {pred}  (probability = {p:.2f}, actual = {actual})")

    # ---- 7. Persist --------------------------------------------------------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": rf,
        "model_name": "Random Forest",
        "feature_cols": FEATURE_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "category_maps": category_maps,
        "defaults": defaults,
        "feature_importances": importances.to_dict(),
        "metrics": {"logistic_regression": metrics_lr, "random_forest": metrics_rf},
        "yes_rate": float(y.mean()),
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)
    print(f"\nSaved model artifact -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
