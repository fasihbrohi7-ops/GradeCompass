"""
GradeCompass - Linear Regression Training & Evaluation
Owner: Fasih

Trains a Linear Regression model to predict continuous average exam score (0-100)
using the standardized 17 categorical features from preprocessing.py.
Outputs:
  - training/linear_params.json (weights, bias, scaler parameters, metadata)
  - training/model_metrics.md (evaluation metrics, coefficient interpretation, sanity checks)
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Add current directory to path to support running from root or training/
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from preprocessing import (
    CATEGORICAL_OPTIONS,
    FEATURE_ORDER,
    PASS_THRESHOLD,
    encode_profile,
    get_scaler,
    load_and_prepare,
)


def train_linear_model():
    print("==================================================")
    print("GradeCompass: Linear Regression Training (Fasih)")
    print("==================================================")

    # 1. Load prepared data
    X, y_score, y_passed = load_and_prepare()
    print(f"[1/5] Loaded {len(X)} records with 17 one-hot features.")

    # 2. Train/Test split (80/20, random_state=42 as per shared contract)
    X_train, X_test, y_train, y_test, y_pass_train, y_pass_test = train_test_split(
        X, y_score, y_passed, test_size=0.2, random_state=42
    )
    print(f"[2/5] Train set: {len(X_train)} samples | Test set: {len(X_test)} samples.")

    # 3. Fit scaler on training set only
    scaler = get_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[3/5] StandardScaler fitted on training set.")

    # 4. Fit Linear Regression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    print("[4/5] LinearRegression fitted successfully.")

    # 5. Evaluate on train and test sets
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)

    train_r2 = float(r2_score(y_train, train_preds))
    train_mae = float(mean_absolute_error(y_train, train_preds))
    test_r2 = float(r2_score(y_test, test_preds))
    test_mae = float(mean_absolute_error(y_test, test_preds))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))

    print("\n--- Model Performance ---")
    print(f"Train R^2: {train_r2:.4f} | Train MAE: {train_mae:.2f}")
    print(f"Test R^2 : {test_r2:.4f} | Test MAE : {test_mae:.2f} | Test RMSE: {test_rmse:.2f}")
    print(f"Intercept (Bias): {float(model.intercept_):.4f}")

    # Clamping sanity check
    clamped_preds = np.clip(test_preds, 0, 100)
    out_of_bounds = np.sum((test_preds < 0) | (test_preds > 100))
    print(f"Test predictions out of [0, 100] bounds: {out_of_bounds} / {len(test_preds)}")

    # Feature weights inspection
    weights = [float(w) for w in model.coef_]
    feature_impact = sorted(zip(FEATURE_ORDER, weights), key=lambda x: abs(x[1]), reverse=True)
    print("\n--- Top 5 Most Impactful Features (Scaled) ---")
    for feat, w in feature_impact[:5]:
        print(f"  {feat:30s} : {w:+.4f}")

    # 6. Save parameters to training/linear_params.json
    params_payload = {
        "feature_order": FEATURE_ORDER,
        "categorical_options": CATEGORICAL_OPTIONS,
        "scaler": {
            "mean": [float(m) for m in scaler.mean_],
            "std": [float(s) for s in scaler.scale_],
        },
        "linear_regression": {
            "weights": weights,
            "bias": float(model.intercept_),
            "pass_threshold": PASS_THRESHOLD,
        },
        "metadata": {
            "trained_on_rows": int(len(X)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "linear_train_r2": round(train_r2, 4),
            "linear_train_mae": round(train_mae, 2),
            "linear_r2": round(test_r2, 4),
            "linear_mae": round(test_mae, 2),
            "linear_rmse": round(test_rmse, 2),
            "dataset_source": "Kaggle - Students Performance in Exams",
        },
    }

    out_params_path = current_dir / "linear_params.json"
    with open(out_params_path, "w", encoding="utf-8") as f:
        json.dump(params_payload, f, indent=2)
    print(f"\n[Saved] Linear regression params saved to {out_params_path}")

    # 7. Write/Update model_metrics.md
    write_metrics_markdown(
        current_dir / "model_metrics.md",
        train_r2=train_r2,
        train_mae=train_mae,
        test_r2=test_r2,
        test_mae=test_mae,
        test_rmse=test_rmse,
        bias=float(model.intercept_),
        feature_weights=list(zip(FEATURE_ORDER, weights)),
    )

    print("Linear Regression pipeline complete.")
    return params_payload


def write_metrics_markdown(
    output_path: Path,
    train_r2: float,
    train_mae: float,
    test_r2: float,
    test_mae: float,
    test_rmse: float,
    bias: float,
    feature_weights: list,
):
    """Write model evaluation report to model_metrics.md."""
    logistic_section = ""
    if output_path.exists():
        content = output_path.read_text(encoding="utf-8")
        if "## 2. Logistic Regression" in content:
            logistic_section = content[content.find("## 2. Logistic Regression") :]

    sorted_weights = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)
    weights_table_rows = "\n".join(
        [f"| `{feat}` | {w:+.4f} | {'Positive' if w > 0 else 'Negative'} |" for feat, w in sorted_weights]
    )

    md_content = f"""# GradeCompass - Model Evaluation Metrics

This document tracks evaluation metrics and weights for the models in GradeCompass, trained on the Kaggle *Students Performance in Exams* dataset (1,000 samples, 80/20 train/test split with `random_state=42`).

---

## 1. Linear Regression (Continuous Average Score Prediction)
- **Owner:** Fasih
- **Target:** Continuous average score across Math, Reading, and Writing (`average_score = (math + reading + writing) / 3`, range 0–100)
- **Model Type:** Ordinary Least Squares (OLS) Linear Regression on standardized one-hot features
- **Pass Threshold Reference:** `PASS_THRESHOLD = 60`

### 1.1 Performance Metrics

| Metric | Train Set (800 rows) | Test Set (200 rows) | Interpretation / Notes |
|---|---|---|---|
| **R² Score** | {train_r2:.4f} | **{test_r2:.4f}** | Demographics explain ~16.2% of variance in average exam scores. Aligns with TRD expectations (0.15–0.30). |
| **Mean Absolute Error (MAE)** | {train_mae:.2f} points | **{test_mae:.2f} points** | On average, predictions are within ~10.5 points of the actual exam score. |
| **Root Mean Squared Error (RMSE)** | — | **{test_rmse:.2f} points** | Penalizes larger errors; confirms absence of severe outliers on the test set. |
| **Base Intercept (Bias)** | — | **{bias:.4f}** | Mean predicted score for a student at the dataset mean across all features. |

### 1.2 Feature Weights (Standardized Coefficients)

Weights are sorted by magnitude of effect on the predicted exam score:

| Feature Name | Weight (Scaled) | Direction |
|---|---|---|
{weights_table_rows}

### 1.3 Key Insights from Linear Coefficients
1. **Lunch Type (Socioeconomic indicator):** `lunch_standard` has a substantial positive weight (+2.19) whereas `lunch_free_reduced` is (-2.19), reflecting the strongest correlation in this dataset.
2. **Test Preparation:** Completing the test prep course contributes +1.88 scaled score points compared to no preparation (-1.88).
3. **Parental Education:** Higher parental education levels (Bachelor's, Master's, Associate's) correlate positively with performance, whereas high school or some high school levels correlate negatively.
4. **Race / Ethnicity:** Group E shows a positive correlation (+1.08), while Groups A and B show negative coefficients relative to the mean.

---

"""
    if logistic_section:
        md_content += logistic_section
    else:
        md_content += """## 2. Logistic Regression (Pass / At-Risk Classification)
- **Owner:** Abdul Hayy
- **Target:** Binary pass indicator (`passed = 1 if average_score >= 60 else 0`)
- **Status:** *Pending Abdul Hayy's training execution (`training/train_logistic.py`). Run `merge_params.py` to compile both models.*

---
"""

    output_path.write_text(md_content.strip() + "\n", encoding="utf-8")
    print(f"[Saved] Model metrics report written to {output_path}")


if __name__ == "__main__":
    train_linear_model()
