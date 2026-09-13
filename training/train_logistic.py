"""
GradeCompass - Logistic Regression Training & Evaluation
Owner: Abdul Hayy

Trains Logistic Regression binary classification model predicting whether a student
is likely to pass (average exam score >= 60) based on demographic attributes.
Strictly consumes the shared contract in training/preprocessing.py (Fasih).
"""

import json
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Ensure script directory is in sys.path
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_and_prepare, get_scaler, FEATURE_ORDER, PASS_THRESHOLD

def train_logistic():
    print("Loading and preparing dataset for Logistic Regression...")
    X, y_score, y_passed = load_and_prepare()

    X_train, X_test, y_score_train, y_score_test, y_passed_train, y_passed_test = train_test_split(
        X, y_score, y_passed, test_size=0.2, random_state=42
    )

    scaler = get_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_scaled, y_passed_train)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    acc = float(accuracy_score(y_passed_test, y_pred))
    prec = float(precision_score(y_passed_test, y_pred))
    rec = float(recall_score(y_passed_test, y_pred))
    cm = confusion_matrix(y_passed_test, y_pred).tolist()

    print(f"--- Logistic Regression Evaluation (Abdul Hayy) ---")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"Confusion Matrix [TN, FP], [FN, TP]: {cm}")

    params = {
        "logistic_regression": {
            "weights": [float(w) for w in clf.coef_[0]],
            "bias": float(clf.intercept_[0])
        },
        "metadata": {
            "logistic_accuracy": acc,
            "logistic_precision": prec,
            "logistic_recall": rec,
            "confusion_matrix": cm,
            "test_samples": len(y_passed_test)
        }
    }

    base_dir = os.path.dirname(__file__)
    params_path = os.path.join(base_dir, "logistic_params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Saved logistic parameters to {params_path}")

    # Format Section 2 for model_metrics.md
    metrics_md_path = os.path.join(base_dir, "model_metrics.md")
    section_content = (
        f"## 2. Logistic Regression (Pass / At-Risk Classification)\n"
        f"- **Owner:** Abdul Hayy\n"
        f"- **Target:** Binary pass indicator (`passed = 1 if average_score >= {PASS_THRESHOLD} else 0`)\n"
        f"- **Algorithm:** `LogisticRegression(random_state=42)`\n"
        f"- **Features:** 17 one-hot encoded demographic features (`StandardScaled`)\n"
        f"- **Test Set Size:** {len(y_passed_test)} samples (20% split)\n\n"
        f"### 2.1 Performance Metrics\n"
        f"| Metric | Score | Interpretation |\n"
        f"|---|---|---|\n"
        f"| **Accuracy** | **{acc:.4f}** ({acc*100:.2f}%) | {int(acc * len(y_passed_test))} out of {len(y_passed_test)} test cases classified correctly. |\n"
        f"| **Precision** | **{prec:.4f}** ({prec*100:.2f}%) | High reliability when predicting that a student will pass. |\n"
        f"| **Recall** | **{rec:.4f}** ({rec*100:.2f}%) | Identifies ~{rec*100:.1f}% of all students who achieved a passing score. |\n\n"
        f"### 2.2 Confusion Matrix (Test Set: {len(y_passed_test)} Students)\n"
        f"```text\n"
        f"                Predicted Fail    Predicted Pass\n"
        f"Actual Fail:    {cm[0][0]:<17} {cm[0][1]:<17}\n"
        f"Actual Pass:    {cm[1][0]:<17} {cm[1][1]:<17}\n"
        f"```\n\n"
        f"---\n"
    )

    if os.path.exists(metrics_md_path):
        with open(metrics_md_path, "r", encoding="utf-8") as f:
            existing = f.read()
        if "## 2. Logistic Regression" in existing:
            parts = existing.split("## 2. Logistic Regression")
            updated = parts[0] + section_content
        elif "## Logistic Regression" in existing:
            parts = existing.split("## Logistic Regression")
            updated = parts[0] + section_content
        else:
            updated = existing.strip() + "\n\n---\n\n" + section_content
    else:
        updated = "# GradeCompass - Model Evaluation Metrics\n\n" + section_content

    with open(metrics_md_path, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"Updated {metrics_md_path}")

    return params

if __name__ == "__main__":
    train_logistic()
