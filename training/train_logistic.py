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
        "weights": [float(w) for w in clf.coef_[0]],
        "bias": float(clf.intercept_[0]),
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "confusion_matrix": cm,
            "test_samples": len(y_passed_test)
        }
    }

    base_dir = os.path.dirname(__file__)
    params_path = os.path.join(base_dir, "logistic_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved logistic parameters to {params_path}")

    # Append to model_metrics.md
    metrics_md_path = os.path.join(base_dir, "model_metrics.md")
    content = (
        f"## Logistic Regression Model (Classification — Abdul Hayy)\n\n"
        f"- **Target**: `passed` (1 if `average_score >= {PASS_THRESHOLD}` else 0)\n"
        f"- **Algorithm**: `LogisticRegression(random_state=42)`\n"
        f"- **Features**: 17 one-hot encoded demographic features (StandardScaled)\n"
        f"- **Test Set Size**: {len(y_passed_test)} samples (20% split)\n\n"
        f"### Performance Metrics\n"
        f"| Metric | Score |\n"
        f"|---|---|\n"
        f"| **Accuracy** | {acc:.4f} ({acc*100:.2f}%) |\n"
        f"| **Precision** | {prec:.4f} ({prec*100:.2f}%) |\n"
        f"| **Recall** | {rec:.4f} ({rec*100:.2f}%) |\n\n"
        f"### Confusion Matrix\n"
        f"```text\n"
        f"                Predicted Fail    Predicted Pass\n"
        f"Actual Fail:        {cm[0][0]:<15}   {cm[0][1]:<15}\n"
        f"Actual Pass:        {cm[1][0]:<15}   {cm[1][1]:<15}\n"
        f"```\n"
    )

    if os.path.exists(metrics_md_path):
        with open(metrics_md_path, "r", encoding="utf-8") as f:
            existing = f.read()
        if "## Logistic Regression Model" in existing:
            # Replace existing section
            parts = existing.split("## Logistic Regression Model")
            updated = parts[0] + content
        else:
            updated = existing + "\n\n" + content
    else:
        updated = "# GradeCompass Model Evaluation Metrics\n\n" + content

    with open(metrics_md_path, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"Updated {metrics_md_path}")

    return params

if __name__ == "__main__":
    train_logistic()
