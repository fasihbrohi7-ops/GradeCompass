"""
GradeCompass - Model Parameter Merging Utility
Owner: Fasih

Merges linear regression parameters (Fasih) and logistic regression parameters (Abdul Hayy)
into the single client-side runtime payload required by the web application:
  web/assets/model_params.json
and syncs:
  web/assets/model_metrics.md

Validates the merged payload against PRD-TRD Section 3.3 schema.
"""

import json
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR = BASE_DIR / "training"
WEB_ASSETS_DIR = BASE_DIR / "web" / "assets"


def merge_model_parameters():
    print("==================================================")
    print("GradeCompass: Model Parameters Merge (Fasih)")
    print("==================================================")

    linear_params_path = TRAINING_DIR / "linear_params.json"
    logistic_params_path = TRAINING_DIR / "logistic_params.json"

    # 1. Load Linear Regression Parameters (Fasih)
    if not linear_params_path.exists():
        print("[!] linear_params.json not found. Running training/train_linear.py...")
        from train_linear import train_linear_model

        linear_data = train_linear_model()
    else:
        with open(linear_params_path, "r", encoding="utf-8") as f:
            linear_data = json.load(f)
        print(f"[OK] Loaded Linear Regression parameters from {linear_params_path}")

    # 2. Check / Load Logistic Regression Parameters (Abdul Hayy)
    logistic_weights = None
    logistic_bias = 0.0
    logistic_accuracy = 0.0
    logistic_precision = 0.0
    logistic_recall = 0.0

    if logistic_params_path.exists():
        with open(logistic_params_path, "r", encoding="utf-8") as f:
            logistic_data = json.load(f)
        logistic_weights = logistic_data.get("logistic_regression", {}).get("weights")
        logistic_bias = float(logistic_data.get("logistic_regression", {}).get("bias", 0.0))
        meta = logistic_data.get("metadata", {})
        logistic_accuracy = float(meta.get("logistic_accuracy", 0.0))
        logistic_precision = float(meta.get("logistic_precision", 0.0))
        logistic_recall = float(meta.get("logistic_recall", 0.0))
        print(f"[OK] Loaded Logistic Regression parameters from {logistic_params_path}")
    else:
        print("[INFO] logistic_params.json not found yet (Abdul Hayy's model pending).")
        print("       Populating schema-compliant placeholder for logistic regression so UI can run.")
        logistic_weights = [0.0] * 17
        logistic_bias = 0.0

    # 3. Assemble final merged payload according to Section 3.3
    feature_order = linear_data["feature_order"]
    categorical_options = linear_data["categorical_options"]
    scaler = linear_data["scaler"]
    linear_reg = linear_data["linear_regression"]
    linear_meta = linear_data.get("metadata", {})

    merged_payload = {
        "feature_order": feature_order,
        "categorical_options": categorical_options,
        "scaler": {
            "mean": scaler["mean"],
            "std": scaler["std"],
        },
        "linear_regression": {
            "weights": linear_reg["weights"],
            "bias": float(linear_reg["bias"]),
            "pass_threshold": int(linear_reg.get("pass_threshold", 60)),
        },
        "logistic_regression": {
            "weights": logistic_weights,
            "bias": float(logistic_bias),
        },
        "metadata": {
            "trained_on_rows": int(linear_meta.get("trained_on_rows", 1000)),
            "linear_r2": float(linear_meta.get("linear_r2", 0.0)),
            "linear_mae": float(linear_meta.get("linear_mae", 0.0)),
            "logistic_accuracy": float(logistic_accuracy),
            "logistic_precision": float(logistic_precision),
            "logistic_recall": float(logistic_recall),
            "dataset_source": "Kaggle - Students Performance in Exams",
        },
    }

    # 4. Validate Schema
    assert len(merged_payload["feature_order"]) == 17, "feature_order length must be 17"
    assert len(merged_payload["scaler"]["mean"]) == 17, "scaler.mean length must be 17"
    assert len(merged_payload["scaler"]["std"]) == 17, "scaler.std length must be 17"
    assert len(merged_payload["linear_regression"]["weights"]) == 17, "linear weights length must be 17"
    assert len(merged_payload["logistic_regression"]["weights"]) == 17, "logistic weights length must be 17"
    print("[OK] Schema validation passed: All 17 feature weights and scaler parameters match.")

    # 5. Write to web/assets/model_params.json
    WEB_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out_model_path = WEB_ASSETS_DIR / "model_params.json"
    with open(out_model_path, "w", encoding="utf-8") as f:
        json.dump(merged_payload, f, indent=2)
    print(f"[OK] Successfully wrote merged params to {out_model_path}")

    # 6. Copy training/model_metrics.md to web/assets/model_metrics.md
    metrics_src = TRAINING_DIR / "model_metrics.md"
    if metrics_src.exists():
        metrics_dst = WEB_ASSETS_DIR / "model_metrics.md"
        shutil.copy2(metrics_src, metrics_dst)
        print(f"[OK] Synced model_metrics.md to {metrics_dst}")

    print("==================================================")
    print("Merge completed successfully!")
    print("==================================================")


if __name__ == "__main__":
    merge_model_parameters()
