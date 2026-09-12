import json
import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_and_prepare, get_scaler, FEATURE_ORDER, PASS_THRESHOLD

def train_linear():
    print("Loading and preparing dataset for Linear Regression...")
    X, y_score, y_passed = load_and_prepare()

    X_train, X_test, y_score_train, y_score_test, y_passed_train, y_passed_test = train_test_split(
        X, y_score, y_passed, test_size=0.2, random_state=42
    )

    scaler = get_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Linear Regression
    reg = LinearRegression()
    reg.fit(X_train_scaled, y_score_train)

    # Evaluate
    y_pred = reg.predict(X_test_scaled)
    r2 = float(r2_score(y_score_test, y_pred))
    mae = float(mean_absolute_error(y_score_test, y_pred))

    print(f"--- Linear Regression Evaluation (Fasih) ---")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE:      {mae:.4f}")

    params = {
        "weights": [float(w) for w in reg.coef_],
        "bias": float(reg.intercept_),
        "pass_threshold": PASS_THRESHOLD,
        "metrics": {
            "r2": r2,
            "mae": mae,
            "test_samples": len(y_score_test)
        }
    }

    base_dir = os.path.dirname(__file__)
    params_path = os.path.join(base_dir, "linear_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved linear parameters to {params_path}")

    # Write to model_metrics.md
    metrics_md_path = os.path.join(base_dir, "model_metrics.md")
    content = (
        f"# GradeCompass Model Evaluation Metrics\n\n"
        f"Generated from `dataset/StudentsPerformance.csv` (1000 samples, 80/20 train/test split, `random_state=42`).\n\n"
        f"## Linear Regression Model (Regression — Fasih)\n\n"
        f"- **Target**: `average_score` (Continuous 0–100: mean of math, reading, and writing scores)\n"
        f"- **Algorithm**: `LinearRegression()`\n"
        f"- **Features**: 17 one-hot encoded demographic features (StandardScaled)\n"
        f"- **Test Set Size**: {len(y_score_test)} samples (20% split)\n\n"
        f"### Performance Metrics\n"
        f"| Metric | Score |\n"
        f"|---|---|\n"
        f"| **R² Score** | {r2:.4f} |\n"
        f"| **MAE (Mean Absolute Error)** | {mae:.4f} |\n\n"
        f"> **Note on R²**: Demographics-only models on this dataset typically yield modest R² (~0.15–0.30) as academic performance depends heavily on unobserved individual factors (study hours, motivation, course difficulty). This reflects realistic statistical correlation rather than deterministic causation.\n"
    )

    if os.path.exists(metrics_md_path):
        with open(metrics_md_path, "r", encoding="utf-8") as f:
            existing = f.read()
        if "## Logistic Regression Model" in existing:
            # Preserve logistic part
            logistic_part = existing[existing.find("## Logistic Regression Model"):]
            updated = content + "\n\n" + logistic_part
        else:
            updated = content
    else:
        updated = content

    with open(metrics_md_path, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"Updated {metrics_md_path}")

    return params

if __name__ == "__main__":
    train_linear()
