import json
import os
import shutil
import sys
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_and_prepare, get_scaler, FEATURE_ORDER, PASS_THRESHOLD

def merge():
    base_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    web_assets_dir = os.path.join(project_root, "web", "assets")
    os.makedirs(web_assets_dir, exist_ok=True)

    linear_params_file = os.path.join(base_dir, "linear_params.json")
    logistic_params_file = os.path.join(base_dir, "logistic_params.json")

    if not os.path.exists(linear_params_file):
        print("linear_params.json not found. Running train_linear.py...")
        from train_linear import train_linear
        train_linear()

    if not os.path.exists(logistic_params_file):
        print("logistic_params.json not found. Running train_logistic.py...")
        from train_logistic import train_logistic
        train_logistic()

    with open(linear_params_file, "r") as f:
        linear_data = json.load(f)

    with open(logistic_params_file, "r") as f:
        logistic_data = json.load(f)

    # Compute scaler on train split
    X, y_score, y_passed = load_and_prepare()
    X_train, X_test, _, _, _, _ = train_test_split(
        X, y_score, y_passed, test_size=0.2, random_state=42
    )
    scaler = get_scaler(X_train)

    categorical_options = {
        "gender": ["female", "male"],
        "race_ethnicity": ["group A", "group B", "group C", "group D", "group E"],
        "parental_education": [
            "some high school",
            "high school",
            "some college",
            "associate's degree",
            "bachelor's degree",
            "master's degree"
        ],
        "lunch": ["standard", "free/reduced"],
        "test_preparation_course": ["none", "completed"]
    }

    merged = {
        "feature_order": FEATURE_ORDER,
        "categorical_options": categorical_options,
        "scaler": {
            "mean": [float(m) for m in scaler.mean_],
            "std": [float(s) for s in scaler.scale_]
        },
        "linear_regression": {
            "weights": linear_data["weights"],
            "bias": linear_data["bias"],
            "pass_threshold": PASS_THRESHOLD
        },
        "logistic_regression": {
            "weights": logistic_data["weights"],
            "bias": logistic_data["bias"]
        },
        "metadata": {
            "trained_on_rows": len(X),
            "linear_r2": linear_data.get("metrics", {}).get("r2", 0.0),
            "linear_mae": linear_data.get("metrics", {}).get("mae", 0.0),
            "logistic_accuracy": logistic_data.get("metrics", {}).get("accuracy", 0.0),
            "logistic_precision": logistic_data.get("metrics", {}).get("precision", 0.0),
            "logistic_recall": logistic_data.get("metrics", {}).get("recall", 0.0),
            "dataset_source": "Kaggle - Students Performance in Exams"
        }
    }

    output_path = os.path.join(web_assets_dir, "model_params.json")
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Successfully generated merged model params at: {output_path}")

    # Copy metrics markdown to web/assets/
    src_metrics = os.path.join(base_dir, "model_metrics.md")
    if os.path.exists(src_metrics):
        dst_metrics = os.path.join(web_assets_dir, "model_metrics.md")
        shutil.copyfile(src_metrics, dst_metrics)
        print(f"Copied model_metrics.md to {dst_metrics}")

if __name__ == "__main__":
    merge()
