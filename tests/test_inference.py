import json
import os
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.preprocessing import load_and_prepare, get_scaler, FEATURE_ORDER, encode_profile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression


def test_inference_parity():
    print("==================================================")
    print("GradeCompass: Inference Parity & Validation Tests")
    print("==================================================")

    # 1. Load parameters JSON
    params_path = os.path.join(os.path.dirname(__file__), "..", "web", "assets", "model_params.json")
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    # 2. Train reference python models
    X, y_score, y_passed = load_and_prepare()
    X_train, X_test, y_score_train, y_score_test, y_passed_train, y_passed_test = train_test_split(
        X, y_score, y_passed, test_size=0.2, random_state=42
    )
    scaler = get_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_score_train)

    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_scaled, y_passed_train)

    # Test profiles
    test_profiles = [
        {
            "name": "Profile 1 - Standard",
            "data": {
                "gender": "female",
                "race_ethnicity": "group B",
                "parental_education": "bachelor's degree",
                "lunch": "standard",
                "test_preparation_course": "none",
            },
        },
        {
            "name": "Profile 2 - Free Lunch with Prep",
            "data": {
                "gender": "male",
                "race_ethnicity": "group A",
                "parental_education": "some high school",
                "lunch": "free/reduced",
                "test_preparation_course": "completed",
            },
        },
        {
            "name": "Profile 3 - High Achievement",
            "data": {
                "gender": "female",
                "race_ethnicity": "group E",
                "parental_education": "master's degree",
                "lunch": "standard",
                "test_preparation_course": "completed",
            },
        },
        {
            "name": "Profile 4 - At-Risk",
            "data": {
                "gender": "male",
                "race_ethnicity": "group C",
                "parental_education": "high school",
                "lunch": "free/reduced",
                "test_preparation_course": "none",
            },
        },
    ]

    scaler_mean = np.array(params["scaler"]["mean"])
    scaler_std = np.array(params["scaler"]["std"])
    lin_weights = np.array(params["linear_regression"]["weights"])
    lin_bias = params["linear_regression"]["bias"]
    log_weights = np.array(params["logistic_regression"]["weights"])
    log_bias = params["logistic_regression"]["bias"]

    for profile in test_profiles:
        raw_vec = np.array(encode_profile(profile["data"])).reshape(1, -1)
        assert raw_vec.sum() == 5, f"Expected 5 active features, got {raw_vec.sum()}"
        raw_df = pd.DataFrame(raw_vec, columns=FEATURE_ORDER)

        # Reference sklearn predictions
        scaled_sklearn = scaler.transform(raw_df)
        sklearn_score = float(np.clip(lin_reg.predict(scaled_sklearn)[0], 0, 100))
        sklearn_prob = float(log_reg.predict_proba(scaled_sklearn)[0][1])

        # Client-side math simulation
        scaled_manual = (raw_vec - scaler_mean) / scaler_std
        manual_score = float(np.clip(np.dot(scaled_manual, lin_weights)[0] + lin_bias, 0, 100))
        z = float(np.dot(scaled_manual, log_weights)[0] + log_bias)
        manual_prob = 1.0 / (1.0 + np.exp(-z))

        assert abs(sklearn_score - manual_score) < 1e-4, f"Score mismatch: {sklearn_score} vs {manual_score}"
        assert abs(sklearn_prob - manual_prob) < 1e-4, f"Prob mismatch: {sklearn_prob} vs {manual_prob}"

        print(f"[PASS] {profile['name']}:")
        print(f"       Score: {manual_score:.2f} (Sklearn: {sklearn_score:.2f})")
        print(f"       Pass Prob: {manual_prob*100:.2f}% (Sklearn: {sklearn_prob*100:.2f}%)")

    # Optional Node.js verification if Node is installed
    if shutil.which("node"):
        js_test_path = os.path.join(os.path.dirname(__file__), "test_inference.js")
        result = subprocess.run(["node", js_test_path], capture_output=True, text=True, check=True)
        print("\nNode.js Test Output:")
        print(result.stdout)
    else:
        print("\n[NOTE] Node.js not detected in system PATH. Pure Python mathematical verification succeeded.")

    print("\n[SUCCESS] Model parity and client inference verified!")


if __name__ == "__main__":
    test_inference_parity()
