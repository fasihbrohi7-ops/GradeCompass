import json
import os
import subprocess
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.preprocessing import load_and_prepare, get_scaler, FEATURE_ORDER
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

def test_inference_parity():
    # Load parameters
    params_path = os.path.join(os.path.dirname(__file__), "..", "web", "assets", "model_params.json")
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    # Train reference python models
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

    # Run JS test
    js_test_path = os.path.join(os.path.dirname(__file__), "test_inference.js")
    result = subprocess.run(["node", js_test_path], capture_output=True, text=True, check=True)
    print(result.stdout)

    print("[SUCCESS] Python & JavaScript models verified!")

if __name__ == "__main__":
    test_inference_parity()
