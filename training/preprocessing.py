"""
GradeCompass - Shared Preprocessing Pipeline
Owner: Fasih (Foundation shared with Abdul Hayy)

This module implements the single source of truth for dataset loading, target computation,
categorical feature one-hot encoding, and feature scaling, following Section 3 of the PRD-TRD.
Both train_linear.py (Fasih) and train_logistic.py (Abdul Hayy) consume this contract.
"""

import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Fixed 17-feature order as defined in PRD-TRD Section 3.2.
FEATURE_ORDER = [
    "gender_female", "gender_male",
    "race_group_A", "race_group_B", "race_group_C", "race_group_D", "race_group_E",
    "parent_edu_some_high_school", "parent_edu_high_school", "parent_edu_some_college",
    "parent_edu_associates_degree", "parent_edu_bachelors_degree", "parent_edu_masters_degree",
    "lunch_standard", "lunch_free_reduced",
    "test_prep_none", "test_prep_completed",
]

# Pass threshold constant for classification
PASS_THRESHOLD = 60

# Categorical options mapping for UI dropdown generation & schema validation
CATEGORICAL_OPTIONS = {
    "gender": ["female", "male"],
    "race_ethnicity": ["group A", "group B", "group C", "group D", "group E"],
    "parental_education": [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree",
    ],
    "lunch": ["standard", "free/reduced"],
    "test_preparation_course": ["none", "completed"],
}

PARENT_EDU_MAP = {
    "some high school": "parent_edu_some_high_school",
    "high school": "parent_edu_high_school",
    "some college": "parent_edu_some_college",
    "associate's degree": "parent_edu_associates_degree",
    "bachelor's degree": "parent_edu_bachelors_degree",
    "master's degree": "parent_edu_masters_degree",
}


def resolve_dataset_path(csv_path: str = "dataset/StudentsPerformance.csv") -> str:
    """Resolve the dataset path whether running from repository root or training/ directory."""
    candidates = [
        Path(csv_path),
        Path(__file__).resolve().parent.parent / csv_path,
        Path(__file__).resolve().parent.parent / "dataset" / "StudentsPerformance.csv",
        Path("dataset/StudentsPerformance.csv"),
        Path("../dataset/StudentsPerformance.csv"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(f"Could not locate StudentsPerformance.csv at {csv_path} or default locations.")


def load_and_prepare(csv_path: str = "dataset/StudentsPerformance.csv"):
    """
    Load the Kaggle StudentsPerformance dataset and prepare features and targets.
    
    Returns:
        X (pd.DataFrame): 17 one-hot encoded binary features in exact FEATURE_ORDER.
        y_score (pd.Series): Continuous average exam score (0-100).
        y_passed (pd.Series): Binary pass indicator (1 if average_score >= 60 else 0).
    """
    resolved_path = resolve_dataset_path(csv_path)
    df = pd.read_csv(resolved_path)

    # Requirement F1: assert no missing values
    assert df.isnull().sum().sum() == 0, "Unexpected missing values in dataset"

    # Requirement F2: average score across math, reading, writing
    df["average_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

    # Requirement F3: binary target passed
    df["passed"] = (df["average_score"] >= PASS_THRESHOLD).astype(int)

    # Requirement F4: one-hot encode the 5 categorical features in exact FEATURE_ORDER
    encoded = pd.DataFrame(0, index=df.index, columns=FEATURE_ORDER)

    encoded["gender_female"] = (df["gender"] == "female").astype(int)
    encoded["gender_male"] = (df["gender"] == "male").astype(int)

    for g in ["A", "B", "C", "D", "E"]:
        encoded[f"race_group_{g}"] = (df["race/ethnicity"] == f"group {g}").astype(int)

    for raw_label, col_name in PARENT_EDU_MAP.items():
        encoded[col_name] = (df["parental level of education"] == raw_label).astype(int)

    encoded["lunch_standard"] = (df["lunch"] == "standard").astype(int)
    encoded["lunch_free_reduced"] = (df["lunch"] == "free/reduced").astype(int)

    encoded["test_prep_none"] = (df["test preparation course"] == "none").astype(int)
    encoded["test_prep_completed"] = (df["test preparation course"] == "completed").astype(int)

    # Verify column count and order matches contract
    assert list(encoded.columns) == FEATURE_ORDER, "Encoded feature columns do not match FEATURE_ORDER"
    assert encoded.shape[1] == 17, f"Expected 17 features, got {encoded.shape[1]}"

    return encoded, df["average_score"], df["passed"]


def get_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """
    Fit StandardScaler on the one-hot encoded training features (Requirement F6).
    
    Returns:
        StandardScaler: Fitted scaler instance with mean_ and scale_ for 17 features.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def encode_profile(profile: dict) -> list:
    """
    Encode a single profile dictionary into the 17-element binary feature list.
    
    Args:
        profile: dict with keys gender, race_ethnicity, parental_education, lunch, test_preparation_course.
    Returns:
        list of 17 ints (0 or 1) in FEATURE_ORDER.
    """
    vector = [0] * len(FEATURE_ORDER)
    col_idx = {col: i for i, col in enumerate(FEATURE_ORDER)}

    if "gender" in profile:
        g = profile["gender"]
        if f"gender_{g}" in col_idx:
            vector[col_idx[f"gender_{g}"]] = 1

    if "race_ethnicity" in profile:
        race = profile["race_ethnicity"].replace("group ", "")
        if f"race_group_{race}" in col_idx:
            vector[col_idx[f"race_group_{race}"]] = 1

    if "parental_education" in profile:
        raw_edu = profile["parental_education"]
        col_name = PARENT_EDU_MAP.get(raw_edu)
        if col_name and col_name in col_idx:
            vector[col_idx[col_name]] = 1

    if "lunch" in profile:
        l = profile["lunch"].replace("/", "_")
        if f"lunch_{l}" in col_idx:
            vector[col_idx[f"lunch_{l}"]] = 1

    if "test_preparation_course" in profile:
        tp = profile["test_preparation_course"]
        if f"test_prep_{tp}" in col_idx:
            vector[col_idx[f"test_prep_{tp}"]] = 1

    return vector


if __name__ == "__main__":
    print("--- Testing Shared Preprocessing Pipeline ---")
    X, y_score, y_passed = load_and_prepare()
    print(f"Loaded dataset with {len(X)} rows.")
    print(f"Feature matrix shape: {X.shape} (Expected: (1000, 17))")
    print(f"Target score range: [{y_score.min():.1f}, {y_score.max():.1f}], Mean: {y_score.mean():.2f}")
    print(f"Target passed balance: Passed={y_passed.sum()} ({y_passed.mean()*100:.1f}%), Failed={len(y_passed) - y_passed.sum()}")
    
    scaler = get_scaler(X)
    print(f"Scaler mean vector length: {len(scaler.mean_)}")
    print(f"Scaler std vector length: {len(scaler.scale_)}")
    print("All preprocessing checks passed successfully!")
