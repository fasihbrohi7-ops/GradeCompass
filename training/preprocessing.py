import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_ORDER = [
    "gender_female", "gender_male",
    "race_group_A", "race_group_B", "race_group_C", "race_group_D", "race_group_E",
    "parent_edu_some_high_school", "parent_edu_high_school", "parent_edu_some_college",
    "parent_edu_associates_degree", "parent_edu_bachelors_degree", "parent_edu_masters_degree",
    "lunch_standard", "lunch_free_reduced",
    "test_prep_none", "test_prep_completed",
]

PASS_THRESHOLD = 60

def load_and_prepare(csv_path="dataset/StudentsPerformance.csv"):
    if not os.path.exists(csv_path):
        # Allow running from within training/ directory
        alt_path = os.path.join("..", csv_path)
        if os.path.exists(alt_path):
            csv_path = alt_path

    df = pd.read_csv(csv_path)
    assert df.isnull().sum().sum() == 0, "Unexpected missing values in dataset"

    df["average_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
    df["passed"] = (df["average_score"] >= PASS_THRESHOLD).astype(int)

    encoded = pd.DataFrame(0, index=df.index, columns=FEATURE_ORDER)
    encoded["gender_female"] = (df["gender"] == "female").astype(int)
    encoded["gender_male"] = (df["gender"] == "male").astype(int)
    for g in ["A", "B", "C", "D", "E"]:
        encoded[f"race_group_{g}"] = (df["race/ethnicity"] == f"group {g}").astype(int)
    edu_map = {
        "some high school": "parent_edu_some_high_school",
        "high school": "parent_edu_high_school",
        "some college": "parent_edu_some_college",
        "associate's degree": "parent_edu_associates_degree",
        "bachelor's degree": "parent_edu_bachelors_degree",
        "master's degree": "parent_edu_masters_degree",
    }
    for raw, col in edu_map.items():
        encoded[col] = (df["parental level of education"] == raw).astype(int)
    encoded["lunch_standard"] = (df["lunch"] == "standard").astype(int)
    encoded["lunch_free_reduced"] = (df["lunch"] == "free/reduced").astype(int)
    encoded["test_prep_none"] = (df["test preparation course"] == "none").astype(int)
    encoded["test_prep_completed"] = (df["test preparation course"] == "completed").astype(int)

    return encoded, df["average_score"], df["passed"]

def get_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler
