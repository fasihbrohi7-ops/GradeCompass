# GradeCompass - Model Evaluation Metrics

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
| **R² Score** | 0.2543 | **0.1622** | Demographics explain ~16.2% of variance in average exam scores. Aligns with TRD expectations (0.15–0.30). |
| **Mean Absolute Error (MAE)** | 9.94 points | **10.49 points** | On average, predictions are within ~10.5 points of the actual exam score. |
| **Root Mean Squared Error (RMSE)** | — | **13.40 points** | Penalizes larger errors; confirms absence of severe outliers on the test set. |
| **Base Intercept (Bias)** | — | **68.1692** | Mean predicted score for a student at the dataset mean across all features. |

### 1.2 Feature Weights (Standardized Coefficients)

Weights are sorted by magnitude of effect on the predicted exam score:

| Feature Name | Weight (Scaled) | Direction |
|---|---|---|
| `lunch_free_reduced` | -2.1904 | Negative |
| `lunch_standard` | +2.1904 | Positive |
| `test_prep_completed` | +1.8772 | Positive |
| `test_prep_none` | -1.8772 | Negative |
| `parent_edu_bachelors_degree` | +1.4903 | Positive |
| `parent_edu_high_school` | -1.4258 | Negative |
| `race_group_E` | +1.3479 | Positive |
| `gender_female` | +1.0216 | Positive |
| `gender_male` | -1.0216 | Negative |
| `race_group_B` | -0.8935 | Negative |
| `parent_edu_some_high_school` | -0.8246 | Negative |
| `race_group_D` | +0.7163 | Positive |
| `parent_edu_masters_degree` | +0.7084 | Positive |
| `race_group_A` | -0.5985 | Negative |
| `race_group_C` | -0.5687 | Negative |
| `parent_edu_associates_degree` | +0.4518 | Positive |
| `parent_edu_some_college` | +0.0964 | Positive |

### 1.3 Key Insights from Linear Coefficients
1. **Lunch Type (Socioeconomic indicator):** `lunch_standard` has a substantial positive weight (+2.19) whereas `lunch_free_reduced` is (-2.19), reflecting the strongest correlation in this dataset.
2. **Test Preparation:** Completing the test prep course contributes +1.88 scaled score points compared to no preparation (-1.88).
3. **Parental Education:** Higher parental education levels (Bachelor's, Master's, Associate's) correlate positively with performance, whereas high school or some high school levels correlate negatively.
4. **Race / Ethnicity:** Group E shows a positive correlation (+1.35), while Groups A, B, and C show negative coefficients relative to the sample mean.

---

## 2. Logistic Regression (Pass / At-Risk Classification)
- **Owner:** Abdul Hayy
- **Target:** Binary pass indicator (`passed = 1 if average_score >= 60 else 0`)
- **Algorithm:** `LogisticRegression(random_state=42)`
- **Features:** 17 one-hot encoded demographic features (`StandardScaled`)
- **Test Set Size:** 200 samples (20% split)

### 2.1 Performance Metrics
| Metric | Score | Interpretation |
|---|---|---|
| **Accuracy** | **0.6800** (68.00%) | 136 out of 200 test cases classified correctly. |
| **Precision** | **0.7229** (72.29%) | High reliability when predicting that a student will pass. |
| **Recall** | **0.8696** (86.96%) | Identifies ~87% of all students who achieved a passing score. |

### 2.2 Confusion Matrix (Test Set: 200 Students)
```text
                Predicted Fail    Predicted Pass
Actual Fail:    16                46
Actual Pass:    18                120
```

---
