# GradeCompass Model Evaluation Metrics

Generated from `dataset/StudentsPerformance.csv` (1000 samples, 80/20 train/test split, `random_state=42`).

## Linear Regression Model (Regression — Fasih)

- **Target**: `average_score` (Continuous 0–100: mean of math, reading, and writing scores)
- **Algorithm**: `LinearRegression()`
- **Features**: 17 one-hot encoded demographic features (StandardScaled)
- **Test Set Size**: 200 samples (20% split)

### Performance Metrics
| Metric | Score |
|---|---|
| **R² Score** | 0.1622 |
| **MAE (Mean Absolute Error)** | 10.4902 |

> **Note on R²**: Demographics-only models on this dataset typically yield modest R² (~0.15–0.30) as academic performance depends heavily on unobserved individual factors (study hours, motivation, course difficulty). This reflects realistic statistical correlation rather than deterministic causation.


## Logistic Regression Model (Classification — Abdul Hayy)

- **Target**: `passed` (1 if `average_score >= 60` else 0)
- **Algorithm**: `LogisticRegression(random_state=42)`
- **Features**: 17 one-hot encoded demographic features (StandardScaled)
- **Test Set Size**: 200 samples (20% split)

### Performance Metrics
| Metric | Score |
|---|---|
| **Accuracy** | 0.6800 (68.00%) |
| **Precision** | 0.7229 (72.29%) |
| **Recall** | 0.8696 (86.96%) |

### Confusion Matrix
```text
                Predicted Fail    Predicted Pass
Actual Fail:        16                46             
Actual Pass:        18                120            
```
