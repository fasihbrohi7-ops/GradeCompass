# GradeCompass 🎓🧭

> Demographic & Background-based Student Exam Performance Predictor

GradeCompass is a lightweight machine learning application that predicts student exam outcomes from background attributes using models trained on the public Kaggle **"Students Performance in Exams"** dataset (1,000 students).

---

## 👥 Team & Responsibilities

| Contributor | Area & Responsibilities |
|---|---|
| **Fasih** | Shared dataset preprocessing pipeline, Linear Regression model (predicting continuous average score), model evaluation & metrics, merging pipeline (`merge_params.py`), Vercel deployment configuration. |
| **Abdul Hayy** | Logistic Regression model (predicting pass/fail likelihood), frontend UI/UX (vanilla HTML/CSS/JS), browser inference layer, dynamic bar chart, UI usage documentation. |

---

## 📊 Dataset & Targets

- **Source:** Kaggle — *Students Performance in Exams* (1,000 samples, 0 missing values).
- **Features (Inputs):** Exclusively 5 categorical attributes:
  1. `gender`: `female`, `male`
  2. `race/ethnicity`: `group A`, `group B`, `group C`, `group D`, `group E`
  3. `parental level of education`: `some high school`, `high school`, `some college`, `associate's degree`, `bachelor's degree`, `master's degree`
  4. `lunch`: `standard`, `free/reduced` (socioeconomic proxy)
  5. `test preparation course`: `none`, `completed`
- **Target Definitions:**
  - `average_score = (math score + reading score + writing score) / 3` (Continuous 0–100, predicted by **Linear Regression**)
  - `passed = 1 if average_score >= 60 else 0` (Binary classification, predicted by **Logistic Regression**)

> *Note: Individual subject scores are used exclusively to compute the target labels and are never used as model inputs.*

---

## ⚙️ Shared Data Contract (17 Features)

Both models strictly adhere to the exact 17-feature one-hot encoded vector order defined in `training/preprocessing.py`:

```text
 1. gender_female
 2. gender_male
 3. race_group_A
 4. race_group_B
 5. race_group_C
 6. race_group_D
 7. race_group_E
 8. parent_edu_some_high_school
 9. parent_edu_high_school
10. parent_edu_some_college
11. parent_edu_associates_degree
12. parent_edu_bachelors_degree
13. parent_edu_masters_degree
14. lunch_standard
15. lunch_free_reduced
16. test_prep_none
17. test_prep_completed
```

All features are standardized with a single shared `StandardScaler` fitted exclusively on the 80% training split (`random_state=42`).

---

## 🚀 Setup & Training Instructions

### 1. Environment Setup
Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/fasihbrohi7-ops/GradeCompass.git
cd GradeCompass
pip install -r training/requirements.txt
```

### 2. Verify Preprocessing Pipeline (Fasih)
Run the shared preprocessing module to verify data validation and encoding:

```bash
python training/preprocessing.py
```

### 3. Train Linear Regression Model (Fasih)
Train the ordinary least squares linear regression model and compute evaluation metrics:

```bash
python training/train_linear.py
```
This produces:
- `training/linear_params.json` (weights, bias, scaler mean/std, metadata)
- `training/model_metrics.md` (detailed evaluation report)

### 4. Train Logistic Regression Model (Abdul Hayy)
Run Abdul Hayy's logistic regression training script (uses the identical split and scaler):

```bash
python training/train_logistic.py
```
This produces `training/logistic_params.json`.

### 5. Merge Model Parameters (Fasih)
Compile both models' parameters into the browser-ready runtime JSON:

```bash
python training/merge_params.py
```
Outputs:
- `web/assets/model_params.json` (validated against PRD Section 3.3 schema)
- `web/assets/model_metrics.md`

---

## 📈 Model Performance Summary

### Linear Regression (Fasih)
- **Train Set (800 rows):** R² = 0.2543, MAE = 9.94 points
- **Test Set (200 rows):** R² = **0.1622**, MAE = **10.49 points**, RMSE = **13.40 points**
- **Baseline Intercept (Bias):** 68.17 points
- **Top Positive Drivers:** `lunch_standard` (+2.19), `test_prep_completed` (+1.88), `parent_edu_bachelors_degree` (+1.49)
- **Top Negative Drivers:** `lunch_free_reduced` (-2.19), `test_prep_none` (-1.88), `parent_edu_high_school` (-1.43)

*(Detailed metrics and coefficient analysis are available in `training/model_metrics.md`.)*

### Logistic Regression (Abdul Hayy)
- *Pending execution of `training/train_logistic.py`.*

---

## 🌐 Deployment Guide (Vercel - Fasih)

GradeCompass runs entirely client-side with static assets. No backend server or serverless functions are required at runtime.

### Deploying to Vercel
1. Push all code to GitHub.
2. In [Vercel Dashboard](https://vercel.com):
   - Click **Add New... -> Project** and import the `GradeCompass` repository.
   - **Root Directory:** `./web` (or leave root with `vercel.json` configured).
   - **Framework Preset:** `Other`.
   - **Build Command:** None (static site).
   - **Output Directory:** `web` (configured via `vercel.json`).
3. Click **Deploy**.

### Verification
After deployment:
1. Open the deployed application URL in your browser.
2. Open DevTools (F12) -> **Network Tab** and verify that `assets/model_params.json` returns **HTTP 200 OK**.
3. Confirm dropdown changes trigger instant client-side predictions without console errors.

---

## ⚠️ Disclaimer
Predictions made by GradeCompass reflect statistical correlations present within a Kaggle sample dataset of 1,000 student records. They are intended strictly for educational and demonstration purposes, and must **never** be interpreted as deterministic claims regarding any individual student or demographic group.
