# GradeCompass 🎓🧭

> Demographic & Background-based Student Exam Performance Predictor

GradeCompass is a lightweight machine learning application that predicts student exam outcomes from background attributes using models trained on the public Kaggle **"Students Performance in Exams"** dataset (1,000 students).

Dual models evaluate continuous expected score (**Linear Regression**) and pass/risk likelihood (**Logistic Regression**) directly in the browser with zero server runtime dependencies.

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

## 📈 Model Performance Summary

Evaluated on the held-out 20% test set (200 students):

### 1. Linear Regression (Fasih)
- **Train Set (800 rows):** R² = 0.2543, MAE = 9.94 points
- **Test Set (200 rows):** R² = **0.1622**, MAE = **10.49 points**, RMSE = **13.40 points**
- **Baseline Intercept (Bias):** 68.17 points
- **Top Drivers:** `lunch_standard` (+2.19), `test_prep_completed` (+1.88), `parent_edu_bachelors_degree` (+1.49)

### 2. Logistic Regression (Abdul Hayy)
- **Accuracy:** **68.00%** (136/200 correct)
- **Precision:** **72.29%**
- **Recall:** **86.96%**
- **Confusion Matrix:**
  ```text
                  Predicted Fail    Predicted Pass
  Actual Fail:    16                46
  Actual Pass:    18                120
  ```

*(Detailed metrics and coefficient analysis are available in `training/model_metrics.md` and `web/about.html`.)*

---

## 🚀 Getting Started & Local Development

### 1. Environment Setup
Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/fasihbrohi7-ops/GradeCompass.git
cd GradeCompass
pip install -r training/requirements.txt
```

### 2. Training Models & Generating Assets
```bash
# Verify shared preprocessing pipeline
python training/preprocessing.py

# Train Linear Regression model (Fasih)
python training/train_linear.py

# Train Logistic Regression model (Abdul Hayy)
python training/train_logistic.py

# Merge parameters into web/assets/model_params.json
python training/merge_params.py
```

### 3. Running Automated Verification
```bash
# Run model parity and mathematical validation tests
python tests/test_inference.py

# Run JavaScript engine tests (requires Node.js)
node tests/test_inference.js
```

### 4. Running the Web Application
Because GradeCompass loads `web/assets/model_params.json` via `fetch()`, run any local static HTTP server:

```bash
# Using Python
python -m http.server 8000 --directory web

# Or using Node
npx serve web
```
Then open `http://localhost:8000` in your browser.

---

## 🖥️ Web UI Features (Abdul Hayy)

- **Dynamic Select Controls:** The 5 dropdown menus are populated directly from `model_params.json`'s categorical options schema (single source of truth).
- **Real-Time Live Inference:** Dropdown changes automatically trigger immediate client-side inference without page reloads or submit buttons.
- **Score Progress Gauge:** Visual circular SVG gauge displaying expected score with color cues (green &ge; 60, red &lt; 60).
- **Pass / At-Risk Classification Badge:** Semantic badge and probability percentage indicator.
- **Sensitivity Analysis Bar Chart:** Custom HTML5 `<canvas>` chart dynamically comparing exam score with vs. without a test prep course, holding all other student attributes constant.
- **Responsive & Accessible:** Built mobile-first using pure CSS flexbox and grid, with responsive scaling for high-DPI (Retina) screens.
- **Reset Functionality:** Restores default input combinations with one click.
- **About Page:** Dedicated documentation page (`web/about.html`) presenting methodology, confusion matrix, metrics table, and algorithmic limitations.

---

## 🌐 Deployment Guide (Vercel - Fasih)

GradeCompass is a 100% static application designed for zero-config Vercel deployment:

1. Connect repository on [Vercel](https://vercel.com).
2. Set **Root Directory** to `web` (or leave at root, since `vercel.json` specifies `"outputDirectory": "web"`).
3. Set **Framework Preset** to `Other` (Static Site).
4. No build command or environment variables required. Click **Deploy**.

---

## ⚠️ Disclaimer
Predictions made by GradeCompass reflect statistical correlations present within a Kaggle sample dataset of 1,000 student records. They are intended strictly for educational and demonstration purposes, and must **never** be interpreted as deterministic claims regarding any individual student or demographic group.
