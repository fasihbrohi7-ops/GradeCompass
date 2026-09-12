# GradeCompass
## Product Requirements Document (PRD) + Technical Requirements Document (TRD)

**Version:** 2.0 (updated for real Kaggle dataset + team split)
**Team:** Fasih (Linear Regression + Deployment) & Abdul Hayy (Logistic Regression + UI)
**Prepared for:** AI coding agent implementation (e.g. Claude Code) — each teammate can hand their section to the agent independently.

---

## 0. Team & Task Split

| Owner | Responsibilities |
|---|---|
| **Fasih** | Dataset preprocessing pipeline (shared foundation), Linear Regression model (predicting average score), model evaluation, contributing `linear_regression` block to `model_params.json`, Vercel deployment configuration & final production deploy. |
| **Abdul Hayy** | Logistic Regression model (predicting pass/fail), contributing `logistic_regression` block to `model_params.json`, full frontend UI/UX (HTML/CSS/JS), integrating both models' parameters into the browser inference layer, chart/visualization, README usage section. |

**Critical shared contract:** Both of you must use the **exact same preprocessing/encoding** (same feature order, same one-hot categories, same scaler) so your two models can be merged into a single `model_params.json` and run correctly by the same JS inference code. Section 3 below is the shared contract — read it together and agree on it **before** either of you starts training. Fasih owns writing/publishing the shared preprocessing script; Abdul Hayy imports it as-is rather than rewriting it.

**Suggested workflow:**
1. Both read Section 3 (Data Contract) together and confirm agreement.
2. Fasih builds `training/preprocessing.py` (shared) + `training/train_linear.py`, pushes to repo.
3. Abdul Hayy pulls Fasih's preprocessing script, builds `training/train_logistic.py` using the same script, pushes his `logistic_regression` params.
4. Either of you runs `training/merge_params.py` to combine both into the final `web/assets/model_params.json`.
5. Abdul Hayy builds the UI against the merged `model_params.json` (can start UI layout/styling in parallel using placeholder/dummy JSON before real params are ready — do not block on Fasih).
6. Fasih handles the Vercel deployment once the UI is functional.

---

## 1. Overview

### 1.1 Problem Statement
Given a student's demographic and preparation background — gender, race/ethnicity group, parental education level, lunch type (a proxy for socioeconomic status), and whether they completed a test preparation course — predict:
1. Their **expected average exam score** (0–100, across math/reading/writing) → **Linear Regression**.
2. Whether they are **likely to pass or be at risk** (based on that average score) → **Logistic Regression**.

Both models are trained on the same dataset and the same input features (the categorical background attributes), giving a natural side-by-side regression vs. classification comparison.

### 1.2 Dataset
`dataset/StudentsPerformance.csv` — the public Kaggle "Students Performance in Exams" dataset. **1000 rows, no missing values.**

| Column (original) | Type | Values |
|---|---|---|
| `gender` | categorical | `female`, `male` |
| `race/ethnicity` | categorical | `group A`, `group B`, `group C`, `group D`, `group E` |
| `parental level of education` | categorical (ordinal-ish) | `some high school`, `high school`, `some college`, `associate's degree`, `bachelor's degree`, `master's degree` |
| `lunch` | categorical | `standard`, `free/reduced` |
| `test preparation course` | categorical | `none`, `completed` |
| `math score` | numeric 0–100 | used only to derive targets, not as a model input |
| `reading score` | numeric 0–100 | used only to derive targets, not as a model input |
| `writing score` | numeric 0–100 | used only to derive targets, not as a model input |

**Important design decision:** the three scores are **not** used as input features for prediction — only to build the targets below. Using one score to predict another would trivially inflate accuracy and defeats the point of a demographics-based predictor. Inputs are exclusively the 5 categorical background attributes.

### 1.3 Goals
- Build and train two models on the same 5 categorical features, engineered identically.
- Export both models' parameters into one JSON consumable by browser JS (client-side inference, no backend at runtime).
- Build a clean, interactive vanilla HTML/CSS/JS UI (dropdowns/selects for each categorical input) with live predictions from both models.
- Deploy as a static site on **Vercel**.

### 1.4 Non-Goals
- No accounts/auth, no database, no server-side inference at runtime.
- No in-browser retraining (stretch goal only).
- Model is descriptive of correlations in this dataset only — **not** a claim that demographics causally determine performance. This must be stated as a disclaimer in the UI (see U10).

---

## 2. Features & Requirements

### 2.1 Data & Model Training (offline, Python) — shared contract owned by Fasih, consumed by Abdul Hayy

| ID | Requirement | Owner |
|----|-------------|---|
| F1 | Load `dataset/StudentsPerformance.csv`, verify no missing values (dataset is clean, but still assert this in code). | Fasih |
| F2 | Compute `average_score = (math score + reading score + writing score) / 3` for every row. | Fasih |
| F3 | Derive binary target `passed = 1 if average_score >= 60 else 0` (threshold is a named constant `PASS_THRESHOLD = 60`, easy to change later). | Fasih |
| F4 | One-hot encode the 5 categorical features in a **fixed, documented order** (see Section 3.2) using `pandas.get_dummies` or `sklearn.preprocessing.OneHotEncoder` with all categories kept (document the exact column order). | Fasih |
| F5 | Split into train/test (80/20), `random_state=42`, shared across both models for consistency. | Fasih |
| F6 | Fit `StandardScaler` on the one-hot encoded training features. Persist mean/std. | Fasih |
| F7 | Train `LinearRegression` on scaled features → `average_score`. | **Fasih** |
| F8 | Evaluate Linear Regression: R² and MAE on the test set. Write to `training/model_metrics.md`. | Fasih |
| F9 | Train `LogisticRegression` on the **same scaled features** (same scaler object, same train/test split) → `passed`. | **Abdul Hayy** |
| F10 | Evaluate Logistic Regression: accuracy, precision, recall, confusion matrix on test set. Append to `training/model_metrics.md`. | Abdul Hayy |
| F11 | Export a merged `model_params.json` (schema in Section 3.3) combining both models' weights/bias + the shared scaler + shared encoding map. | Fasih (final merge step) |

### 2.2 Web UI (vanilla HTML/CSS/JS) — owned by Abdul Hayy

| ID | Requirement |
|----|-------------|
| U1 | Input controls: **5 dropdown/select elements**, one per categorical feature (gender, race/ethnicity, parental education, lunch, test prep course). No sliders — this dataset has no continuous inputs. |
| U2 | Live prediction updates on every dropdown change (`change` event), no submit button required. |
| U3 | Display predicted average score (0–100) with a visual gauge/progress bar. |
| U4 | Display pass/fail classification with a probability percentage and a colored badge (green = "Likely to Pass", red = "At Risk"). |
| U5 | Show a small bar chart comparing predicted average score across the two `test preparation course` options (`none` vs `completed`) holding all other current dropdown selections fixed — a genuinely useful "what if they took the prep course" insight. |
| U6 | "About the Model" section/page showing dataset description, feature list, chosen pass threshold (60), and evaluation metrics pulled from `model_metrics.md`. |
| U7 | Responsive design — usable on mobile and desktop. |
| U8 | All 5 dropdowns default to a sensible starting combination on page load (first option of each) and always have a valid selection (no empty/placeholder state feeding into the model). |
| U9 | Reset button restores default dropdown values. |
| U10 | Visible disclaimer: predictions reflect statistical correlations in a public sample dataset, are for educational purposes only, and must not be interpreted as claims about any individual or demographic group. |

### 2.3 Deployment — owned by Fasih

| ID | Requirement |
|----|-------------|
| D1 | Entire app is static files (`web/`) — no backend server, no serverless functions required. |
| D2 | Deploy on **Vercel**: set the project root/output directory to `web/`, framework preset = "Other" (static site), no build command needed. |
| D3 | Add a `vercel.json` (optional but recommended) to explicitly set `"outputDirectory": "web"` if the repo root isn't `web/`. |
| D4 | Confirm the deployed URL loads `assets/model_params.json` correctly (check browser network tab / console for 404s — common Vercel static-path issue). |
| D5 | Include deployment steps in `README.md`. |

---

## 3. Data Contract (READ TOGETHER BEFORE TRAINING)

This section is the single source of truth both of you must follow exactly, or the two models' outputs will be incompatible when merged into one JSON / run by one JS file.

### 3.1 Target Definitions
```python
PASS_THRESHOLD = 60  # out of 100

average_score = (math_score + reading_score + writing_score) / 3
passed = 1 if average_score >= PASS_THRESHOLD else 0
```

### 3.2 Feature Encoding — fixed column order (do not change)

One-hot encode in this exact order. Every model (linear and logistic) must consume features in this exact order, and the JS inference code will replicate this order.

```
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

That's **17 binary input features** after one-hot encoding, scaled with `StandardScaler`, feeding into both models.

Recommended implementation (`training/preprocessing.py`, owned by Fasih, imported by Abdul Hayy — do not reimplement separately):

```python
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
```

Both `train_linear.py` and `train_logistic.py` must:
```python
from preprocessing import load_and_prepare, get_scaler, FEATURE_ORDER, PASS_THRESHOLD
from sklearn.model_selection import train_test_split

X, y_score, y_passed = load_and_prepare()
X_train, X_test, y_score_train, y_score_test, y_passed_train, y_passed_test = train_test_split(
    X, y_score, y_passed, test_size=0.2, random_state=42
)
scaler = get_scaler(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
This guarantees identical splits and identical scaling for both models.

### 3.3 Merged `model_params.json` Schema

```json
{
  "feature_order": [ /* the 17 strings from Section 3.2, in order */ ],
  "categorical_options": {
    "gender": ["female", "male"],
    "race_ethnicity": ["group A", "group B", "group C", "group D", "group E"],
    "parental_education": ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"],
    "lunch": ["standard", "free/reduced"],
    "test_preparation_course": ["none", "completed"]
  },
  "scaler": {
    "mean": [ /* 17 floats */ ],
    "std": [ /* 17 floats */ ]
  },
  "linear_regression": {
    "weights": [ /* 17 floats */ ],
    "bias": 0.0,
    "pass_threshold": 60
  },
  "logistic_regression": {
    "weights": [ /* 17 floats */ ],
    "bias": 0.0
  },
  "metadata": {
    "trained_on_rows": 1000,
    "linear_r2": 0.0,
    "linear_mae": 0.0,
    "logistic_accuracy": 0.0,
    "logistic_precision": 0.0,
    "logistic_recall": 0.0,
    "dataset_source": "Kaggle - Students Performance in Exams"
  }
}
```

`categorical_options` is included specifically so Abdul Hayy's UI can programmatically build the 5 dropdowns and construct the correct one-hot vector from a user's selection, rather than hardcoding option lists separately in JS (single source of truth).

---

## 4. User Stories

1. *As a visitor*, I want to select a demographic/background profile from dropdowns and instantly see the predicted average score and pass/fail likelihood.
2. *As a visitor*, I want to see how completing a test-prep course would change the predicted outcome for an otherwise-identical profile, so I understand the model's sensitivity to that factor.
3. *As a visitor*, I want to see the models' accuracy/R² so I can judge how much to trust the prediction.
4. *As Fasih*, I want a clean, documented preprocessing pipeline so my linear regression and Abdul's logistic regression stay perfectly compatible.
5. *As Abdul Hayy*, I want the model parameters delivered in one predictable JSON schema so I can build the UI without needing to know Python or re-derive encoding logic.

---

## 5. Success Metrics
- Linear Regression: document actual R² and MAE (demographics-only models on this dataset typically yield modest R², roughly 0.15–0.30 — this is expected and should be described honestly in the UI, not inflated).
- Logistic Regression: document actual accuracy/precision/recall (expect roughly 65–80% accuracy).
- Deployed Vercel site loads with zero console errors and correctly fetches `assets/model_params.json`.
- Lighthouse performance score ≥ 90 on the deployed site.

---

## 6. System Architecture

```
┌──────────────────────────────────────────┐
│  OFFLINE (Python, one-time) — Fasih owns   │
│  shared pipeline, Abdul Hayy owns logistic │
│                                             │
│  dataset/StudentsPerformance.csv            │
│           │                                 │
│           ▼                                 │
│  training/preprocessing.py  (Fasih)          │
│           │                                 │
│     ┌─────┴─────┐                           │
│     ▼           ▼                           │
│ train_linear.py  train_logistic.py          │
│   (Fasih)          (Abdul Hayy)              │
│     │           │                           │
│     └─────┬─────┘                           │
│           ▼                                 │
│  training/merge_params.py (Fasih)            │
│           │                                 │
│           ▼                                 │
│  web/assets/model_params.json                │
│  web/assets/model_metrics.md                 │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│  RUNTIME (Browser, static) — Abdul Hayy    │
│  builds UI, Fasih deploys                   │
│                                             │
│  index.html + style.css + js/*.js            │
│    - fetch model_params.json                 │
│    - build dropdowns from categorical_options│
│    - one-hot encode selection in JS           │
│    - scale, then linear + sigmoid inference   │
│    - render score gauge, badge, chart          │
└──────────────────────────────────────────┘
              │
              ▼
        Deployed on Vercel (Fasih)
```

---

## 7. Repository Structure

```
gradecompass/
├── README.md
├── dataset/
│   └── StudentsPerformance.csv          # already added
├── training/
│   ├── preprocessing.py                  # Fasih — shared, do not duplicate
│   ├── train_linear.py                   # Fasih
│   ├── train_logistic.py                 # Abdul Hayy
│   ├── merge_params.py                   # Fasih — combines both JSON outputs
│   ├── requirements.txt
│   └── model_metrics.md                  # generated (both contribute sections)
├── web/
│   ├── index.html                        # Abdul Hayy
│   ├── about.html                        # Abdul Hayy
│   ├── css/
│   │   └── style.css                     # Abdul Hayy
│   ├── js/
│   │   ├── model.js                      # Abdul Hayy — inference logic
│   │   ├── ui.js                         # Abdul Hayy — DOM binding
│   │   └── chart.js                      # Abdul Hayy — bar chart
│   └── assets/
│       ├── model_params.json             # generated, committed by Fasih after merge
│       └── model_metrics.md              # copied from training/
├── vercel.json                           # Fasih
└── .gitignore
```

---

## 8. Frontend Technical Specification (Abdul Hayy)

### 8.1 `js/model.js` — Inference Logic

```javascript
async function loadModelParams(path = "./assets/model_params.json") { ... }

// Builds the 17-length one-hot vector from the 5 dropdown selections,
// using model_params.categorical_options + feature_order to stay in sync
// with the Python encoding (Section 3.2) — never hardcode this mapping twice.
function encodeSelection(selection, params) { ... }
// selection = { gender, race_ethnicity, parental_education, lunch, test_preparation_course }
// returns a 17-length array of 0/1 in feature_order

function scaleFeatures(rawFeatures, scaler) { ... }

function predictScore(scaledFeatures, linearParams) {
  // dot product + bias, clamp to [0, 100]
}

function predictPassProbability(scaledFeatures, logisticParams) {
  // sigmoid(dot product + bias)
}

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
```

Requirements:
- All math done manually in plain JS — no ML libraries in-browser.
- `encodeSelection` must derive the one-hot mapping from `categorical_options` + `feature_order` in the JSON rather than a separately hardcoded JS object, so Fasih changing category labels never silently breaks the UI.
- Score prediction clamped to [0, 100]. Pass/fail badge threshold: `probability >= 0.5`.

### 8.2 `js/ui.js`
- On load: fetch params, populate all 5 `<select>` elements from `categorical_options`, set defaults to each list's first option, run initial prediction.
- On any `change` event: re-encode selection → scale → predict both → update gauge, badge, probability text.
- Reset button restores first-option defaults for all 5 selects.
- Show a visible error state if `model_params.json` fails to fetch.

### 8.3 `js/chart.js`
- Bar chart (2 bars): predicted average score with `test_preparation_course = none` vs `completed`, all other current selections held fixed. Recompute on every change event.
- Hand-rolled `<canvas>` chart preferred to keep the stack fully vanilla; Chart.js via CDN is acceptable if simpler.

### 8.4 Styling
- Mobile-first, flexbox/grid, green/red semantic colors for pass/at-risk, all styling in `style.css` (no inline styles).

---

## 9. Deployment Specification (Fasih)

1. Confirm `web/assets/model_params.json` and `model_metrics.md` are committed and up to date after any retraining.
2. Test locally: `npx serve web` (or any static server) and verify predictions work with no console errors.
3. On Vercel: **New Project → Import Git Repo → Root Directory = `web/` → Framework Preset = "Other" → Deploy.** No build command, no environment variables needed.
4. Optionally commit `vercel.json`:
```json
{ "outputDirectory": "web" }
```
5. After deploy, open the live URL and confirm `assets/model_params.json` returns 200 in the network tab (case-sensitive paths are a common Vercel gotcha).
6. Add the live URL to `README.md`.

---

## 10. Testing & Validation Checklist

| Area | Check | Owner |
|---|---|---|
| Preprocessing | `preprocessing.py` produces exactly 17 columns in the documented order for any input row. | Fasih |
| Linear model | R² and MAE are printed and written to `model_metrics.md`; sanity-check predictions are within [0,100]. | Fasih |
| Logistic model | Accuracy/precision/recall/confusion matrix printed and written to `model_metrics.md`; uses the *same* train/test split and scaler as the linear model (verify by checking row counts and a couple of shared scaled values match). | Abdul Hayy |
| Merge | `merge_params.py` output matches the Section 3.3 schema exactly (validate with a JSON schema check or manual diff). | Fasih |
| Frontend | All 5 dropdowns populate correctly from `categorical_options`; changing any one updates score, badge, and chart with no reload. | Abdul Hayy |
| Frontend | JS predictions for 3–5 sample profiles match Python model outputs for the same profile (spot-check manually — should match within rounding). | Abdul Hayy + Fasih (joint check before deploy) |
| Deployment | Deployed Vercel URL functions identically to local version, no 404s on `assets/`. | Fasih |

---

## 11. Stretch Goals (Out of Scope for v1)
- In-browser retraining via Pyodide with CSV upload.
- 2D decision-boundary visualization for logistic regression using two selected feature groups.
- Persist last selection in `localStorage`.
- Add a real backend for model versioning / A-B testing thresholds.

---

## 12. Deliverables Summary

| Deliverable | Owner |
|---|---|
| `training/preprocessing.py` (shared contract) | Fasih |
| `training/train_linear.py` + linear metrics | Fasih |
| `training/train_logistic.py` + logistic metrics | Abdul Hayy |
| `training/merge_params.py` + final `model_params.json` | Fasih |
| Full `web/` static app (HTML/CSS/JS, dropdowns, gauge, badge, chart, about page) | Abdul Hayy |
| `vercel.json` + live Vercel deployment | Fasih |
| `README.md` (setup, training, deployment, usage, metrics summary) | Joint (Fasih: setup/deploy sections, Abdul Hayy: usage/UI section) |
