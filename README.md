# GradeCompass 🧭

> A dual-model machine learning application predicting student exam performance and pass/at-risk classification from demographic and academic preparation background attributes.

GradeCompass demonstrates side-by-side comparative machine learning:
1. **Linear Regression (Regression):** Predicts continuous average exam score (0–100) across math, reading, and writing.
2. **Logistic Regression (Binary Classification):** Predicts whether a student is likely to pass (average score &ge; 60/100) with confidence probabilities.

Both models run **100% client-side in the browser** using portable JSON parameter weights with zero backend server dependencies at runtime.

---

## 👥 Team & Task Split

| Teammate | Responsibilities | Status |
|---|---|---|
| **Fasih** | Shared dataset preprocessing foundation (`training/preprocessing.py`), Linear Regression model (`training/train_linear.py`), parameter merge script (`training/merge_params.py`), Vercel deployment configuration (`vercel.json`). | Complete / Integrated |
| **Abdul Hayy** | Logistic Regression model (`training/train_logistic.py`), model evaluation metrics, browser inference engine (`web/js/model.js`), dynamic UI & DOM binding (`web/js/ui.js`), test-prep sensitivity bar chart (`web/js/chart.js`), responsive frontend styling (`web/css/style.css`), About & metrics page (`web/about.html`), usage documentation. | **Complete** |

---

## 📊 Dataset & Feature Architecture

Trained on the public Kaggle [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) dataset (`dataset/StudentsPerformance.csv`, 1,000 records, clean with no missing values).

### Predictor Attributes (5 Categorical Variables)
Individual math, reading, and writing scores are **never** used as inputs for prediction (they only define the training targets). The models strictly predict from demographic background attributes:
- **Gender:** `female`, `male`
- **Race / Ethnicity:** `group A`, `group B`, `group C`, `group D`, `group E`
- **Parental Education:** `some high school`, `high school`, `some college`, `associate's degree`, `bachelor's degree`, `master's degree`
- **Lunch Type:** `standard`, `free/reduced`
- **Test Preparation Course:** `none`, `completed`

### Shared Preprocessing Contract
- Features are one-hot encoded into a fixed 17-column vector.
- Scaled using `StandardScaler` fitted on an 80/20 train/test split (`random_state=42`).
- Target thresholds: `PASS_THRESHOLD = 60`.

---

## 📈 Model Performance Metrics

Evaluated on the held-out 20% test set (200 students):

### 1. Linear Regression (Continuous Score Prediction)
- **R² Score:** `0.1622` (~16.2% variance explained by demographic inputs alone)
- **MAE:** `10.4902` points

### 2. Logistic Regression (Pass/Risk Classification)
- **Accuracy:** `68.00%` (136/200 correct)
- **Precision:** `72.29%`
- **Recall:** `86.96%`
- **Confusion Matrix:**
  ```text
                  Predicted Fail    Predicted Pass
  Actual Fail:    16                46
  Actual Pass:    18                120
  ```

*Full evaluation details can be found in `training/model_metrics.md` and on the web About page.*

---

## 🚀 Getting Started & Local Development

### 1. Offline Model Training (Python)

Requires Python 3.9+ with `pandas` and `scikit-learn`:

```bash
# Install dependencies
pip install -r training/requirements.txt

# Train Linear Regression model
python training/train_linear.py

# Train Logistic Regression model (Abdul Hayy)
python training/train_logistic.py

# Merge parameters into web/assets/model_params.json
python training/merge_params.py
```

### 2. Running Automated Tests

Run the model parity and inference verification tests:

```bash
# Run JavaScript engine verification
node tests/test_inference.js

# Run full cross-language parity suite
python tests/test_inference.py
```

### 3. Running the Web Application

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

## 🌐 Deployment to Vercel

GradeCompass is a 100% static application designed for zero-config Vercel deployment:

1. Connect repository on [Vercel](https://vercel.com).
2. Set **Root Directory** to `web` (or leave at root, since `vercel.json` specifies `"outputDirectory": "web"`).
3. Set **Framework Preset** to `Other` (Static Site).
4. No build command or environment variables required. Click **Deploy**.

---

## ⚠️ Disclaimer

Predictions reflect statistical correlations in a public sample dataset and are provided strictly for educational purposes. They do **not** represent claims that demographic factors determine individual capabilities or academic potential.
