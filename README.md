# GradeCompass 🎓🧭

> **Interactive Client-Side Machine Learning for Student Exam Performance & Pass/Fail Prediction**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JavaScript Vanilla](https://img.shields.io/badge/javascript-vanilla%20ES6+-yellow.svg)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Deploy on Vercel](https://img.shields.io/badge/deploy-vercel-black.svg?logo=vercel)](https://grade-compass-fasih.vercel.app)
[![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)]()

**GradeCompass** is a lightweight, zero-dependency educational machine learning application that predicts student exam outcomes from background attributes using models trained on the public Kaggle **"Students Performance in Exams"** dataset ($1,000$ students).

The application evaluates dual models simultaneously—**Linear Regression** (continuous score from $0$ to $100$) and **Logistic Regression** (pass/at-risk classification with probability)—directly inside the user's browser with **zero runtime backend or serverless infrastructure**.

---

## 🧭 Live Demo & Quick Links

- **Live Production URL**: **[https://grade-compass-fasih.vercel.app](https://grade-compass-fasih.vercel.app)**
- **User Guide**: [Complete End-User Walkthrough](docs/USER_GUIDE.md)
- **Product Requirements**: [PRD Specification](docs/PRD.md)
- **Technical Requirements**: [TRD Specification](docs/TRD.md)
- **Dataset Card**: [Kaggle Dataset Analysis](docs/DATASET_CARD.md)
- **Production Deployment**: [Vercel Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- **API & Schema Reference**: [JSON & Engine Specification](docs/API_AND_SCHEMA.md)

---

## 📑 Table of Contents
1. [Overview & Core Concept](#-overview--core-concept)
2. [Dual Model Comparative Learning](#-dual-model-comparative-learning)
3. [System Architecture](#-system-architecture)
4. [Team & Responsibilities](#-team--responsibilities)
5. [Repository Structure](#-repository-structure)
6. [Shared Data Contract (17 Features)](#-shared-data-contract-17-features)
7. [Installation & Quickstart](#-installation--quickstart)
8. [Reproducible Training Pipeline](#-reproducible-training-pipeline)
9. [Automated Verification & Tests](#-automated-verification--tests)
10. [Model Performance & Evaluation](#-model-performance--evaluation)
11. [Web UI & Features](#-web-ui--features)
12. [Vercel Deployment Guide](#-vercel-deployment-guide)
13. [Documentation Index](#-documentation-index)
14. [Ethical Considerations & Disclaimer](#-ethical-considerations--disclaimer)

---

## 💡 Overview & Core Concept

When assessing educational datasets, machine learning models frequently focus exclusively on either continuous regression or discrete classification. GradeCompass unites both techniques on identical inputs, giving students and educators a direct, side-by-side comparison:

1. **Continuous Regression**: How many points is a student expected to achieve across Math, Reading, and Writing?
2. **Binary Classification**: What is the likelihood that the student crosses the $60$-point passing mark?
3. **Counterfactual Sensitivity**: If a student with an identical demographic profile completes an exam preparation course, how does their expected score and pass probability change?

> **Key Design Decision**: Individual exam scores (Math, Reading, Writing) are used exclusively to calculate target outcomes and are **never fed into the models as predictors**. Predictions rely exclusively on demographic and preparation attributes.

---

## ⚖️ Dual Model Comparative Learning

| Dimension | Linear Regression (Fasih) | Logistic Regression (Abdul Hayy) |
|---|---|---|
| **Problem Type** | Continuous Regression | Binary Classification |
| **Target Variable** | `average_score = (math + reading + writing) / 3` | `passed = 1 if average_score >= 60 else 0` |
| **Output Range** | Clamped $[0, 100]$ score points | Probability $P \in [0.0, 1.0]$ |
| **Core Formula** | $\hat{y} = \min(100, \max(0, \mathbf{w}_{lin}^T \mathbf{z} + b_{lin}))$ | $P(Pass) = \sigma(\mathbf{w}_{log}^T \mathbf{z} + b_{log}) = \frac{1}{1 + e^{-z}}$ |
| **Key Metric** | $R^2 = 0.1622$, $\text{MAE} = 10.49$ points | Accuracy $= 68.00\%$, Precision $= 72.29\%$, Recall $= 86.96\%$ |
| **Decision Cutoff** | $\ge 60$ marks pass threshold | $P \ge 0.50 \implies$ **"Likely to Pass"**; $P < 0.50 \implies$ **"At Risk"** |

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  OFFLINE PHASE (Python 3.10+)                                           │
│                                                                         │
│  dataset/StudentsPerformance.csv (1,000 records)                         │
│         │                                                               │
│         ▼                                                               │
│  training/preprocessing.py (Shared Data Contract & 17 One-Hot Columns)   │
│         │                                                               │
│         ├──► training/train_linear.py (Fasih) ──► linear_params.json   │
│         │                                                               │
│         └──► training/train_logistic.py (Abdul) ─► logistic_params.json│
│                     │                                                   │
│                     ▼                                                   │
│          training/merge_params.py (Schema Validator)                    │
│                     │                                                   │
│                     ▼                                                   │
│          web/assets/model_params.json                                   │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  RUNTIME CLIENT PHASE (Static Browser, 100% Vanilla Web Technologies)   │
│                                                                         │
│  web/index.html + web/css/style.css                                     │
│         │                                                               │
│         ├──► web/js/model.js (Vector Scaling, Dot Product, Sigmoid)     │
│         ├──► web/js/ui.js    (Dynamic Selects, Circular SVG Gauge)     │
│         └──► web/js/chart.js (HTML5 Canvas Prep Course Sensitivity)     │
│                                                                         │
│  Zero serverless functions • Zero runtime backend • Offline capable     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 👥 Team & Responsibilities

| Contributor | Role & Deliverables |
|---|---|
| **Fasih** | Shared dataset preprocessing pipeline ([`preprocessing.py`](training/preprocessing.py)), Linear Regression model training & metrics ([`train_linear.py`](training/train_linear.py)), model parameter merging pipeline ([`merge_params.py`](training/merge_params.py)), and Vercel static deployment configuration ([`vercel.json`](vercel.json)). |
| **Abdul Hayy** | Logistic Regression classification model & evaluation ([`train_logistic.py`](training/train_logistic.py)), client-side JavaScript inference engine ([`model.js`](web/js/model.js)), interactive UI/UX ([`index.html`](web/index.html), [`style.css`](web/css/style.css), [`ui.js`](web/js/ui.js)), dynamic Canvas sensitivity chart ([`chart.js`](web/js/chart.js)), and About dashboard ([`about.html`](web/about.html)). |

---

## 📂 Repository Structure

```text
GradeCompass/
├── .gitignore                      # Git exclusion rules
├── README.md                       # Comprehensive project documentation
├── vercel.json                     # Vercel deployment configuration
├── dataset/
│   └── StudentsPerformance.csv     # Kaggle dataset (1,000 rows, 0 nulls)
├── docs/
│   ├── USER_GUIDE.md               # End-user interactive manual
│   ├── PRD.md                      # Product Requirements Document
│   ├── TRD.md                      # Technical Requirements Document
│   ├── DATASET_CARD.md             # Dataset card & exploratory analysis
│   ├── DEPLOYMENT_GUIDE.md         # Production & Vercel deployment guide
│   ├── API_AND_SCHEMA.md           # API reference & model_params schema
│   └── GradeCompass-PRD-TRD.md     # Original team specification
├── tests/
│   ├── test_inference.py           # Cross-language Python parity test harness
│   └── test_inference.js           # Node.js client-side inference test runner
├── training/
│   ├── preprocessing.py            # Shared data contract & one-hot pipeline
│   ├── train_linear.py             # Linear regression training script
│   ├── train_logistic.py           # Logistic regression training script
│   ├── merge_params.py             # Parameter merge & validation script
│   ├── linear_params.json          # Serialized linear model parameters
│   ├── logistic_params.json        # Serialized logistic model parameters
│   ├── model_metrics.md            # Comprehensive model evaluation report
│   └── requirements.txt            # Python dependencies
└── web/
    ├── index.html                  # Main interactive dashboard
    ├── about.html                  # Methodology, metrics, & confusion matrix
    ├── css/
    │   └── style.css               # Responsive design system & animations
    ├── js/
    │   ├── model.js                # Client-side inference engine (vector math)
    │   ├── ui.js                   # DOM controller & event bindings
    │   └── chart.js                # Custom HTML5 Canvas sensitivity chart
    └── assets/
        ├── model_params.json       # Compiled parameters consumed by browser
        └── model_metrics.md        # Synced markdown evaluation report
```

---

## ⚙️ Shared Data Contract (17 Features)

Both models strictly adhere to the contractual 17-feature column sequence defined in [`training/preprocessing.py`](training/preprocessing.py):

```text
 1. gender_female                10. parent_edu_some_college
 2. gender_male                  11. parent_edu_associates_degree
 3. race_group_A                 12. parent_edu_bachelors_degree
 4. race_group_B                 13. parent_edu_masters_degree
 5. race_group_C                 14. lunch_standard
 6. race_group_D                 15. lunch_free_reduced
 7. race_group_E                 16. test_prep_none
 8. parent_edu_some_high_school  17. test_prep_completed
 9. parent_edu_high_school
```

All features are standardized with a single shared `StandardScaler` fitted exclusively on the 80% training split (`random_state=42`):
$$z_i = \frac{x_i - \mu_i}{\sigma_i}$$

---

## 🚀 Installation & Quickstart

### 1. Prerequisites
- **Python 3.10+** (with pip)
- **Node.js 18+** (optional, for running JS test suites)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### 2. Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/fasihbrohi7-ops/GradeCompass.git
cd GradeCompass

# Install Python training dependencies
pip install -r training/requirements.txt
```

---

## 🔄 Reproducible Training Pipeline

To retrain the models and regenerate parameter assets from scratch:

```bash
# 1. Verify shared preprocessing pipeline
python training/preprocessing.py

# 2. Train Linear Regression model (Fasih)
python training/train_linear.py

# 3. Train Logistic Regression model (Abdul Hayy)
python training/train_logistic.py

# 4. Merge parameters into web/assets/model_params.json
python training/merge_params.py
```

---

## 🧪 Automated Verification & Tests

GradeCompass includes cross-language test suites ensuring complete parity between Python scikit-learn training and JavaScript browser evaluation:

```bash
# Run cross-language parity & validation test suite
python tests/test_inference.py

# Run JavaScript inference engine suite (Node.js)
node tests/test_inference.js
```

### Sample Parity Output
```text
==================================================
GradeCompass: Inference Parity & Validation Tests
==================================================
[PASS] Profile 1 - Standard:
       Score: 72.97 (Sklearn: 72.97)
       Pass Prob: 87.97% (Sklearn: 87.97%)
[PASS] Profile 2 - Free Lunch with Prep:
       Score: 60.91 (Sklearn: 60.91)
       Pass Prob: 49.59% (Sklearn: 49.59%)
[PASS] Profile 3 - High Achievement:
       Score: 85.37 (Sklearn: 85.37)
       Pass Prob: 97.44% (Sklearn: 97.44%)
[PASS] Profile 4 - At-Risk:
       Score: 52.57 (Sklearn: 52.57)
       Pass Prob: 25.50% (Sklearn: 25.50%)

[SUCCESS] Model parity and client inference verified!
```

---

## 📈 Model Performance & Evaluation

Evaluated on the held-out 20% test partition ($200$ student records):

### Linear Regression (Continuous Average Score)
- **Train Set (800 rows)**: $R^2 = 0.2543$, $\text{MAE} = 9.94$ points
- **Test Set (200 rows)**: $R^2 = \mathbf{0.1622}$, $\text{MAE} = \mathbf{10.49}$ points, $\text{RMSE} = \mathbf{13.40}$ points
- **Base Intercept (Bias)**: $68.17$ points
- **Top Positive Drivers**: `lunch_standard` ($+2.19$), `test_prep_completed` ($+1.88$), `parent_edu_bachelors_degree` ($+1.49$), `race_group_E` ($+1.35$).
- **Top Negative Drivers**: `lunch_free_reduced` ($-2.19$), `test_prep_none` ($-1.88$), `parent_edu_high_school` ($-1.43$).

### Logistic Regression (Pass / At-Risk Classification)
- **Pass Threshold**: $\ge 60$ marks out of 100
- **Test Set Accuracy**: $\mathbf{68.00\%}$ ($136 / 200$ correct classifications)
- **Precision**: $\mathbf{72.29\%}$ (high confidence in pass classifications)
- **Recall**: $\mathbf{86.96\%}$ (captures $87\%$ of all passing students)
- **Confusion Matrix**:
  ```text
                  Predicted Fail    Predicted Pass
  Actual Fail:    16                46
  Actual Pass:    18                120
  ```

---

## 🖥️ Web UI & Features

1. **Dynamic Dropdowns**: Generated on runtime load directly from `model_params.categorical_options`.
2. **Real-Time Live Updates**: Reacts immediately to `change` events with zero submit delays.
3. **Score Gauge**: Animated circular SVG gauge color-coded by performance ($\ge 60$ green, $< 60$ red).
4. **Pass/Risk Badge**: Clear semantic badge accompanied by probability progress bar.
5. **Prep Course Sensitivity Chart**: High-DPI HTML5 Canvas comparing scores with and without test preparation holding other background features constant.
6. **Reset Button**: Restores starting profile with one click.
7. **About Dashboard**: Accessible at `about.html` detailing methodology, metrics, and algorithmic limits.

### Local Web Testing
```bash
# Serve the web/ directory using Python
python -m http.server 8000 --directory web

# Or using Node
npx serve web -p 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🌐 Vercel Deployment Guide

GradeCompass is $100\%$ static and pre-configured for zero-config Vercel deployment:

1. Connect your repository on [Vercel](https://vercel.com).
2. **Framework Preset**: Choose **`Other`** (Static Site).
3. **Root Directory**: Leave as `./` (or select `web`).  
   *(Because [`vercel.json`](vercel.json) specifies `{"outputDirectory": "web"}`, Vercel automatically deploys the static files).*
4. **Build Command**: Leave empty / disabled.
5. Click **Deploy**.

For detailed custom domains, edge caching, and CI/CD tips, see the [Vercel Deployment Guide](docs/DEPLOYMENT_GUIDE.md).

---

## 📚 Documentation Index

All extended documentation is stored inside the [`docs/`](docs/) directory:

- 📖 **[User Guide (`docs/USER_GUIDE.md`)](docs/USER_GUIDE.md)**: Visual dashboard guide, step-by-step usage, troubleshooting, and FAQ.
- 📋 **[Product Requirements Document (`docs/PRD.md`)](docs/PRD.md)**: Personas, functional requirements (F1–F11, U1–U10, D1–D5), non-functional requirements, and success criteria.
- 📐 **[Technical Requirements Document (`docs/TRD.md`)](docs/TRD.md)**: System architecture, mathematical formulations, vector encoding, and deployment architecture.
- 📊 **[Dataset Card (`docs/DATASET_CARD.md`)](docs/DATASET_CARD.md)**: Kaggle data dictionary, statistical summaries, class distributions, and bias analysis.
- 🚀 **[Deployment Guide (`docs/DEPLOYMENT_GUIDE.md`)](docs/DEPLOYMENT_GUIDE.md)**: Step-by-step Vercel deployment, local hosting options, and Lighthouse performance auditing.
- 🔍 **[API & Schema Reference (`docs/API_AND_SCHEMA.md`)](docs/API_AND_SCHEMA.md)**: Detailed JSON schema for `model_params.json`, JavaScript inference API, and Python module contracts.
- 📄 **[Original PRD-TRD Specification (`docs/GradeCompass-PRD-TRD.md`)](docs/GradeCompass-PRD-TRD.md)**: The foundational contract defining the team split and requirements.

---

## ⚠️ Ethical Considerations & Disclaimer

> [!IMPORTANT]
> **Correlation Does Not Imply Causation**  
> Predictions generated by GradeCompass reflect statistical correlations identified within a historical public sample dataset of 1,000 student records from Kaggle.
> - Demographic attributes (such as gender, ethnicity, or parental education) **do not determine** innate intelligence, ability, or individual academic potential.
> - Structural inequalities reflected in the dataset (such as lunch subsidies) are systemic social indicators rather than individual deficits.
> - GradeCompass is strictly an **educational tool** demonstrating comparative machine learning principles and must **never** be used for admissions, academic tracking, or high-stakes student evaluations.

---

## 📄 License
This project is open-source under the [MIT License](https://opensource.org/licenses/MIT).
