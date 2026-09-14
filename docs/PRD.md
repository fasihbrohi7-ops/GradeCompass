# GradeCompass - Product Requirements Document (PRD)

**Document Version:** 2.1  
**Product Name:** GradeCompass  
**Authors:** Fasih (ML Training & Deployment) & Abdul Hayy (Classification & Frontend UX)  
**Status:** Approved & Implemented  

---

## 1. Executive Summary & Vision

**GradeCompass** is an educational, browser-native machine learning application that predicts student exam performance outcomes based on background demographic and academic preparation attributes. 

By comparing an ordinary least squares (OLS) **Linear Regression** model (continuous score prediction) against a binary **Logistic Regression** model (pass/at-risk classification) on the exact same dataset, GradeCompass provides a transparent, interactive demonstration of data science fundamentals, model interpretation, and client-side web inference.

---

## 2. Problem Statement & Opportunities

### 2.1 The Problem
Machine learning models are frequently deployed behind opaque server-side APIs or treated as "black boxes" by non-technical audiences. Students, educators, and novice data scientists often struggle to understand:
- How regression differs practically from classification when applied to the same real-world phenomenon.
- How much variance demographic factors actually explain (and how much they do not).
- How one-hot encoding, feature standardization, and linear algebra execute at runtime.

### 2.2 The Opportunity
GradeCompass addresses these challenges by:
- Operating $100\%$ client-side in vanilla web technologies with zero runtime backend dependency.
- Presenting real-time counterfactual analysis (e.g., testing the statistical impact of taking an exam prep course).
- Framing all predictive outputs with statistical modesty, model metrics, and ethical disclaimers.

---

## 3. User Personas & Target Audience

| Persona | Role & Needs | Goal with GradeCompass |
|---|---|---|
| **Alex (Student)** | Secondary or undergraduate student exploring study habits. | Wants to see how much an exam preparation course can improve their expected score and pass probability. |
| **Dr. Elena (Academic Counselor)** | High school counselor advising at-risk cohorts. | Wants to understand statistical correlations in educational outcomes to advocate for subsidized test-prep initiatives. |
| **Marcus (Data Science Learner)** | Aspiring ML developer learning model deployment. | Wants to inspect how trained scikit-learn models are exported to JSON and evaluated in client-side vanilla JavaScript. |

---

## 4. Dataset Specification

The models are trained exclusively on the public Kaggle **"Students Performance in Exams"** dataset:
- **Total Records:** 1,000 students.
- **Missing Values:** 0 (clean dataset).
- **Target Inputs:** Exclusively 5 categorical attributes:
  1. `gender`: `female`, `male`
  2. `race/ethnicity`: `group A`, `group B`, `group C`, `group D`, `group E`
  3. `parental level of education`: `some high school`, `high school`, `some college`, `associate's degree`, `bachelor's degree`, `master's degree`
  4. `lunch`: `standard`, `free/reduced` (socioeconomic proxy)
  5. `test preparation course`: `none`, `completed`

> [!IMPORTANT]
> **Exclusion of Subject Scores from Inputs**:  
> While the dataset includes individual math, reading, and writing scores, these are **never used as predictors**. Using one exam score to predict another would artificially inflate model accuracy and subvert the objective of studying demographic and preparation correlations.

---

## 5. Functional Requirements

### 5.1 Offline Data & Model Training Pipeline (Python)
| Req ID | Description | Owner | Priority |
|---|---|---|---|
| **F1** | Load `dataset/StudentsPerformance.csv` and assert zero null/missing values. | Fasih | P0 |
| **F2** | Derive continuous target `average_score = (math + reading + writing) / 3`. | Fasih | P0 |
| **F3** | Derive binary target `passed = 1 if average_score >= 60 else 0` (`PASS_THRESHOLD = 60`). | Fasih | P0 |
| **F4** | One-hot encode the 5 categorical attributes into a strictly fixed 17-feature column sequence (`FEATURE_ORDER`). | Fasih | P0 |
| **F5** | Split data into 80% train and 20% test partitions using `random_state=42`. | Fasih | P0 |
| **F6** | Fit `StandardScaler` on training split only; persist `mean` and `std` arrays (length 17). | Fasih | P0 |
| **F7** | Train `LinearRegression` on standardized features predicting `average_score`. | Fasih | P0 |
| **F8** | Evaluate Linear Regression: compute $R^2$, MAE, RMSE, and bias; write report to `training/model_metrics.md`. | Fasih | P0 |
| **F9** | Train `LogisticRegression` on the exact same scaled train partition predicting `passed`. | Abdul Hayy | P0 |
| **F10** | Evaluate Logistic Regression: compute Accuracy, Precision, Recall, and Confusion Matrix; append to `training/model_metrics.md`. | Abdul Hayy | P0 |
| **F11** | Compile parameter payloads from both models into a single client-side runtime file: `web/assets/model_params.json`. | Fasih | P0 |

---

### 5.2 Frontend Web Application (Vanilla HTML/CSS/JS)
| Req ID | Description | Owner | Priority |
|---|---|---|---|
| **U1** | Render 5 `<select>` dropdown controls populated from `model_params.categorical_options`. | Abdul Hayy | P0 |
| **U2** | Live predictions triggered instantly on every dropdown `change` event (zero page reloads). | Abdul Hayy | P0 |
| **U3** | Render circular animated score gauge for predicted continuous score ($0$–$100$). | Abdul Hayy | P0 |
| **U4** | Render semantic badge (`Likely to Pass` vs. `At Risk`) and probability progress bar. | Abdul Hayy | P0 |
| **U5** | Render HTML5 `<canvas>` bar chart comparing prep course status (`none` vs. `completed`) holding other factors fixed. | Abdul Hayy | P0 |
| **U6** | Provide dedicated `about.html` page presenting full dataset summary, metrics, and confusion matrix. | Abdul Hayy | P1 |
| **U7** | Fully responsive layout optimized for mobile ($< 768\text{px}$) and desktop screens. | Abdul Hayy | P1 |
| **U8** | Default all dropdowns to valid first options on page load; eliminate empty states. | Abdul Hayy | P0 |
| **U9** | Reset button (`#btn-reset`) to restore default configuration. | Abdul Hayy | P1 |
| **U10** | Display visible educational disclaimer emphasizing correlation vs. causation. | Abdul Hayy | P0 |

---

### 5.3 Deployment & DevOps
| Req ID | Description | Owner | Priority |
|---|---|---|---|
| **D1** | App structure must be 100% static files (`web/`) with no backend runtime server. | Fasih | P0 |
| **D2** | Deployable on Vercel via Git integration with preset "Other" (Static Site). | Fasih | P0 |
| **D3** | Include `vercel.json` declaring `"outputDirectory": "web"`. | Fasih | P0 |
| **D4** | Ensure zero 404 errors on static assets (case-sensitive asset validation). | Fasih | P0 |
| **D5** | Document reproducible training and deployment workflows in `README.md`. | Fasih | P0 |

---

## 6. Non-Functional Requirements

### 6.1 Performance & Latency
- **Inference Latency**: Client-side inference execution must take $< 5\text{ms}$ upon dropdown change.
- **Initial Load Time**: Total transfer size $< 50\text{KB}$ uncompressed; initial render $< 1\text{s}$ on 3G connections.
- **Lighthouse Scores**: Performance $\ge 95$, Accessibility $\ge 95$, Best Practices $\ge 95$, SEO $\ge 95$.

### 6.2 Portability & Zero Dependencies
- Zero external runtime JavaScript frameworks (no React, Vue, or Angular).
- Zero third-party runtime chart libraries (no CDN Chart.js or D3 dependencies; hand-crafted Canvas).
- Offline-capable once loaded.

### 6.3 Accessibility & Usability
- Full keyboard navigation for all interactive controls (`<select>`, `<button>`).
- High-contrast colors conforming to WCAG 2.1 AA standards for text and semantic badges.
- Descriptive `aria-label` attributes on form inputs.

---

## 7. Success Metrics & Key Performance Indicators (KPIs)

1. **Model Baseline Parity**:
   - Linear Regression $R^2 \in [0.15, 0.30]$ on test split (Achieved: **0.1622**).
   - Logistic Regression Accuracy $\in [65\%, 80\%]$ on test split (Achieved: **68.00%**).
2. **Inference Fidelity**:
   - $100\%$ mathematical parity between Python scikit-learn predictions and client-side JavaScript outputs (tolerance $< 10^{-4}$).
3. **Deployment Reliability**:
   - Zero console errors; 100% HTTP 200 responses on all static assets.

---

## 8. Non-Goals (Out of Scope for v1)
- User accounts, authentication, or persistent database storage.
- Runtime server-side model inference or retraining APIs.
- In-browser user model retraining with CSV upload (planned for v2 roadmap).
- Asserting causal claims or deterministic judgments about individuals.
