# GradeCompass - End-User Guide & Interactive Manual 🧭

Welcome to the comprehensive User Guide for **GradeCompass**, an educational machine learning web application designed to explore how background demographic and preparation factors correlate with academic outcomes.

---

## 📑 Table of Contents
1. [Introduction & Purpose](#1-introduction--purpose)
2. [Getting Started & System Requirements](#2-getting-started--system-requirements)
3. [User Interface Overview](#3-user-interface-overview)
4. [Step-by-Step Usage Guide](#4-step-by-step-usage-guide)
   - [Configuring Student Demographics](#41-configuring-student-demographics)
   - [Interpreting Real-Time Predictions](#42-interpreting-real-time-predictions)
   - [Analyzing Test Prep Sensitivity](#43-analyzing-test-prep-sensitivity)
   - [Resetting and Experimenting](#44-resetting-and-experimenting)
5. [Understanding the Predictions](#5-understanding-the-predictions)
   - [Predicted Average Score (Linear Regression)](#51-predicted-average-score-linear-regression)
   - [Pass / At-Risk Classification (Logistic Regression)](#52-pass--at-risk-classification-logistic-regression)
6. [The "About the Model" Evaluation Dashboard](#6-the-about-the-model-evaluation-dashboard)
7. [Ethical Considerations & Disclaimers](#7-ethical-considerations--disclaimers)
8. [Frequently Asked Questions (FAQ)](#8-frequently-asked-questions-faq)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Introduction & Purpose

**GradeCompass** is an educational, interactive analytics tool powered by client-side Machine Learning. Using historical patterns from the public Kaggle *Students Performance in Exams* dataset (1,000 students), GradeCompass demonstrates comparative machine learning by simultaneously evaluating:
1. **Continuous Score Prediction**: An expected average exam score across Math, Reading, and Writing (0–100 scale) calculated via **Linear Regression**.
2. **Outcome Classification**: A probability estimation of whether a student is likely to pass (average score $\ge 60$) or is at academic risk, calculated via **Logistic Regression**.

### Target Audience
- **Educators & Academic Counselors**: Exploring the statistical correlation between preparatory interventions and performance.
- **Students & Self-Learners**: Understanding the power of exam preparation and study support.
- **Machine Learning Students**: Investigating how regression and classification algorithms behave side-by-side on identical input data in a pure client-side environment.

---

## 2. Getting Started & System Requirements

### Browser Compatibility
GradeCompass is built with pure standard Web APIs (HTML5 Canvas, SVG, ES6+ JavaScript, CSS3 Grid/Flexbox) and requires **no external plugins or runtime backend**. It is fully compatible with:
- Google Chrome 90+
- Mozilla Firefox 88+
- Apple Safari 14+
- Microsoft Edge 90+
- Mobile browsers (iOS Safari, Android Chrome)

### Accessing the Application
- **Production URL**: Deployed as a static application on Vercel.
- **Local Host**: Open `web/index.html` via any local static web server:
  ```bash
  python -m http.server 8000 --directory web
  ```
  Navigate to `http://localhost:8000`.

---

## 3. User Interface Overview

The GradeCompass interface is organized into a clean, modern two-column dashboard:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  🧭 GradeCompass                 [ Predictor ]   [ About the Model ]     │
├────────────────────────────────────┬────────────────────────────────────┤
│  STUDENT PROFILE (Inputs)          │  LIVE PREDICTIONS (Dual Inference) │
│                                    │                                    │
│  Gender:                           │  ┌───────────────┐ ┌─────────────┐ │
│  [ Female ▾ ]                      │  │ Score Gauge   │ │ Pass/Risk   │ │
│                                    │  │     72.9      │ │   Badge     │ │
│  Race / Ethnicity:                 │  │  out of 100   │ │ Pass: 88.0% │ │
│  [ Group B ▾ ]                     │  └───────────────┘ └─────────────┘ │
│                                    │                                    │
│  Parental Level of Education:      │  TEST PREP IMPACT (Sensitivity)    │
│  [ Bachelor's Degree ▾ ]           │  ┌───────────────────────────────┐ │
│                                    │  │ Score: None (69.2)            │ │
│  Lunch Program:                    │  │ Score: Completed (76.7)       │ │
│  [ Standard Lunch ▾ ]              │  │ Delta: +7.5 pts advantage     │ │
│                                    │  │ --------- Pass Mark (60) ---- │ │
│  Test Preparation Course:          │  └───────────────────────────────┘ │
│  [ None (No Course) ▾ ]            │                                    │
│                                    │                                    │
│  [ ↺ Reset Selections ]            │                                    │
├────────────────────────────────────┴────────────────────────────────────┤
│  ⚠️ Disclaimer: Predictions reflect statistical correlations in sample...│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Step-by-Step Usage Guide

### 4.1 Configuring Student Demographics
Use the 5 dropdown selectors on the left panel to define a demographic profile:

1. **Gender**:
   - `Female`: Correlates historically with higher reading and writing marks.
   - `Male`: Correlates historically with higher math marks.
2. **Race / Ethnicity**:
   - Categorized from `Group A` through `Group E` (anonymized sociological cohorts from the Kaggle dataset).
3. **Parental Level of Education**:
   - Options: `Some High School`, `High School`, `Some College`, `Associate's Degree`, `Bachelor's Degree`, `Master's Degree`.
4. **Lunch Program**:
   - `Standard Lunch`: Student receives standard meal pricing (socioeconomic stability proxy).
   - `Free / Reduced Lunch`: Student qualifies for subsidized meal support (socioeconomic hardship proxy).
5. **Test Preparation Course**:
   - `None (No Course)`: Student did not complete preparatory coursework.
   - `Completed Course`: Student completed formal exam preparation.

### 4.2 Interpreting Real-Time Predictions
Every change immediately triggers client-side inference without requiring a "Submit" button:

- **Predicted Average Score**:
  - The circular animated gauge visually illustrates the score from $0$ to $100$.
  - Values $\ge 60$ illuminate in **Emerald Green**.
  - Values $< 60$ illuminate in **Crimson Red**.
- **Pass / At-Risk Classification**:
  - Displays a high-contrast badge: **`Likely to Pass`** (Green) or **`At Risk`** (Red).
  - Displays the estimated probability percentage (e.g., `87.97%`).
  - An animated horizontal progress bar represents the probability relative to the $50\%$ decision boundary.

### 4.3 Analyzing Test Prep Sensitivity
The lower right panel features a dynamic HTML5 Canvas chart:
- It isolates the **Test Preparation Course** variable while keeping all other 4 selections identical.
- **Left Bar (Amber/Blue)**: Predicted average score without preparation course.
- **Right Bar (Teal/Emerald)**: Predicted average score with completed preparation course.
- **Red Dashed Line**: The contractual pass threshold ($60$ points).
- **Delta Indicator**: Highlights the exact score advantage gained from taking the test preparation course (typically $+7.5$ points).

### 4.4 Resetting and Experimenting
Click the **"Reset Selections"** button at any time to return all dropdowns to their default starting options and reset the live prediction panels.

---

## 5. Understanding the Predictions

### 5.1 Predicted Average Score (Linear Regression)
- **Mathematical Model**: Ordinary Least Squares (OLS) trained on standardized features.
- **Interpretation**: Represents the expected continuous grade average across Math, Reading, and Writing.
- **Model Confidence**: Demographics explain $\approx 16.2\%$ ($R^2 = 0.1622$) of the variance in score. On average, predictions deviate by approximately $10.49$ points (MAE) from actual exam results.

### 5.2 Pass / At-Risk Classification (Logistic Regression)
- **Mathematical Model**: Binary Logistic Regression utilizing the standard sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$.
- **Threshold**:
  - A student whose average score is expected to be $\ge 60$ is categorized as **Pass** ($1$).
  - If the logistic model yields $P(Pass) \ge 0.50$, the system issues **"Likely to Pass"**.
  - If $P(Pass) < 0.50$, the system issues **"At Risk"**.
- **Model Confidence**: Achieves **$68.00\%$ overall accuracy** with **$86.96\%$ recall** on test data.

---

## 6. The "About the Model" Evaluation Dashboard
Click **"About the Model"** in the top navigation bar to access detailed technical documentation directly in the browser:
- **Dataset Framing**: Explanation of why subject scores are excluded from model inputs.
- **Performance Tables**: Train and test metrics ($R^2$, MAE, Accuracy, Precision, Recall).
- **Confusion Matrix**: Visual table showing true positives ($120$), true negatives ($16$), false positives ($46$), and false negatives ($18$).
- **Team Split**: Attributions for Fasih and Abdul Hayy.

---

## 7. Ethical Considerations & Disclaimers

> [!IMPORTANT]
> **Correlation Does Not Equal Causation**  
> Predictions generated by GradeCompass reflect statistical correlations present within a historical sample dataset of 1,000 students from Kaggle.
> - Demographic attributes (such as gender, ethnicity, or parental education) **do not determine** innate intelligence, ability, or personal potential.
> - Structural disparities reflected in the data (such as lunch subsidies) are systemic social indicators rather than individual deficits.
> - GradeCompass is strictly an **educational tool** demonstrating data science principles and must never be used for admissions, academic tracking, or student evaluations.

---

## 8. Frequently Asked Questions (FAQ)

**Q: Is my student data sent to a remote server?**  
A: **No.** GradeCompass runs $100\%$ client-side inside your browser. Your selections never leave your device.

**Q: Why doesn't the model ask for my homework or study hours?**  
A: The public Kaggle dataset only includes demographic background features and prep course status. This project is specifically designed to explore what demographic signals alone can—and cannot—predict.

**Q: Can I use GradeCompass offline?**  
A: Yes! Once the page and `assets/model_params.json` have loaded, GradeCompass requires no active internet connection.

---

## 9. Troubleshooting

| Issue | Potential Cause | Resolution |
|---|---|---|
| Red error banner: *"Unable to fetch model parameters"* | Running via `file://` protocol where browser blocks `fetch()` | Run via a local HTTP server: `python -m http.server 8000 --directory web`. |
| Predictions do not update when changing dropdowns | JavaScript disabled in browser | Enable JavaScript in your browser preferences. |
| Bar chart appears blurry on high-resolution displays | Browser zoom or canvas rendering quirk | The chart uses automatic `devicePixelRatio` scaling. Refresh the page or reset browser zoom to $100\%$. |
| Dropdown shows empty selection | Corrupted or incomplete JSON payload | Re-run `python training/merge_params.py` to regenerate `model_params.json`. |
