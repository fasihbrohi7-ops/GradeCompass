# GradeCompass - Dataset Card: Students Performance in Exams 📊

**Dataset Name:** Students Performance in Exams  
**Source:** Kaggle Public Datasets ([Kaggle Link](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams))  
**Curator:** Roy (spscientist)  
**File Location:** [`dataset/StudentsPerformance.csv`](../dataset/StudentsPerformance.csv)  
**Dataset Size:** 1,000 observations, 8 attributes  
**Missing Values:** 0 null cells across all columns  

---

## 1. Dataset Overview

The *Students Performance in Exams* dataset contains fictional/anonymized records of high school students' exam scores alongside their demographic and preparatory background attributes. The goal within GradeCompass is to evaluate how much variance in student exam performance can be explained by background characteristics and whether an academic intervention (test preparation course) creates measurable predictive separation.

---

## 2. Data Dictionary

### 2.1 Original Dataset Attributes

| Column Name | Type | Unique Values | Description / Categories | Model Role |
|---|---|---|---|---|
| `gender` | Categorical (Binary) | 2 | `female` (518), `male` (482) | Input Feature (One-Hot) |
| `race/ethnicity` | Categorical (Nominal) | 5 | `group A` (89), `group B` (190), `group C` (319), `group D` (262), `group E` (140) | Input Feature (One-Hot) |
| `parental level of education` | Categorical (Ordinal) | 6 | `some high school` (179), `high school` (196), `some college` (226), `associate's degree` (222), `bachelor's degree` (118), `master's degree` (59) | Input Feature (One-Hot) |
| `lunch` | Categorical (Binary) | 2 | `standard` (645), `free/reduced` (355) | Input Feature (Socioeconomic proxy) |
| `test preparation course` | Categorical (Binary) | 2 | `none` (642), `completed` (358) | Input Feature (Intervention proxy) |
| `math score` | Integer ($0$–$100$) | 81 | Continuous test score ($0$–$100$) | **Target Formulation Only** |
| `reading score` | Integer ($17$–$100$) | 72 | Continuous test score ($0$–$100$) | **Target Formulation Only** |
| `writing score` | Integer ($10$–$100$) | 77 | Continuous test score ($0$–$100$) | **Target Formulation Only** |

---

### 2.2 Derived Target Attributes

| Target Name | Formula | Type | Range / Values | Predictor Model |
|---|---|---|---|---|
| `average_score` | $\frac{\text{math} + \text{reading} + \text{writing}}{3}$ | Float | $[9.0, 100.0]$, Mean: $67.77$ | **Linear Regression** |
| `passed` | $\mathbb{I}(\text{average\_score} \ge 60)$ | Binary | $1$ (Passed: 715, $71.5\%$), $0$ (Failed: 285, $28.5\%$) | **Logistic Regression** |

---

## 3. Summary Statistics

### 3.1 Continuous Score Distribution (1,000 Students)

| Metric | Math Score | Reading Score | Writing Score | Combined Average Score |
|---|---|---|---|---|
| **Mean** | 66.09 | 69.17 | 68.05 | **67.77** |
| **Standard Deviation** | 15.16 | 14.60 | 15.20 | **14.26** |
| **Minimum** | 0 | 17 | 10 | **9.00** |
| **25th Percentile (Q1)** | 57.00 | 59.00 | 57.75 | **58.33** |
| **Median (50th)** | 66.00 | 70.00 | 69.00 | **68.33** |
| **75th Percentile (Q3)** | 77.00 | 79.00 | 79.00 | **77.67** |
| **Maximum** | 100 | 100 | 100 | **100.00** |

---

## 4. Key Subgroup Analysis & Observations

1. **Test Preparation Course Impact**:
   - Students who completed test prep scored an average of **$72.67$**, compared to **$65.04$** for students with no prep ($+7.63$ points average advantage).
   - Pass rate among test prep completers: **$81.8\%$** vs. **$65.7\%$** for non-completers.

2. **Lunch Subsidy (Socioeconomic Indicator)**:
   - Students receiving standard lunch scored an average of **$70.84$** ($78.6\%$ pass rate).
   - Students receiving free/reduced lunch scored an average of **$62.20$** ($58.6\%$ pass rate).

3. **Parental Education**:
   - Master's Degree: Mean score **$73.60$** ($83.1\%$ pass rate).
   - Bachelor's Degree: Mean score **$71.92$** ($82.2\%$ pass rate).
   - High School: Mean score **$63.10$** ($58.7\%$ pass rate).

---

## 5. Ethical & Societal Considerations

- **Anonymization**: The Kaggle dataset does not contain personally identifiable information (PII). Race/ethnicity is labeled generically as groups A through E.
- **Correlation vs. Causation**: While factors like free lunch program eligibility strongly correlate with lower exam scores, this represents socioeconomic hardship and resource disparity—not a lack of innate academic capability.
- **Model Limitations**: Because this dataset excludes crucial variables like student attendance, study time, school funding, and curriculum quality, the models explain roughly $16.2\%$ of score variance. This modesty is an important lesson in ethical machine learning.
