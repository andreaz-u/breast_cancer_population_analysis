# 🎗️ Breast Cancer Survival Prediction
### Statistical Analysis & ML Classification across Two Clinical Populations

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14-4B8BBE)
![Status](https://img.shields.io/badge/status-complete-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## Overview

This project performs a **comparative survival analysis** of breast cancer patients from two independent clinical cohorts — **SEER** (National Cancer Institute, US) and **METABRIC** (Molecular Taxonomy of Breast Cancer International Consortium, Canada–UK). The study spans the full data science pipeline: exploratory analysis, statistical hypothesis testing, and machine learning classification.

The central question is: **which clinical variables best predict whether a patient survives breast cancer, and do those relationships hold across different populations?**

---

## Key Findings

- **Estrogen receptor status** is the single strongest predictor of survival, roughly doubling the odds of survival when positive
- The effect of **positive regional lymph nodes** on survival months differs significantly between populations: each additional node reduces expected survival by ~4.5 months more in METABRIC than in SEER (p < 0.05)
- **Logistic Regression outperforms** both Decision Tree and Neural Network in accuracy (0.80), precision (0.85), and F1-score (0.86), while remaining fully interpretable
- Splitting tumours by the clinical 50mm high-risk threshold does **not** significantly improve survival status prediction (LR test, p > 0.05)
- The Decision Tree severely **overfits** (CV accuracy = 0.36) without depth constraints, highlighting the importance of regularisation

---

## Datasets

| Dataset | Source | Instances | Features |
|---|---|---|---|
| SEER | [NCI / IEEE DataPort](https://ieee-dataport.org/open-access/seer-breast-cancer-data) | 4,005 | 9 |
| METABRIC | [cBioPortal / Nature](https://www.nature.com/articles/nature10983) | 1,406 | 9 |

**Shared features:** Age, T Stage, Grade, Tumor Size, Estrogen Status, Progesterone Status, Regional Node Positive, Survival Months, Status (alive/dead)

> ⚠️ The datasets are **not included** in this repository due to patient privacy and licensing restrictions. Download links are provided above. Place both sheets in a single Excel file named `clean_data_breast_cancer.xlsx` with sheet names `SEER` and `METABRIC`.

---

## Project Structure

```
breast-cancer-survival/
│
├── breast_cancer_analysis.py   # Main analysis script (EDA → Hypothesis tests → ML)
├── clean_data_breast_cancer.xlsx  # ← not included, see Datasets section
├── README.md
└── requirements.txt
```

---

## Methods

### 1. Exploratory Data Analysis
- Descriptive statistics for both cohorts
- Pairwise scatter matrices, histograms (T Stage, Grade), and correlation heatmaps
- Violin plots for categorical vs. numerical variable relationships

### 2. Statistical Hypothesis Testing (α = 0.05)

| # | Hypothesis | Method | Result |
|---|---|---|---|
| H1 | Effect of regional nodes on survival months differs by population | OLS with interaction term | ✅ Rejected H₀ (p < 0.05) |
| H2 | Estrogen receptor effect on survival status differs by population | Logistic regression with interaction | ✅ Rejected H₀ (p < 0.05) |
| H3 | Splitting by tumour size (>50mm) improves survival prediction | Likelihood Ratio Test (χ²) | ❌ Failed to reject H₀ |

### 3. Machine Learning Classification

Three classifiers trained on the combined dataset (70/30 split, 10-fold cross-validation):

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.800** | **0.847** | **0.908** | **0.863** | 0.685 |
| Neural Network | 0.786 | 0.783 | 0.850 | 0.809 | **0.734** |
| Decision Tree | 0.355 | 0.521 | 0.305 | 0.333 | 0.710 |

---

## Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/your-username/breast-cancer-survival.git
cd breast-cancer-survival
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the data file**

Download both datasets and place them in the project root as `clean_data_breast_cancer.xlsx` (see [Datasets](#datasets)).

**4. Update the data path**

In `breast_cancer_analysis.py`, update `DATA_PATH` at the top of the file to point to your local file:
```python
DATA_PATH = "clean_data_breast_cancer.xlsx"
```

**5. Run the analysis**
```bash
python breast_cancer_analysis.py
```

The script runs sequentially through EDA, hypothesis tests, and ML evaluation, printing results and displaying plots at each stage.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
scipy
openpyxl
```

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy openpyxl
```

---

## Limitations & Future Work

- Models only include pre-treatment clinical variables; biomarkers like HER2 status and family history are excluded due to data availability
- SEER is heavily class-imbalanced (85% alive), which may inflate some classification metrics
- The Decision Tree was not tuned (no depth limit or pruning); a grid search over hyperparameters would significantly improve its performance
- A backward stepwise feature selection procedure (AIC-based) was not applied to the logistic regression — this is a natural next step
- With only ~5,400 combined observations, deep learning approaches are constrained; a larger unified dataset would enable more robust modelling

---

## References

- Teng, J. (2019). *SEER Breast Cancer Data*. IEEE DataPort.
- Curtis, C. et al. (2012). *The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups*. Nature.
- Pereira, B. et al. (2016). *The somatic mutation profiles of 2,433 breast cancers refine their genomic and transcriptomic landscapes*. Nature Communications.
- Parkin, D.M. et al. (2001). Estimating the world cancer burden: Globocan 2000. *International Journal of Cancer*, 94(2), 153–6.

---

## Author

**Jia Xin Zhu** — Statistical Methods for Data Science, University of Gothenburg, 2025

---

*If you find this project useful or have suggestions, feel free to open an issue or reach out.*
