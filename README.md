# 🎗️ Breast Cancer Survival Prediction

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14-4B8BBE)
![Status](https://img.shields.io/badge/status-complete-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

Not all breast cancer patients face the same odds. Estrogen receptor status doubles survival chances, while regional node spread hits harder in some populations than others. This project digs into two real clinical cohorts — **SEER** and **METABRIC** — to uncover what actually drives patient outcomes, combining statistical hypothesis testing with machine learning classification.

---

## What's inside

The project follows a four-step pipeline, each with its own notebook:

| Notebook | What it does |
|---|---|
| `00_data_cleaning.ipynb` | Loads raw SEER and METABRIC files, cleans and aligns them to a shared 9-variable schema, exports `clean_data_breast_cancer.xlsx` |
| `01_eda.ipynb` | Descriptive statistics, pairwise scatter plots, correlation heatmaps, violin plots |
| `02_hypothesis_testing.ipynb` | Three statistical tests on survival predictors across populations |
| `03_machine_learning.ipynb` | Trains and compares Logistic Regression, Decision Tree, and Neural Network |

If you just want to run everything at once without the notebooks, `breast_cancer_analysis.py` runs the full pipeline end to end.

---

## Key findings

- **Estrogen receptor status** is the strongest clinical predictor — positive receptors roughly double the odds of survival
- The **regional node penalty differs by population**: each extra positive node costs ~4.5 more survival months in METABRIC than in SEER (p < 0.05)
- **Logistic Regression outperforms** both Decision Tree and Neural Network — accuracy 0.80, F1 0.86 — while remaining the most interpretable
- Splitting patients by the clinical 50mm tumour threshold does **not** improve survival prediction (LR test, p > 0.05)
- The Decision Tree **overfits badly** without depth constraints (CV accuracy = 0.36), a known limitation of single trees on tabular data

---

## Datasets

| Dataset | Source | Instances | Features |
|---|---|---|---|
| SEER | [NCI / IEEE DataPort](https://ieee-dataport.org/open-access/seer-breast-cancer-data) | 4,005 | 9 |
| METABRIC | [cBioPortal / Nature](https://www.nature.com/articles/nature10983) | 1,406 | 9 |

Both datasets share these 9 features: Age, T Stage, Grade, Tumor Size, Estrogen Status, Progesterone Status, Regional Node Positive, Survival Months, and Status (alive/dead).

> ⚠️ Raw data files are not included in this repository due to patient privacy and licensing restrictions. Download them from the links above and place them in `data/` before running the cleaning notebook.

---

## Getting started

**1. Clone the repo**
```bash
git clone https://github.com/andreaz-u/breast_cancer_population_analysis.git
cd breast_cancer_population_analysis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the raw data**

Download both datasets and place them in the `data/` folder:
```
data/
├── SEER_Breast_Cancer_Dataset_.csv
└── METABRIC_raw.csv
```

**4. Run the cleaning notebook first**

Open `notebooks/00_data_cleaning.ipynb` and run all cells. This produces `data/clean_data_breast_cancer.xlsx`, which all other notebooks depend on.

**5. Run the analysis notebooks in order**
```
notebooks/01_eda.ipynb
notebooks/02_hypothesis_testing.ipynb
notebooks/03_machine_learning.ipynb
```

Or run everything at once from the terminal:
```bash
python breast_cancer_analysis.py
```

---

## Statistical tests (α = 0.05)

| # | Hypothesis | Method | Result |
|---|---|---|---|
| H1 | Regional node effect on survival months differs by population | OLS with interaction term | ✅ Rejected H₀ |
| H2 | Estrogen effect on survival outcome differs by population | Logistic regression with interaction | ✅ Rejected H₀ |
| H3 | Splitting by tumour size (>50mm) improves survival prediction | Likelihood Ratio Test (χ²) | ❌ Failed to reject H₀ |

---

## Model performance (10-fold cross-validation)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.800** | **0.847** | **0.908** | **0.863** | 0.685 |
| Neural Network | 0.786 | 0.783 | 0.850 | 0.809 | **0.734** |
| Decision Tree | 0.355 | 0.521 | 0.305 | 0.333 | 0.710 |

---

## Project structure

```
breast_cancer_population_analysis/
│
├── notebooks/
│   ├── 00_data_cleaning.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_machine_learning.ipynb
│
├── data/                          # raw data goes here — not committed
│   └── .gitkeep
│
├── breast_cancer_analysis.py      # full pipeline runner
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Limitations & next steps

- Only pre-treatment clinical variables are used — biomarkers like HER2 status, family history, and treatment type could meaningfully improve predictions
- SEER is heavily class-imbalanced (85% alive), which may inflate accuracy metrics — SMOTE or class-weighted loss functions are worth exploring
- No backward stepwise feature selection was applied to the logistic regression — a leaner model may generalise better
- With ~5,400 combined observations, deep learning is constrained — tens of thousands of records would be needed for neural networks to show a meaningful advantage
- A survival analysis approach (e.g. Cox Proportional Hazards) would make better use of Survival Months as a time-to-event outcome rather than a feature

---

## References

- Teng, J. (2019). *SEER Breast Cancer Data*. IEEE DataPort.
- Curtis, C. et al. (2012). *The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups*. Nature.
- Pereira, B. et al. (2016). *The somatic mutation profiles of 2,433 breast cancers*. Nature Communications.
- Parkin, D.M. et al. (2001). Estimating the world cancer burden: Globocan 2000. *International Journal of Cancer*, 94(2), 153–6.

---

**Jia Xin Zhu** — Statistical Methods for Data Science, University of Gothenburg, 2025

*Feel free to open an issue or reach out with questions or suggestions.*
