"""
================================================================================
  Breast Cancer Survival Analysis: SEER vs METABRIC
  Statistical Methods for Data Science — Project
================================================================================
  Author : Jia Xin Zhu
  Date   : 2025

  Note:
      For a step-by-step walkthrough with commentary, use the notebooks:
        notebooks/01_eda.ipynb
        notebooks/02_hypothesis_testing.ipynb
        notebooks/03_machine_learning.ipynb

  Pipeline:
        1. Exploratory Data Analysis (EDA) with visualisations
        2. Statistical Hypothesis Testing (OLS, Logistic regression)
        3. Machine Learning Classification (Logistic Regression, Decision Tree,
           Neural Network) with cross-validation

  Usage:
      python breast_cancer_analysis.py
================================================================================
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import math
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# Suppress convergence warnings from statsmodels and sklearn during fitting.
# Remove this if you want to debug model convergence issues.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Global plot style ──────────────────────────────────────────────────────────
PALETTE   = {"SEER": "#2C7BB6", "METABRIC": "#D7191C"}
BG_COLOR  = "#F8F9FA"
GRID_COLOR = "#E0E0E0"

plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    BG_COLOR,
    "axes.edgecolor":    "#CCCCCC",
    "axes.grid":         True,
    "grid.color":        GRID_COLOR,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "axes.labelsize":    11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "font.family":       "DejaVu Sans",
})

# ── Configuration ─────────────────────────────────────────────────────────────
# Update this path to point to your local copy of the dataset.
# The file should be an Excel workbook with two sheets: "SEER" and "METABRIC".
# If this file does not exist yet, run notebooks/00_data_cleaning.ipynb first.
DATA_PATH = "data/clean_data_breast_cancer.xlsx"

import os
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Clean data file not found at '{DATA_PATH}'. "
        "Run notebooks/00_data_cleaning.ipynb first to generate it."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load SEER and METABRIC sheets from the Excel workbook."""
    with open(path, "rb") as f:
        seer     = pd.read_excel(f, sheet_name="SEER")
        metabric = pd.read_excel(f, sheet_name="METABRIC")
    print("✔  Data loaded successfully.")
    print(f"   SEER:     {seer.shape[0]:,} rows × {seer.shape[1]} columns")
    print(f"   METABRIC: {metabric.shape[0]:,} rows × {metabric.shape[1]} columns\n")
    return seer, metabric


def print_descriptive_stats(df: pd.DataFrame, label: str) -> None:
    """Print a formatted descriptive statistics table."""
    print(f"\n{'─'*60}")
    print(f"  Descriptive Statistics — {label}")
    print(f"{'─'*60}")
    print(df.describe().round(2).to_string())
    print()


# ── 1a. Pairplots ─────────────────────────────────────────────────────────────

def plot_pairplot(df: pd.DataFrame, label: str, color: str) -> None:
    """Pairwise scatter plot for numeric variables."""
    cols = ["Age", "Tumor Size", "Regional Node 1", "Survival Months"]
    g = sns.pairplot(
        df[cols],
        plot_kws={"s": 4, "alpha": 0.4, "color": color},
        diag_kws={"color": color, "alpha": 0.6},
    )
    g.figure.suptitle(f"{label}: Pairwise Scatter Matrix", y=1.02, fontsize=14, fontweight="bold")
    for ax in g.axes[:, 0]:
        ax.yaxis.label.set_rotation(45)
        ax.yaxis.label.set_ha("right")
    plt.tight_layout()
    plt.show()


# ── 1b. Histograms for ordinal categoricals ────────────────────────────────────

def plot_histograms(df: pd.DataFrame, cols: list[str], label: str, color: str) -> None:
    """Side-by-side histograms for categorical ordinal variables."""
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=20, color=color, alpha=0.75, edgecolor="white")
        ax.set_title(f"{label}: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
    fig.suptitle(f"{label} — Ordinal Variable Distributions", fontweight="bold")
    plt.tight_layout()
    plt.show()


# ── 1c. Violin plots ──────────────────────────────────────────────────────────

def plot_violins(
    df: pd.DataFrame,
    cat_vars: list[str],
    num_vars: list[str],
    label: str,
    color: str,
) -> None:
    """Grid of violin plots: each numerical variable × each categorical variable."""
    n_rows, n_cols = len(num_vars), len(cat_vars)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))

    for r, num_var in enumerate(num_vars):
        for c, cat_var in enumerate(cat_vars):
            ax = axes[r][c] if n_rows > 1 else axes[c]
            sns.violinplot(
                data=df, x=cat_var, y=num_var,
                color=color, alpha=0.7, linewidth=0.8, ax=ax,
            )
            ax.set_title(f"{num_var} vs {cat_var}", fontsize=9)
            ax.set_xlabel("")
            ax.set_ylabel(num_var if c == 0 else "")
            ax.tick_params(axis="x", labelsize=7)

    fig.suptitle(f"{label} — Violin Plots", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ── 1d. Correlation heatmap ───────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, label: str) -> None:
    """Annotated correlation heatmap."""
    corr = df.select_dtypes(include="number").corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))          # upper triangle mask
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.4,
        annot_kws={"size": 8}, ax=ax,
    )
    ax.set_title(f"{label} — Correlation Heatmap", pad=12)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HYPOTHESIS TESTING
# ══════════════════════════════════════════════════════════════════════════════

def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}\n")


def hypothesis_1_ols(data: pd.DataFrame) -> smf.ols:
    """
    H1: The effect of Regional Nodes on Survival Months differs
        between SEER and METABRIC populations (interaction OLS).

    Tests whether the interaction term (Population × Regional_Node) is
    significantly different from zero. A significant result means the
    per-node penalty on survival months is not the same across cohorts.
    Significance level: α = 0.05.
    """
    print_section("Hypothesis 1 — OLS: Survival ~ Population × Regional Nodes")
    model  = smf.ols("Survival_Months ~ Population * Regional_Node", data=data)
    result = model.fit()
    print(result.summary())

    # Extract the interaction coefficient and its standard error directly from
    # the fitted model instead of hardcoding, so results stay accurate if the
    # data changes.
    interaction_term = "Population:Regional_Node"
    beta = result.params[interaction_term]
    se   = result.bse[interaction_term]
    n    = len(data)
    t0 = beta / (se / math.sqrt(n))
    pv = 2 * min(stats.t.cdf(t0, df=n - 1), 1 - stats.t.cdf(t0, df=n - 1))
    print(f"\n  Interaction coefficient : {beta}")
    print(f"  t-statistic            : {t0:.4f}")
    print(f"  p-value                : {pv:.4e}")
    print(f"  Decision               : {'Reject H₀ ✔' if pv < 0.05 else 'Fail to reject H₀'}\n")
    return result


def hypothesis_2_logit(data: pd.DataFrame) -> smf.logit:
    """
    H2: The effect of Estrogen Status on survival Status differs
        between populations (interaction logistic regression).

    Tests whether the interaction term (Population × Estrogen) is
    significantly different from zero. A significant result means that
    the survival benefit of positive estrogen receptors is not consistent
    across cohorts — i.e. the population moderates the estrogen effect.
    Significance level: α = 0.05.
    """
    print_section("Hypothesis 2 — Logit: Status ~ Population × Estrogen")
    model  = smf.logit("Status ~ Population * Estrogen", data=data)
    result = model.fit()
    print(result.summary())

    # Extract the interaction coefficient and its standard error directly from
    # the fitted model instead of hardcoding, so results stay accurate if the
    # data changes.
    interaction_term = "Population:Estrogen"
    beta = result.params[interaction_term]
    se   = result.bse[interaction_term]
    n    = len(data)
    t0 = beta / (se / math.sqrt(n))
    pv = 2 * min(stats.t.cdf(t0, df=n - 1), 1 - stats.t.cdf(t0, df=n - 1))
    print(f"\n  Interaction coefficient : {beta}")
    print(f"  t-statistic            : {t0:.4f}")
    print(f"  p-value                : {pv:.4e}")
    print(f"  Decision               : {'Reject H₀ ✔' if pv < 0.05 else 'Fail to reject H₀'}\n")
    return result


def hypothesis_3_tumor_encoding(data: pd.DataFrame) -> None:
    """
    H3: Continuous vs binary-encoded Tumor Size — compare logistic models
        via AIC and Likelihood Ratio Test.

    Tests whether binarising Tumor Size at the clinical 50mm high-risk
    threshold produces a better-fitting model than treating it as a
    continuous predictor. A non-significant LR test means the simpler
    continuous encoding is preferred. Significance level: α = 0.05.
    """
    print_section("Hypothesis 3 — Tumor Size: Continuous vs Binary Encoding")

    data = data.copy()
    data["TumorSizeMapped"] = (data["Tumor_Size"] >= 50).astype(int)

    formula_base = "Status ~ Age + Estrogen + Progesterone + Grade + Regional_Node"

    m1 = smf.logit(f"{formula_base} + Tumor_Size", data=data).fit(disp=False)
    m2 = smf.logit(f"{formula_base} + C(TumorSizeMapped)", data=data).fit(disp=False)

    print("  Model 1 (continuous Tumor Size):")
    print(m1.summary().tables[1])
    print("\n  Model 2 (binary Tumor Size ≥50mm):")
    print(m2.summary().tables[1])

    lr_stat = 2 * (m2.llf - m1.llf)
    p_lr    = 1 - stats.chi2.cdf(lr_stat, df=1)

    print(f"\n  AIC  — Model 1: {m1.aic:.2f}   |   Model 2: {m2.aic:.2f}")
    print(f"  LR statistic : {lr_stat:.4f}")
    print(f"  LR p-value   : {p_lr:.4e}")
    preferred = "Model 1 (continuous)" if m1.aic < m2.aic else "Model 2 (binary)"
    print(f"  Preferred    : {preferred}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MACHINE LEARNING MODELS
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    X_train, X_test, y_train, y_test, X, y
) -> dict:
    """Train Logistic Regression, Decision Tree, and MLP; return metrics."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Neural Network":      MLPClassifier(hidden_layer_sizes=(50, 50),
                                             max_iter=1000, random_state=42),
    }

    results  = {}
    cv_store = {}

    for name, clf in models.items():
        print(f"\n{'─'*60}")
        print(f"  {name}")
        print(f"{'─'*60}")

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        print(classification_report(y_test, preds))
        auc = roc_auc_score(y_test, preds)
        print(f"  AUC-ROC: {auc:.4f}")

        # 10-fold cross-validation
        cv_metrics = {}
        for metric in ("accuracy", "precision", "recall", "f1"):
            scores = cross_val_score(clf, X, y, cv=10, scoring=metric)
            cv_metrics[metric] = scores.mean()
            print(f"  CV {metric:<10}: {scores.mean():.4f} ± {scores.std():.4f}")

        results[name]  = {"model": clf, "auc": auc}
        cv_store[name] = cv_metrics

    return results, cv_store


def plot_cv_comparison(cv_store: dict) -> None:
    """Grouped bar chart comparing CV metrics across models."""
    metrics  = ["accuracy", "precision", "recall", "f1"]
    n_models = len(cv_store)
    x        = np.arange(len(metrics))
    width    = 0.22
    colors   = ["#2C7BB6", "#1A9641", "#D7191C"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model_name, cv_metrics) in enumerate(cv_store.items()):
        vals = [cv_metrics[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model_name,
                      color=colors[i], alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Score")
    ax.set_title("10-Fold Cross-Validation: Model Comparison", pad=12)
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(results: dict, X: pd.DataFrame) -> None:
    """Feature importance plots for each trained model."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors    = ["#2C7BB6", "#1A9641", "#D7191C"]

    # ── Logistic Regression coefficients ──────────────────────────────────────
    clf_lr = results["Logistic Regression"]["model"]
    fi_lr  = pd.Series(clf_lr.coef_[0], index=X.columns).sort_values()
    fi_lr.plot(kind="barh", ax=axes[0], color=[
        colors[0] if v >= 0 else colors[2] for v in fi_lr
    ], edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("Logistic Regression\nCoefficients")
    axes[0].set_xlabel("Coefficient value")

    # ── Decision Tree importances ──────────────────────────────────────────────
    clf_dt = results["Decision Tree"]["model"]
    fi_dt  = pd.Series(clf_dt.feature_importances_, index=X.columns).sort_values()
    fi_dt.plot(kind="barh", ax=axes[1], color=colors[1], edgecolor="white", alpha=0.85)
    axes[1].set_title("Decision Tree\nFeature Importances")
    axes[1].set_xlabel("Gini importance")

    # ── Neural Network mean absolute input weights ─────────────────────────────
    clf_nn = results["Neural Network"]["model"]
    nn_fi  = (
        pd.DataFrame(clf_nn.coefs_[0].T, columns=X.columns)
        .abs().mean().sort_values()
    )
    nn_fi.plot(kind="barh", ax=axes[2], color=colors[2], edgecolor="white", alpha=0.85)
    axes[2].set_title("Neural Network\nMean |Input Weights|")
    axes[2].set_xlabel("Mean absolute weight")

    for ax in axes:
        ax.grid(axis="x", linestyle="--", linewidth=0.6)

    fig.suptitle("Feature Importance Across Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load data ──────────────────────────────────────────────────────────────
    seer, metabric = load_data(DATA_PATH)

    # ── Descriptive statistics ─────────────────────────────────────────────────
    print_descriptive_stats(seer,     "SEER")
    print_descriptive_stats(metabric, "METABRIC")

    # ── EDA plots ──────────────────────────────────────────────────────────────
    cat_vars = ["T Stage", "Grade", "Estrogen Status", "Progesterone Status", "Status"]
    num_vars = ["Age", "Tumor Size", "Regional Node 1", "Survival Months"]

    for df, label, color in [
        (seer,     "SEER",     PALETTE["SEER"]),
        (metabric, "METABRIC", PALETTE["METABRIC"]),
    ]:
        plot_pairplot(df, label, color)
        plot_histograms(df, ["T Stage", "Grade"], label, color)
        plot_heatmap(df, label)
        plot_violins(df, cat_vars, num_vars, label, color)

    # ── Combine datasets ───────────────────────────────────────────────────────
    seer["Population"]     = 0
    metabric["Population"] = 1
    data = pd.concat([seer, metabric], ignore_index=True)
    data.rename(columns={
        "Survival Months":    "Survival_Months",
        "Regional Node 1":    "Regional_Node",
        "Estrogen Status":    "Estrogen",
        "Progesterone Status":"Progesterone",
        "Tumor Size":         "Tumor_Size",
        "T Stage":            "T_Stage",
    }, inplace=True)

    # ── Hypothesis tests ───────────────────────────────────────────────────────
    hypothesis_1_ols(data)
    hypothesis_2_logit(data)
    hypothesis_3_tumor_encoding(data)

    # ── ML preparation ─────────────────────────────────────────────────────────
    print_section("Machine Learning — Model Training & Evaluation")
    X = data.drop(columns=["Status"])
    y = data["Status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Train, evaluate, cross-validate ───────────────────────────────────────
    results, cv_store = train_and_evaluate(X_train, X_test, y_train, y_test, X, y)

    # ── Summary plots ──────────────────────────────────────────────────────────
    plot_cv_comparison(cv_store)
    plot_feature_importance(results, X)

    # ── Final AUC summary ──────────────────────────────────────────────────────
    print_section("Final Model Performance Summary")
    for name, info in results.items():
        print(f"  {name:<25} AUC-ROC: {info['auc']:.4f}")
    print()


if __name__ == "__main__":
    main()
