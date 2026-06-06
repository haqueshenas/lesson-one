# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 1: Global Architecture and Dynamics of Scientific Correction
#
# Analysis:
# Correlation and Regression Analysis of National Retraction-System Indicators
#
# Correlation, multicollinearity, and regression analyses of country-level
# and article-level indicators associated with retraction burden,
# retraction delay, collaboration structure, and temporal correction dynamics.
#
# Author:
# Abbas Haghshenas
# Independent Researcher in Crop Production, Shiraz, Iran
#
# Project:
# Lesson One
# https://haqueshenas.github.io/lesson-one
#
# AI-Assisted Code Generation:
# This Python code was generated with the assistance of ChatGPT
# (OpenAI; versions 5.3–5.5) under the direction, testing,
# validation, and iterative refinement of the author.
#
# License:
# MIT License
#
# Copyright (c) 2026 Abbas Haghshenas
# =============================================================================

import os
import sys
import tkinter as tk
from tkinter import filedialog

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr, kendalltau

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =========================================================
# GLOBAL SETTINGS
# =========================================================

plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 12

sns.set_style("whitegrid")


# =========================================================
# GUI
# =========================================================


def choose_file(title):
    root = tk.Tk()
    root.withdraw()

    return filedialog.askopenfilename(
        title=title,
        filetypes=[("CSV files", "*.csv")]
    )



def choose_output_folder():
    root = tk.Tk()
    root.withdraw()

    return filedialog.askdirectory(
        title="Select Output Folder"
    )


# =========================================================
# DIRECTORY MANAGEMENT
# =========================================================


def create_directories(base):

    folders = {
        "correlation": os.path.join(base, "01_Correlation_Analysis"),
        "regression": os.path.join(base, "02_Regression_Analysis")
    }

    for f in folders.values():
        os.makedirs(f, exist_ok=True)

    return folders


# =========================================================
# SAVE FIGURES
# =========================================================


def save_figure(fig, folder, filename):

    fig.savefig(
        os.path.join(folder, f"{filename}.png"),
        bbox_inches="tight"
    )

    fig.savefig(
        os.path.join(folder, f"{filename}.pdf"),
        bbox_inches="tight"
    )

    fig.savefig(
        os.path.join(folder, f"{filename}.svg"),
        bbox_inches="tight"
    )


# =========================================================
# CORRELATION MATRICES
# =========================================================


def compute_correlation_matrix(df, variables, method="spearman"):

    corr_matrix = pd.DataFrame(
        index=variables,
        columns=variables
    )

    p_matrix = pd.DataFrame(
        index=variables,
        columns=variables
    )

    for v1 in variables:
        for v2 in variables:

            if method == "spearman":
                corr, pval = spearmanr(
                    df[v1],
                    df[v2]
                )

            elif method == "kendall":
                corr, pval = kendalltau(
                    df[v1],
                    df[v2]
                )

            corr_matrix.loc[v1, v2] = corr
            p_matrix.loc[v1, v2] = pval

    return corr_matrix.astype(float), p_matrix.astype(float)


# =========================================================
# HEATMAP
# =========================================================


def draw_heatmap(
        matrix,
        title,
        output_folder,
        filename
):

    fig, ax = plt.subplots(
        figsize=(14, 12)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={
            "label": "Correlation coefficient"
        },
        ax=ax
    )

    ax.set_title(title)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    save_figure(
        fig,
        output_folder,
        filename
    )


# =========================================================
# CROSS-DOMAIN CORRELATION
# =========================================================


def run_cross_domain_correlations(df, output_folder):

    pairs = [
        ("RR_Total_%", "Collaboration_Share_%"),
        ("Total_Mean_RD_Years", "Collaboration_Share_%"),
        ("NRS13I_Total_RR", "Collaboration_Share_%"),
        ("NRRDI_Total_RR", "Collaboration_Share_%")
    ]

    results = []

    for x, y in pairs:

        s_corr, s_p = spearmanr(
            df[x],
            df[y]
        )

        k_corr, k_p = kendalltau(
            df[x],
            df[y]
        )

        results.append({
            "Variable_1": x,
            "Variable_2": y,
            "Spearman_rho": s_corr,
            "Spearman_p": s_p,
            "Kendall_tau": k_corr,
            "Kendall_p": k_p
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        os.path.join(
            output_folder,
            "Cross_Domain_Correlations.csv"
        ),
        index=False
    )

    results_df.to_excel(
        os.path.join(
            output_folder,
            "Cross_Domain_Correlations.xlsx"
        ),
        index=False
    )

    return results_df


# =========================================================
# VIF
# =========================================================


def calculate_vif(df, predictors):

    X = df[predictors].copy()
    X = sm.add_constant(X)

    vif_df = pd.DataFrame()

    vif_df["Variable"] = X.columns
    vif_df["VIF"] = [
        variance_inflation_factor(
            X.values,
            i
        )
        for i in range(X.shape[1])
    ]

    return vif_df


# =========================================================
# OLS REGRESSION
# =========================================================


def run_ols_models(country_df, output_folder):

    predictors = [
        "RR_Total_%",
        "Collaboration_Share_%"
    ]

    vif_df = calculate_vif(
        country_df,
        predictors
    )

    vif_df.to_csv(
        os.path.join(
            output_folder,
            "VIF_Table.csv"
        ),
        index=False
    )

    # Model A
    model_a = smf.ols(
        formula='''
        Total_Mean_RD_Years ~
        Q("RR_Total_%") +
        Q("Collaboration_Share_%")
        ''',
        data=country_df
    ).fit()

    model_a.summary2().tables[1].to_csv(
        os.path.join(
            output_folder,
            "OLS_Model_A.csv"
        )
    )

    # Model B
    model_b = smf.ols(
        formula='''
        Q("RR_Total_%") ~
        Total_Mean_RD_Years +
        Q("Collaboration_Share_%")
        ''',
        data=country_df
    ).fit()

    model_b.summary2().tables[1].to_csv(
        os.path.join(
            output_folder,
            "OLS_Model_B.csv"
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.scatter(
        model_a.fittedvalues,
        model_a.resid
    )

    ax.axhline(
        y=0,
        linestyle="--"
    )

    ax.set_xlabel(
        "Fitted values"
    )

    ax.set_ylabel(
        "Residuals"
    )

    ax.set_title(
        "Residual Diagnostics (OLS Model A)"
    )

    save_figure(
        fig,
        output_folder,
        "OLS_Residual_Diagnostics"
    )

    return model_a, model_b


# =========================================================
# LOGISTIC REGRESSION
# =========================================================


def run_logistic_regression(article_df, output_folder):

    article_df = article_df.copy()

    article_df["Type_Binary"] = (
        article_df["Type"] == "Collab"
    ).astype(int)

    logit_model = smf.logit(
        formula="""
        Type_Binary ~ RD_Years
        """,
        data=article_df
    ).fit(disp=False)

    logit_summary = logit_model.summary2().tables[1]

    logit_summary.to_csv(
        os.path.join(
            output_folder,
            "Logistic_Model.csv"
        )
    )

    return logit_model


# =========================================================
# QUANTILE REGRESSION
# =========================================================


def run_quantile_regression(article_df, output_folder):

    article_df = article_df.copy()

    article_df["Type_Binary"] = (
        article_df["Type"] == "Collab"
    ).astype(int)

    quantiles = [0.25, 0.50, 0.75]

    all_results = []

    for q in quantiles:

        model = smf.quantreg(
            "RD_Years ~ Type_Binary",
            article_df
        )

        result = model.fit(q=q)

        result_df = result.summary2().tables[1]

        result_df["Quantile"] = q

        all_results.append(result_df)

    final_df = pd.concat(all_results)

    final_df.to_csv(
        os.path.join(
            output_folder,
            "Quantile_Regression.csv"
        )
    )

    return final_df


# =========================================================
# INTERPRETIVE REPORT
# =========================================================


def write_correlation_report(
        output_folder,
        variables
):

    with open(
        os.path.join(
            output_folder,
            "Interpretive_Correlation_Report.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "CORRELATION ANALYSIS REPORT\n\n"
        )

        f.write(
            "Input source: Retraction_Rates_By_Country.csv\n\n"
        )

        f.write(
            "Variables included:\n"
        )

        for v in variables:
            f.write(f"- {v}\n")

        f.write(
            "\nMethods:\n"
        )

        f.write(
            "- Spearman correlation\n"
        )

        f.write(
            "- Kendall rank correlation\n"
        )

        f.write(
            "- Cross-domain restricted correlation analysis\n\n"
        )

        f.write(
            "Important note:\n"
        )

        f.write(
            "Composite indicators were not directly "
            "correlated with their source variables "
            "in inferential interpretation to avoid "
            "part-whole dependency bias.\n"
        )



def write_regression_report(
        output_folder,
        model_a,
        model_b,
        logit_model
):

    with open(
        os.path.join(
            output_folder,
            "Interpretive_Regression_Report.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "REGRESSION ANALYSIS REPORT\n\n"
        )

        f.write(
            "Input sources:\n"
        )

        f.write(
            "- Country-level dataset\n"
        )

        f.write(
            "- Article-level dataset\n\n"
        )

        f.write(
            "Models:\n"
        )

        f.write(
            "Model A: RD as outcome\n"
        )

        f.write(
            "Model B: RR as outcome\n"
        )

        f.write(
            "Model C: Logistic model "
            "(collaboration probability)\n\n"
        )

        f.write(
            "Model A R-squared: "
            f"{model_a.rsquared:.4f}\n"
        )

        f.write(
            "Model B R-squared: "
            f"{model_b.rsquared:.4f}\n"
        )

        f.write(
            "Logistic pseudo R-squared: "
            f"{logit_model.prsquared:.4f}\n"
        )


# =========================================================
# MAIN CORRELATION ANALYSIS
# =========================================================


def run_correlation_analysis(
        country_df,
        output_folder
):

    variables = [
        "RR_Total_%",
        "RR_Single_%",
        "RR_Collab_%",
        "Total_Mean_RD_Years",
        "Single_Mean_RD_Years",
        "Collab_Mean_RD_Years",
        "Total_Slope13",
        "Single_Slope13",
        "Collab_Slope13",
        "Collaboration_Share_%"
    ]

    spearman_corr, spearman_p = compute_correlation_matrix(
        country_df,
        variables,
        method="spearman"
    )

    kendall_corr, kendall_p = compute_correlation_matrix(
        country_df,
        variables,
        method="kendall"
    )

    spearman_corr.to_csv(
        os.path.join(
            output_folder,
            "Spearman_Correlation.csv"
        )
    )

    kendall_corr.to_csv(
        os.path.join(
            output_folder,
            "Kendall_Correlation.csv"
        )
    )

    spearman_p.to_csv(
        os.path.join(
            output_folder,
            "Spearman_Pvalues.csv"
        )
    )

    kendall_p.to_csv(
        os.path.join(
            output_folder,
            "Kendall_Pvalues.csv"
        )
    )

    draw_heatmap(
        spearman_corr,
        "Spearman Correlation Matrix",
        output_folder,
        "Spearman_Heatmap"
    )

    draw_heatmap(
        kendall_corr,
        "Kendall Correlation Matrix",
        output_folder,
        "Kendall_Heatmap"
    )

    run_cross_domain_correlations(
        country_df,
        output_folder
    )

    write_correlation_report(
        output_folder,
        variables
    )


# =========================================================
# MAIN
# =========================================================


def main():

    country_file = choose_file(
        "Select Retraction_Rates_By_Country.csv"
    )

    article_file = choose_file(
        "Select All_papers_for_RD_Comparison.csv"
    )

    output_folder = choose_output_folder()

    if not country_file or not article_file:
        print(
            "Input files not selected."
        )
        sys.exit()

    country_df = pd.read_csv(
        country_file
    )

    article_df = pd.read_csv(
        article_file
    )

    folders = create_directories(
        output_folder
    )

    # Correlation
    run_correlation_analysis(
        country_df,
        folders["correlation"]
    )

    # Regression
    model_a, model_b = run_ols_models(
        country_df,
        folders["regression"]
    )

    logit_model = run_logistic_regression(
        article_df,
        folders["regression"]
    )

    run_quantile_regression(
        article_df,
        folders["regression"]
    )

    write_regression_report(
        folders["regression"],
        model_a,
        model_b,
        logit_model
    )

    print(
        "All analyses completed successfully."
    )


if __name__ == "__main__":
    main()