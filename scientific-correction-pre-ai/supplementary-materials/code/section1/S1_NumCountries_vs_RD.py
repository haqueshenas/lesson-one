# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 1: Global Architecture and Dynamics of Scientific Correction
#
# Analysis:
# Article-Level Analysis: Number of Countries vs Retraction Duration
#
# Descriptive statistics, correlations, Kaplan–Meier curves, Cox proportional hazards models,
# spline-based hazard modeling, quantile regression, and jitter plots
# to assess how the number of contributing countries affects retraction duration.
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
import tkinter as tk
from tkinter import filedialog
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

import statsmodels.formula.api as smf
from patsy import dmatrix
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 12


# ======================================================
# FILE DIALOG
# ======================================================

def choose_input_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV files", "*.csv")]
    )


def choose_output_dir():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(
        title="Select output folder"
    )


# ======================================================
# LOAD DATA
# ======================================================

def load_data(path):
    df = pd.read_csv(path)

    df["RD_Years"] = pd.to_numeric(
        df["RD_Years"],
        errors="coerce"
    )

    df["N_Countries"] = pd.to_numeric(
        df["N_Countries"],
        errors="coerce"
    )

    df["Type_bin"] = np.where(
        df["Type"].str.lower() == "collab",
        1,
        0
    )

    df["Event"] = 1

    df = df.dropna()

    return df


# ======================================================
# SAVE FIGURE
# ======================================================

def save_figure(fig, outdir, name):
    for ext in ["pdf", "svg", "tiff"]:
        fig.savefig(
            os.path.join(outdir, f"{name}.{ext}"),
            bbox_inches="tight"
        )


# ======================================================
# DESCRIPTIVE
# ======================================================

def descriptive_analysis(df, outdir):
    table = df.groupby("N_Countries")["RD_Years"].agg(
        Count="count",
        Mean="mean",
        Median="median",
        SD="std",
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75)
    ).reset_index()

    table.to_csv(
        os.path.join(
            outdir,
            "Table_S1_Descriptive_By_NCountries.csv"
        ),
        index=False
    )

    return table


# ======================================================
# CORRELATION
# ======================================================

def correlation_analysis(df):
    pearson = pearsonr(
        df["N_Countries"],
        df["RD_Years"]
    )

    spearman = spearmanr(
        df["N_Countries"],
        df["RD_Years"]
    )

    return pearson, spearman


# ======================================================
# KM PLOT
# ======================================================

def km_plot(df, outdir):
    bins = [0, 1, 2, 4, 8, np.inf]
    labels = ["1", "2", "3-4", "5-8", "9+"]

    df["NC_Group"] = pd.cut(
        df["N_Countries"],
        bins=bins,
        labels=labels
    )

    kmf = KaplanMeierFitter()

    # full
    fig = plt.figure(figsize=(10, 7))

    for g in labels:
        sub = df[df["NC_Group"] == g]

        if len(sub) == 0:
            continue

        kmf.fit(
            sub["RD_Years"],
            sub["Event"],
            label=f"{g} countries"
        )

        kmf.plot_survival_function()

    plt.xlabel("Retraction duration (years)")
    plt.ylabel("Probability unretracted")

    save_figure(
        fig,
        outdir,
        "Figure_S1_KM_Full"
    )

    plt.close()

    # clipped to 12 years
    fig = plt.figure(figsize=(10, 7))

    for g in labels:
        sub = df[df["NC_Group"] == g]

        if len(sub) == 0:
            continue

        kmf.fit(
            sub["RD_Years"],
            sub["Event"],
            label=f"{g} countries"
        )

        kmf.plot_survival_function()

    plt.xlim(0, 12)
    plt.xlabel("Retraction duration (years)")
    plt.ylabel("Probability unretracted")

    save_figure(
        fig,
        outdir,
        "Figure_S1_KM_Clipped12Y"
    )

    plt.close()


# ======================================================
# COX PRIMARY
# ======================================================

def cox_primary(df):
    model_df = df[
        ["RD_Years", "Event", "N_Countries"]
    ].copy()

    cph = CoxPHFitter(
        penalizer=0.01
    )

    cph.fit(
        model_df,
        duration_col="RD_Years",
        event_col="Event"
    )

    return cph, model_df


# ======================================================
# COX COLLAB ONLY
# ======================================================

def cox_collab_only(df):
    sub = df[
        df["Type"].str.lower() == "collab"
    ].copy()

    model_df = sub[
        ["RD_Years", "Event", "N_Countries"]
    ]

    cph = CoxPHFitter(
        penalizer=0.01
    )

    cph.fit(
        model_df,
        duration_col="RD_Years",
        event_col="Event"
    )

    return cph


# ======================================================
# PH TEST
# ======================================================

def ph_test(cph, model_df):
    ph = proportional_hazard_test(
        cph,
        model_df,
        time_transform="rank"
    )

    return ph.summary

# ======================================================
# RESTRICTED CUBIC SPLINE COX
# ======================================================

def spline_cox_analysis(df, outdir):

    # ==============================================
    # restrict unstable sparse tail
    # ==============================================

    spline_df_source = df[
        df["N_Countries"] <= 10
    ].copy()

    # ==============================================
    # spline basis
    # ==============================================

    spline_basis = dmatrix(
        "bs(N_Countries, df=4, include_intercept=False)",
        {"N_Countries": spline_df_source["N_Countries"]},
        return_type='dataframe'
    )

    spline_df = pd.concat(
        [
            spline_df_source[
                ["RD_Years", "Event"]
            ].reset_index(drop=True),

            spline_basis.reset_index(drop=True)
        ],
        axis=1
    )

    # ==============================================
    # spline Cox model
    # ==============================================

    cph_spline = CoxPHFitter(
        penalizer=0.01
    )

    cph_spline.fit(
        spline_df,
        duration_col="RD_Years",
        event_col="Event"
    )

    # ==============================================
    # prediction grid
    # ==============================================

    x_grid = np.linspace(
        1,
        10,
        200
    )

    grid_df = pd.DataFrame({
        "N_Countries": x_grid
    })

    grid_basis = dmatrix(
        "bs(N_Countries, df=4, include_intercept=False)",
        grid_df,
        return_type='dataframe'
    )

    pred = cph_spline.predict_partial_hazard(
        grid_basis
    )

    # normalize to 1-country baseline
    pred = pred / pred.iloc[0]

    # ==============================================
    # figure
    # ==============================================

    fig = plt.figure(figsize=(10, 7))

    plt.plot(
        x_grid,
        pred,
        linewidth=3
    )

    plt.axhline(
        1,
        linestyle="--",
        linewidth=1
    )

    plt.xlim(1, 10)

    plt.xlabel("Number of contributing countries")
    plt.ylabel("Relative hazard of retraction")

    save_figure(
        fig,
        outdir,
        "Figure_3_Spline_Cox_Hazard"
    )

    plt.close()

    # ==============================================
    # export prediction table
    # ==============================================

    out_table = pd.DataFrame({
        "N_Countries": x_grid,
        "Relative_Hazard": pred.values.flatten()
    })

    out_table.to_csv(
        os.path.join(
            outdir,
            "Table_S4_Spline_Cox.csv"
        ),
        index=False
    )

    # ==============================================
    # model summary
    # ==============================================

    cph_spline.summary.to_csv(
        os.path.join(
            outdir,
            "Table_S5_Spline_Cox_Model.csv"
        )
    )

    return cph_spline

# ======================================================
# SPLINE PLOT
# ======================================================

def spline_plot(df, outdir):
    grouped = df.groupby(
        "N_Countries"
    )["RD_Years"].median()

    # full
    fig = plt.figure(figsize=(10, 7))

    plt.plot(
        grouped.index,
        grouped.values,
        linewidth=2
    )

    plt.xlabel("Number of contributing countries")
    plt.ylabel("Median retraction duration (years)")

    save_figure(
        fig,
        outdir,
        "Figure_1_Spline_Full"
    )

    plt.close()

    # clipped
    fig = plt.figure(figsize=(10, 7))

    plt.plot(
        grouped.index,
        grouped.values,
        linewidth=2
    )

    plt.ylim(0, 12)

    plt.xlabel("Number of contributing countries")
    plt.ylabel("Median retraction duration (years)")

    save_figure(
        fig,
        outdir,
        "Figure_1_Spline_Clipped12Y"
    )

    plt.close()


# ======================================================
# QUANTILE REGRESSION
# ======================================================

def quantile_regression(df):
    quantiles = [
        0.10,
        0.25,
        0.50,
        0.75,
        0.90
    ]

    rows = []

    for q in quantiles:
        model = smf.quantreg(
            "RD_Years ~ N_Countries + Type_bin",
            df
        )

        res = model.fit(q=q)

        rows.append({
            "Quantile": q,
            "Intercept": res.params["Intercept"],
            "N_Countries": res.params["N_Countries"],
            "Type_bin": res.params["Type_bin"],
            "Pseudo_R2": res.prsquared
        })

    return pd.DataFrame(rows)


def quantile_plot(qdf, outdir):
    fig = plt.figure(figsize=(10, 7))

    plt.plot(
        qdf["Quantile"],
        qdf["N_Countries"],
        marker="o"
    )

    plt.xlabel("Quantile")
    plt.ylabel("Coefficient for N_Countries")

    save_figure(
        fig,
        outdir,
        "Figure_2_Quantile_Full"
    )

    plt.close()


# ======================================================
# JITTER PLOT
# ======================================================

def jitter_plot(df, outdir):
    x = df["N_Countries"] + np.random.normal(
        0,
        0.05,
        len(df)
    )

    # full
    fig = plt.figure(figsize=(10, 7))

    plt.scatter(
        x,
        df["RD_Years"],
        alpha=0.15,
        s=8
    )

    plt.xlabel("Number of contributing countries")
    plt.ylabel("Retraction duration (years)")

    save_figure(
        fig,
        outdir,
        "Figure_S2_Jitter_Full"
    )

    plt.close()

    # clipped
    fig = plt.figure(figsize=(10, 7))

    plt.scatter(
        x,
        df["RD_Years"],
        alpha=0.15,
        s=8
    )

    plt.ylim(0, 12)

    plt.xlabel("Number of contributing countries")
    plt.ylabel("Retraction duration (years)")

    save_figure(
        fig,
        outdir,
        "Figure_S2_Jitter_Clipped12Y"
    )

    plt.close()


# ======================================================
# REPORT
# ======================================================

def write_report(
    outdir,
    pearson,
    spearman,
    cph_main,
    cph_collab,
    qdf
):
    with open(
        os.path.join(
            outdir,
            "Article_Level_Report.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write("ARTICLE-LEVEL ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")

        f.write("Correlation\n")
        f.write("-"*20 + "\n")
        f.write(f"Pearson r={pearson[0]:.4f}, p={pearson[1]:.4e}\n")
        f.write(f"Spearman rho={spearman[0]:.4f}, p={spearman[1]:.4e}\n\n")

        f.write("Primary Cox Model\n")
        f.write("-"*20 + "\n")
        f.write(str(cph_main.summary))
        f.write("\n\n")

        f.write("Collaborative-only Cox Model\n")
        f.write("-"*20 + "\n")
        f.write(str(cph_collab.summary))
        f.write("\n\n")

        f.write("Quantile Regression\n")
        f.write("-"*20 + "\n")
        f.write(qdf.to_string(index=False))


# ======================================================
# MAIN
# ======================================================

def main():
    print("Select input CSV file...")
    infile = choose_input_file()

    if not infile:
        return

    print("Select output directory...")
    outdir = choose_output_dir()

    if not outdir:
        return

    print("Loading data...")
    df = load_data(infile)

    print("Descriptive...")
    descriptive_analysis(df, outdir)

    print("Correlations...")
    pearson, spearman = correlation_analysis(df)

    print("KM...")
    km_plot(df, outdir)

    print("Primary Cox...")
    cph_main, model_df = cox_primary(df)

    cph_main.summary.to_csv(
        os.path.join(
            outdir,
            "Table_1_Cox_Primary.csv"
        )
    )

    print("Collaborative Cox...")
    cph_collab = cox_collab_only(df)

    cph_collab.summary.to_csv(
        os.path.join(
            outdir,
            "Table_S2_Cox_CollabOnly.csv"
        )
    )

    print("PH test...")
    ph = ph_test(
        cph_main,
        model_df
    )

    ph.to_csv(
        os.path.join(
            outdir,
            "Table_S3_PH_Test.csv"
        )
    )

    print("Spline Cox model...")
    spline_cox_analysis(df, outdir)

    print("Spline plot...")
    spline_plot(df, outdir)

    print("Quantile regression...")
    qdf = quantile_regression(df)

    qdf.to_csv(
        os.path.join(
            outdir,
            "Table_2_Quantile.csv"
        ),
        index=False
    )

    quantile_plot(
        qdf,
        outdir
    )

    print("Jitter plot...")
    jitter_plot(
        df,
        outdir
    )

    print("Writing report...")
    write_report(
        outdir,
        pearson,
        spearman,
        cph_main,
        cph_collab,
        qdf
    )

    print("Done.")


if __name__ == "__main__":
    main()