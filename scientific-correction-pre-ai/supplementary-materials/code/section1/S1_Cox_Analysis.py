# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 1: Global Architecture and Dynamics of Scientific Correction
#
# Analysis:
# Article-Level Retraction Survival (Cox Analysis)
#
# Kaplan–Meier survival curves and Cox proportional hazards models
# to assess the effect of article type and country on retraction duration.
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

from scipy.stats import chi2


# =====================================================
# GLOBAL SETTINGS
# =====================================================

plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 11


# =====================================================
# GUI
# =====================================================

def choose_input():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select article-level CSV file",
        filetypes=[("CSV files", "*.csv")]
    )

    return file_path


def choose_output():
    root = tk.Tk()
    root.withdraw()

    folder = filedialog.askdirectory(
        title="Select output folder"
    )

    return folder


# =====================================================
# DIRECTORY
# =====================================================

def build_directories(base):

    dirs = {
        "figures": os.path.join(base, "Figures"),
        "tables": os.path.join(base, "Tables"),
        "reports": os.path.join(base, "Reports")
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    return dirs


# =====================================================
# SAVE FIGURE
# =====================================================

def save_figure(fig, folder, name):

    fig.savefig(
        os.path.join(folder, f"{name}.png"),
        bbox_inches="tight"
    )

    fig.savefig(
        os.path.join(folder, f"{name}.pdf"),
        bbox_inches="tight"
    )

    fig.savefig(
        os.path.join(folder, f"{name}.svg"),
        bbox_inches="tight"
    )

    fig.savefig(
        os.path.join(folder, f"{name}.eps"),
        bbox_inches="tight"
    )

    plt.close(fig)


# =====================================================
# DATA PREP
# =====================================================

def prepare_data(df):

    df = df.copy()

    required_cols = [
        "Country",
        "ISO3",
        "Type",
        "RD_Years"
    ]

    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df["event"] = 1

    df["Type_Binary"] = (
        df["Type"]
        .str.lower()
        .map({
            "single": 0,
            "collab": 1
        })
    )

    return df


# =====================================================
# OVERALL KAPLAN-MEIER
# =====================================================

def overall_km(df, fig_dir):

    fig, ax = plt.subplots(figsize=(10, 8))

    kmf = KaplanMeierFitter()

    for label in ["Single", "Collab"]:

        sub = df[df["Type"] == label]

        kmf.fit(
            sub["RD_Years"],
            event_observed=sub["event"],
            label=label
        )

        kmf.plot_survival_function(ax=ax)

    ax.set_xlim(0, 12)

    ax.set_xlabel("Retraction duration (years)")
    ax.set_ylabel("Probability not yet retracted")
    ax.set_title("Kaplan–Meier survival curves")

    save_figure(
        fig,
        fig_dir,
        "Figure_Overall_KM"
    )


# =====================================================
# GLOBAL COX MODEL
# =====================================================

def global_cox(df, table_dir):

    cph = CoxPHFitter()

    model_df = df[
        ["RD_Years", "event", "Type_Binary"]
    ]

    cph.fit(
        model_df,
        duration_col="RD_Years",
        event_col="event"
    )

    cph.summary.to_csv(
        os.path.join(
            table_dir,
            "Global_Cox_Model.csv"
        )
    )

    return cph


# =====================================================
# COUNTRY-INTERACTION COX
# =====================================================

def interaction_cox(df, table_dir):

    country_dummies = pd.get_dummies(
        df["Country"],
        prefix="Country",
        drop_first=True
    )

    interaction_terms = country_dummies.mul(
        df["Type_Binary"],
        axis=0
    )

    interaction_terms.columns = [
        c + "_X_Type"
        for c in interaction_terms.columns
    ]

    model_df = pd.concat(
        [
            df[["RD_Years", "event", "Type_Binary"]],
            country_dummies,
            interaction_terms
        ],
        axis=1
    )

    cph = CoxPHFitter()

    cph.fit(
        model_df,
        duration_col="RD_Years",
        event_col="event"
    )

    cph.summary.to_csv(
        os.path.join(
            table_dir,
            "Interaction_Cox_Model.csv"
        )
    )

    return cph, model_df


# =====================================================
# LIKELIHOOD RATIO TEST
# =====================================================

def likelihood_ratio_test(
        reduced_model,
        full_model,
        df_diff):

    LR = 2 * (
        full_model.log_likelihood_
        - reduced_model.log_likelihood_
    )

    p = chi2.sf(LR, df_diff)

    return LR, p


# =====================================================
# COUNTRY-SPECIFIC HR
# =====================================================

def country_specific_hr(df, table_dir):

    results = []

    countries = sorted(
        df["Country"].unique()
    )

    for country in countries:

        sub = df[
            df["Country"] == country
        ]

        if sub["Type_Binary"].nunique() < 2:
            continue

        cph = CoxPHFitter()

        temp = sub[
            ["RD_Years", "event", "Type_Binary"]
        ]

        cph.fit(
            temp,
            duration_col="RD_Years",
            event_col="event"
        )

        hr = np.exp(
            cph.params_["Type_Binary"]
        )

        p = cph.summary.loc[
            "Type_Binary",
            "p"
        ]

        results.append([
            country,
            hr,
            p
        ])

    out = pd.DataFrame(
        results,
        columns=[
            "Country",
            "HR_Collab_vs_Single",
            "p_value"
        ]
    )

    out.to_csv(
        os.path.join(
            table_dir,
            "Country_Specific_HR.csv"
        ),
        index=False
    )

    return out


# =====================================================
# FOREST PLOT
# =====================================================

def forest_plot(hr_df, fig_dir):

    hr_df = hr_df.sort_values(
        "HR_Collab_vs_Single"
    )

    fig, ax = plt.subplots(
        figsize=(8, 16)
    )

    y = np.arange(len(hr_df))

    ax.scatter(
        hr_df["HR_Collab_vs_Single"],
        y
    )

    ax.axvline(
        1,
        linestyle="--"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        hr_df["Country"]
    )

    ax.set_xlabel(
        "Hazard ratio"
    )

    ax.set_title(
        "Country-specific hazard ratios"
    )

    save_figure(
        fig,
        fig_dir,
        "Figure_Forest_HR"
    )


# =====================================================
# MEDIAN RD TABLE
# =====================================================

def median_rd_table(df, table_dir):

    out = (
        df.groupby(
            ["Country", "Type"]
        )["RD_Years"]
        .median()
        .reset_index()
    )

    out.to_csv(
        os.path.join(
            table_dir,
            "Median_RD_Table.csv"
        ),
        index=False
    )

    return out


# =====================================================
# HEATMAP
# =====================================================

def interaction_heatmap(df, fig_dir):

    pivot = (
        df.groupby(
            ["ISO3", "Type"]
        )["RD_Years"]
        .median()
        .unstack()
    )

    diff = (
        pivot["Collab"]
        - pivot["Single"]
    )

    diff = diff.sort_values()

    fig, ax = plt.subplots(
        figsize=(8, 14)
    )

    matrix = diff.values.reshape(-1, 1)

    im = ax.imshow(
        matrix,
        aspect="auto"
    )

    ax.set_yticks(
        range(len(diff.index))
    )

    ax.set_yticklabels(
        diff.index
    )

    ax.set_xticks([0])
    ax.set_xticklabels(
        ["Collab - Single"]
    )

    plt.colorbar(im)

    ax.set_title(
        "Country-level interaction heatmap"
    )

    save_figure(
        fig,
        fig_dir,
        "Figure_Interaction_Heatmap"
    )


# =====================================================
# COUNTRY KM PANELS
# =====================================================

def country_km_panels(df, fig_dir):

    countries = sorted(
        df["Country"].unique()
    )

    for country in countries:

        sub = df[
            df["Country"] == country
        ]

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        kmf = KaplanMeierFitter()

        for label in ["Single", "Collab"]:

            s = sub[
                sub["Type"] == label
            ]

            if len(s) == 0:
                continue

            kmf.fit(
                s["RD_Years"],
                event_observed=s["event"],
                label=label
            )

            kmf.plot_survival_function(ax=ax)

        ax.set_xlim(0, 12)

        ax.set_title(country)

        save_figure(
            fig,
            fig_dir,
            f"KM_{country}"
        )


# =====================================================
# MODEL COMPARISON TABLE
# =====================================================

def model_comparison(
        global_model,
        interaction_model,
        model_df,
        table_dir):

    df_diff = (
        len(model_df.columns)
        - 3
    )

    LR, p = likelihood_ratio_test(
        global_model,
        interaction_model,
        df_diff
    )

    out = pd.DataFrame({
        "Model": [
            "Global",
            "Interaction"
        ],
        "LogLikelihood": [
            global_model.log_likelihood_,
            interaction_model.log_likelihood_
        ]
    })

    out["LR_Test"] = [np.nan, LR]
    out["p_value"] = [np.nan, p]

    out.to_csv(
        os.path.join(
            table_dir,
            "Model_Comparison.csv"
        ),
        index=False
    )


# =====================================================
# REPORT
# =====================================================

def write_report(
        global_model,
        interaction_model,
        report_dir):

    hr = np.exp(
        global_model.params_["Type_Binary"]
    )

    p = global_model.summary.loc[
        "Type_Binary",
        "p"
    ]

    with open(
        os.path.join(
            report_dir,
            "Interpretive_Report.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "ARTICLE-LEVEL RETRACTION SURVIVAL ANALYSIS\n"
        )

        f.write(
            "===========================================\n\n"
        )

        f.write(
            "Primary question:\n"
        )

        f.write(
            "- Does article type affect RD?\n"
        )

        f.write(
            "- Does country affect RD?\n"
        )

        f.write(
            "- Does the type effect vary by country?\n\n"
        )

        f.write(
            "Primary model:\n"
        )

        f.write(
            "Cox proportional hazards model\n\n"
        )

        f.write(
            f"Global HR (Collab vs Single): {hr:.4f}\n"
        )

        f.write(
            f"p-value: {p:.8f}\n\n"
        )

        f.write(
            "Interpretation:\n"
        )

        f.write(
            "HR > 1 means faster retraction for collaborative papers.\n"
        )

        f.write(
            "HR < 1 means slower retraction for collaborative papers.\n"
        )


# =====================================================
# MAIN
# =====================================================

def main():

    infile = choose_input()

    if not infile:
        sys.exit()

    outdir = choose_output()

    if not outdir:
        sys.exit()

    dirs = build_directories(outdir)

    df = pd.read_csv(infile)

    df = prepare_data(df)

    overall_km(
        df,
        dirs["figures"]
    )

    global_model = global_cox(
        df,
        dirs["tables"]
    )

    interaction_model, model_df = interaction_cox(
        df,
        dirs["tables"]
    )

    model_comparison(
        global_model,
        interaction_model,
        model_df,
        dirs["tables"]
    )

    hr_df = country_specific_hr(
        df,
        dirs["tables"]
    )

    forest_plot(
        hr_df,
        dirs["figures"]
    )

    median_rd_table(
        df,
        dirs["tables"]
    )

    interaction_heatmap(
        df,
        dirs["figures"]
    )

    country_km_panels(
        df,
        dirs["figures"]
    )

    write_report(
        global_model,
        interaction_model,
        dirs["reports"]
    )

    print(
        "Analysis completed successfully."
    )


if __name__ == "__main__":
    main()