# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 2: Macro-Structural Organization of Scientific Correction Systems
#
# Analysis:
# Structural Equation Modeling of Scientific Correction Dynamics (SEM)
#
# Phase:
# Phase 1 – Data Preparation and Harmonization
#
# Description:
# Harmonizes, validates, cleans, and standardizes SEM input variables,
# evaluates multicollinearity using variance inflation factors (VIF),
# prepares publication-ready datasets, and generates descriptive
# distributions for latent-variable modeling of scientific correction
# dynamics.
#
# Author:
# Abbas Haghshenas
# Independent Researcher in Crop Production, Shiraz, Iran
#
# Project:
# Lesson One
# https://haqueshenas.github.io/lesson-one
#
# AI-Assisted Code Statement:
# This Python code was generated with the assistance of ChatGPT
# (OpenAI; versions 5.3–5.5) under the direction, testing,
# validation, and iterative refinement of the author.
#
# License:
# MIT License
#
# Copyright (c) 2026 Abbas Haghshenas
# =============================================================================

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import os


# ==========================================
# Phase 1 — Data Preparation
# Scientific Correction System:
# RR_Total_% + Total_Mean_RD_Years
# + Collaboration_Share_% + NSlope13_Total
# ==========================================

COLUMN_MAPPING = {
    "GDP": "GDP (log_2000_2024)",
    "HDI": "HDI (2000_2023)",

    "CPI": "CPI (2000_2024)",
    "WGI": "WGI (2000_2024)",

    "GEE": "GEE (2000_2024)",

    "RDE": "RDE (2000_2023)",
    "GII": "GII (2000-2025)",

    "AFI": "AFI (2000_2024)",

    # ===== Scientific Correction Indicators =====

    "RR_Total": "RR_Total_%",
    "Mean_RD": "Total_Mean_RD_Years",
    "Collaboration_Share": "Collaboration_Share_%",
    "NSlope13_Total": "NSlope13_Total"
}

def compute_vif(df):
    vif_data = pd.DataFrame()
    vif_data["variable"] = df.columns

    vif_values = []

    for i in range(df.shape[1]):
        vif_values.append(
            variance_inflation_factor(df.values, i)
        )

    vif_data["VIF"] = vif_values

    return vif_data


def run_phase1():

    file_path = filedialog.askopenfilename(
        title="Select Master Dataset",
        filetypes=[("CSV Files", "*.csv")]
    )

    if not file_path:
        return

    out_dir = filedialog.askdirectory(
        title="Select Output Directory"
    )

    if not out_dir:
        return

    os.makedirs(out_dir, exist_ok=True)

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df_raw = pd.read_csv(file_path)

    used_cols = {}
    missing = []

    for internal, master_col in COLUMN_MAPPING.items():

        if master_col in df_raw.columns:
            used_cols[internal] = master_col
        else:
            missing.append(master_col)

    if missing:

        messagebox.showerror(
            "Missing Columns",
            "Required columns not found:\n\n" +
            "\n".join(missing)
        )

        return

    df = df_raw[list(used_cols.values())].copy()
    df.columns = list(used_cols.keys())

    missing_report = df.isna().sum()

    df = df.fillna(
        df.median(numeric_only=True)
    )

    scaler = StandardScaler()

    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns
    )

    vif = compute_vif(df_scaled)

    df.to_csv(
        os.path.join(out_dir, "phase1_harmonized_raw.csv"),
        index=False
    )

    df_scaled.to_csv(
        os.path.join(out_dir, "phase1_cleaned_data.csv"),
        index=False
    )

    vif.to_csv(
        os.path.join(out_dir, "vif.csv"),
        index=False
    )

    for col in df_scaled.columns:

        plt.figure()

        plt.hist(df_scaled[col], bins=30)

        plt.title(f"Distribution: {col}")

        plt.savefig(
            os.path.join(fig_dir, f"{col}.png"),
            dpi=600
        )

        plt.savefig(
            os.path.join(fig_dir, f"{col}.pdf"),
            format="pdf"
        )

        plt.close()

    report = []

    report.append("=== Phase 1 Report ===\n")

    report.append("Scientific Correction System Definition:")
    report.append("- RR_Total_%")
    report.append("- Total_Mean_RD_Years")
    report.append("- Collaboration_Share_%")
    report.append("- NSlope13_Total")

    report.append(
        "\nThese indicators represent four relatively independent "
        "dimensions of scientific correction systems:"
    )

    report.append(
        "1. Correction burden/frequency"
    )

    report.append(
        "2. Correction latency/responsiveness"
    )

    report.append(
        "3. Collaboration-complexity structure"
    )

    report.append(
        "4. Temporal correction dynamics"
    )

    report.append("\nMissing Values:")
    report.append(str(missing_report))

    report.append("\nVIF Analysis:")
    report.append(str(vif))

    with open(
        os.path.join(out_dir, "phase1_report.txt"),
        "w",
        encoding="utf-8"
    ) as f:

        f.write("\n".join(report))

    messagebox.showinfo(
        "Phase 1 Complete",
        "Phase 1 completed successfully."
    )


def main():

    root = tk.Tk()

    root.title("SEM Framework — Phase 1")
    root.geometry("620x420")

    tk.Label(
        root,
        text="Phase 1 — Data Preparation",
        font=("Arial", 13, "bold")
    ).pack(pady=20)

    tk.Button(
        root,
        text="Run Phase 1",
        font=("Arial", 11),
        command=run_phase1
    ).pack(pady=30)

    root.mainloop()


if __name__ == "__main__":
    main()