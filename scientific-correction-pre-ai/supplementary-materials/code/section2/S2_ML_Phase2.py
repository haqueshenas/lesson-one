# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 2: Macro-Structural Organization of Scientific Correction Systems
#
#  Analysis:
#  Predictive Modeling of Scientific Correction Dynamics
#
# Phase:
# ML Phase 2 – Scientific Correction System Engineering
#
# Description:
# Constructs standardized predictor and target matrices, generates the
# integrated correlation structure of the Scientific Correction System,
# visualizes correlation patterns and target-variable distributions,
# and exports analytical outputs for predictive model development.
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

import os
import json
import numpy as np
import pandas as pd
import tkinter as tk

from tkinter import filedialog, messagebox
from datetime import datetime

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# =========================================================
# GUI
# =========================================================

def select_master_file():

    root = tk.Tk()
    root.withdraw()

    return filedialog.askopenfilename(
        title="Select Master.csv (Original Dataset)",
        filetypes=[("CSV files", "*.csv")]
    )


def select_output_dir():

    root = tk.Tk()
    root.withdraw()

    return filedialog.askdirectory(
        title="Select Output Directory for ML Phase 2"
    )

# =========================================================
# VARIABLES
# =========================================================

X_COLUMNS = [

    "GDP (log_2000_2024)",
    "CPI (2000_2024)",
    "RDE (2000_2023)",
    "GEE (2000_2024)",
    "HDI (2000_2023)",
    "WGI (2000_2024)",
    "AFI (2000_2024)",
    "GII (2000-2025)"
]

Y_COLUMNS = [

    "RR_Total_%",
    "Total_Mean_RD_Years",
    "Collaboration_Share_%",
    "NSlope13_Total"
]

# =========================================================
# VALIDATION
# =========================================================

def validate_columns(df):

    missing = []

    for col in X_COLUMNS + Y_COLUMNS:

        if col not in df.columns:
            missing.append(col)

    return missing

# =========================================================
# MAIN
# =========================================================

def main():

    print("\n=================================================")
    print("ML PHASE 2 — SCIENTIFIC CORRECTION ENGINEERING")
    print("=================================================\n")

    # -------------------------------------------------
    # INPUTS
    # -------------------------------------------------

    master_file = select_master_file()

    if not master_file:

        messagebox.showerror(
            "Error",
            "Master.csv was not selected."
        )
        return

    out_dir = select_output_dir()

    if not out_dir:

        messagebox.showerror(
            "Error",
            "Output directory was not selected."
        )
        return

    # -------------------------------------------------
    # OUTPUT STRUCTURE
    # -------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = os.path.join(
        out_dir,
        f"ML_Phase2_Engineering_{timestamp}"
    )

    os.makedirs(base_dir, exist_ok=True)

    figures_dir = os.path.join(base_dir, "figures")

    os.makedirs(figures_dir, exist_ok=True)

    # -------------------------------------------------
    # LOAD
    # -------------------------------------------------

    print("[INFO] Loading Master.csv ...")

    df = pd.read_csv(master_file)

    # -------------------------------------------------
    # VALIDATE
    # -------------------------------------------------

    print("[INFO] Validating columns ...")

    missing = validate_columns(df)

    if missing:

        messagebox.showerror(
            "Missing Columns",
            "\n".join(missing)
        )

        return

    # -------------------------------------------------
    # CLEAN
    # -------------------------------------------------

    print("[INFO] Cleaning data ...")

    df_clean = df.copy()

    df_clean = df_clean.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df_clean = df_clean.dropna()

    # -------------------------------------------------
    # BUILD MATRICES
    # -------------------------------------------------

    X = df_clean[X_COLUMNS]

    Y = df_clean[Y_COLUMNS]

    # -------------------------------------------------
    # STANDARDIZATION
    # -------------------------------------------------

    print("[INFO] Standardizing data ...")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = pd.DataFrame(
        x_scaler.fit_transform(X),
        columns=X_COLUMNS
    )

    Y_scaled = pd.DataFrame(
        y_scaler.fit_transform(Y),
        columns=Y_COLUMNS
    )

    # -------------------------------------------------
    # SAVE MATRICES
    # -------------------------------------------------

    X_scaled.to_csv(
        os.path.join(base_dir, "X_scaled.csv"),
        index=False
    )

    Y_scaled.to_csv(
        os.path.join(base_dir, "Y_scaled.csv"),
        index=False
    )

    # -------------------------------------------------
    # CORRELATION MATRIX
    # -------------------------------------------------

    print("[INFO] Building correlation matrix ...")

    corr = pd.concat(
        [X_scaled, Y_scaled],
        axis=1
    ).corr()

    corr.to_csv(
        os.path.join(base_dir, "correlation_matrix.csv")
    )

    # -------------------------------------------------
    # CORRELATION FIGURE
    # -------------------------------------------------

    plt.figure(figsize=(14,12))

    plt.imshow(
        corr,
        aspect='auto'
    )

    plt.colorbar()

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90
    )

    plt.yticks(
        range(len(corr.columns)),
        corr.columns
    )

    plt.title(
        "Correlation Matrix — Scientific Correction System",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(figures_dir, "correlation_matrix.svg"),
        format="svg"
    )

    plt.savefig(
        os.path.join(figures_dir, "correlation_matrix.pdf"),
        format="pdf"
    )

    plt.savefig(
        os.path.join(figures_dir, "correlation_matrix.png"),
        dpi=600
    )

    plt.close()

    # -------------------------------------------------
    # DISTRIBUTIONS
    # -------------------------------------------------

    print("[INFO] Building variable distributions ...")

    for col in Y_COLUMNS:

        plt.figure(figsize=(8,5))

        plt.hist(
            Y_scaled[col],
            bins=20
        )

        plt.title(col)

        plt.tight_layout()

        safe_name = col.replace("%","pct")

        plt.savefig(
            os.path.join(
                figures_dir,
                f"{safe_name}_distribution.svg"
            ),
            format="svg"
        )

        plt.savefig(
            os.path.join(
                figures_dir,
                f"{safe_name}_distribution.pdf"
            ),
            format="pdf"
        )

        plt.savefig(
            os.path.join(
                figures_dir,
                f"{safe_name}_distribution.png"
            ),
            dpi=600
        )

        plt.close()

    # -------------------------------------------------
    # METADATA
    # -------------------------------------------------

    meta = {

        "phase": "ML Phase 2",

        "architecture": {

            "type":
                "Scientific Correction System Modeling",

            "learning_framework":
                "Multi-Output Regression",

            "prediction_direction":
                "Development Indicators -> Scientific Correction System",

            "target_structure":
                "Integrated Multidimensional Scientific Correction System"
        },

        "X_features": X_COLUMNS,

        "Y_targets": Y_COLUMNS,

        "rows_after_cleaning":
            int(df_clean.shape[0]),

        "date":
            timestamp
    }

    with open(
        os.path.join(base_dir, "metadata.json"),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            meta,
            f,
            indent=4,
            ensure_ascii=False
        )

    # -------------------------------------------------
    # REPORT
    # -------------------------------------------------

    report = f"""
ML PHASE 2 REPORT
=================================================

INPUT:
Master.csv

ROWS AFTER CLEANING:
{df_clean.shape[0]}

X FEATURES:
{X_COLUMNS}

Y TARGETS:
{Y_COLUMNS}

OUTPUT FILES:
- X_scaled.csv
- Y_scaled.csv
- correlation_matrix.csv
- metadata.json

FIGURES:
- correlation_matrix
- target distributions

STATUS:
READY FOR ML PHASE 3
(Model Benchmarking & Predictive Learning)
"""

    with open(
        os.path.join(base_dir, "report.txt"),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    # -------------------------------------------------
    # FINISH
    # -------------------------------------------------

    messagebox.showinfo(
        "Success",
        "ML Phase 2 completed successfully.\n\n"
        "Use the generated files in ML Phase 3."
    )

    print("\n[OK] ML Phase 2 completed successfully.\n")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()