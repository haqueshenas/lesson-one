# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 2: Macro-Structural Organization of Scientific Correction Systems
#
#  Analysis:
#  Predictive Modeling of Scientific Correction Dynamics
#
# Phase:
# ML Phase 1 – Data Engineering & Validation
#
# Description:
# Validates the integrated modeling dataset, performs data cleaning,
# removes missing and non-finite values, standardizes predictor and
# target variables, and exports publication-ready machine learning
# matrices and metadata for downstream predictive modeling analyses.
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

import pandas as pd
import numpy as np
import json
import os

from tkinter import Tk, filedialog, messagebox
from sklearn.preprocessing import StandardScaler
from datetime import datetime


# ==========================================
# GUI
# ==========================================

def select_input_file():

    root = Tk()
    root.withdraw()

    return filedialog.askopenfilename(
        title="Select phase1_cleaned_data.csv",
        filetypes=[("CSV files", "*.csv")]
    )


def select_output_dir():

    root = Tk()
    root.withdraw()

    return filedialog.askdirectory(
        title="Select Output Folder"
    )


# ==========================================
# VARIABLES
# ==========================================

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


# ==========================================
# VALIDATION
# ==========================================

def validate_columns(df):

    required = X_COLUMNS + Y_COLUMNS

    missing = [
        c for c in required
        if c not in df.columns
    ]

    return missing


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n=== ML Phase 1 — Data Engineering ===\n")

    input_file = select_input_file()

    if not input_file:

        messagebox.showerror(
            "Error",
            "No input file selected."
        )

        return

    output_dir = select_output_dir()

    if not output_dir:

        messagebox.showerror(
            "Error",
            "No output directory selected."
        )

        return

    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # LOAD
    # ==========================================

    print("[INFO] Loading dataset...")

    df = pd.read_csv(input_file)

    # ==========================================
    # VALIDATE
    # ==========================================

    print("[INFO] Validating columns...")

    missing = validate_columns(df)

    if missing:

        messagebox.showerror(
            "Missing Columns",
            "\n".join(missing)
        )

        return

    # ==========================================
    # CLEAN
    # ==========================================

    print("[INFO] Cleaning data...")

    df_clean = df.copy()

    df_clean = df_clean.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df_clean = df_clean.dropna()

    # ==========================================
    # MATRICES
    # ==========================================

    X = df_clean[X_COLUMNS]
    Y = df_clean[Y_COLUMNS]

    # ==========================================
    # SCALING
    # ==========================================

    print("[INFO] Scaling data...")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    Y_scaled = y_scaler.fit_transform(Y)

    # ==========================================
    # OUTPUT
    # ==========================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    base_dir = os.path.join(
        output_dir,
        f"ML_Phase1_Output_{timestamp}"
    )

    os.makedirs(base_dir, exist_ok=True)

    pd.DataFrame(
        X_scaled,
        columns=X_COLUMNS
    ).to_csv(

        os.path.join(base_dir, "X_scaled.csv"),
        index=False
    )

    pd.DataFrame(
        Y_scaled,
        columns=Y_COLUMNS
    ).to_csv(

        os.path.join(base_dir, "Y_scaled.csv"),
        index=False
    )

    # ==========================================
    # METADATA
    # ==========================================

    meta = {

        "phase": "ML Phase 1",

        "architecture":

        {
            "learning_type":
            "Multi-output supervised regression",

            "target_system":
            "Scientific_Correction_System",

            "normalization":
            "StandardScaler",

            "predictors":
            X_COLUMNS,

            "targets":
            Y_COLUMNS
        },

        "rows_used":
        int(df_clean.shape[0])
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

    # ==========================================
    # REPORT
    # ==========================================

    report = f"""

ML PHASE 1 REPORT
================================

Input:
{input_file}

Rows after cleaning:
{df_clean.shape[0]}

Predictors:
{X_COLUMNS}

Targets:
{Y_COLUMNS}

Outputs:
- X_scaled.csv
- Y_scaled.csv
- metadata.json

Status:
Ready for ML Phase 2
"""

    with open(
        os.path.join(base_dir, "report.txt"),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    messagebox.showinfo(
        "Success",
        f"ML Phase 1 completed.\n\nOutputs:\n{base_dir}"
    )

    print("\n[OK] ML Phase 1 completed.\n")


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    main()