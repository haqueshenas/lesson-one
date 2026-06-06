# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 2: Macro-Structural Organization of Scientific Correction Systems
#
# Analysis:
# Latent_Structure_PCA
#
# Phase 2 – Structural Dataset & Correlation Matrix
#
# Description:
# Builds the integrated structural dataset from development and correction
# indicators, computes Pearson correlation matrix, and visualizes it as
# high-resolution heatmaps (PNG, SVG, PDF).
#
# Author:
# Abbas Haghshenas, Independent Researcher
#
# Project:
# Lesson One – Multi-Phase Interblock Analysis
# https://haqueshenas.github.io/lesson-one
#
# AI-Assisted Code:
# Generated with ChatGPT GPT-5.5; iterative refinement by author
#
# License:
# MIT License
#
# Copyright (c) 2026 Abbas Haghshenas
# =============================================================================

import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# INDICES
# -----------------------------

DEVELOPMENT_INDICES = [
    "GDP (log_2000_2024)",
    "CPI (2000_2024)",
    "RDE (2000_2023)",
    "GEE (2000_2024)",
    "HDI (2000_2023)",
    "WGI (2000_2024)",
    "AFI (2000_2024)",
    "GII (2000-2025)"
]

RETRACTION_INDEXES = [
    "RR_Total_%",
    "Total_Mean_RD_Years",
    "Collaboration_Share_%",
    "NSlope13_Total"
]

ALL_INDICES = DEVELOPMENT_INDICES + RETRACTION_INDEXES

# -----------------------------
# CORE FUNCTIONS
# -----------------------------

def build_structural_dataset(df):
    return df[ALL_INDICES].copy()


def compute_correlation_matrix(df, output_dir):

    corr = df.corr(method="pearson")

    corr.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True
    )

    plt.title("Integrated Structural Correlation Matrix")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=600)
    plt.savefig(os.path.join(output_dir, "correlation_matrix.svg"))
    plt.savefig(os.path.join(output_dir, "correlation_matrix.pdf"))

    plt.close()

    return corr

# -----------------------------
# GUI
# -----------------------------

def run_phase2():

    root = tk.Tk()
    root.withdraw()

    input_file = filedialog.askopenfilename(
        title="Select Input CSV File",
        filetypes=[("CSV files", "*.csv")]
    )

    if not input_file:
        return

    output_dir = filedialog.askdirectory(title="Select Output Directory")

    if not output_dir:
        return

    try:
        df = pd.read_csv(input_file)

        df_system = build_structural_dataset(df)

        os.makedirs(output_dir, exist_ok=True)

        df_system.to_csv(
            os.path.join(output_dir, "phase2_structural_dataset.csv"),
            index=False
        )

        compute_correlation_matrix(df_system, output_dir)

        messagebox.showinfo("Success", "Phase 2 completed successfully.")

    except Exception as e:
        messagebox.showerror("Execution Error", str(e))


if __name__ == "__main__":
    run_phase2()