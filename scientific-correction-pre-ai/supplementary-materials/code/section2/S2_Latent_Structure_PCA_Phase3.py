# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 2: Macro-Structural Organization of Scientific Correction Systems
#
# Analysis:
# Latent_Structure_PCA
#
# Phase 3 – Latent Structure Discovery (PCA)
#
# Description:
# Extracts latent structural dimensions using PCA on standardized
# development and correction indicators, computes loadings, scores,
# generates scree plots, heatmaps, scatterplots, and metadata JSON.
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
import json
import tkinter as tk
from tkinter import filedialog, messagebox

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# CONFIG
# -------------------------------

GLOBAL_DEV_INDICATORS = [
    "GDP (log_2000_2024)",
    "CPI (2000_2024)",
    "RDE (2000_2023)",
    "GEE (2000_2024)",
    "HDI (2000_2023)",
    "WGI (2000_2024)",
    "AFI (2000_2024)",
    "GII (2000-2025)"
]

RETRACTION_INDICATORS = [
    "RR_Total_%",
    "Total_Mean_RD_Years",
    "Collaboration_Share_%",
    "NSlope13_Total"
]

META_COLUMNS = ["Country", "ISO3"]

# -------------------------------
# GUI FUNCTIONS
# -------------------------------

def select_input_file():
    path = filedialog.askopenfilename(
        title="Select MASTER CSV file",
        filetypes=[("CSV files", "*.csv")]
    )

    input_entry.delete(0, tk.END)
    input_entry.insert(0, path)


def select_output_dir():
    folder = filedialog.askdirectory(title="Select Output Directory")

    output_entry.delete(0, tk.END)
    output_entry.insert(0, folder)

# -------------------------------
# CORE
# -------------------------------

def run_phase3():

    input_path = input_entry.get()
    output_dir = output_entry.get()

    if not input_path or not output_dir:
        messagebox.showerror("Error", "Select input and output.")
        return

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)

    features = GLOBAL_DEV_INDICATORS + RETRACTION_INDICATORS

    required = META_COLUMNS + features

    missing = [c for c in required if c not in df.columns]

    if missing:
        messagebox.showerror("Missing Columns", str(missing))
        return

    df_model = df[required].copy()

    # ---------------- Standardization ----------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(df_model[features])

    # ---------------- PCA ----------------

    pca = PCA()

    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i+1}" for i in range(len(features))]
    )

    scores = pd.DataFrame(
        X_pca,
        columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
    )

    scores = pd.concat([df_model[META_COLUMNS], scores], axis=1)

    # ---------------- Save ----------------

    loadings.to_csv(os.path.join(output_dir, "phase3_pca_loadings.csv"))

    scores.to_csv(
        os.path.join(output_dir, "phase3_pca_scores.csv"),
        index=False
    )

    meta = {
        "features": features,
        "explained_variance_ratio": explained.tolist(),
        "eigenvalues": eigenvalues.tolist()
    }

    with open(os.path.join(output_dir, "phase3_metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)

    # ---------------- Scree Plot ----------------

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained)+1), explained, marker='o')
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot")
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, "scree_plot.png"), dpi=600)
    plt.savefig(os.path.join(output_dir, "scree_plot.svg"))
    plt.savefig(os.path.join(output_dir, "scree_plot.pdf"))
    plt.close()

    # ---------------- Loadings Heatmap ----------------

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        loadings,
        cmap="coolwarm",
        center=0
    )

    plt.title("PCA Loadings Heatmap")

    plt.savefig(os.path.join(output_dir, "loadings_heatmap.png"), dpi=600)
    plt.savefig(os.path.join(output_dir, "loadings_heatmap.svg"))
    plt.savefig(os.path.join(output_dir, "loadings_heatmap.pdf"))

    plt.close()

    # ---------------- Scatter ----------------

    plt.figure(figsize=(10, 8))

    plt.scatter(scores["PC1"], scores["PC2"], s=60)

    for i, iso in enumerate(scores["ISO3"]):
        plt.text(scores["PC1"][i], scores["PC2"][i], iso, fontsize=8)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Latent Structural Space")

    plt.savefig(os.path.join(output_dir, "pc1_pc2_scatter.png"), dpi=600)
    plt.savefig(os.path.join(output_dir, "pc1_pc2_scatter.svg"))
    plt.savefig(os.path.join(output_dir, "pc1_pc2_scatter.pdf"))

    plt.close()

    # ---------------- Report ----------------

    report = []

    report.append("PHASE 3 – LATENT STRUCTURE DISCOVERY REPORT")
    report.append("=" * 80)
    report.append("")

    report.append("This phase extracts latent structural dimensions")
    report.append("governing the joint organization of macro-developmental")
    report.append("conditions and scientific correction-system dynamics.")
    report.append("")

    report.append("The revised orthogonal correction-system specification")
    report.append("reduces recursive covariance inflation and improves")
    report.append("latent interpretability.")

    with open(os.path.join(output_dir, "phase3_report.txt"), "w") as f:
        f.write("\n".join(report))

    messagebox.showinfo("Success", "Phase 3 completed successfully.")

# -------------------------------
# GUI
# -------------------------------

root = tk.Tk()
root.title("Phase 3 – Latent Structure Discovery")
root.geometry("720x320")

input_entry = tk.Entry(root, width=90)
input_entry.pack(pady=5)

tk.Button(root, text="Browse Input", command=select_input_file).pack(pady=5)

output_entry = tk.Entry(root, width=90)
output_entry.pack(pady=5)

tk.Button(root, text="Browse Output", command=select_output_dir).pack(pady=5)

tk.Button(root, text="Run Phase 3", command=run_phase3, height=2).pack(pady=15)

root.mainloop()