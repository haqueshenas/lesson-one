# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 2: Macro-Structural Organization of Scientific Correction Systems
#
# Analysis:
# Latent_Structure_PCA
#
# Phase 4 – Regime Detection & System Typology
#
# Description:
# Detects latent systemic regimes using PCA scores, Hopkins statistic,
# and Gaussian Mixture Models. Outputs regime assignments, cluster centers,
# metadata JSON, visualizations (scatter plots), and interpretive report.
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

from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# GUI FUNCTIONS
# -------------------------------

def select_input_file():
    file_path = filedialog.askopenfilename(
        title="Select phase3_pca_scores.csv",
        filetypes=[("CSV files", "*.csv")]
    )
    input_entry.delete(0, tk.END)
    input_entry.insert(0, file_path)

def select_output_dir():
    folder = filedialog.askdirectory(
        title="Select Output Directory"
    )
    output_entry.delete(0, tk.END)
    output_entry.insert(0, folder)

# -------------------------------
# CORE LOGIC
# -------------------------------

def run_phase4():

    input_path = input_entry.get()
    output_dir = output_entry.get()

    if not input_path or not output_dir:
        messagebox.showerror("Error", "Please select input file and output directory.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # -------- Load PCA Scores --------
    df = pd.read_csv(input_path)

    meta_cols = ["Country", "ISO3"]
    pc_cols = [c for c in df.columns if c.startswith("PC")]

    if len(pc_cols) < 2:
        messagebox.showerror("Error", "Not enough PCA components for regime detection.")
        return

    X = df[pc_cols].values

    # =========================================================
    # STEP 1 — Hopkins Statistic
    # =========================================================

    def hopkins_statistic(X, sampling_size=0.3):

        X = np.array(X)

        n, d = X.shape

        m = int(max(5, sampling_size * n))

        nbrs = NearestNeighbors(n_neighbors=2).fit(X)

        rand_X = np.random.uniform(
            np.min(X, axis=0),
            np.max(X, axis=0),
            (m, d)
        )

        ujd = []
        wjd = []

        random_indices = np.random.choice(n, m, replace=False)

        for j in range(m):
            u_dist, _ = nbrs.kneighbors(rand_X[j].reshape(1, -1), n_neighbors=1)
            ujd.append(u_dist[0][0])

            w_dist, _ = nbrs.kneighbors(X[random_indices[j]].reshape(1, -1), n_neighbors=2)
            wjd.append(w_dist[0][1])

        H = sum(ujd) / (sum(ujd) + sum(wjd))

        return H

    hopkins = hopkins_statistic(X)

    # =========================================================
    # STEP 2 — Evidence-Based Regime Detection
    # =========================================================

    bic_scores = {}

    max_k = min(6, len(df) - 1)

    for k in range(1, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            random_state=42,
            n_init=10
        )

        gmm.fit(X)

        bic_scores[k] = gmm.bic(X)

    best_k = min(bic_scores, key=bic_scores.get)

    # =========================================================
    # STEP 3 — Final GMM Model
    # =========================================================

    gmm_final = GaussianMixture(
        n_components=best_k,
        covariance_type='full',
        random_state=42,
        n_init=20
    )

    gmm_final.fit(X)

    regimes = gmm_final.predict(X)

    df["Regime"] = regimes

    # -------- Save Tables --------
    df_out = df[meta_cols + ["Regime"] + pc_cols]
    df_out.to_csv(os.path.join(output_dir, "phase4_regimes.csv"), index=False)

    centers = pd.DataFrame(
        gmm_final.means_,
        columns=pc_cols
    )
    centers["Regime"] = range(best_k)
    centers.to_csv(os.path.join(output_dir, "phase4_cluster_centers.csv"), index=False)

    # -------- Metadata JSON --------
    meta = {
        "optimal_k": int(best_k),
        "hopkins_statistic": float(hopkins),
        "bic_scores": bic_scores,
        "pc_dimensions_used": pc_cols
    }

    with open(os.path.join(output_dir, "phase4_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    # -------- Visualization --------

    # Regime Map (PC1-PC2)
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=df["PC1"],
        y=df["PC2"],
        hue=df["Regime"],
        palette="tab10",
        s=120
    )

    for i, iso in enumerate(df["ISO3"]):
        plt.text(df["PC1"][i], df["PC2"][i], iso, fontsize=8)

    plt.title("System Regime Map (Latent Space PC1–PC2)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.savefig(os.path.join(output_dir, "regime_map.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "regime_map.svg"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "regime_map.pdf"), bbox_inches="tight")
    plt.close()

    # -------- Report --------
    report = []
    report.append("PHASE 4 – REGIME DETECTION REPORT")
    report.append("="*80)
    report.append("")
    report.append(f"Optimal number of latent regimes detected: {best_k}")
    report.append("")
    report.append(f"Hopkins statistic: {round(hopkins, 4)}")
    report.append("")
    report.append("")
    report.append("Scientific Interpretation:")
    report.append("Countries are grouped not by raw indicators,")
    report.append("but by their position in the latent systemic space.")
    report.append("")
    report.append("Each regime represents a structural configuration of:")
    report.append("- Development capacity")
    report.append("- Correction Burden")
    report.append("- Correction Latency")
    report.append("- International collaboration structure")
    report.append("- Temporal correction dynamics")
    report.append("")
    report.append("This is a system typology, not a statistical clustering.")

    with open(os.path.join(output_dir, "phase4_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    messagebox.showinfo("Success", f"Phase 4 completed.\nDetected regimes: {best_k}")

# -------------------------------
# GUI
# -------------------------------

root = tk.Tk()
root.title("Phase 4 – Regime Detection System")

root.geometry("760x340")

tk.Label(root, text="Input File: phase3_pca_scores.csv").pack(pady=5)
input_entry = tk.Entry(root, width=95)
input_entry.pack(pady=3)
tk.Button(root, text="Browse PCA Scores File", command=select_input_file).pack(pady=3)

tk.Label(root, text="Output Directory:").pack(pady=5)
output_entry = tk.Entry(root, width=95)
output_entry.pack(pady=3)
tk.Button(root, text="Browse Output Folder", command=select_output_dir).pack(pady=3)

tk.Button(
    root,
    text="Run Phase 4 – Regime Detection & System Typology",
    command=run_phase4,
    height=2
).pack(pady=18)

root.mainloop()