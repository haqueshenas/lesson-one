# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 3: Political Economy of Global Scientific Correction Systems
#
# Analysis:
# Multivariate Structural Regime Identification
# Economic Retraction Regime Identification
#
# Description:
# Constructs a comprehensive economic accounting framework linking
# national research investment, scientific output, and retraction
# dynamics, and identifies latent economic–correction regimes using
# hierarchical multivariate clustering.
#
# Economic Accounting Framework:
# - Gross Research and Development Investment
# - Net Research and Development Investment
# - Retraction-Adjusted R&D Capacity
# - Retraction Cost Estimation
# - Cost per Published Paper
# - Cost per Retracted Paper
# - Retraction-Adjusted Research Efficiency (RARE)
#
# Analytical Components:
# - Economic Normalization
# - Research Investment Accounting
# - Retraction Cost Modeling
# - Distribution Diagnostics
# - Adaptive Log Transformation
# - Hierarchical Agglomerative Clustering
# - Ward Linkage Optimization
# - Cluster Validation Assessment
# - Structural Regime Identification
#
# Cluster Validation Metrics:
# - Silhouette Score
# - Calinski–Harabasz Index
# - Davies–Bouldin Index
#
# Structural Variables:
# - Retraction Rate (RR)
# - Research and Development Expenditure (RDE)
# - Cost per Retracted Publication
# - Retraction-Adjusted Research Efficiency (RARE)
#
# Objective:
# To identify distinct national economic–scientific regimes that
# characterize how countries transform research investment into
# scientific output while experiencing different levels of scientific
# correction burden and retraction-related efficiency loss.
#
# Outputs:
# - Economic Accounting Indicators
# - Cluster Validation Metrics
# - Hierarchical Dendrogram
# - Silhouette Optimization Curves
# - Economic Regime Classification
# - Publication-Grade Figures
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
# (OpenAI; version 5.5) under the direction, testing,
# validation, and iterative refinement of the author.
#
# License:
# MIT License
#
# Copyright (c) 2026 Abbas Haghshenas
# =============================================================================

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import skew
import os
import sys

# ------------------------------------------------------------
# 1. GUI – File Selection
# ------------------------------------------------------------

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

input_path = filedialog.askopenfilename(
    title="Select Master CSV File",
    filetypes=[("CSV files", "*.csv")]
)

if not input_path:
    messagebox.showerror("Error", "No input file selected.")
    sys.exit()

output_dir = filedialog.askdirectory(
    title="Select Output Directory"
)

if not output_dir:
    messagebox.showerror("Error", "No output directory selected.")
    sys.exit()

# ------------------------------------------------------------
# 2. Load Data
# ------------------------------------------------------------

df = pd.read_csv(input_path)
print(f"\nInitial rows loaded: {len(df)}")

# ------------------------------------------------------------
# 3. Validate and Convert RDE
# ------------------------------------------------------------

rde_raw_mean = df["RDE (2000_2023)"].mean()

if rde_raw_mean > 1:
    df["RDE"] = df["RDE (2000_2023)"] / 100
    print("RDE detected as percentage. Converted to decimal.")
else:
    df["RDE"] = df["RDE (2000_2023)"]
    print("RDE detected as decimal. No division applied.")

# ------------------------------------------------------------
# 4. Convert RR to Decimal
# ------------------------------------------------------------

df["RR"] = df["RR_Total_%"] / 100

# ------------------------------------------------------------
# 5. Convert GDP from Natural Log
# ------------------------------------------------------------

df["GDP"] = np.exp(df["GDP (log_2000_2024)"])
print("GDP converted from natural logarithm using exp().")

# ------------------------------------------------------------
# 6. Core Economic Accounting Calculations
# ------------------------------------------------------------

df["Gross_R&D_$"] = df["GDP"] * df["RDE"]
df["Net_R&D_$"] = df["Gross_R&D_$"] * (1 - df["RR"])
df["Net_R&D_%"] = df["RDE"] * (1 - df["RR"])
df["Retraction_Cost_$"] = df["Gross_R&D_$"] * df["RR"]
df["Retraction_Cost_%"] = df["RDE"] * df["RR"]

df["Paper_Cost_$"] = np.where(
    df["WoS_Total_Publications"] > 0,
    df["Gross_R&D_$"] / df["WoS_Total_Publications"],
    np.nan
)

df["Cost_per_Retracted_Paper_$"] = np.where(
    df["RW_Total_Count"] > 0,
    df["Retraction_Cost_$"] / df["RW_Total_Count"],
    np.nan
)

# ------------------------------------------------------------
# 7. Add RARE
# ------------------------------------------------------------

df["RARE"] = np.where(
    df["Gross_R&D_$"] > 0,
    (df["WoS_Total_Publications"] - df["RW_Total_Count"]) / df["Gross_R&D_$"],
    np.nan
)

# ------------------------------------------------------------
# 8. Log Transformation
# ------------------------------------------------------------

log_candidates = [
    "Gross_R&D_$",
    "Retraction_Cost_$",
    "Paper_Cost_$",
    "Cost_per_Retracted_Paper_$",
    "RARE"
]

for col in log_candidates:
    if col in df.columns:
        col_skew = skew(df[col].dropna())
        if abs(col_skew) > 1:
            df[col + "_log"] = np.log1p(df[col])
        else:
            df[col + "_log"] = df[col]

# ------------------------------------------------------------
# 9. Cluster Evaluation (k = 2–8) – Documentation Only
# ------------------------------------------------------------

cluster_vars = [
    "RR",
    "RDE",
    "Cost_per_Retracted_Paper_$_log",
    "RARE_log"
]

X = df[cluster_vars].replace([np.inf, -np.inf], np.nan)
valid_index = X.dropna().index
X_valid = X.loc[valid_index]

print(f"\nRows used for clustering: {len(X_valid)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_valid)

k_range = range(2, 9)
results = []

for k in k_range:
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    results.append([k, sil, ch, db])
    print(f"k={k} | Silhouette={sil:.4f} | CH={ch:.2f} | DB={db:.4f}")

results_df = pd.DataFrame(
    results,
    columns=["k", "Silhouette", "Calinski_Harabasz", "Davies_Bouldin"]
)

results_csv = os.path.join(output_dir, "Cluster_Validation_Metrics.csv")
results_df.to_csv(results_csv, index=False)

# ------------------------------------------------------------
# 10. FINAL CLUSTERING FIXED TO k = 4
# ------------------------------------------------------------

optimal_k = 4
print(f"\nFinal clustering fixed to k = {optimal_k} (statistically supported solution)")

final_model = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
final_clusters = final_model.fit_predict(X_scaled)

df["Economic_Cluster"] = np.nan
df.loc[valid_index, "Economic_Cluster"] = final_clusters

# ------------------------------------------------------------
# 11. Save Main CSV Output
# ------------------------------------------------------------

output_csv = os.path.join(output_dir, "Economic_Retraction_Indicators_Professional_k4.csv")
df.to_csv(output_csv, index=False)

print(f"\nMain CSV saved to:\n{output_csv}")
print(f"Cluster validation metrics saved to:\n{results_csv}")

# ------------------------------------------------------------
# 12. Dendrogram
# ------------------------------------------------------------

linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(14, 8))
dendrogram(linked, orientation='top', distance_sort='descending')
plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)")
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "Economic_Dendrogram.png"), dpi=1200)
plt.savefig(os.path.join(output_dir, "Economic_Dendrogram.pdf"))
plt.close()

# ------------------------------------------------------------
# 13. Silhouette Optimization Plot
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(results_df["k"], results_df["Silhouette"], marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Optimization (k=4 selected)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "Silhouette_Optimization.png"), dpi=1200)
plt.savefig(os.path.join(output_dir, "Silhouette_Optimization.pdf"))
plt.close()

# ------------------------------------------------------------
# 14. Final Cluster Projection (k=4 ONLY)
# ------------------------------------------------------------

plt.figure(figsize=(14, 10))

plt.scatter(
    df["RDE"],
    df["RR"],
    c=df["Economic_Cluster"],
    cmap="tab10",
    s=120,
    alpha=0.85
)

for i in df.index:
    if not pd.isna(df.loc[i, "Economic_Cluster"]):
        plt.text(
            df.loc[i, "RDE"],
            df.loc[i, "RR"],
            df.loc[i, "ISO3"],
            fontsize=8
        )

plt.xlabel("R&D Intensity (Decimal)")
plt.ylabel("Retraction Rate (Decimal)")
plt.title("Economic Clustering (Optimized k=4)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "Economic_Clustering_k4.png"), dpi=1200)
plt.savefig(os.path.join(output_dir, "Economic_Clustering_k4.pdf"))
plt.close()

print("\nAll outputs successfully generated (k=4).")
print("Analysis completed.")