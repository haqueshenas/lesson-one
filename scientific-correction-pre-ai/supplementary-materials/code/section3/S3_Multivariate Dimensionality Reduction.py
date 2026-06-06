# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 3: Political Economy of Global Scientific Correction Systems
#
# Analysis:
# Multivariate Dimensionality Reduction
# Economic Retraction Multivariate PCA and Structural Projection
#
# Description:
# Performs a comprehensive publication-ready multivariate
# dimensionality reduction of national economic and scientific
# correction indicators, including Retraction Rate (RR), Net R&D %,
# Retraction Cost, and RARE. PCA is applied to construct a reduced
# feature space, followed by hierarchical clustering and visualization
# of structural economic regimes.
#
# Analytical Components:
# - Principal Component Analysis (PCA) on economic-retraction indicators
# - Feature standardization and log transformations
# - Hierarchical Agglomerative Clustering in PCA space
# - Cluster validation: Silhouette, Calinski-Harabasz, Davies-Bouldin
# - Publication-grade scatter plots and dendrograms
# - Optimal k selection based on silhouette criterion
# - Annotated cluster projection and PCA biplots
# - Enhanced CSV outputs for downstream analyses
#
# Structural Variables:
# - Retraction Rate (RR)
# - Net R&D Percentage (Net_R&D_%)
# - Retraction Cost (Retraction_Cost_$)
# - Retraction-Adjusted Research Efficiency (RARE)
#
# Objective:
# To reveal latent structural patterns in the economic-retraction
# landscape of countries, reduce dimensionality for interpretability,
# and generate publication-grade figures for scientific communication.
#
# Outputs:
# - PCA Loadings
# - Enhanced CSV with PCA coordinates and cluster assignments
# - Hierarchical Dendrogram
# - Silhouette Optimization Plot
# - Cluster Projection Figures
# - Analysis Summary TXT
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
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram
from adjustText import adjust_text
from matplotlib.lines import Line2D
import os
import sys
import sklearn
import warnings

warnings.filterwarnings("ignore")

# =======================================================
# 0) VERSION INFO
# =======================================================

print("=========================================")
print("Economic Retraction Analysis – Final Scientific Edition")
print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("matplotlib:", mpl.__version__)
print("scikit-learn:", sklearn.__version__)
print("=========================================\n")

# =======================================================
# 1) GUI – Select Input & Output
# =======================================================

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

input_csv = filedialog.askopenfilename(
    title="Select the CSV file",
    filetypes=[("CSV files", "*.csv")]
)

if not input_csv:
    messagebox.showerror("Error", "No input file selected.")
    exit()

output_dir = filedialog.askdirectory(
    title="Select output directory"
)

if not output_dir:
    messagebox.showerror("Error", "No output directory selected.")
    exit()

# =======================================================
# 2) Load Data
# =======================================================

df = pd.read_csv(input_csv)

print(f"Loaded {len(df)} rows.")

required_cols = [
    "ISO3",
    "GDP",
    "Retraction_Cost_$",
    "RARE",
    "RR",
    "Net_R&D_%"
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# =======================================================
# 2.1) Unit Standardization (Display Only)
# =======================================================

df["GDP_BillionUSD"] = df["GDP"] / 1e9
df["Retraction_Cost_BillionUSD"] = df["Retraction_Cost_$"] / 1e9

# =======================================================
# 3) Variable Transformations
# =======================================================

df["Retraction_Cost_$_log"] = np.log1p(df["Retraction_Cost_$"])
df["RARE_log"] = np.log1p(df["RARE"])
df["Net_R&D_%_log"] = np.log1p(df["Net_R&D_%"])

# =======================================================
# 3.1) PCA Feature Space
# =======================================================

pca_features = pd.DataFrame({
    "Neg_Retraction_Cost": -df["Retraction_Cost_$_log"],
    "Pos_RARE": df["RARE_log"],
    "Neg_RR": -df["RR"],
    "Pos_Net_R&D": df["Net_R&D_%_log"]
})

X_pca = pca_features.replace(
    [np.inf, -np.inf],
    np.nan
).dropna()

pca_scaler = StandardScaler()

X_pca_scaled = pca_scaler.fit_transform(X_pca)

# =======================================================
# 3.2) PCA Analysis
# =======================================================

pca = PCA(n_components=2)

PC = pca.fit_transform(X_pca_scaled)

pc1_var = pca.explained_variance_ratio_[0] * 100
pc2_var = pca.explained_variance_ratio_[1] * 100
total_var = pc1_var + pc2_var

# =======================================================
# PCA Loadings Export
# =======================================================

loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PC1", "PC2"],
    index=pca_features.columns
)

loadings_path = os.path.join(
    output_dir,
    "PCA_Loadings.txt"
)

with open(loadings_path, "w", encoding="utf-8") as f:

    f.write("PCA Loadings (Standardized Variables)\n")
    f.write("=====================================\n\n")

    f.write(loadings.to_string())

    f.write("\n\nExplained Variance Ratio:\n")

    f.write(f"PC1: {pc1_var:.2f}%\n")
    f.write(f"PC2: {pc2_var:.2f}%\n")

df.loc[X_pca.index, "PC1"] = PC[:, 0]
df.loc[X_pca.index, "PC2"] = PC[:, 1]

# =======================================================
# 4) Clustering Feature Space
# =======================================================

cluster_features = pd.DataFrame({
    "RR": df["RR"],
    "Net_R&D": df["Net_R&D_%_log"],
    "Log_Cost": df["Retraction_Cost_$_log"],
    "Log_RARE": df["RARE_log"]
})

cluster_features = cluster_features.replace(
    [np.inf, -np.inf],
    np.nan
).dropna()

cluster_scaler = StandardScaler()

cluster_scaled = cluster_scaler.fit_transform(
    cluster_features
)

# =======================================================
# 4.1) Cluster Validation
# =======================================================

silhouette_scores = []
calinski_scores = []
davies_scores = []

candidate_k = range(2, 7)

for k in candidate_k:

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    labels = model.fit_predict(cluster_scaled)

    silhouette_scores.append(
        silhouette_score(cluster_scaled, labels)
    )

    calinski_scores.append(
        calinski_harabasz_score(cluster_scaled, labels)
    )

    davies_scores.append(
        davies_bouldin_score(cluster_scaled, labels)
    )

# =======================================================
# 4.2) Automatic Optimal k Selection
# =======================================================

optimal_k = candidate_k[
    np.argmax(silhouette_scores)
]

print(f"\nOptimal k selected automatically: {optimal_k}")

# =======================================================
# 4.3) Final Hierarchical Clustering Model
# =======================================================

hc = AgglomerativeClustering(
    n_clusters=optimal_k,
    linkage="ward"
)

clusters = hc.fit_predict(cluster_scaled)

df.loc[cluster_features.index, "Cluster"] = clusters

# =======================================================
# 4.4) Save Validation Metrics
# =======================================================

validation_path = os.path.join(
    output_dir,
    "Cluster_Validation_Metrics.txt"
)

with open(validation_path, "w", encoding="utf-8") as f:

    f.write("Hierarchical Clustering Validation\n")
    f.write("=================================\n\n")

    for i, k in enumerate(candidate_k):

        f.write(f"k = {k}\n")
        f.write(f"Silhouette Score: {silhouette_scores[i]:.4f}\n")
        f.write(f"Calinski-Harabasz Score: {calinski_scores[i]:.4f}\n")
        f.write(f"Davies-Bouldin Score: {davies_scores[i]:.4f}\n\n")

    f.write(
        f"Optimal k (Silhouette criterion): {optimal_k}\n"
    )

# =======================================================
# 5) Save Enhanced CSV
# =======================================================

enhanced_csv = os.path.join(
    output_dir,
    "Enhanced_Economic_Retraction_Analytics.csv"
)

df.to_csv(enhanced_csv, index=False)

# =======================================================
# 6) Global Plot Settings
# =======================================================

mpl.rcParams.update({
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12
})

def save_all_formats(fig, name):

    fig.savefig(os.path.join(output_dir, name + ".png"))
    fig.savefig(os.path.join(output_dir, name + ".svg"))
    fig.savefig(os.path.join(output_dir, name + ".pdf"))

    plt.close(fig)

median_gdp = df["GDP_BillionUSD"].median()
median_cost = df["Retraction_Cost_BillionUSD"].median()

# =======================================================
# FIGURE 1 — GDP vs Retraction Cost
# =======================================================

fig, ax = plt.subplots(figsize=(12, 9))

scatter = ax.scatter(
    df["GDP_BillionUSD"],
    df["Retraction_Cost_BillionUSD"],
    c=df["RR"],
    cmap="coolwarm",
    s=100,
    edgecolors="black"
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.axvline(
    median_gdp,
    linestyle="--",
    color="gray",
    linewidth=1
)

ax.axhline(
    median_cost,
    linestyle="--",
    color="gray",
    linewidth=1
)

ax.set_xlabel(
    "Gross Domestic Product (Billion USD, log10 scale)"
)

ax.set_ylabel(
    "Total Retraction Economic Cost (Billion USD, log10 scale)"
)

ax.set_title("GDP vs Retraction Economic Cost")

cbar = fig.colorbar(scatter)
cbar.set_label("Retraction Rate (RR)")

texts = [
    ax.text(
        df.loc[i, "GDP_BillionUSD"],
        df.loc[i, "Retraction_Cost_BillionUSD"],
        df.loc[i, "ISO3"],
        fontsize=7
    )
    for i in df.index
]

adjust_text(texts)

save_all_formats(fig, "Fig1_GDP_vs_Cost")

# =======================================================
# FIGURE 2 — Cluster Plot
# =======================================================

fig, ax = plt.subplots(figsize=(12, 9))

# Automatically generate publication-grade,
# colorblind-safe high-contrast colors

cluster_ids = sorted(df["Cluster"].dropna().unique())

cmap = plt.get_cmap("Dark2", len(cluster_ids))

scatter = ax.scatter(
    df["Retraction_Cost_BillionUSD"],
    df["RARE"],
    c=df["Cluster"],
    cmap=cmap,
    s=130,
    edgecolors="black",
    linewidths=0.7,
    alpha=0.92
)

cluster_colors = scatter.cmap(
    scatter.norm(
        sorted(df["Cluster"].dropna().unique())
    )
)

legend_elements = [
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label=f'Cluster {int(cluster)}',
        markerfacecolor=color,
        markeredgecolor='black',
        markersize=10
    )
    for cluster, color in zip(
        sorted(df["Cluster"].dropna().unique()),
        cluster_colors
    )
]

ax.legend(
    handles=legend_elements,
    title="Structural Regimes",
    loc="lower left",
    frameon=True
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(
    "Retraction Economic Cost (Billion USD, log10 scale)"
)

ax.set_ylabel("RARE (log10 scale)")

ax.set_title(
    f"Multidimensional Structural Regimes "
    f"(Hierarchical Clustering, k={optimal_k})"
)

texts = [
    ax.text(
        df.loc[i, "Retraction_Cost_BillionUSD"],
        df.loc[i, "RARE"],
        df.loc[i, "ISO3"],
        fontsize=7
    )
    for i in df.index
]

adjust_text(texts)

plt.figtext(
    0.5,
    -0.02,
    "Clusters were computed in standardized four-dimensional "
    "feature space and projected onto the Retraction Cost–RARE "
    "plane for visualization.",
    ha="center",
    fontsize=9
)

save_all_formats(fig, "Fig2_Clusters")

# =======================================================
# FIGURE — Hierarchical Dendrogram
# =======================================================

fig, ax = plt.subplots(figsize=(12, 8))

Z = linkage(cluster_scaled, method="ward")

dendrogram(
    Z,
    labels=df.loc[cluster_features.index, "ISO3"].values,
    leaf_rotation=90,
    leaf_font_size=7,
    ax=ax
)

ax.set_title(
    f"Hierarchical Clustering Dendrogram "
    f"(Optimal k={optimal_k})"
)

ax.set_ylabel("Ward Linkage Distance")

plt.tight_layout()

save_all_formats(fig, "Fig_Dendrogram")

# =======================================================
# FIGURE — Silhouette Validation
# =======================================================

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    list(candidate_k),
    silhouette_scores,
    marker='o',
    linewidth=2
)

ax.axvline(
    optimal_k,
    linestyle='--',
    color='gray'
)

ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Silhouette Score")

ax.set_title(
    f"Silhouette-Based Cluster Validation "
    f"(Optimal k={optimal_k})"
)

plt.tight_layout()

save_all_formats(fig, "Fig_Silhouette_Validation")

# =======================================================
# FIGURE 3 — PCA Plot
# =======================================================

fig, ax = plt.subplots(figsize=(12, 9))

scatter = ax.scatter(
    df["PC1"],
    df["PC2"],
    s=120,
    edgecolors="black"
)

ax.set_xlabel(
    f"Principal Component 1 "
    f"({pc1_var:.2f}% variance explained)"
)

ax.set_ylabel(
    f"Principal Component 2 "
    f"({pc2_var:.2f}% variance explained)"
)

ax.set_title(
    f"PCA of Economic Retraction Indicators "
    f"({total_var:.2f}% Total Variance Explained)"
)

texts = [
    ax.text(
        df.loc[i, "PC1"],
        df.loc[i, "PC2"],
        df.loc[i, "ISO3"],
        fontsize=7
    )
    for i in df.index
]

adjust_text(texts)

save_all_formats(fig, "Fig3_PCA")

# =======================================================
# 7) Textual Summary
# =======================================================

summary_path = os.path.join(
    output_dir,
    "Analysis_Summary.txt"
)

with open(summary_path, "w", encoding="utf-8") as f:

    f.write("Economic Retraction Analysis Summary\n")
    f.write("====================================\n\n")

    f.write(f"Countries analyzed: {len(df)}\n\n")

    f.write(
        "Units: GDP and Retraction Cost "
        "reported in Billion USD.\n\n"
    )

    f.write(
        f"Optimal number of clusters: {optimal_k}\n\n"
    )

    f.write("PCA Explained Variance:\n")

    f.write(f"PC1: {pc1_var:.2f}%\n")
    f.write(f"PC2: {pc2_var:.2f}%\n")

print("\nAll analyses and publication-ready figures generated successfully.")

print(f"\nOutputs stored in:\n{output_dir}")