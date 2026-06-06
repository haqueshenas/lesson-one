# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 1: Global Architecture and Dynamics of Scientific Correction
#
# Analysis:
# Country-level Multidimensional Clustering of Retraction Systems
#
# Gaussian Mixture Model (GMM) clustering of national retraction-system
# characteristics using multidimensional indicators of retraction burden,
# correction delay, international collaboration, and temporal dynamics.
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
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# =============================================================================

import os
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score

import umap.umap_ as umap

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore")


# ==========================================================
# GUI
# ==========================================================
def select_input_file():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Country-level CSV File",
        filetypes=[("CSV files", "*.csv")]
    )

    return file_path


def select_output_folder():
    root = tk.Tk()
    root.withdraw()

    folder = filedialog.askdirectory(
        title="Select Output Folder"
    )

    return folder


# ==========================================================
# OUTPUT DIRS
# ==========================================================
def create_output_dirs(outdir):

    folders = {
        "tables": os.path.join(outdir, "01_Tables"),
        "figures": os.path.join(outdir, "02_Figures"),
        "reports": os.path.join(outdir, "03_Reports")
    }

    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    return folders


# ==========================================================
# LOAD DATA
# ==========================================================
def load_data(infile):

    df = pd.read_csv(infile)

    required_cols = [
        "Country",
        "ISO3",
        "RR_Total_%",
        "Total_Mean_RD_Years",
        "Collaboration_Share_%",
        "Total_Slope13"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=required_cols)

    return df


# ==========================================================
# FEATURE SELECTION
# ==========================================================
def prepare_features(df):

    features = [
        "RR_Total_%",
        "Total_Mean_RD_Years",
        "Collaboration_Share_%",
        "Total_Slope13"
    ]

    X = df[features].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=features
    )

    return X_scaled, features


# ==========================================================
# MODEL SELECTION
# ==========================================================
def find_optimal_clusters(X_scaled, outdir):

    cluster_range = range(2, 9)

    bic_scores = []
    sil_scores = []

    fitted_models = {}

    for k in cluster_range:

        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=42,
            n_init=20
        )

        gmm.fit(X_scaled)

        labels = gmm.predict(X_scaled)

        bic = gmm.bic(X_scaled)
        sil = silhouette_score(X_scaled, labels)

        bic_scores.append(bic)
        sil_scores.append(sil)

        fitted_models[k] = gmm

    results = pd.DataFrame({
        "K": list(cluster_range),
        "BIC": bic_scores,
        "Silhouette": sil_scores
    })

    results.to_csv(
        os.path.join(outdir, "Cluster_Model_Selection.csv"),
        index=False
    )

    best_k = results.loc[
        results["BIC"].idxmin(),
        "K"
    ]

    return best_k, results, fitted_models[best_k]


# ==========================================================
# BOOTSTRAP STABILITY
# ==========================================================
def bootstrap_stability(X_scaled, best_k, outdir):

    n_boot = 100

    ari_scores = []

    base_model = GaussianMixture(
        n_components=best_k,
        covariance_type="full",
        random_state=42,
        n_init=20
    )

    base_model.fit(X_scaled)

    base_labels = base_model.predict(X_scaled)

    for i in range(n_boot):

        sample_idx = np.random.choice(
            len(X_scaled),
            len(X_scaled),
            replace=True
        )

        X_boot = X_scaled.iloc[sample_idx]

        model = GaussianMixture(
            n_components=best_k,
            covariance_type="full",
            random_state=i,
            n_init=10
        )

        model.fit(X_boot)

        boot_labels = model.predict(X_scaled)

        ari = adjusted_rand_score(
            base_labels,
            boot_labels
        )

        ari_scores.append(ari)

    stability_df = pd.DataFrame({
        "Bootstrap": range(1, n_boot + 1),
        "ARI": ari_scores
    })

    stability_df.to_csv(
        os.path.join(outdir, "Bootstrap_Stability.csv"),
        index=False
    )

    return stability_df


# ==========================================================
# FINAL CLUSTERING
# ==========================================================
def run_final_clustering(df, X_scaled, gmm, outdir):

    labels = gmm.predict(X_scaled)

    probs = gmm.predict_proba(X_scaled)

    df_out = df.copy()

    df_out["Cluster"] = labels

    for i in range(probs.shape[1]):

        df_out[f"Cluster_{i}_Prob"] = probs[:, i]

    df_out.to_csv(
        os.path.join(outdir, "Country_Cluster_Assignment.csv"),
        index=False
    )

    return df_out


# ==========================================================
# CLUSTER PROFILES
# ==========================================================
def create_cluster_profiles(df_clustered, features, outdir):

    profile = df_clustered.groupby("Cluster")[features].mean()

    profile.to_csv(
        os.path.join(outdir, "Cluster_Profiles.csv")
    )

    return profile


# ==========================================================
# UMAP VISUALIZATION
# ==========================================================
def create_umap_plot(df_clustered, X_scaled, outdir):

    reducer = umap.UMAP(
        n_neighbors=12,
        min_dist=0.25,
        random_state=42
    )

    embedding = reducer.fit_transform(X_scaled)

    df_clustered["UMAP1"] = embedding[:, 0]
    df_clustered["UMAP2"] = embedding[:, 1]

    plt.figure(figsize=(10, 8))

    clusters = sorted(df_clustered["Cluster"].unique())

    for cluster in clusters:

        sub = df_clustered[
            df_clustered["Cluster"] == cluster
        ]

        plt.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            s=120,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.6,
            label=f"Cluster {cluster}"
        )

        for _, r in sub.iterrows():

            plt.text(
                r["UMAP1"],
                r["UMAP2"],
                r["ISO3"],
                fontsize=7
            )

    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    plt.legend(frameon=False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(outdir, "UMAP_Clusters.pdf")
    )

    plt.savefig(
        os.path.join(outdir, "UMAP_Clusters.svg")
    )

    plt.savefig(
        os.path.join(outdir, "UMAP_Clusters.eps")
    )

    plt.close()


# ==========================================================
# HEATMAP
# ==========================================================
def create_cluster_heatmap(profile, outdir):

    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(
        profile.values,
        aspect="auto"
    )

    ax.set_xticks(
        range(len(profile.columns))
    )

    ax.set_xticklabels(
        profile.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(
        range(len(profile.index))
    )

    ax.set_yticklabels(
        [f"Cluster {i}" for i in profile.index]
    )

    plt.colorbar(im)

    plt.tight_layout()

    plt.savefig(
        os.path.join(outdir, "Cluster_Profile_Heatmap.pdf")
    )

    plt.savefig(
        os.path.join(outdir, "Cluster_Profile_Heatmap.svg")
    )

    plt.savefig(
        os.path.join(outdir, "Cluster_Profile_Heatmap.eps")
    )

    plt.close()


# ==========================================================
# MODEL SELECTION PLOT
# ==========================================================
def create_bic_plot(results, outdir):

    plt.figure(figsize=(8, 5))

    plt.plot(
        results["K"],
        results["BIC"],
        marker="o"
    )

    plt.xlabel("Number of Clusters")
    plt.ylabel("BIC")

    plt.tight_layout()

    plt.savefig(
        os.path.join(outdir, "BIC_Curve.pdf")
    )

    plt.savefig(
        os.path.join(outdir, "BIC_Curve.svg")
    )

    plt.savefig(
        os.path.join(outdir, "BIC_Curve.eps")
    )

    plt.close()


# ==========================================================
# STABILITY PLOT
# ==========================================================
def create_stability_plot(stability_df, outdir):

    plt.figure(figsize=(8, 5))

    plt.hist(
        stability_df["ARI"],
        bins=20
    )

    plt.xlabel("Adjusted Rand Index")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        os.path.join(outdir, "Stability_Distribution.pdf")
    )

    plt.savefig(
        os.path.join(outdir, "Stability_Distribution.svg")
    )

    plt.savefig(
        os.path.join(outdir, "Stability_Distribution.eps")
    )

    plt.close()


# ==========================================================
# REPORT
# ==========================================================
def write_report(best_k, model_selection, stability_df, outdir):

    report = []

    report.append(
        "LATENT RETRACTION SYSTEM CLUSTERING REPORT\n\n"
    )

    report.append(
        "Method: Gaussian Mixture Model (GMM)\n"
    )

    report.append(
        f"Optimal number of clusters (BIC): {best_k}\n"
    )

    report.append(
        f"Minimum BIC: {model_selection['BIC'].min():.4f}\n"
    )

    report.append(
        "Cluster number was selected by BIC; "
        "silhouette values were used only as auxiliary support.\n"
    )

    report.append(
        f"Mean bootstrap ARI: "
        f"{stability_df['ARI'].mean():.4f}\n"
    )

    report.append(
        "\nInterpretation:\n"
    )

    report.append(
        "Countries organize into latent correction-system "
        "architectures characterized by distinct burden, "
        "latency, collaboration structure, and temporal "
        "correction dynamics.\n"
    )

    with open(
        os.path.join(outdir, "Clustering_Report.txt"),
        "w",
        encoding="utf-8"
    ) as f:

        f.writelines(report)


# ==========================================================
# MAIN
# ==========================================================
def main():

    infile = select_input_file()

    if not infile:
        return

    outdir = select_output_folder()

    if not outdir:
        return

    folders = create_output_dirs(outdir)

    df = load_data(infile)

    X_scaled, features = prepare_features(df)

    best_k, model_selection, best_model = find_optimal_clusters(
        X_scaled,
        folders["tables"]
    )

    stability_df = bootstrap_stability(
        X_scaled,
        best_k,
        folders["tables"]
    )

    df_clustered = run_final_clustering(
        df,
        X_scaled,
        best_model,
        folders["tables"]
    )

    profile = create_cluster_profiles(
        df_clustered,
        features,
        folders["tables"]
    )

    create_umap_plot(
        df_clustered,
        X_scaled,
        folders["figures"]
    )

    create_cluster_heatmap(
        profile,
        folders["figures"]
    )

    create_bic_plot(
        model_selection,
        folders["figures"]
    )

    create_stability_plot(
        stability_df,
        folders["figures"]
    )

    write_report(
        best_k,
        model_selection,
        stability_df,
        folders["reports"]
    )

    messagebox.showinfo(
        "Done",
        "Clustering analysis completed successfully."
    )


if __name__ == "__main__":
    main()