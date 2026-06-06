# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 2: Macro-Structural Organization of Scientific Correction Systems
#
#  Analysis:
#  Predictive Modeling of Scientific Correction Dynamics
#
# Phase:
# ML Phase 5 – Statistical Inference & Robustness Validation
#
# Description:
# Evaluates the robustness, stability, uncertainty, and reproducibility
# of predictive modeling results using bootstrap resampling, feature
# stability assessment, prediction uncertainty estimation, cross-model
# agreement analysis, and outlier influence diagnostics. Produces
# publication-ready robustness evidence for scientific inference.
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
import pickle
import warnings
import numpy as np
import pandas as pd
import tkinter as tk

from tkinter import filedialog, messagebox
from datetime import datetime

import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

from scipy.stats import zscore

warnings.filterwarnings("ignore")

# =========================================================
# GUI
# =========================================================


def select_file(title, filetypes):

    root = tk.Tk()
    root.withdraw()

    return filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )



def select_directory(title):

    root = tk.Tk()
    root.withdraw()

    return filedialog.askdirectory(
        title=title
    )

# =========================================================
# FIGURE SAVE
# =========================================================


def save_figure(output_dir, name):

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"{name}.svg"),
        format="svg",
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(output_dir, f"{name}.pdf"),
        format="pdf",
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(output_dir, f"{name}.png"),
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

# =========================================================
# LOAD MODEL
# =========================================================


def load_best_model(best_model_txt, models_dir):

    with open(best_model_txt, "r", encoding="utf-8") as f:

        best_model_name = f.read().strip()

    model_path = os.path.join(
        models_dir,
        f"{best_model_name}_trained.pkl"
    )

    with open(model_path, "rb") as f:

        model = pickle.load(f)

    return best_model_name, model

# =========================================================
# BOOTSTRAP FEATURE IMPORTANCE
# =========================================================


def bootstrap_feature_importance(
    model,
    X,
    Y,
    n_bootstrap=200
):

    records = []

    for i in range(n_bootstrap):

        idx = np.random.choice(
            len(X),
            len(X),
            replace=True
        )

        Xb = X.iloc[idx]
        Yb = Y.iloc[idx]

        mb = clone(model)

        mb.fit(Xb, Yb)

        r = permutation_importance(
            mb,
            Xb,
            Yb,
            n_repeats=5,
            random_state=i,
            scoring="neg_mean_squared_error"
        )

        for f, val in zip(X.columns, r.importances_mean):

            records.append({
                "Bootstrap": i,
                "Feature": f,
                "Importance": val
            })

    return pd.DataFrame(records)

# =========================================================
# FEATURE STABILITY
# =========================================================


def compute_feature_stability(bootstrap_df):

    grouped = bootstrap_df.groupby("Feature")

    rows = []

    for feature, g in grouped:

        mean_val = g["Importance"].mean()
        std_val = g["Importance"].std()

        stability = mean_val / (std_val + 1e-6)

        rows.append({
            "Feature": feature,
            "Mean": mean_val,
            "STD": std_val,
            "Stability_Score": stability
        })

    df = pd.DataFrame(rows)

    return df.sort_values(
        by="Stability_Score",
        ascending=False
    )

# =========================================================
# PREDICTION UNCERTAINTY
# =========================================================


def compute_prediction_uncertainty(
    model,
    X,
    Y,
    n_bootstrap=100
):

    all_predictions = []

    for i in range(n_bootstrap):

        idx = np.random.choice(
            len(X),
            len(X),
            replace=True
        )

        Xb = X.iloc[idx]
        Yb = Y.iloc[idx]

        mb = clone(model)

        mb.fit(Xb, Yb)

        pred = mb.predict(X)

        all_predictions.append(pred)

    all_predictions = np.stack(all_predictions)

    pred_std = np.std(
        all_predictions,
        axis=0
    )

    return pd.DataFrame({
        "Prediction_STD": pred_std.mean(axis=1)
    })

# =========================================================
# CROSS MODEL AGREEMENT
# =========================================================


def cross_model_agreement(models_dir, X):

    model_files = [
        f for f in os.listdir(models_dir)
        if f.endswith(".pkl")
    ]

    predictions = {}

    for mf in model_files:

        with open(os.path.join(models_dir, mf), "rb") as f:

            m = pickle.load(f)

        predictions[mf] = m.predict(X).flatten()

    pred_df = pd.DataFrame(predictions)

    return pred_df.corr()

# =========================================================
# OUTLIER INFLUENCE
# =========================================================


def outlier_influence_analysis(model, X, Y):

    baseline_pred = model.predict(X)

    baseline_rmse = np.sqrt(
        mean_squared_error(Y, baseline_pred)
    )

    zscores = np.abs(zscore(X))

    outlier_rows = np.where(
        np.max(zscores, axis=1) > 3
    )[0]

    if len(outlier_rows) == 0:

        return pd.DataFrame({
            "Status": ["No strong outliers detected"]
        })

    X_clean = X.drop(index=outlier_rows)
    Y_clean = Y.drop(index=outlier_rows)

    mc = clone(model)

    mc.fit(X_clean, Y_clean)

    pred_clean = mc.predict(X_clean)

    clean_rmse = np.sqrt(
        mean_squared_error(Y_clean, pred_clean)
    )

    return pd.DataFrame({
        "Baseline_RMSE": [baseline_rmse],
        "Cleaned_RMSE": [clean_rmse],
        "Improvement": [baseline_rmse - clean_rmse],
        "Outlier_Count": [len(outlier_rows)]
    })

# =========================================================
# MAIN
# =========================================================


def main():

    print("\n=================================================")
    print("ML PHASE 5 — ROBUSTNESS VALIDATION")
    print("=================================================\n")

    X_file = select_file(
        "Select X_scaled.csv",
        [("CSV", "*.csv")]
    )

    Y_file = select_file(
        "Select Y_scaled.csv",
        [("CSV", "*.csv")]
    )

    best_model_txt = select_file(
        "Select best_model.txt",
        [("TXT", "*.txt")]
    )

    models_dir = select_directory(
        "Select trained_models folder"
    )

    out_dir = select_directory(
        "Select Output Directory"
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    base_dir = os.path.join(
        out_dir,
        f"ML_Phase5_Robustness_{timestamp}"
    )

    os.makedirs(base_dir, exist_ok=True)

    figures_dir = os.path.join(
        base_dir,
        "figures"
    )

    supplementary_dir = os.path.join(
        base_dir,
        "supplementary"
    )

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(supplementary_dir, exist_ok=True)

    X = pd.read_csv(X_file)
    Y = pd.read_csv(Y_file)

    best_model_name, model = load_best_model(
        best_model_txt,
        models_dir
    )

    # -------------------------------------------------
    # BOOTSTRAP
    # -------------------------------------------------

    bootstrap_df = bootstrap_feature_importance(
        model,
        X,
        Y
    )

    bootstrap_df.to_csv(
        os.path.join(
            base_dir,
            "bootstrap_feature_importance.csv"
        ),
        index=False
    )

    # -------------------------------------------------
    # STABILITY
    # -------------------------------------------------

    stability_df = compute_feature_stability(
        bootstrap_df
    )

    stability_df.to_csv(
        os.path.join(
            base_dir,
            "feature_stability.csv"
        ),
        index=False
    )

    # -------------------------------------------------
    # UNCERTAINTY
    # -------------------------------------------------

    uncertainty_df = compute_prediction_uncertainty(
        model,
        X,
        Y
    )

    uncertainty_df.to_csv(
        os.path.join(
            base_dir,
            "prediction_uncertainty.csv"
        ),
        index=False
    )

    # -------------------------------------------------
    # AGREEMENT
    # -------------------------------------------------

    agreement_df = cross_model_agreement(
        models_dir,
        X
    )

    agreement_df.to_csv(
        os.path.join(
            supplementary_dir,
            "cross_model_agreement.csv"
        )
    )

    # -------------------------------------------------
    # OUTLIERS
    # -------------------------------------------------

    outlier_df = outlier_influence_analysis(
        model,
        X,
        Y
    )

    outlier_df.to_csv(
        os.path.join(
            supplementary_dir,
            "outlier_influence.csv"
        ),
        index=False
    )

    # -------------------------------------------------
    # MAIN FIGURE OUTPUTS ONLY
    # -------------------------------------------------

    plt.figure(figsize=(9,6))

    plt.imshow(
        stability_df[["Mean", "STD", "Stability_Score"]],
        aspect='auto'
    )

    plt.yticks(
        range(len(stability_df)),
        stability_df["Feature"]
    )

    plt.xticks(
        range(3),
        ["Mean", "STD", "Stability"]
    )

    plt.colorbar()

    plt.title("Feature Stability")

    save_figure(
        figures_dir,
        "MAIN_feature_stability"
    )

    plt.figure(figsize=(8,5))

    plt.hist(
        uncertainty_df["Prediction_STD"],
        bins=25
    )

    plt.title(
        "Bootstrap Prediction Uncertainty"
    )

    save_figure(
        figures_dir,
        "MAIN_prediction_uncertainty"
    )

    # -------------------------------------------------
    # METADATA
    # -------------------------------------------------

    metadata = {
        "phase": "ML Phase 5",
        "architecture": "Robustness Validation",
        "best_model": best_model_name,
        "main_manuscript_outputs": [
            "MAIN_feature_stability",
            "MAIN_prediction_uncertainty"
        ]
    }

    with open(
        os.path.join(base_dir, "metadata.json"),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )

    messagebox.showinfo(
        "Success",
        "ML Phase 5 completed successfully."
    )


if __name__ == "__main__":

    main()