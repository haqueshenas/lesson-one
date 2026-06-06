# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 2: Macro-Structural Organization of Scientific Correction Systems
#
# Analysis:
# Predictive Modeling of Scientific Correction Dynamics
#
# Phase:
# ML Phase 4 – Explainable Scientific Correction AI
#
# Description:
# Interprets the best-performing predictive model using explainable
# artificial intelligence (XAI) techniques, including permutation
# importance, SHAP analysis, partial dependence plots (PDPs), and
# residual diagnostics. Generates publication-ready explainability
# outputs for mechanistic interpretation of relationships between
# development indicators and scientific correction dynamics.
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
import numpy as np
import pandas as pd
import tkinter as tk

from tkinter import filedialog, messagebox
from datetime import datetime

import matplotlib.pyplot as plt

import shap

from sklearn.inspection import (
    permutation_importance,
    PartialDependenceDisplay
)

from scipy import stats

# =========================================================
# GUI
# =========================================================

def select_x_file():

    root = tk.Tk()
    root.withdraw()

    return filedialog.askopenfilename(
        title=(
            "Select X_scaled.csv "
            "(Generated in ML Phase 2)"
        ),
        filetypes=[("CSV files", "*.csv")]
    )


def select_y_file():

    root = tk.Tk()
    root.withdraw()

    return filedialog.askopenfilename(
        title=(
            "Select Y_scaled.csv "
            "(Generated in ML Phase 2)"
        ),
        filetypes=[("CSV files", "*.csv")]
    )


def select_models_dir():

    root = tk.Tk()
    root.withdraw()

    return filedialog.askdirectory(
        title=(
            "Select trained_models folder "
            "(Generated in ML Phase 3)"
        )
    )


def select_output_dir():

    root = tk.Tk()
    root.withdraw()

    return filedialog.askdirectory(
        title="Select Output Directory for ML Phase 4"
    )

# =========================================================
# SAVE FIGURE
# =========================================================

def save_figure(figures_dir, name):

    plt.tight_layout()

    plt.savefig(
        os.path.join(figures_dir, f"{name}.svg"),
        format="svg"
    )

    plt.savefig(
        os.path.join(figures_dir, f"{name}.pdf"),
        format="pdf"
    )

    plt.savefig(
        os.path.join(figures_dir, f"{name}.png"),
        dpi=600
    )

    plt.close()

# =========================================================
# SAFE FILE NAME
# =========================================================

def safe_filename(text):

    return (
        text
        .replace("%", "pct")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )

# =========================================================
# MAIN
# =========================================================

def main():

    print("\n=================================================")
    print("ML PHASE 4 — EXPLAINABLE SCIENTIFIC AI")
    print("=================================================\n")

    # -------------------------------------------------
    # INPUTS
    # -------------------------------------------------

    X_file = select_x_file()

    if not X_file:

        messagebox.showerror(
            "Error",
            "X_scaled.csv was not selected."
        )
        return

    Y_file = select_y_file()

    if not Y_file:

        messagebox.showerror(
            "Error",
            "Y_scaled.csv was not selected."
        )
        return

    models_dir = select_models_dir()

    if not models_dir:

        messagebox.showerror(
            "Error",
            "trained_models folder was not selected."
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
        f"ML_Phase4_Explainability_{timestamp}"
    )

    os.makedirs(base_dir, exist_ok=True)

    figures_dir = os.path.join(
        base_dir,
        "figures"
    )

    os.makedirs(figures_dir, exist_ok=True)

    # -------------------------------------------------
    # SUPPLEMENTARY DIRECTORIES
    # -------------------------------------------------

    supp_dir = os.path.join(
        base_dir,
        "supplementary"
    )

    os.makedirs(supp_dir, exist_ok=True)

    shap_supp_dir = os.path.join(
        supp_dir,
        "shap_per_target"
    )

    os.makedirs(shap_supp_dir, exist_ok=True)

    pdp_supp_dir = os.path.join(
        supp_dir,
        "pdp_per_target"
    )

    os.makedirs(pdp_supp_dir, exist_ok=True)

    residual_supp_dir = os.path.join(
        supp_dir,
        "residual_diagnostics"
    )

    os.makedirs(residual_supp_dir, exist_ok=True)

    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------

    print("[INFO] Loading data ...")

    X = pd.read_csv(X_file)

    Y = pd.read_csv(Y_file)

    # -------------------------------------------------
    # LOAD BEST MODEL
    # -------------------------------------------------

    print("[INFO] Loading best model ...")

    root = tk.Tk()
    root.withdraw()

    ranking_file = filedialog.askopenfilename(

        title=(
            "Select best_model.txt "
            "(Generated in ML Phase 3)"
        ),

        filetypes=[("Text files", "*.txt")]
    )

    if not ranking_file:

        messagebox.showerror(
            "Error",
            "best_model.txt was not selected."
        )

        return

    with open(
        ranking_file,
        "r",
        encoding="utf-8"
    ) as f:

        best_model_name = f.read().strip()

    best_model_file = (
        f"{best_model_name}_trained.pkl"
    )

    model_path = os.path.join(
        models_dir,
        best_model_file
    )

    if not os.path.exists(model_path):

        messagebox.showerror(
            "Error",
            f"Best model file not found:\n{best_model_file}"
        )

        return

    with open(
        model_path,
        "rb"
    ) as f:

        model = pickle.load(f)

    print(
        f"[INFO] Using BEST model: "
        f"{best_model_name}"
    )

    # -------------------------------------------------
    # PREDICTIONS
    # -------------------------------------------------

    y_pred = model.predict(X)

    # -------------------------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------------------------

    print("[INFO] Computing permutation importance ...")

    r = permutation_importance(
        model,
        X,
        Y,
        n_repeats=20,
        random_state=42,
        scoring="neg_mean_squared_error"
    )

    importance_df = pd.DataFrame({

        "Feature":
            X.columns,

        "Importance":
            r.importances_mean
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    importance_df.to_csv(
        os.path.join(
            base_dir,
            "feature_importance.csv"
        ),
        index=False
    )

    # -------------------------------------------------
    # FEATURE IMPORTANCE FIGURE
    # -------------------------------------------------

    plt.figure(figsize=(10,6))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.gca().invert_yaxis()

    plt.title(
        "Global Feature Importance"
    )

    save_figure(
        figures_dir,
        "global_feature_importance"
    )

    # -------------------------------------------------
    # SHAP
    # -------------------------------------------------

    print("[INFO] Computing SHAP values ...")

    try:

        explainer = shap.Explainer(
            model.predict,
            X
        )

        shap_values = explainer(X)

        np.save(
            os.path.join(
                base_dir,
                "shap_values.npy"
            ),
            shap_values.values
        )

        # ---------------------------------------------
        # GLOBAL SHAP SUMMARY (MAIN FIGURE)
        # ---------------------------------------------

        mean_abs_shap = np.mean(
            np.abs(shap_values.values),
            axis=2
        )

        plt.figure(figsize=(10, 6))

        shap.summary_plot(
            mean_abs_shap,
            X,
            show=False
        )

        save_figure(
            figures_dir,
            "SHAP_GLOBAL_SUMMARY"
        )

    except Exception as e:

        print(
            f"[WARNING] SHAP failed:\n{e}"
        )

    # ---------------------------------------------
    # TARGET-SPECIFIC SHAP
    # SUPPLEMENTARY ONLY
    # ---------------------------------------------

    for i, target in enumerate(Y.columns):
        print(f"[INFO] SHAP for {target}")

        plt.figure(figsize=(10, 6))

        shap.summary_plot(
            shap_values[:, :, i],
            X,
            show=False
        )

        save_figure(
            shap_supp_dir,
            f"SHAP_summary_{safe_filename(target)}"
        )

    # -------------------------------------------------
    # PDP (MULTI-OUTPUT SAFE)
    # -------------------------------------------------

    print("[INFO] Generating PDP plots ...")

    # MultiOutputRegressor stores
    # one estimator per target

    if hasattr(model, "estimators_"):

        estimators = model.estimators_

    else:

        estimators = [model]

    for target_index, target_name in enumerate(Y.columns):

        print(
            f"[INFO] PDP for target: "
            f"{target_name}"
        )

        target_dir = os.path.join(
            pdp_supp_dir,
            f"PDP_{safe_filename(target_name)}"
        )

        os.makedirs(
            target_dir,
            exist_ok=True
        )

        estimator = estimators[target_index]

        for feature in X.columns:

            try:

                fig, ax = plt.subplots(
                    figsize=(8, 5)
                )

                PartialDependenceDisplay.from_estimator(

                    estimator=estimator,

                    X=X,

                    features=[feature],

                    ax=ax
                )

                ax.set_title(
                    f"PDP\n"
                    f"Target: {target_name}\n"
                    f"Feature: {feature}"
                )

                safe_target = safe_filename(
                    target_name
                )

                safe_feature = safe_filename(
                    feature
                )

                # -------------------------------------
                # save BEFORE close
                # -------------------------------------

                fig.tight_layout()

                fig.savefig(
                    os.path.join(
                        target_dir,
                        f"PDP_{safe_target}_{safe_feature}.svg"
                    ),
                    format="svg",
                    bbox_inches="tight"
                )

                fig.savefig(
                    os.path.join(
                        target_dir,
                        f"PDP_{safe_target}_{safe_feature}.pdf"
                    ),
                    format="pdf",
                    bbox_inches="tight"
                )

                fig.savefig(
                    os.path.join(
                        target_dir,
                        f"PDP_{safe_target}_{safe_feature}.png"
                    ),
                    dpi=600,
                    bbox_inches="tight"
                )

                plt.close(fig)

            except Exception as e:

                print(
                    f"[WARNING] PDP failed "
                    f"for {feature} / "
                    f"{target_name}"
                )

                print(e)

    # -------------------------------------------------
    # RESIDUAL ANALYSIS
    # -------------------------------------------------

    print("[INFO] Residual diagnostics ...")

    for i, target in enumerate(Y.columns):

        residuals = (
            Y.iloc[:,i].values -
            y_pred[:,i]
        )

        # ---------------------------------------------
        # Residual Scatter
        # ---------------------------------------------

        plt.figure(figsize=(7,5))

        plt.scatter(
            y_pred[:,i],
            residuals
        )

        plt.axhline(
            y=0,
            linestyle="--"
        )

        plt.xlabel("Predicted")

        plt.ylabel("Residuals")

        plt.title(
            f"Residual Analysis — {target}"
        )

        save_figure(
            residual_supp_dir,
            f"Residuals_{safe_filename(target)}"
        )

        # ---------------------------------------------
        # QQ Plot
        # ---------------------------------------------

        plt.figure(figsize=(6,6))

        stats.probplot(
            residuals,
            dist="norm",
            plot=plt
        )

        plt.title(
            f"QQ Plot — {target}"
        )

        save_figure(
            residual_supp_dir,
            f"QQ_{safe_filename(target)}"
        )

    # -------------------------------------------------
    # METADATA
    # -------------------------------------------------

    metadata = {

        "phase":
            "ML Phase 4",

        "architecture":
            "Explainable Scientific Correction AI",

        "methods": [

            "Permutation Importance",
            "SHAP",
            "Partial Dependence Plots",
            "Residual Diagnostics",
            "QQ Diagnostics"
        ],

        "prediction_direction":
            "Development Indicators -> Scientific Correction System",

        "date":
            timestamp
    }

    with open(
        os.path.join(
            base_dir,
            "metadata.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )

    # -------------------------------------------------
    # REPORT
    # -------------------------------------------------

    report = f"""
ML PHASE 4 REPORT
=================================================

INPUT FILES:
- X_scaled.csv
- Y_scaled.csv
- trained_models/

TASK:
Explainable AI for Scientific Correction System

METHODS:
- Permutation Importance
- SHAP
- PDP
- Residual Diagnostics
- QQ Diagnostics

OUTPUTS:
- feature_importance.csv
- shap_values.npy
- vector figures
- explainability diagnostics
- metadata.json

STATUS:
READY FOR SCIENTIFIC INTERPRETATION
"""

    with open(
        os.path.join(
            base_dir,
            "report.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    # -------------------------------------------------
    # FINISH
    # -------------------------------------------------

    messagebox.showinfo(
        "Success",
        "ML Phase 4 completed successfully.\n\n"
        "Publication-grade explainability generated."
    )

    print("\n[OK] ML Phase 4 completed successfully.\n")

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()