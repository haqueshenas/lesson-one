# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 2: Macro-Structural Organization of Scientific Correction Systems
#
#  Analysis:
#  Predictive Modeling of Scientific Correction Dynamics
#
# Phase:
# ML Phase 3 – Predictive Modeling & Benchmarking Engine
#
# Description:
# Benchmarks multiple machine-learning algorithms for predicting
# multidimensional scientific correction outcomes from development
# indicators using multi-output regression, cross-validation,
# performance ranking, model persistence, and diagnostic visualization.
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

from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import KFold, cross_val_predict

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    ElasticNet
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR

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


def select_output_dir():

    root = tk.Tk()
    root.withdraw()

    return filedialog.askdirectory(
        title="Select Output Directory for ML Phase 3"
    )

# =========================================================
# MODELS
# =========================================================

def build_models():

    return {

        "LinearRegression":
            LinearRegression(),

        "Ridge":
            Ridge(alpha=1.0),

        "ElasticNet":
            ElasticNet(
                alpha=0.01,
                l1_ratio=0.5
            ),

        "RandomForest":
            RandomForestRegressor(
                n_estimators=300,
                random_state=42
            ),

        "GradientBoosting":
            GradientBoostingRegressor(
                random_state=42
            ),

        "SVR":
            SVR(kernel="rbf"),

        "NeuralNetwork":
            MLPRegressor(
                hidden_layer_sizes=(128,64),
                max_iter=5000,
                random_state=42
            )
    }

# =========================================================
# METRICS
# =========================================================

def compute_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred,
        multioutput='raw_values'
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
            multioutput='raw_values'
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
        multioutput='raw_values'
    )

    return {

        "MAE_mean":
            float(np.mean(mae)),

        "RMSE_mean":
            float(np.mean(rmse)),

        "R2_mean":
            float(np.mean(r2)),

        "MAE_outputs":
            mae.tolist(),

        "RMSE_outputs":
            rmse.tolist(),

        "R2_outputs":
            r2.tolist()
    }

# =========================================================
# MAIN
# =========================================================

def main():

    print("\n=================================================")
    print("ML PHASE 3 — PREDICTIVE MODELING ENGINE")
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
        f"ML_Phase3_Modeling_{timestamp}"
    )

    os.makedirs(base_dir, exist_ok=True)

    figures_dir = os.path.join(
        base_dir,
        "figures"
    )

    os.makedirs(figures_dir, exist_ok=True)

    models_dir = os.path.join(
        base_dir,
        "trained_models"
    )

    os.makedirs(models_dir, exist_ok=True)

    # -------------------------------------------------
    # LOAD
    # -------------------------------------------------

    print("[INFO] Loading data ...")

    X = pd.read_csv(X_file)

    Y = pd.read_csv(Y_file)

    # -------------------------------------------------
    # MODELS
    # -------------------------------------------------

    models = build_models()

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []

    best_rmse = np.inf
    best_model_name = None
    best_model = None
    best_predictions = None

    # -------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------

    for model_name, base_model in models.items():

        print(f"\n[MODEL] {model_name}")

        model = MultiOutputRegressor(base_model)

        # ---------------------------------------------
        # Cross-validated predictions
        # ---------------------------------------------

        y_pred = cross_val_predict(
            model,
            X,
            Y,
            cv=kf
        )

        # ---------------------------------------------
        # Metrics
        # ---------------------------------------------

        metrics = compute_metrics(
            Y.values,
            y_pred
        )

        metrics["Model"] = model_name

        results.append(metrics)

        print(
            f"RMSE = {metrics['RMSE_mean']:.4f}"
        )

        # ---------------------------------------------
        # Final fit
        # ---------------------------------------------

        model.fit(X, Y)

        with open(
            os.path.join(
                models_dir,
                f"{model_name}_trained.pkl"
            ),
            "wb"
        ) as f:

            pickle.dump(model, f)

        # ---------------------------------------------
        # Best model tracking
        # ---------------------------------------------

        if metrics["RMSE_mean"] < best_rmse:

            best_rmse = metrics["RMSE_mean"]

            best_model_name = model_name

            best_model = model

            best_predictions = y_pred

    # -------------------------------------------------
    # RESULTS TABLE
    # -------------------------------------------------

    df_results = pd.DataFrame(results)

    ranking = df_results.sort_values(
        by="RMSE_mean"
    )

    ranking.to_csv(
        os.path.join(
            base_dir,
            "model_ranking.csv"
        ),
        index=False
    )

    # -------------------------------------------------
    # BEST MODEL
    # -------------------------------------------------

    with open(
        os.path.join(
            base_dir,
            "best_model.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(best_model_name)

    # -------------------------------------------------
    # SAVE BEST MODEL PREDICTIONS
    # REQUIRED FOR PHASE 5 ROBUSTNESS
    # -------------------------------------------------

    best_predictions_df = pd.DataFrame(
        best_predictions,
        columns=[
            f"{col}_predicted"
            for col in Y.columns
        ]
    )

    observed_df = pd.DataFrame(
        Y.values,
        columns=[
            f"{col}_observed"
            for col in Y.columns
        ]
    )

    prediction_export = pd.concat(
        [observed_df, best_predictions_df],
        axis=1
    )

    prediction_export.to_csv(
        os.path.join(
            base_dir,
            "best_model_predictions.csv"
        ),
        index=False
    )

    # -------------------------------------------------
    # PREDICTION FIGURES
    # -------------------------------------------------

    print("[INFO] Generating prediction figures ...")

    for i, col in enumerate(Y.columns):

        # ---------------------------------------------
        # Predicted vs Observed
        # ---------------------------------------------

        plt.figure(figsize=(7,7))

        plt.scatter(
            Y.iloc[:,i],
            best_predictions[:,i]
        )

        plt.xlabel("Observed")

        plt.ylabel("Predicted")

        plt.title(
            f"{col}\nPredicted vs Observed"
        )

        plt.tight_layout()

        safe_col = col.replace("%","pct")

        plt.savefig(
            os.path.join(
                figures_dir,
                f"{safe_col}_predicted_vs_observed.svg"
            ),
            format="svg"
        )

        plt.savefig(
            os.path.join(
                figures_dir,
                f"{safe_col}_predicted_vs_observed.pdf"
            ),
            format="pdf"
        )

        plt.savefig(
            os.path.join(
                figures_dir,
                f"{safe_col}_predicted_vs_observed.png"
            ),
            dpi=600
        )

        plt.close()

        # ---------------------------------------------
        # Residuals
        # ---------------------------------------------

        residuals = (
            Y.iloc[:,i].values -
            best_predictions[:,i]
        )

        plt.figure(figsize=(7,5))

        plt.hist(
            residuals,
            bins=20
        )

        plt.title(
            f"{col}\nResidual Distribution"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                figures_dir,
                f"{safe_col}_residuals.svg"
            ),
            format="svg"
        )

        plt.savefig(
            os.path.join(
                figures_dir,
                f"{safe_col}_residuals.pdf"
            ),
            format="pdf"
        )

        plt.savefig(
            os.path.join(
                figures_dir,
                f"{safe_col}_residuals.png"
            ),
            dpi=600
        )

        plt.close()

    # -------------------------------------------------
    # METADATA
    # -------------------------------------------------

    metadata = {

        "phase":
            "ML Phase 3",

        "architecture":
            "Publication-Grade Predictive Benchmarking",

        "prediction_direction":
            "Development Indicators -> Scientific Correction System",

        "cross_validation":
            "KFold(5)",

        "models":
            list(models.keys()),

        "best_model":
            best_model_name,

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
ML PHASE 3 REPORT
=================================================

INPUT FILES:
- X_scaled.csv
- Y_scaled.csv

TASK:
Predicting Scientific Correction System
from Development Indicators

CROSS VALIDATION:
5-Fold CV

MODELS:
{list(models.keys())}

BEST MODEL:
{best_model_name}

OUTPUTS:
- model_ranking.csv
- trained_models/
- prediction figures
- residual figures
- metadata.json

STATUS:
READY FOR ML PHASE 4
(Explainable AI & Scientific Interpretation)
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
        "ML Phase 3 completed successfully.\n\n"
        "Use outputs in ML Phase 4."
    )

    print("\n[OK] ML Phase 3 completed successfully.\n")

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()