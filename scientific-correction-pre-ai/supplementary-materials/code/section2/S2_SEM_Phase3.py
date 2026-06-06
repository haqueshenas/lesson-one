# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 2: Macro-Structural Organization of Scientific Correction Systems
#
# Analysis:
# Structural Equation Modeling of Scientific Correction Dynamics (SEM)
#
# Phase:
# Phase 3 – Confirmatory Factor Analysis (CFA)
#
# Description:
# Fits and evaluates the confirmatory factor analysis (CFA)
# measurement model, estimates latent-variable loadings,
# computes model fit statistics, and visualizes the empirical
# measurement structure underlying scientific correction dynamics.
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

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import semopy
from semopy import Model
from semopy.inspector import inspect
import matplotlib.pyplot as plt
import networkx as nx
import os


def run_phase3():

    data_path = filedialog.askopenfilename(
        title="Select phase1_cleaned_data.csv",
        filetypes=[("CSV Files", "*.csv")]
    )

    model_path = filedialog.askopenfilename(
        title="Select cfa_model.txt",
        filetypes=[("Text Files", "*.txt")]
    )

    out_dir = filedialog.askdirectory(
        title="Select Output Folder"
    )

    if not data_path or not model_path or not out_dir:
        return

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    with open(model_path, "r", encoding="utf-8") as f:
        model_desc = f.read()

    # ==========================================
    # Fit CFA
    # ==========================================

    model = Model(model_desc)
    model.fit(df)

    # ==========================================
    # Fit statistics
    # ==========================================

    stats = semopy.calc_stats(model)
    stats.to_csv(
        os.path.join(out_dir, "fit_indices.csv")
    )

    # ==========================================
    # Estimates
    # ==========================================

    estimates = inspect(model)

    estimates.to_csv(
        os.path.join(out_dir, "cfa_results.csv"),
        index=False
    )

    # ==========================================
    # Factor loadings only
    # observed ~ latent
    # ==========================================

    factor_loadings = estimates[
        estimates["op"] == "~"
    ].copy()

    factor_loadings.to_csv(
        os.path.join(out_dir, "factor_loadings.csv"),
        index=False
    )

    # ==========================================
    # Graph
    # ==========================================

    G = nx.DiGraph()

    for _, row in factor_loadings.iterrows():

        observed = row["lval"]
        latent = row["rval"]
        weight = row["Estimate"]

        G.add_edge(
            latent,
            observed,
            weight=weight
        )

    plt.figure(figsize=(22, 18))

    pos = nx.kamada_kawai_layout(G)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=3200
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=1.6
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10
    )

    edge_labels = {
        (u, v): f"{d['weight']:.2f}"
        for u, v, d in G.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8
    )

    plt.title(
        "Phase 3 — CFA Measurement Model"
    )

    plt.savefig(
        os.path.join(out_dir, "cfa_model.png"),
        dpi=600
    )

    plt.savefig(
        os.path.join(out_dir, "cfa_model.svg"),
        format="svg"
    )

    plt.savefig(
        os.path.join(out_dir, "cfa_model.pdf"),
        format="pdf"
    )

    plt.close()

    # ==========================================
    # Report
    # ==========================================

    with open(
        os.path.join(out_dir, "phase3_report.txt"),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=== Phase 3: Confirmatory Factor Analysis ===\n\n"
        )

        f.write(
            "Scientific Correction System Definition:\n"
        )

        f.write(
            "Indicators:\n"
        )

        f.write(
            "- RR_Total_%\n"
        )

        f.write(
            "- Total_Mean_RD_Years\n"
        )

        f.write(
            "- Collaboration_Share_%\n"
        )

        f.write(
            "- NSlope13_Total\n\n"
        )

        f.write(
            "The latent construct captures multidimensional scientific correction dynamics.\n\n"
        )

        f.write(
            "Fit Indices:\n"
        )

        f.write(str(stats))

        f.write("\n\nFactor Loadings:\n")

        f.write(
            factor_loadings.to_string()
        )

    messagebox.showinfo(
        "Done",
        "Phase 3 completed successfully."
    )


def main():

    root = tk.Tk()

    root.title("SEM Phase 3 — CFA")
    root.geometry("560x340")

    tk.Label(
        root,
        text="Phase 3 — CFA",
        font=("Arial", 13, "bold")
    ).pack(pady=20)

    tk.Button(
        root,
        text="Run Phase 3",
        command=run_phase3
    ).pack(pady=30)

    root.mainloop()


if __name__ == "__main__":
    main()