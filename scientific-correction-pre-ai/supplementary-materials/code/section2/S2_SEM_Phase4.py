# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 2: Macro-Structural Organization of Scientific Correction Systems
#
# Analysis:
# Structural Equation Modeling of Scientific Correction Dynamics (SEM)
#
# Phase:
# Phase 4 – Structural Equation Modeling
#
# Description:
# Fits the full structural equation model (SEM), estimates direct
# relationships among latent systems, evaluates overall model fit,
# quantifies path coefficients, and visualizes the structural
# architecture linking development, governance, innovation,
# academic freedom, and scientific correction dynamics.
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


def build_sem_model():

    model = """

    # Measurement model

    Development_System =~ GDP + HDI

    Governance_System =~ CPI + WGI

    Education_Investment_System =~ GEE

    Innovation_System =~ RDE + GII

    Academic_Freedom =~ AFI

    Scientific_Correction_System =~ RR_Total + Mean_RD + Collaboration_Share + NSlope13_Total


    # Structural model

    Governance_System ~ Development_System

    Education_Investment_System ~ Development_System

    Innovation_System ~ Development_System
    Innovation_System ~ Governance_System
    Innovation_System ~ Education_Investment_System

    Scientific_Correction_System ~ Innovation_System
    Scientific_Correction_System ~ Academic_Freedom
    Scientific_Correction_System ~ Governance_System
    Scientific_Correction_System ~ Development_System

    """

    return model

def run_phase4():

    data_path = filedialog.askopenfilename(
        title="Select phase1_cleaned_data.csv",
        filetypes=[("CSV Files", "*.csv")]
    )

    out_dir = filedialog.askdirectory(
        title="Select Output Folder"
    )

    if not data_path or not out_dir:
        return

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    model_desc = build_sem_model()

    # ==========================================
    # Fit model
    # ==========================================

    model = Model(model_desc)

    model.fit(df)

    # ==========================================
    # Fit indices
    # ==========================================

    stats = semopy.calc_stats(model)

    stats.to_csv(
        os.path.join(out_dir, "sem_fit_indices.csv")
    )

    # ==========================================
    # Estimates
    # ==========================================

    estimates = inspect(model)

    estimates.to_csv(
        os.path.join(out_dir, "sem_results.csv"),
        index=False
    )

    # structural only
    structural_paths = estimates[
        (estimates["op"] == "~") &
        (estimates["lval"].isin([
            "Governance_System",
            "Education_Investment_System",
            "Innovation_System",
            "Scientific_Correction_System"
        ]))
    ]

    structural_paths.to_csv(
        os.path.join(out_dir, "path_coefficients.csv"),
        index=False
    )

    # ==========================================
    # Graph
    # ==========================================

    G = nx.DiGraph()

    for _, row in structural_paths.iterrows():

        source = row["rval"]
        target = row["lval"]
        weight = row["Estimate"]

        G.add_edge(
            source,
            target,
            weight=weight
        )

    plt.figure(figsize=(22, 18))

    pos = nx.kamada_kawai_layout(G)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=3500
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=1.8
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
        "Phase 4 — Structural SEM"
    )

    plt.savefig(
        os.path.join(out_dir, "sem_model.png"),
        dpi=600
    )

    plt.savefig(
        os.path.join(out_dir, "sem_model.svg"),
        format="svg"
    )

    plt.savefig(
        os.path.join(out_dir, "sem_model.pdf"),
        format="pdf"
    )

    plt.close()

    # ==========================================
    # Report
    # ==========================================

    with open(
        os.path.join(out_dir, "phase4_report.txt"),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=== Phase 4 Structural SEM ===\n\n"
        )

        f.write(
            "Scientific Correction System = Multidimensional correction architecture\n"
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
            "The latent construct integrates correction burden, "
            "latency, collaboration complexity, and temporal correction dynamics.\n\n"
        )

        f.write("Fit indices:\n")
        f.write(str(stats))

        f.write("\n\nStructural Paths:\n")
        f.write(
            structural_paths.to_string()
        )

    messagebox.showinfo(
        "Done",
        "Phase 4 completed successfully."
    )


def main():

    root = tk.Tk()

    root.title("SEM Phase 4 — Structural Model")
    root.geometry("560x340")

    tk.Label(
        root,
        text="Phase 4 — Structural SEM",
        font=("Arial", 13, "bold")
    ).pack(pady=20)

    tk.Button(
        root,
        text="Run Phase 4",
        command=run_phase4
    ).pack(pady=30)

    root.mainloop()


if __name__ == "__main__":
    main()