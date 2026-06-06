# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 2: Macro-Structural Organization of Scientific Correction Systems
#
# Analysis:
# Structural Equation Modeling of Scientific Correction Dynamics (SEM)
#
# Phase:
# Phase 2 – Latent Construct Definition and Measurement Architecture
#
# Description:
# Defines latent constructs, specifies the SEM measurement model,
# generates confirmatory factor analysis (CFA) specifications,
# documents theoretical construct definitions, and produces graphical
# representations of the measurement architecture.
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
import json
import networkx as nx
import matplotlib.pyplot as plt
import os


# ==========================================
# Phase 2 — Construct Definition (Final SEM)
# ==========================================

LATENT_MODEL = {

    "Development_System": [
        "GDP",
        "HDI"
    ],

    "Governance_System": [
        "CPI",
        "WGI"
    ],

    "Education_Investment_System": [
        "GEE"
    ],

    "Innovation_System": [
        "RDE",
        "GII"
    ],

    "Academic_Freedom": [
        "AFI"
    ],

    "Scientific_Correction_System": [
        "RR_Total",
        "Mean_RD",
        "Collaboration_Share",
        "NSlope13_Total"
    ]
}


THEORY_DEFINITION = {

    "Development_System":
        "Macro-developmental capacity.",

    "Governance_System":
        "Institutional governance quality.",

    "Education_Investment_System":
        "Public education investment.",

    "Innovation_System":
        "Research and innovation capacity.",

    "Academic_Freedom":
        "Epistemic openness and academic autonomy.",

    "Scientific_Correction_System":
        "Multidimensional scientific correction architecture measured across burden, latency, collaboration complexity, and temporal responsiveness."
}

THEORY_DEFINITION = {
    "Development_System":
        "Macro-developmental capacity.",

    "Governance_System":
        "Institutional governance quality.",

    "Education_Investment_System":
        "Public education investment.",

    "Innovation_System":
        "Research and innovation capacity.",

    "Academic_Freedom":
        "Epistemic openness and academic autonomy.",

    "Scientific_Correction_System":
        "Observed scientific correction dynamics, measured by retraction frequency and retraction latency."
}


def build_cfa_model(latent_model):
    lines = []

    for latent, indicators in latent_model.items():
        lines.append(
            f"{latent} =~ " + " + ".join(indicators)
        )

    return "\n".join(lines)


def build_measurement_graph(latent_model):
    G = nx.DiGraph()

    for latent, indicators in latent_model.items():
        G.add_node(latent, type="latent")

        for indicator in indicators:
            G.add_node(indicator, type="observed")
            G.add_edge(latent, indicator)

    return G


def run_phase2():
    file_path = filedialog.askopenfilename(
        title="Select Phase1 Cleaned Data",
        filetypes=[("CSV Files", "*.csv")]
    )

    if not file_path:
        return

    out_dir = filedialog.askdirectory(
        title="Select Output Directory"
    )

    if not out_dir:
        return

    os.makedirs(out_dir, exist_ok=True)

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(file_path)

    missing = []

    for latent, indicators in LATENT_MODEL.items():
        for var in indicators:
            if var not in df.columns:
                missing.append(var)

    if missing:
        messagebox.showerror(
            "Missing Variables",
            "\n".join(missing)
        )
        return

    with open(
        os.path.join(out_dir, "latent_model.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(LATENT_MODEL, f, indent=4)

    cfa_model = build_cfa_model(LATENT_MODEL)

    with open(
        os.path.join(out_dir, "cfa_model.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(cfa_model)

    G = build_measurement_graph(LATENT_MODEL)

    plt.figure(figsize=(20, 16))

    pos = nx.kamada_kawai_layout(G)

    nx.draw_networkx(
        G,
        pos,
        with_labels=True
    )

    plt.title("Measurement Model")

    plt.savefig(
        os.path.join(fig_dir, "measurement_model.pdf"),
        format="pdf",
        bbox_inches="tight"
    )

    plt.close()

    report = []

    report.append("=== Phase 2 Report ===\n")
    report.append("Scientific Correction System uses:")
    report.append("- RR_Total_%")
    report.append("- Total_Mean_RD_Years")
    report.append("- Collaboration_Share_%")
    report.append("- NSlope13_Total")

    report.append(
        "\nComposite indicators excluded."
    )

    with open(
        os.path.join(out_dir, "phase2_report.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write("\n".join(report))

    messagebox.showinfo(
        "Done",
        "Phase 2 completed successfully."
    )


def main():
    root = tk.Tk()
    root.title("SEM Framework — Phase 2")
    root.geometry("620x420")

    tk.Label(
        root,
        text="Phase 2 — Construct Definition",
        font=("Arial", 13, "bold")
    ).pack(pady=20)

    tk.Button(
        root,
        text="Run Phase 2",
        font=("Arial", 11),
        command=run_phase2
    ).pack(pady=30)

    root.mainloop()


if __name__ == "__main__":
    main()