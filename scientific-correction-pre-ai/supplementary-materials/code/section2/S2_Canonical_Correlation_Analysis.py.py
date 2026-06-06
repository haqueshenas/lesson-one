# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 2: Macro-Structural Organization of Scientific Correction Systems
#
# Analysis:
# Interblock Structural Analysis Using Canonical Correlation Analysis
#
# Canonical Correlation Analysis (CCA), canonical loadings,
# structural correlation network analysis, and diagnostic visualization
# to evaluate latent relationships between macro-developmental indicators
# and national scientific correction-system characteristics.
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
# =============================================================================

# ============================================================
# INTERBLOCK STRUCTURAL ANALYSIS SYSTEM
# Canonical Correlation Analysis + Structural Diagnostics
# ============================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from tkinter import Tk, filedialog, messagebox

# ============================================================
# HIGH-QUALITY FIGURE SETTINGS
# ============================================================

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 12
plt.rcParams["figure.figsize"] = (8, 7)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ============================================================
# VARIABLE BLOCKS
# ============================================================

GLOBAL_INDEXES = [
    "GDP (log_2000_2024)",
    "CPI (2000_2024)",
    "RDE (2000_2023)",
    "GEE (2000_2024)",
    "HDI (2000_2023)",
    "WGI (2000_2024)",
    "AFI (2000_2024)",
    "GII (2000-2025)"
]

# ---------- REVISED Y BLOCK ----------
RETRACTION_INDEXES = [
    "RR_Total_%",
    "Total_Mean_RD_Years",
    "Collaboration_Share_%",
    "NSlope13_Total"
]

# ============================================================
# GUI HELPERS
# ============================================================

def gui_file(title):
    r = Tk()
    r.withdraw()
    return filedialog.askopenfilename(
        title=title,
        filetypes=[("CSV files", "*.csv")]
    )

def gui_folder(title):
    r = Tk()
    r.withdraw()
    return filedialog.askdirectory(title=title)

# ============================================================
# MAIN
# ============================================================

def main():

    inp = gui_file("Select master dataset CSV")
    out = gui_folder("Select output folder")

    if not inp or not out:
        return

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = pd.read_csv(inp)

    # IMPORTANT:
    # keep only rows complete across BOTH blocks
    full_df = df[GLOBAL_INDEXES + RETRACTION_INDEXES].dropna()

    X = full_df[GLOBAL_INDEXES]
    Y = full_df[RETRACTION_INDEXES]

    # ========================================================
    # STANDARDIZATION
    # ========================================================

    scalerX = StandardScaler()
    scalerY = StandardScaler()

    Xs = scalerX.fit_transform(X)
    Ys = scalerY.fit_transform(Y)

    # ========================================================
    # CANONICAL CORRELATION ANALYSIS
    # ========================================================

    cca = CCA(n_components=2)

    Xc, Yc = cca.fit_transform(Xs, Ys)

    # ========================================================
    # CANONICAL CORRELATIONS
    # ========================================================

    corr1, p1 = pearsonr(Xc[:, 0], Yc[:, 0])
    corr2, p2 = pearsonr(Xc[:, 1], Yc[:, 1])

    # ========================================================
    # SAVE SCORES
    # ========================================================

    cca_df = pd.DataFrame({
        "CCA1_Global": Xc[:, 0],
        "CCA1_Correction": Yc[:, 0],
        "CCA2_Global": Xc[:, 1],
        "CCA2_Correction": Yc[:, 1]
    })

    cca_df.to_csv(
        os.path.join(out, "cca_scores.csv"),
        index=False
    )

    # ========================================================
    # LOADINGS
    # ========================================================

    loadings_X = pd.DataFrame(
        np.corrcoef(Xs.T, Xc.T)[:len(GLOBAL_INDEXES), len(GLOBAL_INDEXES):],
        index=GLOBAL_INDEXES,
        columns=["CCA1", "CCA2"]
    )

    loadings_Y = pd.DataFrame(
        np.corrcoef(Ys.T, Yc.T)[:len(RETRACTION_INDEXES), len(RETRACTION_INDEXES):],
        index=RETRACTION_INDEXES,
        columns=["CCA1", "CCA2"]
    )

    loadings_X.to_csv(os.path.join(out, "cca_loadings_X.csv"))
    loadings_Y.to_csv(os.path.join(out, "cca_loadings_Y.csv"))

    # ========================================================
    # FIGURE 1 — CANONICAL STRUCTURE
    # ========================================================

    plt.figure(figsize=(8, 7))

    sns.regplot(
        x=Xc[:, 0],
        y=Yc[:, 0],
        scatter_kws={"s": 70},
        line_kws={"linewidth": 2}
    )

    plt.xlabel("Canonical Variate 1 — Development System")
    plt.ylabel("Canonical Variate 1 — Scientific Correction System")

    plt.title("Canonical Correlation Structure")

    txt = (
        f"Canonical correlation (r) = {corr1:.3f}\n"
        f"p-value = {p1:.4g}"
    )

    plt.text(
        0.05,
        0.95,
        txt,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    plt.tight_layout()

    plt.savefig(os.path.join(out, "Fig_CCA_Structure.pdf"))
    plt.savefig(os.path.join(out, "Fig_CCA_Structure.png"))

    plt.close()

    # ========================================================
    # NETWORK ANALYSIS
    # ========================================================

    G = nx.Graph()

    corr_matrix = full_df[GLOBAL_INDEXES + RETRACTION_INDEXES].corr()

    for g in GLOBAL_INDEXES:
        for r in RETRACTION_INDEXES:

            c = corr_matrix.loc[g, r]

            if abs(c) > 0.40:
                G.add_edge(g, r, weight=c)

    plt.figure(figsize=(16, 16))

    pos = nx.spring_layout(G, seed=42, k=0.7)

    edge_weights = [
        abs(G[u][v]["weight"]) * 3
        for u, v in G.edges()
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=2500
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_weights,
        alpha=0.7
    )

    plt.title("Interblock Structural Correlation Network")

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(os.path.join(out, "interblock_network.pdf"))
    plt.savefig(os.path.join(out, "interblock_network.png"))

    plt.close()

    # ========================================================
    # INTERPRETIVE REPORT
    # ========================================================

    with open(
        os.path.join(out, "interblock_report.txt"),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "INTERBLOCK STRUCTURAL ANALYSIS REPORT\n"
        )

        f.write("=" * 90 + "\n\n")

        f.write(
            "This analysis evaluates the latent structural coupling "
            "between macro-developmental conditions and scientific "
            "correction-system dynamics using Canonical Correlation "
            "Analysis (CCA).\n\n"
        )

        f.write(
            f"First canonical correlation (CCA1): "
            f"r = {corr1:.4f}, "
            f"p = {p1:.4g}\n"
        )

        f.write(
            f"Second canonical correlation (CCA2): "
            f"r = {corr2:.4f}, "
            f"p = {p2:.4g}\n\n"
        )

        f.write(
            "CCA1 represents the dominant shared structural dimension "
            "linking macro-developmental organization to scientific "
            "correction dynamics.\n\n"
        )

        f.write(
            "Variable loadings indicate the relative contribution of "
            "each indicator to the latent canonical dimensions.\n\n"
        )

        f.write(
            "The network graph visualizes strong cross-domain "
            "correlations (|r| > 0.40), interpreted as structural "
            "dependencies rather than causal pathways.\n"
        )

    # ========================================================
    # FINISHED
    # ========================================================

    messagebox.showinfo(
        "Done",
        "Revised interblock structural analysis completed."
    )

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()