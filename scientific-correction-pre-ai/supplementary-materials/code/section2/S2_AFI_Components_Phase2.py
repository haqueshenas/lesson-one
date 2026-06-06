# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 2: Macro-Structural Organization of Scientific Correction Systems
#
# Analysis:
# AFI Components Analysis
#
# Phase:
# AFI Components Analysis – Phase 2
# Multivariate Analysis of Academic Freedom Components and
# Scientific Correction Dynamics
#
# Description:
# Performs a comprehensive investigation of the
# relationships between Academic Freedom Index (AFI) components and
# multidimensional indicators of scientific correction systems.
#
# Analytical Modules:
# - Pearson Correlation Analysis
# - Spearman Correlation Analysis
# - Significance-Annotated Correlation Heatmaps
# - Publication-Grade Regression Models
# - Bootstrap Confidence Intervals
# - Principal Component Analysis (PCA)
# - Joint AFI–Correction System Embedding
# - Multiple Regression Modeling
# - Partial Correlation Analysis
# - High-Resolution Figure Generation
#
# Academic Freedom Components:
# - Freedom to Research and Teach
# - Freedom of Academic Exchange and Dissemination
# - Institutional Autonomy
# - Campus Integrity
# - Freedom of Academic and Cultural Expression
#
# Scientific Correction Indicators:
# - Retraction Rate
# - Collaboration Share
# - Retraction Delay
# - Temporal Retraction Dynamics
#
# Objective:
# To identify which dimensions of academic freedom are most strongly
# associated with national scientific correction patterns and to
# characterize the latent structure linking epistemic openness and
# scientific self-correction mechanisms.
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

import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk

from tkinter import filedialog, messagebox
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

from adjustText import adjust_text

import statsmodels.api as sm
import pingouin as pg

warnings.filterwarnings("ignore")


# ============================================================
# GLOBAL STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 600,
})

sns.set_style("whitegrid")

# ============================================================
# SHORT LABELS FOR PCA VECTORS
# ============================================================

SHORT_LABELS = {
    "Freedom_Research_Teach_Mean_2000_2024": "FRT",
    "Freedom_Academic_Exchange_Mean_2000_2024": "FAE",
    "Institutional_Autonomy_Mean_2000_2024": "IA",
    "Campus_Integrity_Mean_2000_2024": "CI",
    "Academic_Cultural_Expression_Mean_2000_2024": "ACE",
    "RR_Total_%": "RR",
    "Collaboration_Share_%": "COL",
    "Total_Mean_RD_Years": "RDY",
    "NSlope13_Total": "NS"
}

# ============================================================
# GUI FUNCTIONS
# ============================================================


def select_input_file():
    root = tk.Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        title="Select AFI-Correction CSV Dataset",
        filetypes=[("CSV Files", "*.csv")]
    )

    if not path:
        raise SystemExit("❌ No input file selected.")

    return path



def select_output_folder():
    root = tk.Tk()
    root.withdraw()

    path = filedialog.askdirectory(
        title="Select Output Directory"
    )

    if not path:
        raise SystemExit("❌ No output folder selected.")

    return path


# ============================================================
# SAFE CSV LOADER
# ============================================================


def safe_read_csv(path):

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1"
    ]

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] Loaded using encoding: {enc}")
            return df
        except:
            continue

    raise ValueError("❌ Unable to load CSV file.")


# ============================================================
# SIGNIFICANCE STARS
# ============================================================


def significance_stars(p):

    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


# ============================================================
# CORRELATION MATRIX WITH STARS
# ============================================================

def correlation_matrix(df, variables, method="pearson"):

    corr = pd.DataFrame(
        index=variables,
        columns=variables,
        dtype=float
    )

    pvals = pd.DataFrame(
        index=variables,
        columns=variables,
        dtype=float
    )

    annotated = pd.DataFrame(
        index=variables,
        columns=variables
    )

    for i in variables:
        for j in variables:

            try:

                # --------------------------------------------
                # Extract only required columns
                # --------------------------------------------

                temp = df[[i, j]].copy()

                # --------------------------------------------
                # Handle duplicated column names safely
                # --------------------------------------------

                temp = temp.loc[:, ~temp.columns.duplicated()]

                # --------------------------------------------
                # Convert to numeric
                # --------------------------------------------

                temp[i] = pd.to_numeric(
                    temp[i],
                    errors="coerce"
                )

                temp[j] = pd.to_numeric(
                    temp[j],
                    errors="coerce"
                )

                # --------------------------------------------
                # Pairwise deletion
                # --------------------------------------------

                temp = temp[[i, j]].dropna()

                # --------------------------------------------
                # Minimum sample check
                # --------------------------------------------

                if len(temp) < 3:

                    r = np.nan
                    p = np.nan

                # --------------------------------------------
                # Constant-variable check
                # --------------------------------------------

                elif float(temp[i].nunique()) <= 1 or \
                     float(temp[j].nunique()) <= 1:

                    r = np.nan
                    p = np.nan

                else:

                    x = temp[i].astype(float).values
                    y = temp[j].astype(float).values

                    if method == "pearson":

                        r, p = pearsonr(x, y)

                    else:

                        r, p = spearmanr(x, y)

            except Exception as e:

                print(f"[WARNING] Correlation failed:")
                print(f"{i} vs {j}")
                print(e)

                r = np.nan
                p = np.nan

            # --------------------------------------------
            # Save results
            # --------------------------------------------

            corr.loc[i, j] = r
            pvals.loc[i, j] = p

            # --------------------------------------------
            # Annotation
            # --------------------------------------------

            if np.isnan(r):

                annotated.loc[i, j] = ""

            else:

                annotated.loc[i, j] = (
                    f"{r:.2f}{significance_stars(p)}"
                )

    return corr, pvals, annotated

# ============================================================
# CORRELATION HEATMAP
# ============================================================


def save_heatmap(corr, annotated, outpath, title):

    plt.figure(figsize=(12, 10))

    plt.grid(False)

    mask = np.triu(np.ones_like(corr, dtype=bool))

    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        annot=annotated,
        fmt="",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0,
        linecolor=None,
        square=True,
        cbar_kws={"shrink": 0.8}
    )

    ax = plt.gca()
    ax.grid(False)

    plt.title(title, fontsize=16, weight="bold")

    plt.tight_layout()

    plt.savefig(outpath + ".png", bbox_inches="tight")
    plt.savefig(outpath + ".pdf", bbox_inches="tight")
    plt.savefig(outpath + ".svg", bbox_inches="tight")

    plt.close()

# ============================================================
# PUBLICATION-GRADE REGRESSION PLOT
# ============================================================


def advanced_regplot(df, x, y, outpath):

    temp = df[[x, y, "ISO3"]].copy()

    temp[x] = pd.to_numeric(temp[x], errors="coerce")
    temp[y] = pd.to_numeric(temp[y], errors="coerce")

    temp = temp.dropna()

    if len(temp) < 3:
        return

    r, p = pearsonr(
        temp[x].astype(float),
        temp[y].astype(float)
    )

    plt.figure(figsize=(7, 6))

    sns.regplot(
        data=temp,
        x=x,
        y=y,
        ci=95,
        scatter_kws={
            "s": 80,
            "alpha": 0.9,
            "edgecolors": "black"
        },
        line_kws={
            "linewidth": 2.5
        }
    )

    # --------------------------------------------
    # ISO3 labels
    # --------------------------------------------

    texts = []

    for i in range(len(temp)):

        texts.append(
            plt.text(
                temp[x].iloc[i],
                temp[y].iloc[i],
                temp["ISO3"].iloc[i],
                fontsize=7,
                weight="bold"
            )
        )

    adjust_text(texts)

    plt.title(f"{x} vs {y}", weight="bold")

    plt.text(
        0.03,
        0.03,
        f"r = {r:.3f}\np = {p:.4g}",
        transform=plt.gca().transAxes,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8
        )
    )

    plt.grid(False)

    plt.tight_layout()

    plt.savefig(outpath + ".png", bbox_inches="tight")
    plt.savefig(outpath + ".pdf", bbox_inches="tight")
    plt.savefig(outpath + ".svg", bbox_inches="tight")

    plt.close()

# ============================================================
# PCA ANALYSIS
# ============================================================


# ============================================================
# GENERIC PCA FUNCTION
# ============================================================

def run_pca_analysis(
        df,
        variables,
        outdir,
        filename,
        title,
        draw_vectors=True
):

    # ------------------------------------------------
    # Prepare data
    # ------------------------------------------------

    temp = df[variables + ["ISO3"]].copy()

    for col in variables:
        temp[col] = pd.to_numeric(
            temp[col],
            errors="coerce"
        )

    temp = temp.dropna()

    if len(temp) < 3:
        print(f"[WARNING] PCA skipped: {filename}")
        return None, None

    X = temp[variables]

    # ------------------------------------------------
    # Standardization
    # ------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------
    # PCA
    # ------------------------------------------------

    pca = PCA(n_components=2)

    pcs = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_

    # ------------------------------------------------
    # PCA dataframe
    # ------------------------------------------------

    pca_df = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "ISO3": temp["ISO3"].values
    })

    # ------------------------------------------------
    # Loadings
    # ------------------------------------------------

    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        columns=["PC1", "PC2"],
        index=variables
    )

    loadings.to_csv(
        os.path.join(outdir, f"{filename}_Loadings.csv"),
        encoding="utf-8-sig"
    )

    # ------------------------------------------------
    # Figure
    # ------------------------------------------------

    plt.figure(figsize=(9, 8))

    # ------------------------
    # Vectors FIRST (background)
    # ------------------------

    if draw_vectors:

        for i, var in enumerate(variables):

            xvec = loadings.iloc[i, 0] * 3
            yvec = loadings.iloc[i, 1] * 3

            plt.arrow(
                0,
                0,
                xvec,
                yvec,
                color="red",
                alpha=0.35,
                linewidth=1.2,
                head_width=0.05,
                zorder=1
            )

            plt.text(
                xvec * 1.08,
                yvec * 1.08,
                SHORT_LABELS.get(var, var),
                fontsize=8,
                color="darkred",
                alpha=0.85,
                weight="bold",
                zorder=2
            )

    # ------------------------
    # Scatter ON TOP
    # ------------------------

    plt.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        s=85,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
        zorder=3
    )

    # ------------------------
    # ISO labels
    # ------------------------

    texts = []

    for i in range(len(pca_df)):

        texts.append(
            plt.text(
                pca_df["PC1"].iloc[i],
                pca_df["PC2"].iloc[i],
                pca_df["ISO3"].iloc[i],
                fontsize=7,
                weight="bold",
                zorder=4
            )
        )

    adjust_text(
        texts,
        arrowprops=dict(
            arrowstyle="-",
            color="gray",
            lw=0.5,
            alpha=0.5
        )
    )

    # ------------------------------------------------
    # Axes
    # ------------------------------------------------

    plt.xlabel(
        f"PC1 ({explained[0]*100:.1f}% variance)"
    )

    plt.ylabel(
        f"PC2 ({explained[1]*100:.1f}% variance)"
    )

    plt.title(title, weight="bold")

    plt.grid(False)

    plt.tight_layout()

    outpath = os.path.join(outdir, filename)

    plt.savefig(outpath + ".png", bbox_inches="tight")
    plt.savefig(outpath + ".pdf", bbox_inches="tight")
    plt.savefig(outpath + ".svg", bbox_inches="tight")

    plt.close()

    return pca_df, loadings

# ============================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================


def bootstrap_correlation(x, y, n_boot=5000):

    vals = []

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (~np.isnan(x)) & (~np.isnan(y))

    x = x[mask]
    y = y[mask]

    if len(x) < 4:
        return np.nan, np.nan

    for _ in range(n_boot):

        idx = np.random.choice(
            len(x),
            len(x),
            replace=True
        )

        xb = x[idx]
        yb = y[idx]

        # constant vector protection
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue

        try:
            r, _ = pearsonr(xb, yb)

            if np.isfinite(r):
                vals.append(r)

        except:
            continue

    if len(vals) < 10:
        return np.nan, np.nan

    low = np.percentile(vals, 2.5)
    high = np.percentile(vals, 97.5)

    return low, high

# ============================================================
# MULTIPLE REGRESSION
# ============================================================


def run_multiple_regression(df, predictors, target, outdir):

    X = df[predictors]
    y = df[target]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HC3")

    report_path = os.path.join(
        outdir,
        f"Regression_{target}.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    return model


# ============================================================
# PARTIAL CORRELATION
# ============================================================


# ============================================================
# PARTIAL CORRELATION
# ============================================================

def run_partial_correlations(df, outdir):

    results = []

    afi_components = [
        "Freedom_Research_Teach_Mean_2000_2024",
        "Freedom_Academic_Exchange_Mean_2000_2024",
        "Institutional_Autonomy_Mean_2000_2024",
        "Campus_Integrity_Mean_2000_2024",
        "Academic_Cultural_Expression_Mean_2000_2024"
    ]

    targets = [
        "RR_Total_%",
        "Total_Mean_RD_Years",
        "NSlope13_Total",
        "Collaboration_Share_%"
    ]

    for comp in afi_components:

        for target in targets:

            try:

                covars = [v for v in afi_components if v != comp]

                pcorr = pg.partial_corr(
                    data=df,
                    x=comp,
                    y=target,
                    covar=covars,
                    method="pearson"
                )

                # ------------------------------------------------
                # Robust extraction across pingouin versions
                # ------------------------------------------------

                r_value = pcorr["r"].values[0]

                # Different versions of pingouin
                if "p-val" in pcorr.columns:
                    p_value = pcorr["p-val"].values[0]

                elif "p_unc" in pcorr.columns:
                    p_value = pcorr["p_unc"].values[0]

                else:
                    p_value = np.nan

                # Confidence intervals if available
                if "CI95%" in pcorr.columns:

                    ci = pcorr["CI95%"].values[0]

                    if isinstance(ci, (list, tuple, np.ndarray)):
                        ci_low, ci_high = ci[0], ci[1]
                    else:
                        ci_low, ci_high = np.nan, np.nan

                else:
                    ci_low, ci_high = np.nan, np.nan

                results.append({
                    "Component": comp,
                    "Target": target,
                    "Partial_r": r_value,
                    "p_value": p_value,
                    "CI95_low": ci_low,
                    "CI95_high": ci_high,
                    "Significance": significance_stars(
                        p_value if not np.isnan(p_value) else 1
                    )
                })

            except Exception as e:

                print(f"[WARNING] Partial correlation failed:")
                print(f"Component: {comp}")
                print(f"Target: {target}")
                print(f"Error: {e}")

                results.append({
                    "Component": comp,
                    "Target": target,
                    "Partial_r": np.nan,
                    "p_value": np.nan,
                    "CI95_low": np.nan,
                    "CI95_high": np.nan,
                    "Significance": ""
                })

    out_df = pd.DataFrame(results)

    out_csv = os.path.join(
        outdir,
        "Partial_Correlations.csv"
    )

    out_df.to_csv(
        out_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"[SUCCESS] Partial correlations saved:")
    print(out_csv)

    return out_df

# ============================================================
# FOREST PLOT FOR AFI COMPONENTS VS RR
# ============================================================

def create_rr_forest_plot(boot_df, outdir):

    temp = boot_df[
        boot_df["Outcome"] == "RR_Total_%"
    ].copy()

    short_names = {
        "AFI_Mean_2000_2024": "AFI",
        "Freedom_Research_Teach_Mean_2000_2024": "FRT",
        "Freedom_Academic_Exchange_Mean_2000_2024": "FAE",
        "Institutional_Autonomy_Mean_2000_2024": "IA",
        "Campus_Integrity_Mean_2000_2024": "CI",
        "Academic_Cultural_Expression_Mean_2000_2024": "ACE"
    }

    temp["Short"] = temp["Predictor"].map(short_names)

    temp = temp.sort_values("r")

    plt.figure(figsize=(7, 5))

    y_pos = np.arange(len(temp))

    plt.errorbar(
        temp["r"],
        y_pos,
        xerr=[
            temp["r"] - temp["CI_low"],
            temp["CI_high"] - temp["r"]
        ],
        fmt='o',
        capsize=4,
        linewidth=1.5
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1,
        color="gray"
    )

    plt.yticks(y_pos, temp["Short"])

    plt.xlabel("Pearson r (95% CI)")

    plt.title(
        "Bootstrap Correlations:\nAFI Components vs Retraction Rate",
        weight="bold"
    )

    plt.grid(False)

    plt.tight_layout()

    outpath = os.path.join(
        outdir,
        "Forest_AFI_vs_RR"
    )

    plt.savefig(outpath + ".png", bbox_inches="tight")
    plt.savefig(outpath + ".pdf", bbox_inches="tight")
    plt.savefig(outpath + ".svg", bbox_inches="tight")

    plt.close()

# ============================================================
# PANEL FIGURE
# ============================================================

def create_main_panel_figure(df, outdir):

    x = "Academic_Cultural_Expression_Mean_2000_2024"
    y = "RR_Total_%"

    temp = df[[x, y, "ISO3"]].dropna()

    r, p = pearsonr(
        temp[x],
        temp[y]
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    sns.regplot(
        data=temp,
        x=x,
        y=y,
        ci=95,
        scatter_kws={
            "s": 80,
            "edgecolors": "black",
            "alpha": 0.9
        },
        line_kws={
            "linewidth": 2.5
        },
        ax=ax
    )

    texts = []

    for i in range(len(temp)):

        texts.append(
            ax.text(
                temp[x].iloc[i],
                temp[y].iloc[i],
                temp["ISO3"].iloc[i],
                fontsize=7,
                weight="bold"
            )
        )

    adjust_text(texts)

    ax.text(
        0.03,
        0.03,
        f"r = {r:.3f}\np = {p:.3g}",
        transform=ax.transAxes,
        bbox=dict(
            facecolor="white",
            alpha=0.8
        )
    )

    ax.set_title(
        "ACE vs Retraction Rate",
        weight="bold"
    )

    ax.grid(False)

    plt.tight_layout()

    outpath = os.path.join(
        outdir,
        "Main_Regression_Panel"
    )

    plt.savefig(outpath + ".png", bbox_inches="tight")
    plt.savefig(outpath + ".pdf", bbox_inches="tight")
    plt.savefig(outpath + ".svg", bbox_inches="tight")

    plt.close()

# ============================================================
# MASTER MULTI-PANEL REGRESSION FIGURE
# 24-PANEL SUPPLEMENTARY FIGURE
# ============================================================

def create_master_regression_panel(
        df,
        afi_vars,
        correction_vars,
        outdir
):


    # ------------------------------------------------
    # Figure layout
    # ------------------------------------------------

    nrows = len(afi_vars)
    ncols = len(correction_vars)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.2*ncols, 4.8*nrows),
        constrained_layout=True
    )

    # ------------------------------------------------
    # Handle single-row/col edge cases
    # ------------------------------------------------

    if nrows == 1:
        axes = np.array([axes])

    if ncols == 1:
        axes = axes.reshape(-1, 1)

    # ------------------------------------------------
    # Panel lettering
    # ------------------------------------------------

    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    counter = 0

    # ------------------------------------------------
    # Loop panels
    # ------------------------------------------------

    for r, afi in enumerate(afi_vars):

        for c, outcome in enumerate(correction_vars):

            ax = axes[r, c]

            # ----------------------------------------
            # Data prep
            # ----------------------------------------

            temp = df[[afi, outcome, "ISO3"]].copy()

            temp[afi] = pd.to_numeric(
                temp[afi],
                errors="coerce"
            )

            temp[outcome] = pd.to_numeric(
                temp[outcome],
                errors="coerce"
            )

            temp = temp.dropna()

            # ----------------------------------------
            # Remove inf values safely
            # ----------------------------------------

            temp = temp[
                np.isfinite(temp[afi]) &
                np.isfinite(temp[outcome])
                ]

            # ----------------------------------------
            # Pearson correlation
            # ----------------------------------------

            try:

                if len(temp) >= 3 and \
                        temp[afi].nunique() > 1 and \
                        temp[outcome].nunique() > 1:

                    xvals = temp[afi].astype(float).values
                    yvals = temp[outcome].astype(float).values

                    r_value, p_value = pearsonr(
                        xvals,
                        yvals
                    )

                else:

                    r_value = np.nan
                    p_value = np.nan

            except Exception as e:

                print(
                    f"[WARNING] Correlation failed: "
                    f"{afi} vs {outcome}"
                )

                print(e)

                r_value = np.nan
                p_value = np.nan

            # ----------------------------------------
            # Regression plot
            # ----------------------------------------

            sns.regplot(
                data=temp,
                x=afi,
                y=outcome,
                ci=95,
                ax=ax,

                scatter_kws={
                    "s": 34,
                    "alpha": 0.90,
                    "edgecolors": "black",
                    "linewidths": 0.35
                },

                line_kws={
                    "linewidth": 1.8
                }
            )

            # ----------------------------------------
            # ISO labels
            # ----------------------------------------

            texts = []

            for i in range(len(temp)):

                texts.append(

                    ax.text(
                        temp[afi].iloc[i],
                        temp[outcome].iloc[i],
                        temp["ISO3"].iloc[i],

                        fontsize=4.8,
                        weight="bold"
                    )
                )

            try:

                adjust_text(
                    texts,
                    ax=ax,

                    arrowprops=dict(
                        arrowstyle="-",
                        color="gray",
                        lw=0.25,
                        alpha=0.4
                    )
                )

            except:
                pass

            # ----------------------------------------
            # Statistics box
            # ----------------------------------------

            ax.text(
                0.03,
                0.03,

                f"r = {r_value:.2f}",
                # f"p = {p_value:.3g}",

                transform=ax.transAxes,

                fontsize=7,

                bbox=dict(
                    facecolor="white",
                    alpha=0.82,
                    edgecolor="lightgray",
                    boxstyle="round,pad=0.25"
                )
            )

            # ----------------------------------------
            # Titles only on top row
            # ----------------------------------------

            if r == 0:
                ax.set_title(
                    outcome,
                    fontsize=11,
                    weight="bold"
                )

            # ----------------------------------------
            # Y labels only first column
            # ----------------------------------------

            if c == 0:

                ax.set_ylabel(
                    afi,
                    fontsize=10,
                    weight="bold"
                )

            else:

                ax.set_ylabel("")

            # ----------------------------------------
            # X labels only bottom row
            # ----------------------------------------

            if r == nrows - 1:

                ax.set_xlabel(
                    outcome,
                    fontsize=10
                )

            else:

                ax.set_xlabel("")

            # ----------------------------------------
            # Remove grids
            # ----------------------------------------

            ax.grid(False)

            # ----------------------------------------
            # Cleaner spines
            # ----------------------------------------
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.tick_params(
                axis='both',
                labelsize=7
            )

            # ----------------------------------------
            # Panel letters
            # ----------------------------------------

            ax.text(
                -0.12,
                1.05,

                letters[counter],

                transform=ax.transAxes,

                fontsize=13,
                weight="bold"
            )

            counter += 1

    # ------------------------------------------------
    # Global title
    # ------------------------------------------------

    fig.suptitle(
        "Associations Between Academic Freedom and "
        "Scientific Correction Metrics",

        fontsize=20,
        weight="bold"
    )

    plt.subplots_adjust(
        top=0.93,
        wspace=0.22,
        hspace=0.22
    )

    # ------------------------------------------------
    # Save outputs
    # ------------------------------------------------

    outpath = os.path.join(
        outdir,
        "SUPPLEMENTARY_MASTER_REGRESSION_PANEL"
    )

    # Vector outputs
    plt.savefig(
        outpath + ".pdf",
        bbox_inches="tight"
    )

    plt.savefig(
        outpath + ".svg",
        bbox_inches="tight"
    )

    # PNG preview
    plt.savefig(
        outpath + ".png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    print("\n[SUCCESS] Master panel figure generated.")

# ============================================================
# MAIN
# ============================================================


def main():

    print("="*70)
    print(" AFI vs Scientific Correction System ")
    print(" Advanced Publication-Grade Analysis Pipeline ")
    print("="*70)

    infile = select_input_file()
    outdir = select_output_folder()

    os.makedirs(outdir, exist_ok=True)

    print("\n[INFO] Loading dataset...")

    df = safe_read_csv(infile)

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    afi_vars = [
        "AFI_Mean_2000_2024",
        "Freedom_Research_Teach_Mean_2000_2024",
        "Freedom_Academic_Exchange_Mean_2000_2024",
        "Institutional_Autonomy_Mean_2000_2024",
        "Campus_Integrity_Mean_2000_2024",
        "Academic_Cultural_Expression_Mean_2000_2024"
    ]

    correction_vars = [
        "RR_Total_%",
        "Collaboration_Share_%",
        "Total_Mean_RD_Years",
        "NSlope13_Total"
    ]

    all_vars = afi_vars + correction_vars

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    print("\n[INFO] Running Pearson correlations...")

    corr_p, pvals_p, ann_p = correlation_matrix(
        df,
        all_vars,
        method="pearson"
    )

    corr_p.to_csv(
        os.path.join(outdir, "Pearson_Correlation_Matrix.csv"),
        encoding="utf-8-sig"
    )

    save_heatmap(
        corr_p,
        ann_p,
        os.path.join(outdir, "Pearson_Heatmap"),
        "Pearson Correlation Matrix"
    )

    print("\n[INFO] Running Spearman correlations...")

    corr_s, pvals_s, ann_s = correlation_matrix(
        df,
        all_vars,
        method="spearman"
    )

    corr_s.to_csv(
        os.path.join(outdir, "Spearman_Correlation_Matrix.csv"),
        encoding="utf-8-sig"
    )

    save_heatmap(
        corr_s,
        ann_s,
        os.path.join(outdir, "Spearman_Heatmap"),
        "Spearman Correlation Matrix"
    )

    # --------------------------------------------------------
    # Scatterplots
    # --------------------------------------------------------

    print("\n[INFO] Generating regression figures...")

    for afi in afi_vars:
        for corr_var in correction_vars:

            fname = f"{afi}_vs_{corr_var}"

            advanced_regplot(
                df,
                afi,
                corr_var,
                os.path.join(outdir, fname)
            )

    # --------------------------------------------------------
    # MASTER SUPPLEMENTARY PANEL
    # --------------------------------------------------------

    print("\n[INFO] Creating master supplementary panel...")

    create_master_regression_panel(
        df,
        afi_vars,
        correction_vars,
        outdir
    )

    # --------------------------------------------------------
    # AFI PCA
    # --------------------------------------------------------

    print("\n[INFO] Running AFI PCA...")

    afi_components = [
        "Freedom_Research_Teach_Mean_2000_2024",
        "Freedom_Academic_Exchange_Mean_2000_2024",
        "Institutional_Autonomy_Mean_2000_2024",
        "Campus_Integrity_Mean_2000_2024",
        "Academic_Cultural_Expression_Mean_2000_2024"
    ]

    run_pca_analysis(
        df=df,
        variables=afi_components,
        outdir=outdir,
        filename="AFI_PCA",
        title="PCA of AFI Components"
    )

    # --------------------------------------------------------
    # AFI Components + RR PCA
    # --------------------------------------------------------

    print("\n[INFO] Running AFI + RR PCA...")

    rr_pca_vars = afi_components + [
        "RR_Total_%"
    ]

    run_pca_analysis(
        df=df,
        variables=rr_pca_vars,
        outdir=outdir,
        filename="AFI_RR_PCA",
        title="PCA: AFI Components and Retraction Rate"
    )

    # --------------------------------------------------------
    # Joint PCA
    # --------------------------------------------------------

    print("\n[INFO] Running Joint PCA...")

    joint_vars = afi_components + [
        "RR_Total_%",
        "Collaboration_Share_%",
        "Total_Mean_RD_Years",
        "NSlope13_Total"
    ]

    run_pca_analysis(
        df=df,
        variables=joint_vars,
        outdir=outdir,
        filename="Joint_PCA",
        title="Joint PCA: AFI and Scientific Correction System"
    )

    # --------------------------------------------------------
    # Bootstrap confidence intervals
    # --------------------------------------------------------

    print("\n[INFO] Bootstrap confidence intervals...")

    bootstrap_results = []

    for afi in afi_vars:
        for corr_var in correction_vars:

            x = df[afi].values
            y = df[corr_var].values

            r, p = pearsonr(x, y)

            low, high = bootstrap_correlation(x, y)

            bootstrap_results.append({
                "Predictor": afi,
                "Outcome": corr_var,
                "r": r,
                "p": p,
                "CI_low": low,
                "CI_high": high
            })

    boot_df = pd.DataFrame(bootstrap_results)

    boot_df.to_csv(
        os.path.join(outdir, "Bootstrap_Correlation_CI.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Forest plot
    # --------------------------------------------------------

    create_rr_forest_plot(
        boot_df,
        outdir
    )

    # --------------------------------------------------------
    # Main panel regression
    # --------------------------------------------------------

    create_main_panel_figure(
        df,
        outdir
    )

    # --------------------------------------------------------
    # Multiple regression
    # --------------------------------------------------------

    print("\n[INFO] Multiple regression models...")

    predictors = afi_vars[1:]

    for target in correction_vars:
        run_multiple_regression(
            df,
            predictors,
            target,
            outdir
        )

    # --------------------------------------------------------
    # Partial correlations
    # --------------------------------------------------------

    print("\n[INFO] Partial correlations...")

    run_partial_correlations(df, outdir)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    report = os.path.join(outdir, "Analysis_Report.txt")

    with open(report, "w", encoding="utf-8") as f:

        f.write("AFI vs Scientific Correction System\n")
        f.write("="*70 + "\n\n")

        f.write("Generated outputs:\n\n")

        f.write("1. Pearson correlation matrices\n")
        f.write("2. Spearman correlation matrices\n")
        f.write("3. Significance-star heatmaps\n")
        f.write("4. Publication-grade regression plots\n")
        f.write("5. PCA embedding of AFI components\n")
        f.write("6. Bootstrap confidence intervals\n")
        f.write("7. Multiple regression models\n")
        f.write("8. Partial correlations\n")
        f.write("9. Vector graphics outputs\n")

    print("\n[SUCCESS] Analysis completed successfully.")
    print(f"[OUTPUT] Results saved in:\n{outdir}")

    messagebox.showinfo(
        "Completed",
        "AFI–Correction analysis completed successfully."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()