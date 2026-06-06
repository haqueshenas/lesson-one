# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 2: Macro-Structural Organization of Scientific Correction Systems
#
# Analysis:
# Single-Index Correlation and Regression Analysis
#
# Description:
# Performs independent bivariate analyses between each development
# indicator and multiple scientific correction indicators using
# Pearson correlation, Spearman rank correlation, and ordinary least
# squares (OLS) regression. Generates publication-ready scatterplots,
# correlation heatmaps, statistical summaries, and interpretive reports
# for exploratory assessment of pairwise relationships.
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
#
# License:
# MIT License
#
# Copyright (c) 2026 Abbas Haghshenas
# =============================================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from tkinter import Tk, filedialog, messagebox

# ===== High Quality Graphics Settings =====
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 11
plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

GLOBAL_INDEXES = [
    "GDP (log_2000_2024)","CPI (2000_2024)","RDE (2000_2023)",
    "GEE (2000_2024)","HDI (2000_2023)","WGI (2000_2024)",
    "AFI (2000_2024)","GII (2000-2025)"
]

RETRACTION_INDEXES = [
    "RR_Single_%","RR_Collab_%","RR_Total_%",
    "Total_Mean_RD_Years","Single_Mean_RD_Years","Collab_Mean_RD_Years",
    "NRRDI_Total_RR","NRRDI_Single_RR","NRRDI_Collab_RR",
    "NRS13I_Total_RR","NRS13I_Single_RR","NRS13I_Collab_RR"
]

def gui_file(title):
    root = Tk(); root.withdraw()
    return filedialog.askopenfilename(title=title, filetypes=[("CSV","*.csv")])

def gui_folder(title):
    root = Tk(); root.withdraw()
    return filedialog.askdirectory(title=title)

def strength_label(r):
    ar = abs(r)
    if ar >= 0.7: return "Strong"
    elif ar >= 0.4: return "Moderate"
    elif ar >= 0.2: return "Weak"
    else: return "Negligible"

def analyze(df, gcol, out_root):

    safe = gcol.replace(" ","_").replace("(","").replace(")","").replace("-","_")
    out_dir = os.path.join(out_root, safe)
    os.makedirs(out_dir, exist_ok=True)

    corr_rows, reg_rows, desc_lines = [], [], []

    for rcol in RETRACTION_INDEXES:
        if gcol not in df.columns or rcol not in df.columns:
            continue

        sub = df[[gcol,rcol]].dropna()
        if len(sub) < 6: continue

        x,y = sub[gcol], sub[rcol]

        pr,pp = pearsonr(x,y)
        sr,sp = spearmanr(x,y)

        strength = strength_label(pr)

        corr_rows.append({
            "Global":gcol,"Retraction":rcol,
            "Pearson_r":pr,"Pearson_p":pp,
            "Spearman_r":sr,"Spearman_p":sp,
            "Strength":strength,"N":len(sub)
        })

        X = sm.add_constant(x)
        model = sm.OLS(y,X).fit()

        reg_rows.append({
            "Global":gcol,"Retraction":rcol,
            "beta":model.params[1],
            "p":model.pvalues[1],
            "R2":model.rsquared,
            "N":len(sub)
        })

        interpret = "NOT suitable for main article discussion."
        if strength in ["Strong","Moderate"] and pp < 0.05:
            interpret = "RELEVANT for scientific reporting and theoretical interpretation."

        desc_lines.append(
            f"{gcol} vs {rcol}: r={pr:.3f}, p={pp:.4f} → {strength} relationship. {interpret}"
        )

        # plot
        plt.figure()
        sns.regplot(x=x,y=y,ci=95)
        plt.title(f"{gcol} vs {rcol}")
        plt.tight_layout()

        base = os.path.join(out_dir,f"{safe}__{rcol}")
        plt.savefig(base+".svg")
        plt.savefig(base+".pdf")
        plt.savefig(base+".png", dpi=600)
        plt.close()

    corr_df = pd.DataFrame(corr_rows)
    reg_df  = pd.DataFrame(reg_rows)

    corr_df.to_csv(os.path.join(out_dir,"correlation_results.csv"),index=False)
    reg_df.to_csv(os.path.join(out_dir,"regression_results.csv"),index=False)

    # heatmap
    if not corr_df.empty:
        heat = corr_df.copy()
        heat["abs"] = heat["Pearson_r"].abs()
        heat = heat.sort_values("abs",ascending=False)
        pivot = heat.pivot(index="Retraction",columns="Global",values="Pearson_r")

        plt.figure(figsize=(10,12))
        sns.heatmap(pivot,annot=True,cmap="coolwarm",center=0)
        plt.title(f"Correlation Structure – {gcol}")
        plt.tight_layout()

        base = os.path.join(out_dir,"heatmap")
        plt.savefig(base+".svg")
        plt.savefig(base+".pdf")
        plt.savefig(base+".png", dpi=600)
        plt.close()

    with open(os.path.join(out_dir,"analysis_report.txt"),"w",encoding="utf-8") as f:
        f.write(f"ANALYSIS REPORT: {gcol}\n"+"="*90+"\n\n")
        f.write("CORRELATIONS\n")
        f.write(corr_df.to_string(index=False))
        f.write("\n\nREGRESSIONS\n")
        f.write(reg_df.to_string(index=False))

    with open(os.path.join(out_dir,"interpretive_report.txt"),"w",encoding="utf-8") as f:
        f.write(f"INTERPRETIVE SUMMARY: {gcol}\n"+"="*90+"\n\n")
        for line in desc_lines:
            f.write(line+"\n")

def main():
    print("\n--- Single Index Analysis ---\n")
    inp = gui_file("Select Master Dataset CSV")
    if not inp: return
    out = gui_folder("Select Output Folder")
    if not out: return

    df = pd.read_csv(inp)

    for g in GLOBAL_INDEXES:
        analyze(df,g,out)

    messagebox.showinfo("Done","Single Index Analysis Completed.")

if __name__=="__main__":
    main()
