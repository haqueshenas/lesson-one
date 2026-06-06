# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 3: Political Economy of Global Scientific Correction Systems
#
# Analysis:
# Structural Econometric Modeling
# GDP–R&D Interaction Model of Retraction Cost
#
# Description:
# Estimates a structural econometric model linking national economic
# capacity, research investment intensity, and scientific correction
# costs through a centered interaction specification.
#
# The model evaluates whether the relationship between economic scale
# (GDP) and retraction-associated economic costs varies as a function
# of national R&D investment intensity.
#
# Analytical Framework:
# - Ordinary Least Squares (OLS)
# - HC3 Heteroskedasticity-Robust Standard Errors
# - Mean-Centered Interaction Modeling
# - Conditional Elasticity Estimation
# - Variance Inflation Factor (VIF) Assessment
# - Breusch–Pagan Heteroskedasticity Testing
# - Influence Diagnostics (Cook’s Distance)
# - Residual Diagnostics and Normality Assessment
#
# Dependent Variable:
# - Log(Retraction Economic Cost)
#
# Independent Variables:
# - Log(GDP)
# - Net R&D Intensity
# - GDP × Net R&D Interaction
#
# Research Objective:
# To quantify how national economic scale and research investment
# jointly influence the economic burden associated with scientific
# correction systems, and to evaluate whether R&D intensity moderates
# the elasticity of retraction costs with respect to GDP.
#
# Model Specification:
#
# log(Retraction Cost) =
# β0 + β1·log(GDP)
#    + β2·Net_R&D
#    + β3·[log(GDP) × Net_R&D]
#    + ε
#
# Outputs:
# - Full OLS Regression Summary
# - Publication-Ready Coefficient Table
# - HC3 Robust Inference Results
# - VIF Diagnostics
# - Breusch–Pagan Test
# - Residuals vs Fitted Plot
# - Q–Q Residual Diagnostics
# - Cook’s Distance Analysis
# - Conditional Elasticity Interpretation
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
import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics.gofplots import qqplot
from scipy import stats

# ============================================================
# 1. GUI
# ============================================================

root = tk.Tk()
root.withdraw()

messagebox.showinfo("Select CSV File", "Select the Enhanced Economic CSV file.")
input_file = filedialog.askopenfilename(
    title="Select CSV File",
    filetypes=[("CSV Files", "*.csv")]
)

if not input_file:
    raise Exception("No input file selected.")

messagebox.showinfo("Select Output Folder", "Select output folder for results.")
output_folder = filedialog.askdirectory(title="Select Output Folder")

if not output_folder:
    raise Exception("No output folder selected.")

print("Input file:", input_file)
print("Output folder:", output_folder)

# ============================================================
# 2. Load Data
# ============================================================

df = pd.read_csv(input_file)
print("Rows loaded:", len(df))

required_cols = [
    "GDP_BillionUSD",
    "Retraction_Cost_BillionUSD",
    "Net_R&D_%"
]

for col in required_cols:
    if col not in df.columns:
        raise Exception(f"Missing required column: {col}")

# ============================================================
# 3. Data Preparation
# ============================================================

df = df.dropna(subset=required_cols)

df["log_GDP"] = np.log(df["GDP_BillionUSD"])
df["log_RetractionCost"] = np.log(df["Retraction_Cost_BillionUSD"])

# 🔴 Centering (reduces multicollinearity)
df["c_log_GDP"] = df["log_GDP"] - df["log_GDP"].mean()
df["c_Net_RD"] = df["Net_R&D_%"] - df["Net_R&D_%"].mean()

# 🔵 Interaction term (centered)
df["Interaction"] = df["c_log_GDP"] * df["c_Net_RD"]

X = df[["c_log_GDP", "c_Net_RD", "Interaction"]]
X = sm.add_constant(X)

y = df["log_RetractionCost"]

# ============================================================
# 4. OLS with Robust SE
# ============================================================

model = sm.OLS(y, X).fit(cov_type="HC3")

print(model.summary())

with open(os.path.join(output_folder, "Model_Full_Summary.txt"),
          "w", encoding="utf-8") as f:
    f.write(model.summary().as_text())

# ============================================================
# 5. Publication-Ready Table
# ============================================================

coef_table = model.summary2().tables[1]
coef_table["Significance"] = coef_table["P>|z|"].apply(
    lambda p: "***" if p < 0.01 else
              "**" if p < 0.05 else
              "*" if p < 0.10 else ""
)

coef_table.to_csv(
    os.path.join(output_folder, "Table1_GDP_RD_Interaction_Model.csv"),
    encoding="utf-8"
)

# Title and notes (ASCII safe)
title = "Table 1. Structural Econometric Model of Retraction Cost\n"
notes = (
    "\nNotes: Dependent variable is log(Retraction Cost). "
    "Variables are mean-centered. "
    "HC3 heteroskedasticity-robust standard errors reported. "
    "Interaction captures moderating effect of R&D intensity. "
    "Significance levels: *** p<0.01, ** p<0.05, * p<0.10."
)

with open(os.path.join(output_folder, "Table1_Title_and_Notes.txt"),
          "w", encoding="utf-8") as f:
    f.write(title)
    f.write(notes)

# ============================================================
# 6. VIF
# ============================================================

vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

vif_data.to_csv(
    os.path.join(output_folder, "VIF_Table.csv"),
    index=False,
    encoding="utf-8"
)

# ============================================================
# 7. Breusch-Pagan
# ============================================================

bp_test = het_breuschpagan(model.resid, model.model.exog)
bp_labels = ["LM Statistic", "LM p-value", "F Statistic", "F p-value"]

with open(os.path.join(output_folder, "Breusch_Pagan_Test.txt"),
          "w", encoding="utf-8") as f:
    for label, value in zip(bp_labels, bp_test):
        f.write(f"{label}: {value}\n")

# ============================================================
# 8. Diagnostic Plots
# ============================================================

plt.rcParams.update({
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "font.size": 12
})

# Residuals vs Fitted
plt.figure()
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(0)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Figure 1. Residuals vs Fitted (Centered Interaction Model)")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Figure1_Residuals_vs_Fitted.png"))
plt.savefig(os.path.join(output_folder, "Figure1_Residuals_vs_Fitted.pdf"))
plt.close()

# QQ Plot
plt.figure()
qqplot(model.resid, line="s")
plt.title("Figure 2. Q-Q Plot of Residuals")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Figure2_QQ_Plot.png"))
plt.savefig(os.path.join(output_folder, "Figure2_QQ_Plot.pdf"))
plt.close()

# Cook's Distance
influence = model.get_influence()
cooks = influence.cooks_distance[0]

plt.figure()
plt.stem(cooks)
plt.title("Figure 3. Cook's Distance")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Figure3_Cooks_Distance.png"))
plt.savefig(os.path.join(output_folder, "Figure3_Cooks_Distance.pdf"))
plt.close()

# ============================================================
# 9. Conditional Elasticity Interpretation (ASCII Safe)
# ============================================================

b1 = model.params["c_log_GDP"]
b3 = model.params["Interaction"]

interpretation = (
    "Baseline elasticity (at mean R&D): " + str(b1) + "\n"
    "Interaction coefficient: " + str(b3) + "\n\n"
    "Conditional elasticity formula:\n"
    "d log(Cost) / d log(GDP) = b1 + b3 * (Net_RD_centered)\n"
)

with open(os.path.join(output_folder, "Elasticity_Interpretation.txt"),
          "w", encoding="utf-8") as f:
    f.write(interpretation)

print("\nCentered Interaction Model Completed Successfully.")
print("All outputs saved to:", output_folder)
