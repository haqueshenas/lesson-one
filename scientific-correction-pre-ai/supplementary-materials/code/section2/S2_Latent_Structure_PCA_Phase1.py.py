# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 2: Macro-Structural Organization of Scientific Correction Systems
#
# Analysis:
# Latent_Structure_PCA
#
# Phase 1 – Data Ontology & Structural System Mapping
#
# Description:
# Generates an ontology of observed indicators, maps them to system layers
# (Development and Scientific Correction), defines roles, and builds
# a basic data profile including types, missingness, and ranges.
#
# Author:
# Abbas Haghshenas, Independent Researcher
#
# Project:
# Lesson One – Multi-Phase Interblock Analysis
# https://haqueshenas.github.io/lesson-one
#
# AI-Assisted Code:
# Generated with ChatGPT GPT-5.5; iterative refinement by author
#
# License:
# MIT License
#
# Copyright (c) 2026 Abbas Haghshenas
# =============================================================================

import pandas as pd
import json
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

# -----------------------------
# System Definitions
# -----------------------------

DEVELOPMENT_INDICES = [
    "GDP",
    "CPI",
    "RDE",
    "GEE",
    "HDI",
    "WGI",
    "AFI",
    "GII"
]

RETRACTION_INDEXES = [
    "RR_Total_%",
    "Total_Mean_RD_Years",
    "Collaboration_Share_%",
    "NSlope13_Total"
]

# -----------------------------
# Layer Mapping
# -----------------------------

LAYER_MAP = {
    "Macro-Development Layer": DEVELOPMENT_INDICES,
    "Scientific Correction Layer": RETRACTION_INDEXES
}

ROLE_MAP = {
    "Developmental Structural Indicator": DEVELOPMENT_INDICES,
    "Scientific Correction Indicator": RETRACTION_INDEXES
}

# Orthogonal design → no recursive dependencies
DEPENDENCIES = {}

# -----------------------------
# Core Functions
# -----------------------------

def detect_layer(col):
    for layer, items in LAYER_MAP.items():
        if col in items:
            return layer
    return "Unmapped Layer"


def detect_role(col):
    for role, items in ROLE_MAP.items():
        if col in items:
            return role
    return "Unclassified"


def build_ontology(columns):
    ontology = {}

    for col in columns:
        ontology[col] = {
            "name": col,
            "layer": detect_layer(col),
            "role": detect_role(col),
            "dependencies": DEPENDENCIES.get(col, []),
            "type": "observed",
            "systemic": True,
            "orthogonal_design": col in RETRACTION_INDEXES
        }

    return ontology


def build_system_map():
    return {
        "Development_System": DEVELOPMENT_INDICES,
        "Scientific_Correction_System": RETRACTION_INDEXES,
        "Latent_Coupling": "CCA/PCA/Clustering Space"
    }


def build_data_profile(df):
    profile = []

    for col in df.columns:
        profile.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isna().sum()),
            "unique": int(df[col].nunique()),
            "min": float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None,
            "max": float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None
        })

    return pd.DataFrame(profile)

# -----------------------------
# GUI Application
# -----------------------------

def run_phase1():
    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo(
        "Route 1 SCoRe Phase 1",
        "Phase 1 — Data Ontology & Structural System Mapping"
    )

    input_file = filedialog.askopenfilename(
        title="Select Input CSV File",
        filetypes=[("CSV files", "*.csv")]
    )

    if not input_file:
        return

    output_dir = filedialog.askdirectory(title="Select Output Directory")

    if not output_dir:
        return

    try:
        df = pd.read_csv(input_file)

        ontology = build_ontology(df.columns)
        system_map = build_system_map()
        profile_df = build_data_profile(df)

        timestamp = datetime.now().isoformat()

        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "ontology.json"), "w", encoding="utf-8") as f:
            json.dump({"generated": timestamp, "ontology": ontology}, f, indent=4, ensure_ascii=False)

        with open(os.path.join(output_dir, "system_map.json"), "w", encoding="utf-8") as f:
            json.dump({"generated": timestamp, "system_map": system_map}, f, indent=4, ensure_ascii=False)

        profile_df.to_csv(os.path.join(output_dir, "data_profile.csv"), index=False)

        messagebox.showinfo("Success", "Phase 1 completed successfully.")

    except Exception as e:
        messagebox.showerror("Execution Error", str(e))


if __name__ == "__main__":
    run_phase1()