# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 2: Macro-Structural Organization of Scientific Correction Systems
#
# Analysis:
# AFI Components Analysis
#
# Phase:
# AFI Components Analysis – Phase 1
# Academic Freedom Component Extraction
#
# Description:
# Extracts the Academic Freedom Index (AFI) and its five constituent
# dimensions from the V-Dem v15 dataset, performs quality-control
# validation, generates longitudinal panel datasets and country-level
# aggregates, and prepares publication-ready inputs for downstream
# analyses of academic freedom and scientific correction dynamics.
#
# Data Source:
# Varieties of Democracy (V-Dem) Dataset v15
#
# Components:
# - Academic Freedom Index (AFI)
# - Freedom to Research and Teach
# - Freedom of Academic Exchange and Dissemination
# - Institutional Autonomy
# - Campus Integrity
# - Freedom of Academic and Cultural Expression
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

import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

START_YEAR = 2000
END_YEAR = 2024

AFI_COLUMNS = {
    "v2xca_academ": "AFI",
    "v2cafres": "Freedom_Research_Teach",
    "v2cafexch": "Freedom_Academic_Exchange",
    "v2cainsaut": "Institutional_Autonomy",
    "v2casurv": "Campus_Integrity",
    "v2clacfree": "Academic_Cultural_Expression"
}

ID_COLUMNS = {
    "country_name": "Country",
    "country_text_id": "ISO3",
    "year": "Year"
}

# ============================================================
# SAFE CSV LOADER
# ============================================================

def safe_read_csv(path):

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin1",
        "cp1252",
        "cp1256"
    ]

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"[INFO] Loaded with encoding: {enc}")
            return df

        except Exception:
            print(f"[WARN] Failed encoding: {enc}")

    raise ValueError("❌ Unable to read CSV file with known encodings.")


# ============================================================
# GUI HELPERS
# ============================================================

def select_file(title):

    root = tk.Tk()
    root.withdraw()

    return filedialog.askopenfilename(
        title=title,
        filetypes=[("CSV files", "*.csv")]
    )


def select_folder(title):

    root = tk.Tk()
    root.withdraw()

    return filedialog.askdirectory(title=title)


# ============================================================
# COLUMN CHECKER
# ============================================================

def validate_columns(df):

    required = list(ID_COLUMNS.keys()) + list(AFI_COLUMNS.keys())

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "\n❌ Missing required columns:\n"
            + "\n".join(missing)
        )

    print("[INFO] All required columns detected.")


# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

def process_vdem(vdem_path, countries_path, output_dir):

    print("\n====================================================")
    print(" LOADING DATA ")
    print("====================================================")

    vdem = safe_read_csv(vdem_path)
    countries = safe_read_csv(countries_path)

    print("\n====================================================")
    print(" VALIDATING STRUCTURE ")
    print("====================================================")

    validate_columns(vdem)

    # --------------------------------------------------------
    # Keep only required columns
    # --------------------------------------------------------

    keep_cols = (
        list(ID_COLUMNS.keys()) +
        list(AFI_COLUMNS.keys())
    )

    vdem = vdem[keep_cols].copy()

    # --------------------------------------------------------
    # Rename columns
    # --------------------------------------------------------

    rename_dict = {}
    rename_dict.update(ID_COLUMNS)
    rename_dict.update(AFI_COLUMNS)

    vdem.rename(columns=rename_dict, inplace=True)

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    vdem["Year"] = pd.to_numeric(vdem["Year"], errors="coerce")

    for col in AFI_COLUMNS.values():
        vdem[col] = pd.to_numeric(vdem[col], errors="coerce")

    # --------------------------------------------------------
    # Drop invalid rows
    # --------------------------------------------------------

    vdem = vdem.dropna(subset=["Year", "ISO3"])

    # --------------------------------------------------------
    # Year filtering
    # --------------------------------------------------------

    vdem = vdem[
        (vdem["Year"] >= START_YEAR) &
        (vdem["Year"] <= END_YEAR)
    ]

    print(f"[INFO] Years filtered: {START_YEAR}-{END_YEAR}")

    # --------------------------------------------------------
    # Country filtering
    # --------------------------------------------------------

    countries.columns = (
        countries.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if "iso3" not in countries.columns:
        raise ValueError(
            "❌ Countries file must contain column: iso3"
        )

    target_iso = (
        countries["iso3"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    target_iso = set(target_iso)

    # Remove Taiwan
    target_iso.discard("TWN")

    vdem["ISO3"] = vdem["ISO3"].astype(str).str.upper()

    vdem = vdem[vdem["ISO3"].isin(target_iso)]

    # Explicit Taiwan removal
    vdem = vdem[vdem["ISO3"] != "TWN"]

    print(f"[INFO] Countries retained: {len(vdem['ISO3'].unique())}")

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    dup_count = vdem.duplicated(
        subset=["Country", "ISO3", "Year"]
    ).sum()

    print(f"[INFO] Duplicate country-year rows: {dup_count}")

    if dup_count > 0:

        vdem = (
            vdem
            .drop_duplicates(
                subset=["Country", "ISO3", "Year"]
            )
            .copy()
        )

        print("[INFO] Duplicates removed.")

    # ========================================================
    # PANEL DATASET
    # ========================================================

    print("\n====================================================")
    print(" CREATING PANEL DATASET ")
    print("====================================================")

    panel_df = vdem.sort_values(
        ["Country", "Year"]
    ).reset_index(drop=True)

    # ========================================================
    # COUNTRY MEAN DATASET
    # ========================================================

    print("\n====================================================")
    print(" CREATING COUNTRY MEAN DATASET ")
    print("====================================================")

    mean_df = (
        panel_df
        .groupby(["Country", "ISO3"], as_index=False)
        [list(AFI_COLUMNS.values())]
        .mean()
    )

    # --------------------------------------------------------
    # Rename mean columns
    # --------------------------------------------------------

    mean_rename = {}

    for col in AFI_COLUMNS.values():
        mean_rename[col] = f"{col}_Mean_2000_2024"

    mean_df.rename(columns=mean_rename, inplace=True)

    # ========================================================
    # QC SUMMARY
    # ========================================================

    print("\n====================================================")
    print(" QUALITY CONTROL ")
    print("====================================================")

    qc_rows = []

    for col in AFI_COLUMNS.values():

        qc_rows.append({
            "Variable": col,
            "Missing_Values": panel_df[col].isna().sum(),
            "Mean": panel_df[col].mean(),
            "SD": panel_df[col].std(),
            "Min": panel_df[col].min(),
            "Max": panel_df[col].max()
        })

    qc_df = pd.DataFrame(qc_rows)

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print("\n====================================================")
    print(" SAVING OUTPUTS ")
    print("====================================================")

    os.makedirs(output_dir, exist_ok=True)

    # Panel dataset
    panel_path = os.path.join(
        output_dir,
        "AFI_Components_Panel_2000_2024.csv"
    )

    panel_df.to_csv(
        panel_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Mean dataset
    mean_path = os.path.join(
        output_dir,
        "AFI_Components_Mean_2000_2024.csv"
    )

    mean_df.to_csv(
        mean_path,
        index=False,
        encoding="utf-8-sig"
    )

    # QC dataset
    qc_path = os.path.join(
        output_dir,
        "AFI_Components_QC_Report.csv"
    )

    qc_df.to_csv(
        qc_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # TEXT REPORT
    # ========================================================

    report_path = os.path.join(
        output_dir,
        "AFI_Extraction_Report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:

        f.write("====================================================\n")
        f.write(" STEP07 – AFI COMPONENT EXTRACTION\n")
        f.write("====================================================\n\n")

        f.write(f"Execution time: {datetime.now()}\n\n")

        f.write("SOURCE:\n")
        f.write("V-Dem Core Dataset v15\n\n")

        f.write("YEARS:\n")
        f.write(f"{START_YEAR}-{END_YEAR}\n\n")

        f.write("AFI COMPONENTS INCLUDED:\n")

        for k, v in AFI_COLUMNS.items():
            f.write(f"- {k} --> {v}\n")

        f.write("\n")

        f.write("PROCESSING PIPELINE:\n")
        f.write("- Encoding-safe CSV loading\n")
        f.write("- Column validation\n")
        f.write("- Numeric conversion\n")
        f.write("- Year filtering\n")
        f.write("- 57-country filtering\n")
        f.write("- Taiwan exclusion\n")
        f.write("- Duplicate removal\n")
        f.write("- Panel dataset generation\n")
        f.write("- Country-level mean aggregation\n")
        f.write("- QC diagnostics\n\n")

        f.write("OUTPUT FILES:\n")
        f.write(f"- {panel_path}\n")
        f.write(f"- {mean_path}\n")
        f.write(f"- {qc_path}\n\n")

        f.write("COUNTRY COUNT:\n")
        f.write(f"{mean_df.shape[0]}\n")

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n====================================================")
    print(" COMPLETED SUCCESSFULLY ")
    print("====================================================")

    print(f"\n[SAVED] PANEL DATASET:\n{panel_path}")
    print(f"\n[SAVED] COUNTRY MEANS:\n{mean_path}")
    print(f"\n[SAVED] QC REPORT:\n{qc_path}")
    print(f"\n[SAVED] TEXT REPORT:\n{report_path}")

    messagebox.showinfo(
        "Completed",
        "AFI component extraction completed successfully."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("====================================================")
    print(" STEP07 – AFI COMPONENT EXTRACTION ")
    print(" V-Dem v15 | Research-Grade Pipeline ")
    print("====================================================")

    # --------------------------------------------------------
    # Select V-Dem file
    # --------------------------------------------------------

    vdem_path = select_file(
        "Select V-Dem v15 CSV File"
    )

    if not vdem_path:
        raise SystemExit("❌ No V-Dem file selected.")

    # --------------------------------------------------------
    # Select country file
    # --------------------------------------------------------

    countries_path = select_file(
        "Select Countries_57 CSV File"
    )

    if not countries_path:
        raise SystemExit("❌ No country file selected.")

    # --------------------------------------------------------
    # Select output directory
    # --------------------------------------------------------

    output_dir = select_folder(
        "Select Output Directory"
    )

    if not output_dir:
        raise SystemExit("❌ No output directory selected.")

    # --------------------------------------------------------
    # Run pipeline
    # --------------------------------------------------------

    process_vdem(
        vdem_path,
        countries_path,
        output_dir
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()