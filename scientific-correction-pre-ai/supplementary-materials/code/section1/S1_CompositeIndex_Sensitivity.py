# =============================================================================
# Scientific Correction in the Pre-AI Era: Lessons from Global Retraction Dynamics
#
# Section 1: Global Architecture and Dynamics of Scientific Correction
#
# Analysis: Country-Level Sensitivity Analysis of Composite Index
#
# Interactive sensitivity analysis of country rankings for a linear composite index
# of the form: Index = w × X + (1 − w) × Y. Users can vary weights systematically
# to assess ranking stability using Spearman correlation.
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
# This Python Streamlit application was generated with the assistance of ChatGPT
# (OpenAI; versions 5.3–5.5) under the direction, testing,
# validation, and iterative refinement of the author.
#
# License:
# MIT License
#
# Copyright (c) 2026 Abbas Haghshenas
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# ===============================
# Page configuration
# ===============================
st.set_page_config(
    page_title="Composite Index Sensitivity Analysis",
    layout="centered"
)

st.title("Composite Index Sensitivity Analysis")
st.markdown(
    """
    This tool performs a **sensitivity analysis of country rankings**
    for a **linear composite index** of the form:

    **Index = w × X + (1 − w) × Y**

    where:
    - **X** and **Y** are user-selected normalized indicators
    - **w** varies systematically to assess ranking stability
    """
)

# ===============================
# Upload data
# ===============================
uploaded_file = st.file_uploader(
    "Upload CSV file (country-level indicators)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    # ===============================
    # Column selection
    # ===============================
    st.subheader("Select columns")

    country_col = st.selectbox(
        "Country column",
        df.columns,
        help="Column containing country names"
    )

    x_col = st.selectbox(
        "Component X (e.g., NRR, NRR_Total_RR, etc.)",
        df.columns,
        help="First normalized component of the composite index"
    )

    y_col = st.selectbox(
        "Component Y (e.g., NRD, NSlope13, etc.)",
        df.columns,
        help="Second normalized component of the composite index"
    )

    # ===============================
    # Sensitivity settings
    # ===============================
    st.subheader("Sensitivity settings")

    weight_step = st.slider(
        "Weight step for component X (weight of Y = 1 − X)",
        min_value=0.05,
        max_value=0.25,
        value=0.10,
        step=0.05
    )

    base_weight = 0.5

    # ===============================
    # Run analysis
    # ===============================
    if st.button("Run Sensitivity Analysis"):

        results = []

        # ---------- Base ranking ----------
        df_base = df[[country_col, x_col, y_col]].dropna().copy()

        df_base["Index_base"] = (
            base_weight * df_base[x_col]
            + (1 - base_weight) * df_base[y_col]
        )

        df_base["Rank_base"] = df_base["Index_base"].rank(
            ascending=False,
            method="min"
        )

        # ---------- Sensitivity loop ----------
        for w in np.arange(0.1, 0.91, weight_step):
            df_temp = df_base.copy()

            df_temp["Index"] = (
                w * df_temp[x_col]
                + (1 - w) * df_temp[y_col]
            )

            df_temp["Rank"] = df_temp["Index"].rank(
                ascending=False,
                method="min"
            )

            rho, pval = spearmanr(
                df_base["Rank_base"],
                df_temp["Rank"]
            )

            results.append({
                "Weight_X": round(w, 2),
                "Weight_Y": round(1 - w, 2),
                "Spearman_rho_vs_0.5": rho,
                "p_value": pval
            })

        results_df = pd.DataFrame(results)

        # ===============================
        # Display results
        # ===============================
        st.subheader("Sensitivity results (ranking stability)")
        st.dataframe(results_df)

        # ===============================
        # Download results
        # ===============================
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="composite_index_sensitivity_results.csv",
            mime="text/csv"
        )

        st.success("Sensitivity analysis completed successfully.")
