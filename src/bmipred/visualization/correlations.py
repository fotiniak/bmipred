#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def get_filtered_df(df: pd.DataFrame, specific_columns: list = None) -> pd.DataFrame:
    # Select numeric columns, drop unwanted prefixes, and clean inf/nan values
    df = df.copy()
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].astype(int, errors="ignore")
    if "target" in df.columns:
        df["target"] = df["target"].astype(int, errors="ignore")

    if specific_columns is None:
        specific_columns = ["DailyDosage_weighted_mean"]

    prefixes = [
        "PatientDurableKey",
        "RateOfBMIChange",
        "StrengthNumeric_daily",
        "bmi_diff_from_next",
        "BodySurfaceArea",
        "bmi_diff_from_first_measurement",
        "NumberOfDaysOffMedication",
    ]

    cols_to_exclude = [
        col
        for col in df.columns
        if any(col.startswith(pref) for pref in prefixes) or col in specific_columns
    ]

    numeric_cols = df.select_dtypes(include=["number"]).columns
    selected_cols = [col for col in numeric_cols if col not in cols_to_exclude]

    # clean invalid values
    df = df[selected_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def plot_top_correlations(df: pd.DataFrame, df_name: str, out_dir: str) -> None:
    # Plot top positive/negative spearman correlations with target
    corr_target = df.corr(method="spearman")["target"].drop("target", errors="ignore")
    top_positive = corr_target.sort_values(ascending=False).head(25)
    top_negative = corr_target.sort_values().head(25)

    results = pd.concat([top_positive, top_negative], axis=0).reset_index()
    results.columns = ["Feature", "Correlation"]
    results = results.sort_values(by="Correlation")

    plt.figure(figsize=(5, 10))
    sns.barplot(
        x="Correlation",
        y="Feature",
        data=results,
        palette="bwr",
        legend=False,
    )
    plt.title("Top Most Correlated Features\nwith >5% Weight Increase", fontsize=14)
    plt.xlabel("Spearman Correlation Coefficient", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()

    out_file = os.path.join(out_dir, f"{df_name}_top_correlations.png")
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved barplot -> {out_file}")


def plot_clustered_heatmap(df: pd.DataFrame, df_name: str, out_dir: str, rename_dict: dict = None) -> None:
    # Plot clustered heatmap of spearman correlations
    corr = df.corr(method="spearman")
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0)

    if rename_dict:
        corr = corr.rename(columns=rename_dict, index=rename_dict)

    grid = sns.clustermap(corr, cmap="bwr", center=0, figsize=(20, 18))
    grid.ax_heatmap.set_title("Spearman correlations plot", pad=20, fontsize=10)

    out_file = os.path.join(out_dir, f"{df_name}_heatmap.png")
    grid.fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(grid.figure)
    print(f"[INFO] Saved heatmap -> {out_file}")


def plot_clustered_heatmap_with_pvalues(
    df: pd.DataFrame, df_name: str, out_dir: str, rename_dict: dict = None
) -> None:
    # Plot clustered heatmap of spearman correlations with significance stars
    corr_array, p_array = spearmanr(df, nan_policy="omit")
    corr_df = pd.DataFrame(corr_array, index=df.columns, columns=df.columns)
    p_values_df = pd.DataFrame(p_array, index=df.columns, columns=df.columns)

    corr_df = corr_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    p_values_df = p_values_df.replace([np.inf, -np.inf], np.nan).fillna(1)

    if rename_dict:
        corr_df = corr_df.rename(columns=rename_dict, index=rename_dict)
        p_values_df = p_values_df.rename(columns=rename_dict, index=rename_dict)

    grid = sns.clustermap(corr_df, cmap="bwr", center=0, figsize=(20, 18))
    grid.cax.set_position([0.85, 0.85, 0.03, 0.1])
    grid.cax.tick_params(labelsize=10)

    reordered_corr = grid.data2d
    reordered_p = p_values_df.loc[reordered_corr.index, reordered_corr.columns]

    ax = grid.ax_heatmap
    for i in range(reordered_corr.shape[0]):
        for j in range(reordered_corr.shape[1]):
            p_val = reordered_p.iloc[i, j]
            if p_val < 0.001:
                marker = "***"
            elif p_val < 0.01:
                marker = "**"
            elif p_val < 0.05:
                marker = "*"
            else:
                marker = ""
            ax.text(j + 0.5, i + 0.5, marker, ha="center", va="center", fontsize=8)

    grid.ax_heatmap.set_title(
        "Spearman Correlations with Significance", pad=20, fontsize=10
    )

    out_file = os.path.join(out_dir, f"{df_name}_heatmap_pvalues.png")
    grid.figure.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(grid.figure)
    print(f"[INFO] Saved heatmap with p-values -> {out_file}")
