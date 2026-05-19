#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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


def filter_sparse_columns_pairwise(
    df: pd.DataFrame,
    min_pairwise: int = 1000,
    protected_cols: set[str] | None = None,
) -> pd.DataFrame:
    """Drop columns where any pair has fewer than min_pairwise jointly non-missing rows.

    Keeps columns listed in protected_cols (e.g. the target variable) regardless.
    """
    if protected_cols is None:
        protected_cols = {"target"}
    bad: set[str] = set()
    for c1, c2 in combinations(df.columns, 2):
        if df[[c1, c2]].dropna().shape[0] < min_pairwise:
            sparser = c1 if df[c1].notna().sum() <= df[c2].notna().sum() else c2
            if sparser not in protected_cols:
                bad.add(sparser)
    return df.drop(columns=list(bad))


def compute_pairwise_spearman(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Compute pairwise Spearman r and p-values, handling NaN per pair.
    cols = list(df.columns)
    n = len(cols)
    corr_array = np.eye(n)
    p_array    = np.ones((n, n))

    for i, j in combinations(range(n), 2):
        pair = df.iloc[:, [i, j]].dropna()
        if len(pair) < 3:
            corr_array[i, j] = corr_array[j, i] = np.nan
            p_array[i, j]    = p_array[j, i]    = np.nan
        else:
            r, p = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
            corr_array[i, j] = corr_array[j, i] = r
            p_array[i, j]    = p_array[j, i]    = p

    corr_df     = pd.DataFrame(corr_array, index=cols, columns=cols)
    p_values_df = pd.DataFrame(p_array,    index=cols, columns=cols)
    return corr_df, p_values_df


def plot_lower_triangle_heatmap(
    corr_df: pd.DataFrame,
    p_values_df: pd.DataFrame,
    title: str = "Spearman Correlation Heatmap",
    cbar_label: str = "Spearman\ncorrelation coefficient (ρ)",
    target_col: str | None = "target",
    figsize: tuple = (20, 17),
    out_path: str | None = None,
    dpi: int = 500,
) -> None:
    # Lower-triangle clustered heatmap with correlation labels and target highlighting.
    # Uses seaborn clustermap for row/column ordering, then renders a plain heatmap
    # of the lower triangle with significance-masked cell labels.
    # Cluster for ordering only
    cluster_grid = sns.clustermap(corr_df.fillna(0), cmap="bwr", center=0, figsize=(20, 18))
    row_order = cluster_grid.dendrogram_row.reordered_ind
    col_order = cluster_grid.dendrogram_col.reordered_ind
    plt.close()

    corr_reordered = corr_df.iloc[row_order, col_order]
    p_reordered    = p_values_df.iloc[row_order, col_order]
    # Mask upper triangle
    mask = np.triu(np.ones_like(corr_reordered, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_reordered.fillna(0), mask=mask, cmap="bwr", center=0,
        linewidths=0.1, linecolor="lightgrey",
        xticklabels=True, yticklabels=True, ax=ax,
        cbar=False,
    )

    # White-out upper triangle cells
    n_rows, n_cols = corr_reordered.shape
    for i in range(n_rows):
        for j in range(i + 1, n_cols):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color="white", zorder=2))

    # Inline colorbar in top-right empty space
    cbar_ax = inset_axes(
        ax, width="15%", height="2%", loc="upper right",
        bbox_to_anchor=(-0.05, -0.05, 1, 1), bbox_transform=ax.transAxes,
    )
    fig.colorbar(ax.collections[0], cax=cbar_ax, orientation="horizontal")
    cbar_ax.set_title(cbar_label, fontsize=11, pad=6)

    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    # Cell labels for significant pairs
    for i in range(n_rows):
        for j in range(n_cols):
            if mask[i, j]:
                continue
            p_val = p_reordered.iloc[i, j]
            r_val = corr_reordered.iloc[i, j]
            if pd.isna(p_val) or pd.isna(r_val) or p_val >= 0.05:
                continue
            text_color = "white" if abs(r_val) > 0.6 else "black"
            ax.text(j + 0.5, i + 0.5, f"{r_val:.2f}",
                    ha="center", va="center", color=text_color, fontsize=6)

    # Highlight target row/column
    if target_col and target_col in list(corr_reordered.index):
        labels = list(corr_reordered.index)
        idx = labels.index(target_col)
        n = len(labels)
        ax.add_patch(plt.Rectangle((0, idx), n, 1, fill=False, edgecolor="black", lw=2.5, clip_on=False))
        ax.add_patch(plt.Rectangle((idx, 0), 1, n, fill=False, edgecolor="black", lw=2.5, clip_on=False))

    ax.set_title(f"{title}\n", fontsize=18, pad=15)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


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
