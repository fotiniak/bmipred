#!/usr/bin/env python3
# src/bmipred/analysis/olanzapine_bmi_boxplot.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove outliers from df[column] using the IQR rule."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)]


def prepare_groups(
    on_df: pd.DataFrame,
    off_df: pd.DataFrame,
    value_col: str = "BodyMassIndex_recalc",
    label_col: str = "Olanzapine",
    on_label=True,
    off_label=False,
) -> tuple[pd.DataFrame, float, float]:
    """
    - Adds boolean label_col for on/off groups
    - Removes outliers per group using IQR on value_col
    - Returns concatenated df + Mann-Whitney U p-value and statistic
    """
    # assign labels
    on_df = on_df.copy()
    off_df = off_df.copy()
    on_df[label_col] = on_label
    off_df[label_col] = off_label

    # keep minimal columns for the test/plot
    on_min = on_df[[value_col, label_col]].dropna(subset=[value_col])
    off_min = off_df[[value_col, label_col]].dropna(subset=[value_col])

    # remove outliers separately in each group
    on_clean = remove_outliers_iqr(on_min, value_col)
    off_clean = remove_outliers_iqr(off_min, value_col)

    # mann–whitney (two-sided)
    u_stat, p_value = stats.mannwhitneyu(
        on_clean[value_col],
        off_clean[value_col],
        alternative="two-sided",
    )

    combined = pd.concat([on_clean, off_clean], ignore_index=True)
    return combined, u_stat, p_value


def plot_box_q2_to_q3(
    df_combined: pd.DataFrame,
    *,
    value_col: str = "BodyMassIndex_recalc",
    label_col: str = "Olanzapine",
    figsize: tuple = (2, 4),
    palette=("tab:blue", "tab:orange"),
    title: str = "BMI (Median to 75th Percentile)",
    xlabel: str = "Olanzapine Treatment",
    ylabel: str = "BMI (kg/m²)",
    annotate_text: str = "Mann-Whitney U-test *** ",
    out_path: str | None = None,
    dpi: int = 600,
):
    """Minimal boxplot (median to Q3) with zoomed y-limits and fixed annotation text."""
    plt.figure(figsize=figsize)

    ax = sns.boxplot(
        data=df_combined,
        x=label_col,
        y=value_col,
        whis=[50, 75],# from median to 75th percentile
        showfliers=False,
        width=0.5,
        palette=list(palette),
    )

    # zoom y-axis using group medians and Q3s
    grouped = df_combined.groupby(label_col)[value_col]
    medians = grouped.quantile(0.50)
    q3s = grouped.quantile(0.75)
    min_median = medians.min()
    max_q3 = q3s.max()
    margin = (max_q3 - min_median) * 0.2 if (max_q3 > min_median) else 1.0
    ax.set_ylim(min_median - margin, max_q3 + margin)

    y_pos = max_q3 + margin * 0.5
    ax.text(0.5, y_pos, annotate_text, ha="center", va="bottom", fontsize=6)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=6)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=dpi)
    plt.show()


def plot_box_full_range(
    df_combined: pd.DataFrame,
    *,
    value_col: str = "BodyMassIndex_recalc",
    label_col: str = "Olanzapine",
    figsize: tuple = (2, 4),
    palette=("tab:blue", "tab:orange"),
    title: str = "Boxplot of BMI by olanzapine status*",
    xlabel: str = "Olanzapine Treatment",
    ylabel: str = "BMI (kg/m²)",
    annotate_text: str = "Mann-Whitney U-test ***",
    out_path: str | None = None,
    dpi: int = 600,
):
    """Boxplot spanning min–max of the cleaned data (no outliers), matching your second script."""
    plt.figure(figsize=figsize)

    ax = sns.boxplot(
        data=df_combined,
        x=label_col,
        y=value_col,
        whis=[0, 100],# min to max of cleaned data
        showfliers=False,
        width=0.5,
        palette=list(palette),
    )

    grouped = df_combined.groupby(label_col)[value_col]
    mins = grouped.min()
    maxs = grouped.max()
    margin = (maxs.max() - mins.min()) * 0.2 if (maxs.max() > mins.min()) else 1.0
    ax.set_ylim(mins.min() - margin, maxs.max() + margin)

    y_pos = maxs.max() + margin * 0.5
    ax.text(0.5, y_pos, annotate_text, ha="center", va="bottom", fontsize=6)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=6)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=dpi)
    plt.show()
