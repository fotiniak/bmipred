#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from statsmodels.nonparametric.smoothers_lowess import lowess

SEX_PALETTE = {0: "#9fb3ef", 1: "#f2a9bf"}  # male=0, female=1
HIST_BINS, FIGSIZE = 40, (7, 7)

MALE_FILL   = "#9fb3ef"
FEMALE_FILL = "#f2a9bf"
MALE_LINE   = "#8da2e5"
FEMALE_LINE = "#e4a9b7"

# helper
def compute_pseudo_stats(series: pd.Series) -> dict:
    # Compute pseudo min, median, max from 5 values (handles datetime).
    s = series.dropna()
    if s.empty:
        return {"pseudo_min": np.nan, "pseudo_median": np.nan, "pseudo_max": np.nan}
    is_datetime = np.issubdtype(s.dtype, np.datetime64)
    s_numeric = s.astype("int64").to_numpy() if is_datetime else s.to_numpy()
    s_numeric.sort()
    n = len(s_numeric)
    if n < 5:
        return {"pseudo_min": np.nan, "pseudo_median": np.nan, "pseudo_max": np.nan}
    pseudo_min = np.mean(s_numeric[:5])
    mid = n // 2
    pseudo_median = np.mean(s_numeric[max(0, mid - 2):min(n, mid + 3)])
    pseudo_max = np.mean(s_numeric[-5:])
    if is_datetime:
        unit = str(s.dtype).split("[")[-1].split("]")[0] if "[" in str(s.dtype) else "ns"
        return {
            "pseudo_min": pd.to_datetime(pseudo_min, unit=unit),
            "pseudo_median": pd.to_datetime(pseudo_median, unit=unit),
            "pseudo_max": pd.to_datetime(pseudo_max, unit=unit),
        }
    return {"pseudo_min": pseudo_min, "pseudo_median": pseudo_median, "pseudo_max": pseudo_max}


def plot_histogram_with_medians(df: pd.DataFrame, column: str, title: str, x_label: str, save_path: str) -> None:
    # Plot histogram + kde stratified by sex, with pseudo medians.
    plt.figure(figsize=FIGSIZE)
    sns.histplot(data=df, x=column, hue="Sex", multiple="layer", bins=HIST_BINS, kde=True, palette=SEX_PALETTE)
    for sex, group in df.groupby("Sex"):
        stats = compute_pseudo_stats(group[column])
        if pd.isna(stats["pseudo_median"]):
            continue
        val = stats["pseudo_median"]
        color, label = SEX_PALETTE.get(sex, "black"), ("Male" if sex == 0 else "Female")
        formatted = val.strftime("%Y-%m-%d") if isinstance(val, pd.Timestamp) else f"{val:.2f}"
        plt.axvline(val, color=color, ls="--", lw=1.5, label=f"{label} Pseudo Median: {formatted}")
    plt.title(title, fontsize=22)
    plt.xlabel(x_label, fontsize=22)
    plt.ylabel("Number of Individuals", fontsize=22)
    plt.tick_params(axis="both", labelsize=16)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def aggregate_in_groups_of_k(values: np.ndarray, k: int = 5) -> np.ndarray:
    # GDPR-safe aggregation: sort values, split into groups of k, return per-group median.
    values = np.sort(np.asarray(values, dtype=float))
    values = values[~np.isnan(values)]
    if len(values) < k:
        return np.array([])
    grouped: list[float] = []
    for start in range(0, len(values), k):
        group = values[start:start + k]
        if len(group) < k:
            if grouped:
                prev = max(0, start - k)
                grouped[-1] = float(np.median(values[prev:]))
            break
        grouped.append(float(np.median(group)))
    return np.array(grouped)


def pseudo_median_from_values(values: np.ndarray, k: int = 5) -> float:
    # Compute GDPR-safe pseudo-median via aggregate_in_groups_of_k.
    grouped = aggregate_in_groups_of_k(values, k=k)
    return float(np.median(grouped)) if len(grouped) > 0 else np.nan


def plot_histogram_two_cohorts_by_sex(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    value_col: str,
    sex_col: str = "Sex",
    label1: str = "Cohort 1",
    label2: str = "Cohort 2",
    xlabel: str = "",
    ylabel: str = "Number of Individuals",
    out_dir: str | None = None,
    filename1: str = "hist_cohort1.png",
    filename2: str = "hist_cohort2.png",
    dpi: int = 600,
) -> None:
    # Plot side-by-side histograms (Male/Female) with KDE and pseudo-median lines.
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[sex_col] = out[sex_col].astype("string").str.strip()
        out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
        out = out.dropna(subset=[value_col, sex_col])
        return out[out[sex_col].isin(["Male", "Female"])].copy()

    p1, p2 = _prep(df1), _prep(df2)
    all_vals = pd.concat([p1[value_col], p2[value_col]])
    bins = np.arange(np.floor(all_vals.min()), np.ceil(all_vals.max()) + 1, 1)

    datasets = [(p1, label1, filename1), (p2, label2, filename2)]

    for df_plot, title, fname in datasets:
        male   = df_plot.loc[df_plot[sex_col] == "Male",   value_col].dropna().to_numpy()
        female = df_plot.loc[df_plot[sex_col] == "Female", value_col].dropna().to_numpy()

        male_pm   = pseudo_median_from_values(male)
        female_pm = pseudo_median_from_values(female)

        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.set_box_aspect(1)
        ax.set_facecolor("#f0f0f0")

        ax.hist(male,   bins=bins, color=MALE_FILL,   alpha=0.45, edgecolor="black", linewidth=0.8)
        ax.hist(female, bins=bins, color=FEMALE_FILL, alpha=0.45, edgecolor="black", linewidth=0.8)

        clip = (bins.min(), bins.max())
        if len(male)   > 1:
            sns.kdeplot(x=male,   ax=ax, color=MALE_LINE,   linewidth=1.2, fill=False, bw_adjust=0.9, clip=clip)
        if len(female) > 1:
            sns.kdeplot(x=female, ax=ax, color=FEMALE_LINE, linewidth=1.2, fill=False, bw_adjust=0.9, clip=clip)

        if pd.notna(male_pm):
            ax.axvline(male_pm,   color=MALE_LINE,   linestyle="--", linewidth=1.0)
        if pd.notna(female_pm):
            ax.axvline(female_pm, color=FEMALE_LINE, linestyle="--", linewidth=1.0)

        total_n  = len(df_plot)
        male_n   = (df_plot[sex_col] == "Male").sum()
        female_n = (df_plot[sex_col] == "Female").sum()

        ax.set_title(f"{title}\nn = {total_n:,}  (♂{male_n:,}, ♀{female_n:,})", fontsize=11, pad=10)
        ax.set_xlabel(xlabel or value_col, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)

        legend_handles = [
            Patch(facecolor=MALE_FILL,   edgecolor="none", alpha=0.7,
                  label=f"Male Pseudo Median: {male_pm:.2f}"   if pd.notna(male_pm)   else "Male Pseudo Median: NA"),
            Patch(facecolor=FEMALE_FILL, edgecolor="none", alpha=0.7,
                  label=f"Female Pseudo Median: {female_pm:.2f}" if pd.notna(female_pm) else "Female Pseudo Median: NA"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=True, fancybox=False, edgecolor="0.7", fontsize=8)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color("0.6")

        fig.subplots_adjust(left=0.14, right=0.96, bottom=0.14, top=0.90)
        if out_dir:
            import os
            fig.savefig(os.path.join(out_dir, fname), dpi=dpi, facecolor="white")
        plt.close(fig)


def plot_bmi_over_age(df: pd.DataFrame, save_path: str) -> None:
    # Plot BMI across age with rolling average + LOWESS smoothing.
    avg = df.groupby("healthAssesment_age")["BodyMassIndex_recalc"].mean().reset_index()
    avg["BMI_smooth"] = avg["BodyMassIndex_recalc"].rolling(6, min_periods=1).mean()
    lowess_fit = lowess(avg["BodyMassIndex_recalc"], avg["healthAssesment_age"], frac=0.1)
    plt.figure(figsize=FIGSIZE)
    plt.plot(avg["healthAssesment_age"], avg["BMI_smooth"], color="blue", lw=2, label="Rolling Avg Trend")
    plt.plot(lowess_fit[:, 0], lowess_fit[:, 1], color="red", lw=2, label="LOWESS Trend")
    plt.title("BMI across Age", fontsize=20)
    plt.xlabel("Age (years)", fontsize=22)
    plt.ylabel("BMI (kg/m²)", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(title="Trendline", fontsize=10, title_fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
