#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

SEX_PALETTE = {0: "#9fb3ef", 1: "#f2a9bf"}  # male=0, female=1
HIST_BINS, FIGSIZE = 40, (7, 7)

# helper
def compute_pseudo_stats(series: pd.Series) -> dict:
    """compute pseudo min, median, max from 5 values (handles datetime)."""
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
    """plot histogram + kde stratified by sex, with pseudo medians."""
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


def plot_bmi_over_age(df: pd.DataFrame, save_path: str) -> None:
    """plot BMI across age with rolling average + LOWESS smoothing."""
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
