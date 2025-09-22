#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

def plot_lowess_two(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    age_col: str,
    value_col: str,
    label1: str,
    label2: str,
    frac: float,
    figsize: tuple,
    save_path: str) -> None:
    # Overlay LOWESS trendlines from two cohorts for BMI vs Age.
    m1 = df1.groupby(age_col)[value_col].mean().reset_index()
    m2 = df2.groupby(age_col)[value_col].mean().reset_index()
    lo1 = lowess(m1[value_col], m1[age_col], frac=frac)
    lo2 = lowess(m2[value_col], m2[age_col], frac=frac)

    plt.figure(figsize=figsize)
    plt.plot(lo1[:, 0], lo1[:, 1], lw=2, label=label1)
    plt.plot(lo2[:, 0], lo2[:, 1], lw=2, label=label2)
    plt.xlabel("Age (years)", fontsize=12)
    plt.ylabel("BMI (kg/m²)", fontsize=12)
    plt.title("BMI Trend Across Age - Comparison", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
