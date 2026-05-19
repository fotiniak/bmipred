#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
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


# BMI slope analysis

def _fit_slope(sub_df: pd.DataFrame, x_col: str, y_col: str, min_points: int = 2) -> float:
    tmp = sub_df[[x_col, y_col]].dropna()
    if len(tmp) < min_points or tmp[x_col].nunique() < 2:
        return np.nan
    return float(np.polyfit(tmp[x_col].to_numpy(), tmp[y_col].to_numpy(), 1)[0])


def compute_patient_bmi_slopes(
    df: pd.DataFrame,
    pre_days: int = 365,
    post_days: int = 365,
    patient_col: str = "PatientDurableKey",
    start_col: str = "StartInstant",
    date_col: str = "CreateInstant",
    bmi_col: str = "BodyMassIndex_recalc",
) -> pd.DataFrame:
    # Compute per-patient OLS BMI slopes in a pre- and post-treatment window.
    df = df.copy()
    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    df[date_col]  = pd.to_datetime(df[date_col],  errors="coerce")
    df[bmi_col]   = pd.to_numeric(df[bmi_col],    errors="coerce")
    df = df.dropna(subset=[patient_col, start_col, date_col, bmi_col])
    df = df.sort_values([patient_col, date_col]).reset_index(drop=True)

    df["_time_rel_days"]  = (df[date_col] - df[start_col]).dt.days
    df["_time_rel_years"] = df["_time_rel_days"] / 365.25

    df = df[df["_time_rel_days"].between(-pre_days, post_days)].copy()

    # Baseline = last BMI before treatment start
    baseline = (
        df[df["_time_rel_days"] < 0]
        .sort_values([patient_col, "_time_rel_days"])
        .groupby(patient_col, as_index=False)
        .tail(1)
        [[patient_col, bmi_col]]
        .rename(columns={bmi_col: "baseline_bmi"})
    )

    rows: list[dict] = []
    for pid, g in df.groupby(patient_col):
        pre  = g[(g["_time_rel_days"] >= -pre_days)  & (g["_time_rel_days"] < 0)]
        post = g[(g["_time_rel_days"] >= 0)           & (g["_time_rel_days"] <= post_days)]

        pre_slope  = _fit_slope(pre,  "_time_rel_years", bmi_col)
        post_slope = _fit_slope(post, "_time_rel_years", bmi_col)

        bl_rows = baseline[baseline[patient_col] == pid]
        bl = float(bl_rows["baseline_bmi"].iloc[0]) if not bl_rows.empty else np.nan

        rows.append({
            patient_col: pid,
            "n_pre": len(pre),
            "n_post": len(post),
            "pre_slope_bmi_per_year": pre_slope,
            "post_slope_bmi_per_year": post_slope,
            "slope_change_bmi_per_year": (
                post_slope - pre_slope
                if pd.notna(pre_slope) and pd.notna(post_slope)
                else np.nan
            ),
            "baseline_bmi": bl,
        })

    return pd.DataFrame(rows)


def plot_bmi_slope_comparison(
    slopes_df: pd.DataFrame,
    patient_col: str = "PatientDurableKey",
    pre_col: str = "pre_slope_bmi_per_year",
    post_col: str = "post_slope_bmi_per_year",
    pre_label: str = "Pre-treatment",
    post_label: str = "Post-treatment",
    figsize: tuple = (4, 5),
    palette: tuple = ("lightblue", "orange"),
    out_path: str | None = None,
    dpi: int = 600,
) -> None:
    # Paired Wilcoxon boxplot comparing pre- and post-treatment BMI slopes.
    slopes_test = slopes_df.dropna(subset=[pre_col, post_col]).copy()

    w_stat, p_value = wilcoxon(slopes_test[pre_col], slopes_test[post_col])
    p_label = "Wilcoxon signed-rank: p < 0.001" if p_value < 0.001 else f"Wilcoxon signed-rank: p = {p_value:.3f}"

    slopes_long = slopes_test.melt(
        id_vars=patient_col,
        value_vars=[pre_col, post_col],
        var_name="window",
        value_name="bmi_slope_per_year",
    )
    slopes_long["window"] = slopes_long["window"].replace({pre_col: pre_label, post_col: post_label})

    summary = (
        slopes_long.groupby("window")["bmi_slope_per_year"]
        .agg(median="median", mean="mean")
        .reindex([pre_label, post_label])
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=slopes_long, x="window", y="bmi_slope_per_year",
        showfliers=False, ax=ax, palette=list(palette),
        order=[pre_label, post_label],
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel("BMI slope (kg/m² per year)")
    ax.set_title(f"Patient-level BMI slopes\nn = {len(slopes_test):,} | {p_label}", fontsize=10)

    for i, window in enumerate(summary.index):
        ax.text(
            i, 0.95, f"Median: {summary.loc[window, 'median']:.2f}\nMean: {summary.loc[window, 'mean']:.2f}",
            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.9),
        )

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


# Delta-BMI trajectory

def build_delta_bmi_trajectory(
    df: pd.DataFrame,
    anchor_offset_days: int = 0,
    max_months: int = 24,
    min_patients_per_bin: int = 10,
    patient_col: str = "PatientDurableKey",
    start_col: str = "StartInstant",
    date_col: str = "CreateInstant",
    bmi_col: str = "BodyMassIndex_recalc",
) -> tuple[pd.DataFrame, int]:
    # Compute median ΔBMI trajectory around an anchor date.
    max_days = int(round(max_months * 365.25 / 12))

    df = df.copy()
    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    df[date_col]  = pd.to_datetime(df[date_col],  errors="coerce")
    df[bmi_col]   = pd.to_numeric(df[bmi_col],    errors="coerce")
    df = df.dropna(subset=[patient_col, start_col, date_col, bmi_col])

    anchor = df[start_col] + pd.Timedelta(days=anchor_offset_days)
    df["_time_rel_days"] = (df[date_col] - anchor).dt.days
    df = df[df["_time_rel_days"].between(-max_days, max_days)].copy()

    # Baseline BMI = last BMI before the anchor
    baseline = (
        df[df["_time_rel_days"] < 0]
        .sort_values([patient_col, "_time_rel_days"])
        .groupby(patient_col, as_index=False)
        .tail(1)
        [[patient_col, bmi_col]]
        .rename(columns={bmi_col: "_baseline_bmi"})
    )
    df = df.merge(baseline, on=patient_col, how="left")
    df["_delta_bmi"] = df[bmi_col] - df["_baseline_bmi"]

    df["_time_bin_30d"] = (df["_time_rel_days"] // 30).astype("Int64")
    df["_bin_center"]   = df["_time_bin_30d"] + 0.5

    patient_bin = (
        df.dropna(subset=["_delta_bmi", "_bin_center"])
        .groupby([patient_col, "_bin_center"], as_index=False)
        .agg(_delta_bmi=("_delta_bmi", "median"))
    )

    traj = (
        patient_bin.groupby("_bin_center", as_index=False)["_delta_bmi"]
        .agg(
            median="_delta_bmi",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            n_patients="count",
        )
        .rename(columns={"_bin_center": "time_bin_center_months", "_delta_bmi": "median"})
        .sort_values("time_bin_center_months")
    )
    traj = traj[traj["n_patients"] >= min_patients_per_bin].copy()

    n_patients_plot = patient_bin.loc[
        patient_bin["_bin_center"].isin(traj["time_bin_center_months"]),
        patient_col,
    ].nunique()

    return traj, n_patients_plot


def _plot_single_trajectory(
    ax: plt.Axes,
    traj: pd.DataFrame,
    n_patients: int,
    title: str,
    show_auc: bool = False,
    ylabel: str = "ΔBMI from last pre-anchor BMI",
) -> None:
    ax.plot(traj["time_bin_center_months"], traj["median"], linewidth=2, label="Median ΔBMI")
    ax.fill_between(traj["time_bin_center_months"], traj["q1"], traj["q3"], alpha=0.2, label="IQR")
    ax.fill_between(traj["time_bin_center_months"], 0, traj["median"], alpha=0.15)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="gray",  linestyle=":",  linewidth=1)
    ax.set_xlim(-24, 24)
    ax.set_xlabel("Months relative to anchor", fontsize=10)
    ax.set_title(f"{title}, n = {n_patients:,}", fontsize=10)

    if show_auc:
        auc_abs = float(np.trapezoid(np.abs(traj["median"]), traj["time_bin_center_months"]))
        ax.text(
            0.03, 0.97, f"Cumulative median\nΔBMI: {auc_abs:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.9),
        )


def plot_delta_bmi_trajectory(
    df: pd.DataFrame,
    anchor_offset_days: int = 0,
    max_months: int = 24,
    min_patients_per_bin: int = 10,
    xlabel: str = "Months relative to treatment start",
    ylabel: str = "ΔBMI from last pre-treatment measurement",
    title: str = "BMI change from baseline",
    figsize: tuple = (5, 5),
    out_path: str | None = None,
    dpi: int = 600,
    **trajectory_kwargs,
) -> None:
    # Single-panel delta-BMI trajectory plot.
    traj, n_patients = build_delta_bmi_trajectory(
        df, anchor_offset_days=anchor_offset_days, max_months=max_months,
        min_patients_per_bin=min_patients_per_bin, **trajectory_kwargs,
    )
    fig, ax = plt.subplots(figsize=figsize)
    _plot_single_trajectory(ax, traj, n_patients, title, show_auc=False, ylabel=ylabel)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_delta_bmi_trajectory_3panel(
    df: pd.DataFrame,
    anchor_offsets: tuple[int, int, int] = (0, -365, 365),
    panel_titles: tuple[str, str, str] = (
        "At treatment start",
        "1 year before treatment start",
        "1 year after treatment start",
    ),
    max_months: int = 24,
    min_patients_per_bin: int = 10,
    figsize: tuple = (15, 5),
    out_path: str | None = None,
    dpi: int = 600,
    **trajectory_kwargs,
) -> None:
    # Three-panel delta-BMI trajectory plot with AUC annotations.
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)

    for ax, offset, title in zip(axes, anchor_offsets, panel_titles):
        traj, n_patients = build_delta_bmi_trajectory(
            df, anchor_offset_days=offset, max_months=max_months,
            min_patients_per_bin=min_patients_per_bin, **trajectory_kwargs,
        )
        _plot_single_trajectory(ax, traj, n_patients, title, show_auc=True)

    axes[0].set_ylabel("ΔBMI from last pre-anchor BMI", fontsize=10)
    axes[0].legend()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
