#!/usr/bin/env python3
# bmipred/modeling/reports.py

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import os
import logging


def _is_binary(series: pd.Series) -> bool:
    """Check if series is binary."""
    if pd.api.types.is_bool_dtype(series):
        return True
    s = series.dropna().unique()
    return set(s).issubset({0, 1})


def _pseudo_stats(arr: np.ndarray) -> tuple:
    """Compute pseudo-statistics for small arrays."""
    n = len(arr)
    if n < 5:
        return (np.nan, np.nan, np.nan)
    arr = np.sort(arr)
    pmin = arr[:5].mean()
    centre = arr[max(0, n // 2 - 2) : max(0, n // 2 - 2) + 5].mean()
    pmax = arr[-5:].mean()
    return (pmin, centre, pmax)


def summarise_column(col: pd.Series) -> Dict[str, Any]:
    """Create summary statistics for a column."""
    total_n = len(col)
    miss_n = col.isna().sum()
    out = {"N": total_n, "Missing_%": round(miss_n / total_n * 100, 2)}

    if total_n == miss_n:  # everything missing
        return out

    if _is_binary(col):
        t = ((col == 1) | (col == True)).sum()
        f = ((col == 0) | (col == False)).sum()
        out.update(
            True_count=int(t),
            False_count=int(f),
            True_percentage=round(t / (total_n - miss_n) * 100, 2),
            False_percentage=round(f / (total_n - miss_n) * 100, 2),
        )
    else:
        num = pd.to_numeric(col, errors="coerce").dropna().to_numpy()
        if len(num) > 0:
            pmin, pmed, pmax = _pseudo_stats(num)
            out.update(
                Min=round(pmin, 2),
                Max=round(pmax, 2),
                Mean=round(num.mean(), 2),
                Median=round(pmed, 2),
            )
    return out


def save_feature_summary(df: pd.DataFrame, csv_path: str):
    """Save feature summary statistics to CSV."""
    rows = []
    for col in df.columns:
        stats = summarise_column(df[col])
        stats["Variable"] = col  # Use column name directly, no pretty function
        rows.append(stats)

    summary = pd.DataFrame(rows)[[
        "Variable", "N", "Missing_%",
        "True_count", "False_count", "True_percentage", "False_percentage",
        "Min", "Max", "Mean", "Median"
    ]]
    summary.to_csv(csv_path, index=False)
    logging.info(f"Feature summary saved → {csv_path}")


def create_summary_metrics(metrics_df: pd.DataFrame, keep_metrics: Dict[str, str]) -> pd.DataFrame:
    """Create summary metrics across splits."""
    metric_cols = list(keep_metrics.keys())
    summary_rows = []
    
    for model, grp in metrics_df.groupby("Model"):
        row = {"Model": model}
        for metric in metric_cols:
            if metric in grp.columns:
                vals = grp[metric].values
                row[f"{metric} Mean"] = np.mean(vals)
                row[f"{metric} 95% CI Lower"], row[f"{metric} 95% CI Upper"] = np.percentile(vals, [2.5, 97.5])
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


def create_compact_metrics(summary_df: pd.DataFrame, keep_metrics: Dict[str, str]) -> pd.DataFrame:
    """Create compact metrics with confidence intervals."""
    compact_cols = ["Model"]
    compact_df = summary_df[["Model"]].copy()
    
    for base, pretty_name in keep_metrics.items():
        if f"{base} Mean" in summary_df.columns:
            compact_df[pretty_name] = (
                summary_df[f"{base} Mean"].round(3).astype(str)
                + " (" + summary_df[f"{base} 95% CI Lower"].round(3).astype(str)
                + "–" + summary_df[f"{base} 95% CI Upper"].round(3).astype(str) + ")"
            )
    
    return compact_df


def save_all_reports(
    table_name: str,
    all_metrics: List[pd.DataFrame],
    output_dir: str,
    keep_metrics: Dict[str, str]
):
    """Save all summary reports for a table."""
    # Combine all splits
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_metrics.to_csv(
        os.path.join(output_dir, f"{table_name}_all_splits_metrics.csv"),
        index=False
    )
    
    # Create summary metrics
    summary_metrics = create_summary_metrics(combined_metrics, keep_metrics)
    summary_metrics["Table"] = table_name
    summary_path = os.path.join(output_dir, f"{table_name}_summary_metrics.csv")
    summary_metrics.to_csv(summary_path, index=False)
    
    # Create compact metrics
    compact_metrics = create_compact_metrics(summary_metrics, keep_metrics)
    compact_metrics["Table"] = table_name
    compact_metrics.to_csv(
        os.path.join(output_dir, f"{table_name}_summary_metrics_compact.csv"),
        index=False
    )
    
    logging.info(f"Reports saved for {table_name}")