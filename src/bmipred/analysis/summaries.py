# src/bmipred/analysis/summaries.py

import numpy as np
import pandas as pd

def is_binary(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    uniq = series.dropna().unique()
    return set(uniq).issubset({0, 1})

def pseudo_stats(arr: np.ndarray) -> tuple[float, float, float]:
    n = len(arr)
    if n < 5:
        return (np.nan, np.nan, np.nan)
    arr_sorted = np.sort(arr)
    return (
        arr_sorted[:5].mean(),
        arr_sorted[max(0, n // 2 - 2): max(0, n // 2 - 2) + 5].mean(),
        arr_sorted[-5:].mean()
    )

def summarise_column(col: pd.Series) -> dict:
    total_n, missing = len(col), col.isna().sum()
    out = {"N": total_n, "Missing_%": round(missing / total_n * 100, 2)}
    if total_n == missing:
        return out
    if is_binary(col):
        true_count = ((col == 1) | (col == True)).sum()
        false_count = ((col == 0) | (col == False)).sum()
        out.update({
            "True_count": int(true_count),
            "False_count": int(false_count),
            "True_percentage": round(true_count / (total_n - missing) * 100, 2),
            "False_percentage": round(false_count / (total_n - missing) * 100, 2),
        })
    else:
        numeric = pd.to_numeric(col, errors="coerce").dropna().to_numpy()
        pmin, pmed, pmax = pseudo_stats(numeric)
        out.update({
            "Min": round(pmin, 2),
            "Max": round(pmax, 2),
            "Mean": round(numeric.mean(), 2),
            "Median": round(pmed, 2),
        })
    return out

def summarise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for col in df.columns:
        stats = summarise_column(df[col])
        stats["Column"] = col
        summary.append(stats)
    cols = [
        "Column", "N", "Missing_%", "True_count", "False_count",
        "True_percentage", "False_percentage", "Min", "Max", "Mean", "Median"
    ]
    return pd.DataFrame(summary)[cols]
