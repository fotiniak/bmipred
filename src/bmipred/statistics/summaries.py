# src/bmipred/analysis/summaries.py

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

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

def pseudo_quantile(v: pd.Series, q: float, k: int = 5) -> float:
    # GDPR-safe quantile: mean of the k values closest to the q-th quantile.
    v = v.dropna().reset_index(drop=True)
    if len(v) < k:
        return np.nan
    target = v.quantile(q)
    return float(v.iloc[(v - target).abs().argsort()[:k]].mean())


def round5(n: int) -> int:
    # Round count to the nearest 5 for GDPR disclosure.
    return int(round(n / 5) * 5)


def _gdpr_desc(s: pd.Series) -> dict:
    # GDPR-safe descriptive stats for a single numeric series.
    v = s.dropna().reset_index(drop=True)
    n_miss = round5(s.isna().sum())
    pct_miss = 100 * s.isna().mean()
    if len(v) < 5:
        return {
            "mean_sd": "insufficient n",
            "median_iqr": "insufficient n",
            "range": "insufficient n",
            "missing": f"~{n_miss} ({pct_miss:.1f}%)",
        }
    p50 = pseudo_quantile(v, 0.50)
    p25 = pseudo_quantile(v, 0.25)
    p75 = pseudo_quantile(v, 0.75)
    p_min = float(v.nsmallest(5).mean())
    p_max = float(v.nlargest(5).mean())
    return {
        "mean_sd":    f"{v.mean():.2f} ({v.std():.2f})",
        "median_iqr": f"{p50:.2f} [{p25:.2f}, {p75:.2f}]",
        "range":      f"{p_min:.2f}–{p_max:.2f}",
        "missing":    f"~{n_miss} ({pct_miss:.1f}%)",
    }


def build_publication_table(
    df_on: pd.DataFrame,
    df_off: pd.DataFrame,
    columns: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    # Build a GDPR-safe publication comparison table for two cohorts.
    if columns is None:
        columns = [
            ("BMI", "BodyMassIndex_recalc"),
            ("BMI variance", "bmi_variance_history"),
        ]

    n_on  = round5(len(df_on))
    n_off = round5(len(df_off))
    rows: list[tuple] = []

    for label, col in columns:
        a = df_on[col] if col in df_on.columns else pd.Series(dtype=float)
        b = df_off[col] if col in df_off.columns else pd.Series(dtype=float)
        d_on  = _gdpr_desc(a)
        d_off = _gdpr_desc(b)

        a_clean, b_clean = a.dropna(), b.dropna()
        if len(a_clean) >= 3 and len(b_clean) >= 3:
            u, p = scipy_stats.mannwhitneyu(a_clean, b_clean, alternative="two-sided")
            r = abs(1 - (2 * u) / (len(a_clean) * len(b_clean)))
            p_fmt = "<0.001" if p < 0.001 else f"{p:.3f}"
            r_fmt = f"{r:.3f}"
        else:
            p_fmt = r_fmt = "—"

        rows += [
            (label,             "",                    "",                    "",     ""),
            ("  Mean (SD)",     d_on["mean_sd"],       d_off["mean_sd"],      p_fmt,  r_fmt),
            ("  Median [IQR]",  d_on["median_iqr"],    d_off["median_iqr"],   "",     ""),
            ("  Range*",        d_on["range"],          d_off["range"],        "",     ""),
            ("  Missing n (%)", d_on["missing"],        d_off["missing"],      "",     ""),
        ]

    col_names = [
        "Characteristic",
        f"On treatment (~n={n_on:,})",
        f"Off treatment (~n={n_off:,})",
        "p-value",
        "Effect size (r)",
    ]
    return pd.DataFrame(rows, columns=col_names).set_index("Characteristic")


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
