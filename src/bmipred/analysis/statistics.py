#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, t
from math import sqrt

def compare_two_groups_with_normality_check(
    arr1,
    arr2,
    label1="Group1",
    label2="Group2",
    alpha=0.05,
    equal_var=False) -> pd.DataFrame:
    # Compare two groups with Shapiro test to perform either t-test or Mann-Whitney depending on normality.
    x1, x2 = np.array(arr1, float), np.array(arr2, float)
    x1, x2 = x1[~np.isnan(x1)], x2[~np.isnan(x2)]
    n1, n2 = len(x1), len(x2)
    mean1, mean2, std1, std2 = x1.mean(), x2.mean(), x1.std(ddof=1), x2.std(ddof=1)
    sem1, sem2 = std1 / sqrt(n1), std2 / sqrt(n2)

    row1, row2 = {"group": label1, "n": n1, "mean": mean1, "median": np.median(x1),
                  "std_dev": std1, "std_err": sem1}, \
                 {"group": label2, "n": n2, "mean": mean2, "median": np.median(x2),
                  "std_dev": std2, "std_err": sem2}

    p_shap1, p_shap2 = shapiro(x1)[1], shapiro(x2)[1]
    both_normal = (p_shap1 >= alpha) and (p_shap2 >= alpha)

    if both_normal:
        t_stat, p_val = ttest_ind(x1, x2, equal_var=equal_var)
        if equal_var:
            df_approx = n1 + n2 - 2
        else:
            var1, var2 = std1**2, std2**2
            num = (var1/n1 + var2/n2)**2
            den = (var1**2 / ((n1**2)*(n1 - 1))) + (var2**2 / ((n2**2)*(n2 - 1)))
            df_approx = num / den
        mean_diff = mean1 - mean2
        se_diff = sqrt((std1**2)/n1 + (std2**2)/n2)
        ci = t.ppf(1 - alpha/2, df_approx) * se_diff
        updates = {"test": "Independent t-test", "stat": t_stat, "df": df_approx,
                   "p_value": p_val, "mean_diff": mean_diff, "ci_lower": mean_diff - ci,
                   "ci_upper": mean_diff + ci, "u_stat": np.nan, "z_score": np.nan, "rbc": np.nan}
    else:
        u_stat, p_val = mannwhitneyu(x1, x2, alternative="two-sided")
        mean_U, stdev_U = (n1 * n2) / 2.0, sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        z_approx, rbc = (u_stat - mean_U) / stdev_U, (2 * (u_stat / (n1*n2))) - 1
        updates = {"test": "Mann-Whitney U", "stat": np.nan, "df": np.nan,
                   "p_value": p_val, "mean_diff": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                   "u_stat": u_stat, "z_score": z_approx, "rbc": rbc}

    for row in (row1, row2):
        row.update(updates)

    cols = ["group", "n", "mean", "median", "std_dev", "std_err", "test",
            "stat", "df", "p_value", "mean_diff", "ci_lower", "ci_upper", "u_stat", "z_score", "rbc"]
    return pd.DataFrame([row1, row2], columns=cols)
