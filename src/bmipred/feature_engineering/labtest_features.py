#!/usr/bin/env python3
# src/bmipred/feature_engineering/labtest_features.py

import pandas as pd
import numpy as np


def normalize_lab_tests(df: pd.DataFrame,
                        patientid_col: str = "PatientDurableKey",
                        timestamp_col: str = "CollectionInstant",
                        labanalysis_col: str = "Labanalysis",
                        value_col: str | None = "Value",
                        flag_col: str = "Flag",
                        abnormal_flags: list = ["Høj", "Lav"],) -> pd.DataFrame:
    out = df.copy()

    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    
    # Conditionally include value_col in dropna subset
    dropna_subset = [timestamp_col]
    
    if value_col is not None: # Only drop rows with missing values if value_col is specified
        dropna_subset.append(value_col)

    out = out.dropna(subset=dropna_subset)
    out = out.sort_values([patientid_col, timestamp_col]).reset_index(drop=True)

    # Normalize lab analysis names
    out[labanalysis_col] = out[labanalysis_col].str.replace(r'[^a-zA-Z0-9]', '', regex=True)

    # Standardize flag: keep Høj/Lav, everything else → Normal
    out[flag_col] = np.where(out[flag_col].isin(abnormal_flags), out[flag_col], "Normal")

    # Remove purely letter-based values that are not abnormal
    if value_col is not None:
        has_letters = (
            out[value_col].astype(str).str.contains(r"[^\W\d_]", regex=True, na=False) &
            ~out[value_col].astype(str).str.contains(r"\d", regex=True, na=False)
        )
        to_remove = has_letters & (out[flag_col] == "Normal")
        print(f"Rows removed (non-numeric normal values): {to_remove.sum()}")
        out = out.loc[~to_remove].copy()

    out["IsAbnormal"] = out[flag_col].isin(abnormal_flags).astype("int8")

    return out



def lab_history_features(df: pd.DataFrame,
                         patientid_col: str = "PatientDurableKey",
                         timestamp_col: str = "CollectionInstant",) -> pd.DataFrame:
    out = df.copy()

    # Aggregate counts per patient+timestamp
    instance = (
        out
        .groupby([patientid_col, timestamp_col], as_index=False)
        .agg(
            tests_at_instant=(timestamp_col, "size"),
            abnormal_at_instant=("IsAbnormal", "sum"),
        )
    )

    # Cumulative counts strictly before current timestamp
    instance["number_of_prev_tests"] = (
        instance.groupby(patientid_col)["tests_at_instant"].cumsum()
        - instance["tests_at_instant"]
    ).astype(int)

    instance["number_of_prev_abnormal_tests"] = (
        instance.groupby(patientid_col)["abnormal_at_instant"].cumsum()
        - instance["abnormal_at_instant"]
    ).astype(int)

    # Days since last test
    instance["days_since_last_test"] = (
        (instance[timestamp_col] - instance.groupby(patientid_col)[timestamp_col].shift(1))
        .dt.total_seconds() / 86400.0
    )

    # Merge features back to original rows
    out = out.merge(
        instance[[patientid_col, timestamp_col,
                  "number_of_prev_tests",
                  "number_of_prev_abnormal_tests",
                  "days_since_last_test"]],
        on=[patientid_col, timestamp_col],
        how="left",
    )

    # Abnormal test rate
    out["proportion_of_abnormal_tests"] = (
        out["number_of_prev_abnormal_tests"] /
        out["number_of_prev_tests"].replace(0, np.nan)
    )

    return out
