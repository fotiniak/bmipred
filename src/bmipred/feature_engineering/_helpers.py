#!/usr/bin/env python3
# src/bmipred/feature_engineering/_helpers.py

import pandas as pd



def filter_timestamps(df: pd.DataFrame,
                      timestamp_col: str = "CreateInstant",
                      date_lo: pd.Timestamp = pd.Timestamp("2016-01-01"),
                      date_hi: pd.Timestamp = pd.Timestamp("2026-01-01"),) -> pd.DataFrame:
   # Filter df to include only rows with CreateInstant between date_lo and date_hi.
    print("Rows before date filtering:", len(df)) 
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df[df[timestamp_col].between(date_lo, date_hi, inclusive='left')]
    print("Rows after date filtering:", len(df))
    return df



def cumulative_past_events(df: pd.DataFrame,
                           patientid_col: str = "PatientDurableKey",
                           timestamp_col: str = "Timestamp",
                           out_col: str = "number_of_previous_X",) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values([patientid_col, timestamp_col])
    out[out_col] = out.groupby(patientid_col).cumcount() + 1
    return out
