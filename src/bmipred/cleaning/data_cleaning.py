#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/bmipred/data_cleaning.py

import pandas as pd
import time

def clean_table(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    '''Standardise EHR dataframe (renaming, dtype conversions, missing values to NA).'''
    start_time = time.time()
    # Rename ID columns
    if 'DurableKey' in df.columns:
        df.rename(columns={'DurableKey': 'PatientDurableKey'}, inplace=True)

        # numeric columns
    for col in config["columns"].get("numeric", []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # datetime columns
    for col in config["columns"].get("datetime", []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # categorical columns
    for col in config["columns"].get("categorical", []):
        if col in df.columns:
            df[col] = df[col].astype("category")

    # boolean columns
    for col in config["columns"].get("boolean", []):
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # replace missing values
    df.replace(config.get("missing", []), pd.NA, inplace=True)

    elapsed_time = time.time() - start_time
    print(f"Data cleaning completed in {elapsed_time:.2f} seconds.")
    return df
