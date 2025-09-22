#!/usr/bin/env python3
# src/bmipred/integrated_data.py

import pandas as pd
import numpy as np

def refine_integrated_tables(df: pd.DataFrame,
                              patientid_col: str = "PatientDurableKey",
                              labanalysis_col: str = "Labanalysis",
                              ) -> pd.DataFrame:


    # Filter some sparse columns out of the dataset
    
    # familyHistory "Relative_with"
    relative_columns = [col for col in df.columns if col.startswith("Relative_with")]
    true_frequencies = df[relative_columns].mean()
    columns_to_keep = true_frequencies[true_frequencies > 0.05].index
    df = df[list(columns_to_keep) + [col for col in df.columns if not col.startswith("Relative_with")]]
    print("df.shape:", df.shape, df[patientid_col].nunique())

    # "lab_analysis"
    lab_analysis_columns = [col for col in df.columns if col.startswith(labanalysis_col)]
    na_frequencies = df[lab_analysis_columns].isna().mean()
    columns_to_remove = na_frequencies[na_frequencies > 0.5].index
    df = df.drop(columns=columns_to_remove)
    print("df", df.shape, df[patientid_col].nunique())

    #
    # 1) Replace string "None" with np.nan
    df.replace("None", np.nan, inplace=True)

    # 2) Replace actual Python None with np.nan
    df.replace({None: np.nan}, inplace=True)

    # 3) Convert True/False or "True"/"False" to 1/0
    df.replace({True: 1, "True": 1, False: 0, "False": 0}, inplace=True)

    #===============================================Create Medication Columns from ATCs
    # Antipsychotics
    # we want anything that starts with "N05A" but not "N05AN01" 
    # (that might be e.g. lithium"s code, adapt as needed).
    def has_antipsychotics(codes):
        if not isinstance(codes, str):
            return 0
        code_list = [c.strip() for c in codes.split(",")]
        return int(any(c.startswith("N05A") and not c.startswith("N05AN01") 
                       for c in code_list))

    df["antipsychotic_medications"] = df["ATCs"].apply(has_antipsychotics)

    # Olanzapine
    def has_olanzapine(codes):
        if not isinstance(codes, str):
            return 0
        return int(any(code.strip().startswith("N05AH03") for code in codes.split(",")))

    df["olanzapine_medication"] = df["ATCs"].apply(has_olanzapine)

    #===============================================Create Diagnosis Columns from SKSCodes
    # Schizophrenia Diagnosis
    # "DF2" or "DF20", "DF25", etc. -> if ANY code in SKSCodes starts with "DF2"
    def has_schizophrenia(codes):
        # handle missing or empty strings
        if not isinstance(codes, str):
            return 0
        # split by comma and check if any code starts with "DF2"
        return int(any(code.strip().startswith("DF2") for code in codes.split(",")))

    df["schizophrenia_diagnosis"] = df["SKSCode_all"].apply(has_schizophrenia)

    # Cancer Diagnosis
    # you said any prefix in ["SKS_DC","SKS_DD0","SKS_DD1","SKS_DD2","SKS_DD3","SKS_DD4"]
    # but your actual codes might look like "DC", "DD0", "DD1", etc., *without* the "SKS_" prefix.
    cancer_prefixes = ["DC","DD0","DD1","DD2","DD3","DD4"]

    def has_cancer(codes):
        if not isinstance(codes, str):
            return 0
        # check if any code starts with any of those prefixes
        return int(any(any(code.strip().startswith(pref) for pref in cancer_prefixes)
                       for code in codes.split(",")))

    df["cancer_diagnosis"] = df["SKSCode_all"].apply(has_cancer)

    #================================================One-Hot Encode Columns SmokingStatus, MaritalStatus
    cols_to_encode = ["SmokingStatus", "MaritalStatus"]
    # Only encode if they exist
    available = [c for c in cols_to_encode if c in df.columns]

    if available:
        dummies = pd.get_dummies(df[available], prefix=available, drop_first=False)
        df = pd.concat([df.drop(available, axis=1), dummies], axis=1)

    #================================================Convert 0/1 Columns to Boolean
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0,1}):
                df[col] = df[col].astype("boolean")

    #================================================Truncate the ATCs and SKSCodes to a length of 4 characters
    def truncate_codes_to_4_chars(df, atc_col="ATCs", sks_col="SKSCode_all"):
        """
        For the given "atc_col" and "sks_col" in df (containing comma-separated codes),
        create new columns (e.g. "ATCs_4" and "SKSCodes_4") with the first 4 chars of each code. 
        This reduces the code granularity and helps mitigate sparsity.
        """
        # --- Truncate ATC codes ---
        if atc_col in df.columns:
            def truncate_atc_codes(codes):
                # handle missing or non-string
                if not isinstance(codes, str):
                    return ""
                # for each code, keep only first 4 characters
                truncated_list = [code.strip()[:4] for code in codes.split(",") if code.strip()]
                return ",".join(truncated_list)

            df["ATCs_short"] = df[atc_col].apply(truncate_atc_codes)

        # --- Truncate SKS codes ---
        if sks_col in df.columns:
            def truncate_sks_codes(codes):
                if not isinstance(codes, str):
                    return ""
                truncated_list = [code.strip()[:4] for code in codes.split(",") if code.strip()]
                return ",".join(truncated_list)

            df["SKSCodes_short"] = df[sks_col].apply(truncate_sks_codes)

        return df

    df = truncate_codes_to_4_chars(df, atc_col="ATCs", sks_col="SKSCode_all")

    #================================================Filter Top 200 Codes ATC and SKS Codes to create dummies
    #------------------------------------------------DECIDE HERE WHETHER TO USE SHORT OR LONG FORM CODES:
    # Decide whether to use the long of the short version of the data ATC and SKS codes 

    atcs = "ATCs_short" # here i am choosing the short forms of ATC codes, long form would be: atcs = "ATCs"
    skscodes = "SKSCodes_short" # here i am choosing the short forms of SKS codes, long form would be: atcs = "SKSCodes"

    # For ATCs:
    code_counts_atc = df[atcs].str.split(",", expand=True).stack().value_counts()
    top_atc_codes = code_counts_atc.nlargest(200).index.tolist()

    df["ATCs_filtered"] = df[atcs].apply(lambda x: ",".join(
        c.strip() for c in x.split(",") 
        if c.strip() in top_atc_codes
    ) if pd.notnull(x) else "")

    atc_dummies = df["ATCs_filtered"].str.get_dummies(sep=",").add_prefix("ATC_")
    df = pd.concat([df, atc_dummies], axis=1)

    # For SKSCodes:
    code_counts_sks = df[skscodes].str.split(",", expand=True).stack().value_counts()
    top_sks_codes = code_counts_sks.nlargest(200).index.tolist()

    df["SKSCodes_filtered"] = df[skscodes].apply(lambda x: ",".join(
        c.strip() for c in x.split(",") 
        if c.strip() in top_sks_codes
    ) if pd.notnull(x) else "")

    sks_dummies = df["SKSCodes_filtered"].str.get_dummies(sep=",").add_prefix("SKS_")
    df = pd.concat([df, sks_dummies], axis=1)

    #=================================================Convert 0/1 Columns to Boolean
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0,1}):
                df[col] = df[col].astype("boolean")

    #==================================================Drop Unused Columns
    
    df.drop(columns=["BirthDate",
                     "DeathDate",
                     "weight_diff_from_next",
                     "bmi_diff_from_previous",
                     "ATCs",
                     "ATCs_filtered",
                     "ATCs_short",
                     "SKSCodes_all",
                     "SKSCodes_range_all",
                     "SKSCodes_filtered",
                     "SKSCodes_short",
                     "CollectionInstant",
                     "BodyMassIndex_classification",
                     "nan",
                     "ATC_nan"], inplace=True, errors="ignore")
    
    print(df.shape, df[patientid_col].nunique())

    print("Preprocessing of the integrated data table Finished Successfully!")

    return df
