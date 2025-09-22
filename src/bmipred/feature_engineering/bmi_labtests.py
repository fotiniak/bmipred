#!/usr/bin/env python3
# src/bmipred/feature_engineering/bmi_medication.py


import pandas as pd
import numpy as np
from intervaltree import Interval, IntervalTree
from multiprocessing import Pool, cpu_count
from functools import partial


def integrate_lab_tests(lab_tests: pd.DataFrame,
                         bmi_medication_diagnosis: pd.DataFrame,
                         patientid_col: str = "PatientDurableKey",
                         bmi_instant_col: str = "CreateInstant",
                         collection_instant_col: str = "CollectionInstant",
                         labanalysis_col: str = "Labanalysis",
                         value_col: str = "Value",
                         flag_col: str = "Flag") -> pd.DataFrame:

    # Filter lab tests for patients with BMI medication diagnosis
    valid_patients = set(bmi_medication_diagnosis[patientid_col].unique())
    lab_tests = lab_tests[lab_tests[patientid_col].isin(valid_patients)]

    # Drop rows with missing values in key columns
    lab_tests.dropna(subset=[collection_instant_col, labanalysis_col], inplace=True)

    # Convert CollectionInstant to datetime
    lab_tests[collection_instant_col] = pd.to_datetime(lab_tests[collection_instant_col])

    # Remove duplicates
    lab_tests.drop_duplicates(inplace=True)

    # Standardize Labanalysis codes by removing non-alphanumeric characters
    lab_tests[labanalysis_col] = lab_tests[labanalysis_col].str.replace(r'[^a-zA-Z0-9]', '', regex=True)

    # Removing entries with missing or non-numeric values in the Value column which means that the test was not performed
    lab_tests.dropna(subset=[value_col], inplace=True)
    lab_tests[collection_instant_col] = pd.to_datetime(lab_tests[collection_instant_col])
    print("Lab tests table:",lab_tests.shape, lab_tests[patientid_col].nunique())
    
    # The flag column indicates if the lab result is high, low, or normal. We will convert it to numeric values: High=3, Normal=2, Low=1
    # We do that because the simple categories Normal, High, Low, are not preserving the values hierarchy when pivoting the table
    lab_tests[flag_col] = lab_tests[flag_col].astype("category")
    lab_tests[flag_col] = lab_tests[flag_col].cat.add_categories([1,2,3])
    lab_tests[flag_col] = lab_tests[flag_col].replace({"Høj": 3, "Lav": 1}).fillna(2) # The normal range is set to 2
    lab_tests[flag_col] = pd.to_numeric(lab_tests[flag_col], errors="coerce")
    print("Lab tests table:", lab_tests.shape, lab_tests[patientid_col].nunique())
    
    # Pivoting the lab results to have one column per lab test
    lab_tests_pivot = lab_tests.pivot_table(index=lab_tests.index, columns="Labanalysis", values=flag_col)
    lab_tests_pivot.columns = lab_tests_pivot.columns.map(lambda x: f"Labanalysis_{x}")
    lab_tests = pd.concat([lab_tests, lab_tests_pivot], axis=1)
    lab_tests.drop(["Labanalysis", flag_col], axis=1, inplace=True)
    
    # Aggregate lab results by patient and collection instant
    lab_tests = lab_tests.sort_values(by=[patientid_col, collection_instant_col], ascending=[True, False], na_position="last")
    
    lab_tests_group = lab_tests.groupby([patientid_col, collection_instant_col], as_index=False).agg("first")
    print("lab_tests_group", lab_tests_group.shape, lab_tests_group[patientid_col].nunique())

    # Merge lab tests with bmi_medication_diagnosis on patient ID
    bmi_med_diag_lab = pd.merge(bmi_medication_diagnosis, lab_tests_group, on=[patientid_col], how="left")
    print("Combined table:", bmi_med_diag_lab.shape, bmi_med_diag_lab[patientid_col].nunique())

    # Filter lab tests to only include those taken before the BMI measurement
    bmi_med_diag_lab["CollectionInstant_from_CreateInstant"] = bmi_med_diag_lab[collection_instant_col] - bmi_med_diag_lab[bmi_instant_col]
    bmi_med_diag_lab["IsLabTestBeforeBMI"] = bmi_med_diag_lab["CollectionInstant_from_CreateInstant"] <= pd.Timedelta("0 days")
    bmi_med_diag_lab = bmi_med_diag_lab[(bmi_med_diag_lab["IsLabTestBeforeBMI"] == True)]
    bmi_med_diag_lab.drop_duplicates(inplace=True)
    print("Combined table after filtering:", bmi_med_diag_lab.shape, bmi_med_diag_lab[patientid_col].nunique())

    # For each patient and BMI instant, keep only the most recent lab test before the BMI measurement (discarding older ones)
    bmi_med_diag_lab = bmi_med_diag_lab.sort_values(by=[patientid_col, collection_instant_col], ascending=[True, False], na_position="last")
    bmi_med_diag_lab = bmi_med_diag_lab.groupby([patientid_col, bmi_instant_col], as_index=False).agg("first") # keep the most recent lab test before BMI
    print("Final combined table after keeping the most recent past values per BMI:", bmi_med_diag_lab.shape, bmi_med_diag_lab[patientid_col].nunique())

    # Final table wrangling to keep only relevant columns and results
    selected_columns = [patientid_col, bmi_instant_col, collection_instant_col] + [col for col in bmi_med_diag_lab.columns if col.startswith(labanalysis_col)]
    valid_lab_tests = bmi_med_diag_lab[selected_columns]
    bmi_med_diag_lab = pd.merge(bmi_medication_diagnosis, valid_lab_tests, on=[patientid_col, bmi_instant_col], how="left")

    return bmi_med_diag_lab