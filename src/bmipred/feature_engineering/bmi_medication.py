#!/usr/bin/env python3
# src/bmipred/feature_engineering/bmi_medication.py

import pandas as pd
import numpy as np
from intervaltree import Interval, IntervalTree
from multiprocessing import Pool, cpu_count
from functools import partial


def refine_medication_ids(medication: pd.DataFrame, 
                          bmi_demographics: pd.DataFrame,
                          patientid_col: str = "PatientDurableKey",
                          bmi_instant_col: str = "CreateInstant",
                          atc_col: str = "ATC",
                          start_col: str = "StartInstant",
                          discontinued_col: str = "DiscontinuedInstant") -> pd.DataFrame:
    
    # retain only patients present in bmi_demographics that have BMI data
    valid_patients = set(bmi_demographics[patientid_col].unique())
    # filter for only medications for patients that have BMI data
    medication = medication[medication[patientid_col].isin(valid_patients)]
    medication = medication.dropna(subset=[start_col, atc_col])
    medication = medication[medication[atc_col].astype(str).str.lower() != "nan"]

    # refine the medication table (consider the last medication interval without discontinuation date as ongoing)
    # sort and fill DiscontinuedInstant
    medication = medication.sort_values(by=[patientid_col, atc_col, start_col])
    medication["DiscontinuedInstant_last"] = medication.groupby([patientid_col, atc_col])[discontinued_col].transform("last")
    condition = (
        (medication[start_col] == medication["DiscontinuedInstant_last"]) &
        (medication[discontinued_col] == medication["DiscontinuedInstant_last"])
    )
    medication.loc[condition, discontinued_col] = pd.Timestamp.now()
    medication["MedicationDuration"] = (medication[discontinued_col] - medication[start_col]).dt.days

    print("Medication table:",medication.shape, medication[patientid_col].nunique())

    medication = medication[medication[start_col] < medication[discontinued_col]] # filter out invalid intervals (where end date before start date)
    
    # Make sure that all date columns are converted to datetime if not already
    bmi_demographics[bmi_instant_col] = pd.to_datetime(bmi_demographics[bmi_instant_col])
    medication[start_col] = pd.to_datetime(medication[start_col])
    medication[discontinued_col] = pd.to_datetime(medication[discontinued_col])
    medication.drop(["DiscontinuedInstant_last"], axis=1, inplace=True)
    return medication




def process_id(current_id: int, 
               bmi_demographics: pd.DataFrame, 
               medication: pd.DataFrame, 
               patientid_col: str = "PatientDurableKey",
               bmi_instant_col: str = "CreateInstant",
               start_col: str = "StartInstant",
               discontinued_col: str = "DiscontinuedInstant",
               atc_col: str = "ATC") -> pd.DataFrame:
    """Process a single PatientDurableKey to build an interval tree and map ATC codes to the BMI timestamp.""" 
    
    df1_id = bmi_demographics[bmi_demographics[patientid_col] == current_id].copy()
    df2_id = medication[medication[patientid_col] == current_id].copy()
    
    tree = IntervalTree()
    for _, row in df2_id.iterrows():
        start = row[start_col]
        end = row[discontinued_col]
        code = row[atc_col]
        if pd.notna(start) and pd.notna(end):
            tree[start:end] = code
    
    def find_codes(ts: pd.Timestamp) -> str:
        intervals = tree[ts]
        codes = {interval.data for interval in intervals}
        return ",".join(sorted(codes)) if codes else ""

    df1_id["ATCs"] = df1_id[bmi_instant_col].apply(find_codes)
    return df1_id




def map_atc_to_bmi(bmi_demographics: pd.DataFrame,
                   medication: pd.DataFrame,
                   patientid_col: str = "PatientDurableKey",
                   bmi_instant_col: str = "CreateInstant",
                   start_col: str = "StartInstant",
                   discontinued_col: str = "DiscontinuedInstant",
                   atc_col: str = "ATC",
                   num_processes: int = 16) -> pd.DataFrame:
    """Map ATC codes from medications to BMI timestamps in health assessments using parallel processing."""
    unique_ids = bmi_demographics[patientid_col].unique()
    num_processes = num_processes or cpu_count()

    process_func = partial(process_id, bmi_demographics=bmi_demographics, medication=medication, patientid_col=patientid_col, bmi_instant_col=bmi_instant_col, start_col=start_col, discontinued_col=discontinued_col, atc_col=atc_col)
    with Pool(processes=num_processes) as pool:
        result_list = pool.map(process_func, unique_ids)

    result = pd.concat(result_list, ignore_index=True)
    return result