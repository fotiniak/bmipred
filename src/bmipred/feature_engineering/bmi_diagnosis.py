#!/usr/bin/env python3
# src/bmipred/feature_engineering/bmi_medication.py

import pandas as pd
import numpy as np
from intervaltree import Interval, IntervalTree
from multiprocessing import Pool, cpu_count
from functools import partial


def refine_diagnosis_ids(diagnosis: pd.DataFrame, 
                         bmi_medication: pd.DataFrame,
                         patientid_col: str = "PatientDurableKey",
                         sks_col: str = "SKSCodes_range",
                         start_col: str = "DiagnosisStartDate",
                         discontinued_col: str = "DiagnosisEndDate") -> pd.DataFrame:

    # retain only patients present in bmi_medications that have BMI data
    valid_patients = set(bmi_medication[patientid_col].unique())
    # filter for only medications for patients that have BMI data
    diagnosis = diagnosis[diagnosis[patientid_col].isin(valid_patients)]
    diagnosis = diagnosis.dropna(subset=[start_col, sks_col])
    diagnosis = diagnosis[diagnosis[sks_col].astype(str).str.lower() != "nan"]

    # refine the medication table (consider the last medication interval without discontinuation date as ongoing)
    # sort and fill DiscontinuedInstant
    diagnosis = diagnosis.sort_values(by=[patientid_col, sks_col, start_col])
    diagnosis.dropna(subset=[start_col], inplace=True)
    diagnosis.dropna(subset=[sks_col], inplace=True)
    diagnosis[discontinued_col] = diagnosis[discontinued_col].fillna(pd.Timestamp.now())

    print("Diagnosis table:",diagnosis.shape, diagnosis[patientid_col].nunique())

    diagnosis = diagnosis[diagnosis['DiagnosisStartDate'] < diagnosis['DiagnosisEndDate']]# filter out invalid intervals (where end date before start date)

    # convert to datetime if not already
    diagnosis[start_col] = pd.to_datetime(diagnosis[start_col])
    diagnosis[discontinued_col] = pd.to_datetime(diagnosis[discontinued_col])

    print("Diagnosis table:",diagnosis.shape, diagnosis[patientid_col].nunique())

    return diagnosis




def process_id(current_id: int, 
               bmi_medication: pd.DataFrame, 
               diagnosis: pd.DataFrame, 
               patientid_col: str = "PatientDurableKey",
               bmi_instant_col: str = "CreateInstant",
               start_col: str = "DiagnosisStartDate",
               discontinued_col: str = "DiagnosisEndDate",
               sks_col: str = "SKSCode") -> pd.DataFrame:
    """Process a single PatientDurableKey to build an interval tree and map ATC codes to the BMI timestamp.""" 

    df1_id = bmi_medication[bmi_medication[patientid_col] == current_id].copy()
    df2_id = diagnosis[diagnosis[patientid_col] == current_id].copy()

    tree = IntervalTree()
    for _, row in df2_id.iterrows():
        start = row[start_col]
        end = row[discontinued_col]
        code = row[sks_col]
        if pd.notna(start) and pd.notna(end):
            tree[start:end] = code
    
    def find_codes(ts: pd.Timestamp) -> str:
        intervals = tree[ts]
        codes = {interval.data for interval in intervals}
        return ",".join(sorted(codes)) if codes else ""

    # dynamic column name: "<code_col>_all"
    new_col = f"{sks_col}_all"
    df1_id[new_col] = df1_id[bmi_instant_col].apply(find_codes)
    return df1_id




def map_skscode_to_bmi(bmi_medication: pd.DataFrame,
                       diagnosis: pd.DataFrame,
                       patientid_col: str = "PatientDurableKey",
                       bmi_instant_col: str = "CreateInstant",
                       start_col: str = "DiagnosisStartDate",
                       discontinued_col: str = "DiagnosisEndDate",
                       sks_col: str = "SKSCode",
                       num_processes: int = 16) -> pd.DataFrame:
    """Map SKS codes from medications to BMI timestamps in health assessments using parallel processing."""
    unique_ids = bmi_medication[patientid_col].unique()
    num_processes = num_processes or cpu_count()

    process_func = partial(process_id, bmi_medication=bmi_medication, diagnosis=diagnosis, patientid_col=patientid_col, bmi_instant_col=bmi_instant_col, start_col=start_col, discontinued_col=discontinued_col, sks_col=sks_col)
    with Pool(processes=num_processes) as pool:
        result_list = pool.map(process_func, unique_ids)

    result = pd.concat(result_list, ignore_index=True)
    return result