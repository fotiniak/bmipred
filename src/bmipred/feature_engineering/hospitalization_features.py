#!/usr/bin/env python3
# src/bmipred/feature_engineering/hospitalization_features.py

import pandas as pd

def hospitalization_history_features(df: pd.DataFrame,
                                     patientid_col: str = "PatientDurableKey",
                                     admission_col: str = "InpatientAdmissionInstant",
                                     discharge_col: str = "DischargeInstant",) -> pd.DataFrame:
    out = df.copy()
    out[admission_col] = pd.to_datetime(out[admission_col], errors="coerce")
    out[discharge_col] = pd.to_datetime(out[discharge_col], errors="coerce")

    out["LengthOfStayDays"] = (out[discharge_col] - out[admission_col]).dt.days

    out = out.sort_values([patientid_col, admission_col])

    out["number_of_prev_hosp"] = out.groupby(patientid_col).cumcount()
    out["number_of_prev_hosp_days"] = (out.groupby(patientid_col)["LengthOfStayDays"].cumsum() - out["LengthOfStayDays"])

    return out
