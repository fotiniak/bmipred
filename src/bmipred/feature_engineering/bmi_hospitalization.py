#!/usr/bin/env python3
# src/bmipred/feature_engineering/bmi_medication.py

import pandas as pd
import numpy as np

def integrate_hospitalizations(hospitalization: pd.DataFrame,
                               bmi_integrated_df: pd.DataFrame,
                               patientid_col: str = "PatientDurableKey",
                               bmi_instant_col: str = "CreateInstant",
                               start_col: str = "InpatientAdmissionInstant",
                               end_col: str = "DischargeInstant",
                               hosp_key_col: str = "HospitalAdmissionKey",
                               hosp_service_col: str = "HospitalService") -> pd.DataFrame:

    bmi_integrated_df[bmi_instant_col] = pd.to_datetime(bmi_integrated_df[bmi_instant_col])
    print("Unique patients:", bmi_integrated_df.shape, bmi_integrated_df[patientid_col].nunique())

    # Filter hospital admission data to include only relevant patients.
    hospitalization = hospitalization[hospitalization[patientid_col].isin(bmi_integrated_df[patientid_col].unique())]

    # Merge a simple version of the health assesment with the hospital admission data.
    bmi_integrated_df_hospitalization = pd.merge(bmi_integrated_df, hospitalization, on=[patientid_col], how="left")

    # Add a column indicating whether the patient is currently hospitalized (currently means at the timestamp of CreateInstant)
    bmi_integrated_df_hospitalization["IsCurrentlyHospitalized"] = (
        (bmi_integrated_df_hospitalization[bmi_instant_col] >= bmi_integrated_df_hospitalization[start_col]) &
        (
            (bmi_integrated_df_hospitalization[bmi_instant_col] <= bmi_integrated_df_hospitalization[end_col]) |
            (bmi_integrated_df_hospitalization[end_col].isna())
        )
    )

    # Filter hospital admission data to include only diagnoses before medication start.
    bmi_integrated_df_hospitalization["AdmissionStartTimeFromBMI"] = bmi_integrated_df_hospitalization[start_col] - bmi_integrated_df_hospitalization[bmi_instant_col]
    bmi_integrated_df_hospitalization["IsAdmissionBeforeBMI"] = (bmi_integrated_df_hospitalization["AdmissionStartTimeFromBMI"] <= pd.Timedelta("0 days"))
    bmi_integrated_df_hospitalization = bmi_integrated_df_hospitalization[bmi_integrated_df_hospitalization["IsAdmissionBeforeBMI"]]
    print("Integrated hospitalizations tables:", bmi_integrated_df_hospitalization.shape, bmi_integrated_df_hospitalization[patientid_col].nunique())

    # Filter hospital admission data to include only columns needed for the analysis
    stats = bmi_integrated_df_hospitalization.loc[:,[hosp_key_col, patientid_col, bmi_instant_col, "IsCurrentlyHospitalized", start_col, end_col, hosp_service_col]]

    # Calculate the number of hospitalization days for each admission
    stats["HospitalizationDuration"] = (stats[end_col] - stats[start_col]).dt.days

    # Sort the table based on patient and hospitalization date
    stats = stats.sort_values(by=[patientid_col, start_col])

    # number each admission based on 1st, 2nd, 3rd etc
    stats["AdmissionNumber"] = stats.groupby(patientid_col).cumcount() + 1
    print("stats", stats.shape, stats[patientid_col].nunique())

    stats = stats.groupby([patientid_col, bmi_instant_col]).agg(
        IsCurrentlyHospitalized=("IsCurrentlyHospitalized", "max"),
        NumberOfTotalHospitalizations=(hosp_key_col, "size"),
        TotalHospitalizationDays=("HospitalizationDuration", "sum")
    ).reset_index()

    bmi_integrated_df_hospitalization = pd.merge(bmi_integrated_df, stats, on=[patientid_col, bmi_instant_col], how="left")

    bmi_integrated_df_hospitalization['IsCurrentlyHospitalized'] = bmi_integrated_df_hospitalization['IsCurrentlyHospitalized'].fillna(False)
    bmi_integrated_df_hospitalization['NumberOfTotalHospitalizations'] = bmi_integrated_df_hospitalization['NumberOfTotalHospitalizations'].fillna(0)
    bmi_integrated_df_hospitalization['TotalHospitalizationDays'] = bmi_integrated_df_hospitalization['TotalHospitalizationDays'].fillna(0)

    return bmi_integrated_df_hospitalization


