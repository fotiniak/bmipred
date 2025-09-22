#!/usr/bin/env python3
# src/bmipred/feature_engineering/bmi_medication.py

import pandas as pd
import numpy as np

def integrate_family_history(family_history: pd.DataFrame,
                             bmi_integrated_df: pd.DataFrame,
                             patientid_col: str = "PatientDurableKey",
                             bmi_instant_col: str = "CreateInstant",
                             relation_col: str = "Relation",
                             medical_cond_col: str = "MedicalCondition") -> pd.DataFrame:

    # Preprocess family_history
    bmi_integrated_df[bmi_instant_col] = pd.to_datetime(bmi_integrated_df[bmi_instant_col])
    print("BMI table dimensions:", bmi_integrated_df.shape, bmi_integrated_df[patientid_col].nunique())

    # Filter family history data to include only relevant patients and process medical conditions.
    family_history = family_history[family_history[patientid_col].isin(bmi_integrated_df[patientid_col])]
    family_history = family_history[[patientid_col, relation_col, medical_cond_col]]
    family_history.dropna(subset=[medical_cond_col], inplace=True)
    family_history[medical_cond_col] = family_history[medical_cond_col].str.replace(r'[^a-zA-Z0-9]', '', regex=True)

    # Pivot family history data to have one column per medical condition.
    dummies = pd.get_dummies(family_history[medical_cond_col], prefix="Relative_with")
    family_history = pd.concat([family_history, dummies], axis=1)
    family_history.drop([relation_col, medical_cond_col], axis=1, inplace=True)
    family_history = family_history.groupby(patientid_col).agg("max")

    integrated_data = pd.merge(bmi_integrated_df, family_history, on=[patientid_col], how="left")

    # Fill missing values with 0 for columns starting with the given prefix.
    for col in integrated_data.columns:
        if col.startswith("Relative_with"):
            integrated_data[col] = integrated_data[col].fillna(False)

    return integrated_data


