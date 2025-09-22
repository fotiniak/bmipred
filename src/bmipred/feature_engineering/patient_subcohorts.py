#!/usr/bin/env python3
# src/bmipred/feature_engineering/patient_subcohorts.py

import pandas as pd
import numpy as np


def create_patient_subcohorts(
    df: pd.DataFrame,
    medication: pd.DataFrame,
    olanzapine_info: pd.DataFrame,
    patientid_col: str = "PatientDurableKey",
    sex_col: str = "Sex",
    atc_col: str = "ATC",
    bmi_instant_col: str = "CreateInstant",
    *,
    olz_mode: str = "relevant",  # "relevant" (-100/+1 days window) or "while_on" (strictly on meds)
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns 5 DataFrames that are subcohorts of the input df depending on olanzapine medication status:
      (cohort_wot_olanzapine_first,
       cohort_wot_olanzapine_last,
       cohort_on_olanzapine_first,
       cohort_on_olanzapine_last,
       cohort_on_olanzapine_before)
    """

    # Filter the data appropriately (remove observations where next BMI is unknown)
    df = df.copy()
    df.dropna(subset=["bmi_diff_from_next"], inplace=True)  # removing the last measurements with NA for next BMI

    # Correcting the column RateOfBMIChange because of 0 division
    df["RateOfBMIChange"] = df["bmi_diff_from_next"] / df["bmi_diff_timepass_from_next"]
    df = df[(df["RateOfBMIChange"] > -0.6) & (df["RateOfBMIChange"] < 0.6)]
    df["RateOfBMIChange_month"] = df["RateOfBMIChange"] * 30
    df["RateOfBMIChange_year"] = df["RateOfBMIChange"] * 365
    df["RateOfBMIChange_classification"] = df["RateOfBMIChange"].apply(lambda x: 1 if x > 0 else 0)

    # Choose the target variable # Create it if necessary
    df["bmi_diff_percent"] = (
        ((df["BodyMassIndex_recalc"] + df["bmi_diff_from_next"]) - df["BodyMassIndex_recalc"])
        / df["BodyMassIndex_recalc"]
    ) * 100
    df["bmi_increase_over5percent"] = df["bmi_diff_percent"] > 5
    # df["bmi_increase_over10percent"] = df["bmi_diff_percent"] > 10

    df = df[df["bmi_diff_timepass_from_next"] >= 1]
    print(
        "Observations/Features:",
        df.shape,
        "Number of unique patient ids:",
        df[patientid_col].nunique(),
    )
    print(
        "The number of unique patients stratified by sex are:",
        df[[sex_col, patientid_col]].drop_duplicates().value_counts(sex_col),
    )

    # Identify patients ever on olanzapine (ATC N05AH03)
    olanzapine_ids = medication.loc[medication[atc_col] == "N05AH03", patientid_col].unique()

    # Subcohort without olanzapine (first/last)
    cohort_wot_olanzapine = df[~df[patientid_col].isin(olanzapine_ids)]
    cohort_wot_olanzapine = cohort_wot_olanzapine.sort_values(
        by=[patientid_col, bmi_instant_col], ascending=[True, True]
    )
    cohort_wot_olanzapine_first = cohort_wot_olanzapine.drop_duplicates(subset=[patientid_col], keep="first")
    cohort_wot_olanzapine_last = cohort_wot_olanzapine.drop_duplicates(subset=[patientid_col], keep="last")

    # Subcohort on/around olanzapine at BMI timestamp
    cohort_on_olanzapine = pd.merge(
        df[df[patientid_col].isin(olanzapine_ids)],
        olanzapine_info,
        on=[patientid_col, bmi_instant_col],
        how="left",
    )

    # Inclusion mode
    if olz_mode == "while_on":
        # strictly while patient is on medication
        cohort_on_olanzapine = cohort_on_olanzapine[cohort_on_olanzapine["BMI_WhileOnMeds"] == True]
    else:
        # flexible window (-100/+1 days) around the exposure
        cohort_on_olanzapine = cohort_on_olanzapine[cohort_on_olanzapine["BMI_relevant_to_olanzapine"] == True]

    print(cohort_on_olanzapine.shape, cohort_on_olanzapine[patientid_col].nunique())

    # Extra variables (same as your scripts)
    cohort_on_olanzapine["NumberOfDaysOnMedication"] = (
        cohort_on_olanzapine["BMI_distance_from_StartInstant"] + cohort_on_olanzapine["bmi_diff_timepass_from_next"]
    )
    cohort_on_olanzapine["NumberOfDaysOffMedication"] = (
        cohort_on_olanzapine["BMI_distance_from_StartInstant"]
        + cohort_on_olanzapine["bmi_diff_timepass_from_next"]
        - cohort_on_olanzapine["MedicationDuration"]
    )
    cohort_on_olanzapine["StillOnOlanzapine"] = (
        cohort_on_olanzapine["bmi_diff_timepass_from_next"]
        < (cohort_on_olanzapine["BMI_distance_from_DiscontinuedInstant"] * -1)
    )
    cohort_on_olanzapine.loc[
        cohort_on_olanzapine["StillOnOlanzapine"] == True, "NumberOfDaysOffMedication"
    ] = 0  # if StillOnOlanzapine==True, set NumberOfDaysOffMedication to 0 (no days off medication)
    cohort_on_olanzapine.loc[
        cohort_on_olanzapine["StillOnOlanzapine"] == False, "NumberOfDaysOnMedication"
    ] = cohort_on_olanzapine["MedicationDuration"]  # if StillOnOlanzapine==False, set NumberOfDaysOnMedication to MedicationDuration
    print(cohort_on_olanzapine.shape)

    # Filter cases where both measurements might be before the patients start Olanzapine:
    cohort_on_olanzapine = cohort_on_olanzapine[
        ~(
            (cohort_on_olanzapine["BMI_WhileOnMeds"] == False)
            & (cohort_on_olanzapine["StillOnOlanzapine"] == False)
        )
    ]

    cohort_on_olanzapine = cohort_on_olanzapine.sort_values(
        by=[patientid_col, bmi_instant_col], ascending=[True, True]
    )

    # Before starting olanzapine
    cohort_on_olanzapine["BMI_before_olanzapine"] = (
        (cohort_on_olanzapine["BMI_relevant_to_olanzapine"] == True)
        & (cohort_on_olanzapine["BMI_WhileOnMeds"] == False)
    )
    cohort_on_olanzapine_before = cohort_on_olanzapine[cohort_on_olanzapine["BMI_before_olanzapine"] == True]
    print(cohort_on_olanzapine_before.shape, cohort_on_olanzapine_before[patientid_col].nunique())

    # After starting olanzapine (first/last)
    cohort_on_olanzapine_after = cohort_on_olanzapine[cohort_on_olanzapine["BMI_WhileOnMeds"] == True]
    cohort_on_olanzapine_first = cohort_on_olanzapine_after.drop_duplicates(subset=[patientid_col], keep="first")
    print(cohort_on_olanzapine_first.shape, cohort_on_olanzapine_first[patientid_col].nunique())
    cohort_on_olanzapine_last = cohort_on_olanzapine_after.drop_duplicates(subset=[patientid_col], keep="last")
    print(cohort_on_olanzapine_last.shape, cohort_on_olanzapine_last[patientid_col].nunique())

    return (
        cohort_wot_olanzapine_first,
        cohort_wot_olanzapine_last,
        cohort_on_olanzapine_first,
        cohort_on_olanzapine_last,
        cohort_on_olanzapine_before,
    )


def finalize_cohort(
    df: pd.DataFrame,
    *,
    target_mode: str = "percent5",  # "percent5", "slope_sign", or "regression_diff"
    target_col: str = "bmi_increase_over5percent",
    atc_col: str = "ATC",
) -> pd.DataFrame:
    """
    - For "percent5": renames target_col -> 'target'
    - For "slope_sign": target = 1: RateOfBMIChange > 0, 0: RateOfBMIChange <= 0
    - For "regression_diff": target = bmi_diff_from_next
    - Creates 'Smoking' from SmokingStatus_* dummies
    - Drops marital columns and fixed set of feature columns (errors='ignore')
    """
    out = df.copy()

    # --- target creation/selection ---
    if target_mode == "slope_sign":
        if "RateOfBMIChange" not in out.columns:
            out["RateOfBMIChange"] = out["bmi_diff_from_next"] / out["bmi_diff_timepass_from_next"]
        out = out[(out["RateOfBMIChange"] > -0.6) & (out["RateOfBMIChange"] < 0.6)]
        out = out[out["bmi_diff_timepass_from_next"] >= 1]
        out["target"] = (out["RateOfBMIChange"] > 0).astype(int)

    elif target_mode == "regression_diff":
        if "bmi_diff_from_next" not in out.columns:
            raise KeyError("Column 'bmi_diff_from_next' is required for regression_diff mode.")
        out["target"] = out["bmi_diff_from_next"].astype(float)
        out["RateOfBMIChange"] = out["bmi_diff_from_next"] / out["bmi_diff_timepass_from_next"]
        out = out[(out["RateOfBMIChange"] > -0.6) & (out["RateOfBMIChange"] < 0.6)]
        out = out[out["bmi_diff_timepass_from_next"] >= 1]

    else:  # "percent5" default
        if target_col in out.columns and "target" not in out.columns:
            out = out.rename(columns={target_col: "target"})

    # --- columns to drop/aggregate (same lists you used) ---
    marital_columns = [c for c in out.columns if c.startswith("MaritalStatus")]
    smoking_candidates = [
        "SmokingStatus_Rarely",
        "SmokingStatus_Previously",
        "SmokingStatus_Passive",
        "SmokingStatus_Frequently",
        "SmokingStatus_Never",
    ]
    smoking_cols = [c for c in smoking_candidates if c in out.columns]

    columns_to_remove = [
        "RateOfBMIChange",
        "RateOfBMIChange_month",
        "RateOfBMIChange_year",
        "RateOfBMIChange_classification",
        "bmi_diff_from_next",
        "bmi_percent_change",
        "bmi_diff_percent",
        "bmi_diff_from_first_measurement",
        "BodyMassIndex_diff_last_first",
        "StrengthNumeric_daily",
        "StrengthNumeric",
        "DailyDosage_weighted_mean",
        "MedicationDuration",
        atc_col,
        "NumberOfDaysOffMedication",
        "BMI_distance_from_DiscontinuedInstant",
        "WeightInGrams",
        "HeightInCentimeters_median",
        "BodySurfaceArea",
        "BMI_relevant_to_olanzapine",
        "BMI_WhileOnMeds",
        *smoking_cols,
    ]

    # Aggregated Smoking flag
    out["Smoking"] = 0
    if smoking_cols:
        out["Smoking"] = out[smoking_cols].fillna(0).astype(int).any(axis=1).astype(int)

    # Drop columns
    out = out.drop(columns=marital_columns, errors="ignore")
    out = out.drop(columns=columns_to_remove, errors="ignore")
    return out
