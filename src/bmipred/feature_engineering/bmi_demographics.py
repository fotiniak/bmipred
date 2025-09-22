#!/usr/bin/env python3

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from typing import List, Dict
warnings.filterwarnings('ignore') # deactivate warnings


def filter_adults(df: pd.DataFrame, 
                  min_age: int = 18, 
                  max_age: int = 120, 
                  bmi_instant_col: str = "CreateInstant",
                  birth_date_col: str = "BirthDate") -> pd.DataFrame:
    """Filter df to include only adults (Age >= min_age)."""
    df["healthAssesment_age"] = ((df[bmi_instant_col] - df[birth_date_col]) / pd.Timedelta(days=365.25)).astype(float)
    df = df[(df["healthAssesment_age"] >= min_age)]
    df = df[(df["healthAssesment_age"] <= max_age)]
    return df

def recalculate_bmi(df: pd.DataFrame, 
                    min_height_cm: int = 100, 
                    max_height_cm: int = 220, 
                    patientid_col: str = "PatientDurableKey",
                    heightcm_col: str = "HeightInCentimeters",
                    weightg_col: str = "WeightInGrams") -> pd.DataFrame:
    """Recalculate BMI based on median height."""
    df["HeightInCentimeters_median"] = df.groupby(patientid_col)[heightcm_col].transform("median") # median height per patient
    # Filter out unrealistic heights
    df = df[(df["HeightInCentimeters_median"] >= min_height_cm) & (df["HeightInCentimeters_median"] <= max_height_cm)]
    df["BodyMassIndex_recalc"] = df[weightg_col] * 0.001 / (df["HeightInCentimeters_median"] * 0.01) ** 2
    return df[df["BodyMassIndex_recalc"].notna()]

def filter_outliers(df: pd.DataFrame, 
                    lower_quantile: float = 0.001, 
                    upper_quantile: float = 0.999) -> pd.DataFrame:
    """Filter top 0.1% lower and higher values of recalculated BMI."""
    lower_percentile = df["BodyMassIndex_recalc"].quantile(lower_quantile)
    higher_percentile = df["BodyMassIndex_recalc"].quantile(upper_quantile)
    return df[(df["BodyMassIndex_recalc"] > lower_percentile) & (df["BodyMassIndex_recalc"] < higher_percentile)]

def filter_min_bmi_measurements(df: pd.DataFrame, 
                                min_bmi_measurements: int = 1,
                                patientid_col: str = "PatientDurableKey") -> pd.DataFrame:
    """Filter df to include only patients with at least min_bmi_measurements BMI measurements."""
    if min_bmi_measurements is not None:
        df["healthAssesment_count"] = df.groupby(patientid_col)["BodyMassIndex_recalc"].transform("count")
        return df[df["healthAssesment_count"] >= min_bmi_measurements]
    return df

def calculate_bmi_features(df: pd.DataFrame, 
                           patientid_col: str = "PatientDurableKey",
                           bmi_instant_col: str = "CreateInstant",
                           weightg_col: str = "WeightInGrams") -> pd.DataFrame:
    """Calculate diffs in bmi from previous measurement to the next."""
    df_sorted = df.sort_values(by=[patientid_col, bmi_instant_col], ascending=True)
    df_sorted["bmi_diff_from_previous"] = df_sorted.groupby(patientid_col)["BodyMassIndex_recalc"].diff() # if positive it increased if negative it decreased
    df_sorted["bmi_diff_from_next"] = df_sorted.groupby(patientid_col)["BodyMassIndex_recalc"].diff(-1).mul(-1) # positive if they gain weight negative if they loose
    df_sorted["weight_diff_from_next"] = df_sorted.groupby(patientid_col)[weightg_col].diff(-1).mul(-1) # positive if they gain weight negative if they loose
    df_sorted["bmi_diff_timepass_from_next"] = df_sorted.groupby(patientid_col)[bmi_instant_col].diff(-1).mul(-1) # positive because the next date is always bigger
    df_sorted["bmi_diff_timepass_from_previous"] = df_sorted.groupby(patientid_col)[bmi_instant_col].diff() # from current substracts the previous so it is a positive number

    # Group by PatientDurableKey and calculate the expanding variance of BMI including the current timepoint
    df_sorted['BMI_Variance_History'] = df_sorted.groupby(patientid_col)["BodyMassIndex_recalc"].expanding().var().reset_index(level=0, drop=True)
    
    # Calculate cumulative BMI difference from the baseline (up to the current time point)
    df_sorted['bmi_diff_from_first_measurement'] = df_sorted.groupby(patientid_col)['BodyMassIndex_recalc'].transform(lambda x: x - x.iloc[0])

    # Calculate cumulative percentage change in BMI from the baseline (up to the current time point)
    df_sorted['bmi_diff_from_first_measurement_perc'] = df_sorted.groupby(patientid_col)['BodyMassIndex_recalc'].transform(lambda x: ((x - x.iloc[0]) / x.iloc[0]) * 100)

    # Calculate maximum BMI increase (up to the current time point)
    df_sorted['max_bmi_difference'] = (
        df_sorted.groupby(patientid_col)['BodyMassIndex_recalc']
        .apply(lambda x: x.diff().expanding().max())
        .reset_index(level=0, drop=True)
    )
    
    # Calculate maximum BMI decrease (up to the current time point)
    df_sorted['min_bmi_difference'] = (
        df_sorted.groupby(patientid_col)['BodyMassIndex_recalc']
        .apply(lambda x: x.diff().expanding().min())
        .reset_index(level=0, drop=True)
    )

    # Classification of BMI changes
    df_sorted['BodyMassIndex_classification'] = df_sorted['bmi_diff_from_next'].apply(lambda x: 1 if pd.notna(x) and x > 0 else (0 if pd.notna(x) else pd.NA)) # 1: weight gain, 0: weight loss or no weight change
    
    # Transform the datetime dt date data...
    df_sorted['bmi_diff_timepass_from_next'] = df_sorted['bmi_diff_timepass_from_next'].dt.days
    df_sorted['bmi_diff_timepass_from_previous'] = df_sorted['bmi_diff_timepass_from_previous'].dt.days

    # Calculate the Rate of BMI change
    df_sorted["RateOfBMIChange"] = df_sorted["bmi_diff_from_next"]/df_sorted["bmi_diff_timepass_from_next"]
    df_sorted['RateOfBMIChange_classification'] = df_sorted['RateOfBMIChange'].apply(lambda x: 1 if pd.notna(x) and x > 0 else (0 if pd.notna(x) else pd.NA))

    return df_sorted


def map_smoking_status(df: pd.DataFrame, 
                       smoking_status_col: str = "SmokingStatus") -> pd.DataFrame:
    """Map smoking status to numerical values."""
    smoking_status_mapping = {
        'Hver dag': 'Frequently',
        'Storryger': 'Frequently',
        'Aktuel hverdagsryger': 'Frequently',
        'Nogle dage': 'Rarely',
        'Lejlighedsvis ryger': 'Rarely',
        'Aldrig': 'Never',
        'Aldrig været ryger': 'Never',
        'Tidligere': 'Previously',
        'Tidligere ryger': 'Previously',
        'Udsat for passiv rygning - aldrig været ryger': 'Passive',
        'Ryger, aktuel status ukendt': 'Frequently'  # Optional if you want to handle the unknown case
    }
    df[smoking_status_col] = df[smoking_status_col].map(smoking_status_mapping)
    return df


def map_marital_status(df: pd.DataFrame,
                       marital_status_col: str = "MaritalStatus") -> pd.DataFrame:
    """Map marital status to numerical values."""
    marital_status_mapping = {
        'Ugift': 'Single',
        'Fraskilt': 'Separated',
        'Gift': 'Together',
        'Død': 'Widowed',
        'Enke/enkemand': 'Widowed',
        'Opbrudt partnerskab': 'Separated',
        'Registreret partnerskab': 'Together',
        'Længst levende partner': 'Widowed'
    }

    df[marital_status_col] = df[marital_status_col].map(marital_status_mapping)
    return df


def map_sex(df: pd.DataFrame, 
            sex_col: str = "Sex") -> pd.DataFrame:
    """Map sex to numerical values."""
    sex_mapping = {
        'Mand': 0,
        'Kvinde': 1
    }
    df[sex_col] = df[sex_col].map(sex_mapping)
    return df

