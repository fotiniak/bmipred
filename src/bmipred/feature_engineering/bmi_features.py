#!/usr/bin/env python3

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore') # deactivate warnings



def filter_adults(df: pd.DataFrame, 
                  min_age: int = 18, 
                  max_age: int = 120, 
                  instant_col: str = "CreateInstant",
                  birth_date_col: str = "BirthDate") -> pd.DataFrame:
    # Filter df to include only adults (Age >= min_age).
    df["healthAssesment_age"] = ((df[instant_col] - df[birth_date_col]) / pd.Timedelta(days=365.25)).astype(float)
    df = df[(df["healthAssesment_age"] >= min_age)]
    df = df[(df["healthAssesment_age"] <= max_age)]
    return df



def recalculate_bmi(df: pd.DataFrame, 
                    min_height_cm: int = 100, 
                    max_height_cm: int = 220, 
                    patientid_col: str = "PatientDurableKey",
                    height_col: str = "HeightInCentimeters",
                    weight_col: str = "WeightInGrams") -> pd.DataFrame:
    # Recalculate BMI based on median height.
    df["HeightInCentimeters_median"] = df.groupby(patientid_col)[height_col].transform("median") # median height per patient
    # Filter out unrealistic heights
    df = df[(df["HeightInCentimeters_median"] >= min_height_cm) & (df["HeightInCentimeters_median"] <= max_height_cm)]
    df["BodyMassIndex_recalc"] = df[weight_col] * 0.001 / (df["HeightInCentimeters_median"] * 0.01) ** 2
    return df[df["BodyMassIndex_recalc"].notna()]



def deduplicate_bmi(df: pd.DataFrame,
                    patientid_col: str = "PatientDurableKey",
                    instant_col: str = "CreateInstant",
                    bmi_col: str = "BodyMassIndex_recalc",) -> pd.DataFrame:
    print("Rows before deduplication:", len(df)) 
    df = (df.sort_values(by=[patientid_col, instant_col], ascending=True, na_position="last")
          .dropna(subset=[instant_col, bmi_col])
          .drop_duplicates())
    df = df.groupby([patientid_col, instant_col], as_index=False).agg("first")
    print("Rows after deduplication and aggregation:", len(df))
    return df



def filter_outliers(df: pd.DataFrame, 
                    lower_quantile: float = 0.001, 
                    upper_quantile: float = 0.999) -> pd.DataFrame:
    # Filter top 0.1% lower and higher values of recalculated BMI.
    lower_percentile = df["BodyMassIndex_recalc"].quantile(lower_quantile)
    higher_percentile = df["BodyMassIndex_recalc"].quantile(upper_quantile)
    return df[(df["BodyMassIndex_recalc"] > lower_percentile) & (df["BodyMassIndex_recalc"] < higher_percentile)]



def filter_min_bmi_measurements(df: pd.DataFrame, 
                                min_bmi_measurements: int = 1,
                                patientid_col: str = "PatientDurableKey") -> pd.DataFrame:
    # Filter df to include only patients with at least min_bmi_measurements BMI measurements.
    if min_bmi_measurements is not None:
        df["healthAssesment_count"] = df.groupby(patientid_col)["BodyMassIndex_recalc"].transform("count")
        return df[df["healthAssesment_count"] >= min_bmi_measurements]
    return df



def calculate_bmi_features(df: pd.DataFrame,
                           patientid_col: str = "PatientDurableKey",
                           instant_col: str = "CreateInstant",
                           weight_col: str = "WeightInGrams",) -> pd.DataFrame:
    
    df_sorted = df.sort_values(by=[patientid_col, instant_col], ascending=True)
    g = df_sorted.groupby(patientid_col, sort=False)

    # Count of prior BMI measurements (0-indexed: 0 means first measurement)
    df_sorted["n_bmi_before"] = g.cumcount()

    # Diffs from previous
    df_sorted["days_since_prev_bmi"] = g[instant_col].diff().dt.total_seconds() / 86400.0
    df_sorted["bmi_diff_from_previous"] = g["BodyMassIndex_recalc"].diff()

    # Diffs from next - target variables for prediction
    df_sorted["days_to_next_bmi"] = g[instant_col].diff(-1).mul(-1).dt.total_seconds() / 86400.0
    df_sorted["weight_diff_from_next"] = g[weight_col].diff(-1).mul(-1)
    df_sorted["bmi_diff_from_next"] = g["BodyMassIndex_recalc"].diff(-1).mul(-1)
    df_sorted["bmi_diff_from_next_pct"] = (df_sorted["bmi_diff_from_next"] / df_sorted["BodyMassIndex_recalc"]) * 100.0
    df_sorted["bmi_diff_from_next_5pct"] = (df_sorted["bmi_diff_from_next_pct"] >= 5.0).astype("boolean")

    # Expanding variance
    df_sorted["BMI_Variance_History"] = (g["BodyMassIndex_recalc"].expanding().var().reset_index(level=0, drop=True))

    # Cumulative diff from first measurement (absolute and percentage)
    df_sorted["bmi_diff_from_first_measurement"] = g["BodyMassIndex_recalc"].transform(lambda x: x - x.iloc[0])
    df_sorted["bmi_diff_from_first_measurement_perc"] = g["BodyMassIndex_recalc"].transform(lambda x: ((x - x.iloc[0]) / x.iloc[0]) * 100)

    # Expanding max and min single-step BMI change
    df_sorted["max_bmi_difference"] = (
        g["BodyMassIndex_recalc"]
        .apply(lambda x: x.diff().expanding().max())
        .reset_index(level=0, drop=True)
    )
    df_sorted["min_bmi_difference"] = (
        g["BodyMassIndex_recalc"]
        .apply(lambda x: x.diff().expanding().min())
        .reset_index(level=0, drop=True)
    )

    # BMI slope previous→current (BMI points per year)
    df_sorted["bmi_slope_prev_per_year"] = (
        df_sorted["bmi_diff_from_previous"] / (df_sorted["days_since_prev_bmi"] / 365.25)
    )
    df_sorted.loc[df_sorted["days_since_prev_bmi"] <= 0, "bmi_slope_prev_per_year"] = np.nan
    df_sorted["bmi_slope_prev_per_year_pos"] = (df_sorted["bmi_slope_prev_per_year"] > 0).astype(int)

    # Classification of BMI direction to next measurement
    df_sorted["BodyMassIndex_classification"] = df_sorted["bmi_diff_from_next"].apply(
        lambda x: 1 if pd.notna(x) and x > 0 else (0 if pd.notna(x) else pd.NA)
    )

    # Rate of BMI change (BMI points per day)
    df_sorted["RateOfBMIChange"] = df_sorted["bmi_diff_from_next"] / df_sorted["days_to_next_bmi"]
    df_sorted["RateOfBMIChange_classification"] = df_sorted["RateOfBMIChange"].apply(
        lambda x: 1 if pd.notna(x) and x > 0 else (0 if pd.notna(x) else pd.NA)
    )

    return df_sorted



def calculate_bmi_trajectory(df: pd.DataFrame,
                             patientid_col: str = "PatientDurableKey",
                             instant_col: str = "CreateInstant",
                             bmi_col: str = "BodyMassIndex_recalc",) -> pd.DataFrame:
    # Fits an expanding OLS linear regression of BMI ~ time for each patient.
    # bmi_slope_hist_per_year: BMI points per year up to current row
    # bmi_intercept_hist: predicted BMI at first measurement time
    # bmi_trend_r2_hist: R² of the expanding fit (requires >= 2 points)
    g = df.groupby(patientid_col, sort=False)

    # x-axis: years since first measurement for each patient
    df["t_years"] = (
        (df[instant_col] - g[instant_col].transform("first"))
        .dt.total_seconds() / (365.25 * 24 * 3600)
    )

    # Expanding sufficient statistics
    df["_n"]      = g.cumcount() + 1
    df["_sum_x"]  = df["t_years"].groupby(df[patientid_col]).cumsum()
    df["_sum_y"]  = df[bmi_col].groupby(df[patientid_col]).cumsum()
    df["_sum_x2"] = df["t_years"].pow(2).groupby(df[patientid_col]).cumsum()
    df["_sum_xy"] = (df["t_years"] * df[bmi_col]).groupby(df[patientid_col]).cumsum()
    df["_sum_y2"] = df[bmi_col].pow(2).groupby(df[patientid_col]).cumsum()

    num = df["_n"] * df["_sum_xy"] - df["_sum_x"] * df["_sum_y"]
    den = df["_n"] * df["_sum_x2"] - df["_sum_x"] ** 2
    syy = df["_n"] * df["_sum_y2"] - df["_sum_y"] ** 2

    df["bmi_slope_hist_per_year"] = np.where(
        (df["_n"] >= 2) & (den > 0), num / den, np.nan
    )
    # extra calculation to get the intercept of the expanding fit, which can be more interpretable than the slope alone
    df["bmi_intercept_hist"] = np.where(
        df["bmi_slope_hist_per_year"].notna(),
        (df["_sum_y"] - df["bmi_slope_hist_per_year"] * df["_sum_x"]) / df["_n"],
        np.nan
    )
    df["bmi_trend_r2_hist"] = np.where(
        (df["_n"] >= 2) & (den > 0) & (syy > 0),
        (num ** 2) / (den * syy),
        np.nan
    )

    df = df.drop(columns=["t_years", "_n", "_sum_x", "_sum_y", "_sum_x2", "_sum_xy", "_sum_y2"])
    return df



def map_smoking_status(df: pd.DataFrame, 
                       smoking_status_col: str = "SmokingStatus") -> pd.DataFrame:
    # Map smoking status to numerical values.
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



def map_sex(df: pd.DataFrame, 
            sex_col: str = "Sex") -> pd.DataFrame:
    # Map sex to numerical values.
    sex_mapping = {
        'Mand': 0,
        'Kvinde': 1
    }
    df[sex_col] = df[sex_col].map(sex_mapping)
    return df

'''
def map_marital_status(df: pd.DataFrame,
                       marital_status_col: str = "MaritalStatus") -> pd.DataFrame:
    # Map marital status to numerical values.
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
'''
