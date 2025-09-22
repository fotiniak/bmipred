#!/usr/bin/env python3
# src/bmipred/integrated_df.py


import pandas as pd
import numpy as np


def preprocess_olanzapine_features(bmi_demographics: pd.DataFrame,
                                   medication: pd.DataFrame,
                                   patientid_col: str = "PatientDurableKey",
                                   bmi_instant_col: str = "CreateInstant",
                                   atc_col: str = "ATC",
                                   start_col: str = "StartInstant",
                                   discontinued_col: str = "DiscontinuedInstant",
                                   dosage_col: str = "StrengthNumeric",
                                   dosage_daily_col: str = "StrengthNumeric_daily",
                                   ) -> pd.DataFrame:


    medication = medication[medication[patientid_col].isin(bmi_demographics[patientid_col].unique())]
    print(f"After filtering for patients with BMI records: {medication.shape}, Unique patients: {medication[patientid_col].nunique()}")

    # Merging BMI and medication tables
    df = pd.merge(bmi_demographics, medication, on=patientid_col, how="inner")
    print(f"After merging BMI and medications: {df.shape}, Unique patients: {df[patientid_col].nunique()}")
    print(df.shape)

    # Filtering the df for olanzapine medication
    df = df[df[atc_col]=="N05AH03"] # filter for olanzapine
    print(f"After filtering for Olanzapine: {df.shape}, Unique patients: {df[patientid_col].nunique()}, Unique CreateInstant Timestamp: {df[bmi_instant_col].nunique()}")

    # Calculate relevant timestamp variables
    df["BMI_relevant_to_olanzapine"] = ((df[bmi_instant_col] >= (df[start_col] - pd.Timedelta(days=100))) & (df[bmi_instant_col] <= (df[discontinued_col] + pd.Timedelta(days=1))))
    df["BMI_WhileOnMeds"] = ((df[bmi_instant_col] >= (df[start_col] - pd.Timedelta(days=1))) & (df[bmi_instant_col] <= (df[discontinued_col] + pd.Timedelta(days=1))))
    df = df[df["BMI_relevant_to_olanzapine"]==True]
    print(f"After filtering for Olanzapine: {df.shape}, Unique patients: {df[patientid_col].nunique()}, Unique CreateInstant: {df[bmi_instant_col].nunique()}")

    df["BMI_distance_from_StartInstant"] = (df[bmi_instant_col] - df[start_col]).dt.days
    df["BMI_distance_from_DiscontinuedInstant"] = (df[bmi_instant_col] - df[discontinued_col]).dt.days
    #df = df[((df["BMI_distance_from_StartInstant"] >= 0) & (df["BMI_distance_from_DiscontinuedInstant"] <= 0))] # extra filtering in case we want to only keep BMIs taken while on medication

    df = df[[patientid_col, bmi_instant_col, atc_col, dosage_col, dosage_daily_col, start_col, discontinued_col,
                 "MedicationDuration", "DailyDosage_weighted_mean", "DailyDosage_weighted_mean_nafilled", 
                 "NumberOfUniqueMedicationsPerscribed_BeforeNow", "NumberOfTimesOnAntipsychotics",
                 "NumberOfTimesOnAntidepressants", "NumberOfTimesOnAnxiolytics", "TimesCurrentATCTaken_BeforeNow", 
                 "NumberOfOverlappingATC", "BMI_WhileOnMeds", "BMI_relevant_to_olanzapine","BMI_distance_from_StartInstant", 
                 "BMI_distance_from_DiscontinuedInstant"]]
    
    return df




def combine_olanzapine_info(integrated_data: pd.DataFrame,
                            olanzapine_info: pd.DataFrame,
                            patientid_col: str = "PatientDurableKey",
                            bmi_instant_col: str = "CreateInstant",
                            atc_col: str = "ATC",
                            dosage_col: str = "StrengthNumeric",
                            dosage_daily_col: str = "StrengthNumeric_daily",) -> pd.DataFrame:

    # Load the df - All patients with BMI (latest snapshot per PatientDurableKey)
    print("Observations/Features:", integrated_data.shape, "Number of unique patient ids:", integrated_data[patientid_col].nunique())
    print("Patients stratified by sex are:", integrated_data[["Sex", patientid_col]].drop_duplicates().value_counts("Sex"))

    # Filter the df appropriately (remove observations where NEXT BMI is unknown)
    integrated_data.dropna(subset=["bmi_diff_from_next"], inplace=True) # removing the last measurements with NA for next BMI

    # Correcting the column RateOfBMIChange because of 0 division
    integrated_data["RateOfBMIChange"] = integrated_data["bmi_diff_from_next"]/integrated_data["bmi_diff_timepass_from_next"]
    integrated_data = integrated_data[(integrated_data["RateOfBMIChange"]>-0.6) & (integrated_data["RateOfBMIChange"]<0.6)]
    integrated_data["RateOfBMIChange_classification"] = integrated_data["RateOfBMIChange"].apply(lambda x: 1 if x > 0 else 0)
    integrated_data = integrated_data[integrated_data["bmi_diff_timepass_from_next"]>=1] # at least one day appart between BMI measurements

    # Choose the target variable # Create it if necessary
    integrated_data["bmi_diff_percent"] = (((integrated_data["BodyMassIndex_recalc"]+integrated_data["bmi_diff_from_next"])-integrated_data["BodyMassIndex_recalc"])/integrated_data["BodyMassIndex_recalc"])*100
    integrated_data["bmi_diff_over_percent"] = integrated_data["bmi_diff_percent"] > 5

    print("Number of patients in each class:", integrated_data.value_counts("bmi_diff_over_percent"))

    # Merge the data with extra olanzapine specific info
    integrated_data = pd.merge(integrated_data, olanzapine_info, on=[patientid_col, bmi_instant_col], how="left")
    #integrated_data = integrated_data[integrated_data["BMI_WhileOnMeds"]==True]
    integrated_data = integrated_data[integrated_data["BMI_relevant_to_olanzapine"]==True]
    #integrated_data = integrated_data[integrated_data["olanzapine_medication"]==True]
    #integrated_data = integrated_data[integrated_data["ATC"]=="N05AH03"] # (26422, 476) 7646
    print("After merging integrated BMI data with olanzapine specific info:", integrated_data.shape, integrated_data[patientid_col].nunique())

    # Creating Extra Variables
    integrated_data["NumberOfDaysOnMedication"] = integrated_data["BMI_distance_from_StartInstant"] + integrated_data["bmi_diff_timepass_from_next"]
    integrated_data["NumberOfDaysOffMedication"] = integrated_data["BMI_distance_from_StartInstant"] + integrated_data["bmi_diff_timepass_from_next"] - integrated_data["MedicationDuration"]
    #integrated_data["NumberOfDaysOffMedication"] = integrated_data["bmi_diff_timepass_from_next"] - (integrated_data["BMI_distance_from_DiscontinuedInstant"]*-1) #laternative way of calculation
    integrated_data["StillOnOlanzapine"] = integrated_data["bmi_diff_timepass_from_next"] < (integrated_data["BMI_distance_from_DiscontinuedInstant"]*-1)
    # where StillOnOlanzapine is True, set NumberOfDaysOffMedication to 0
    integrated_data.loc[integrated_data["StillOnOlanzapine"] == True, "NumberOfDaysOffMedication"] = 0
    # where StillOnOlanzapine is False, set NumberOfDaysOnMedication to MedicationDuration
    integrated_data.loc[integrated_data["StillOnOlanzapine"] == False, "NumberOfDaysOnMedication"] = integrated_data["MedicationDuration"]
    print("After creating extra variables:", integrated_data.shape)

    # Filter cases where both measurements might be before the patient starts Olanzapine:
    integrated_data = integrated_data[~((integrated_data["BMI_WhileOnMeds"] == False) & (integrated_data["StillOnOlanzapine"] == False))]

    # Filtering based on duration of medication - Optional
    #integrated_data = integrated_data[integrated_data["MedicationDuration"]>=1]

    integrated_data = integrated_data.drop(columns=["RateOfBMIChange", "RateOfBMIChange_classification", "bmi_diff_from_next", "BodySurfaceArea",
                                            "bmi_percent_change", "bmi_diff_from_first_measurement", "BodyMassIndex_diff_last_first",
                                            "bmi_diff_percent", dosage_daily_col, dosage_col,"DailyDosage_weighted_mean",
                                            "MedicationDuration", atc_col, "BMI_distance_from_DiscontinuedInstant", "WeightInGrams", "HeightInCentimeters_median"], errors="ignore")
                                            #"BMI_distance_from_StartInstant", "NumberOfDaysOffMedication"], errors="ignore")

    # Drop columns that might not be useful for modeling/ were found to have high correlations with other variables
    marital_columns = [col for col in integrated_data.columns if col.startswith("MaritalStatus")]
    # drop both the fixed columns and the marital status columns
    integrated_data = integrated_data.drop(columns=marital_columns)

    target_col = "bmi_diff_over_percent"
    integrated_data.rename(columns={target_col:"target"}, inplace=True)

    # Keep the latest record for each patient before starting olanzapine up to 100 days before...
    integrated_data = integrated_data.sort_values(by=[patientid_col, bmi_instant_col], ascending=[True, True])
    integrated_data["BMI_before_olanzapine"] = ((integrated_data["BMI_relevant_to_olanzapine"]==True) & (integrated_data["BMI_WhileOnMeds"]==False))
    before_df = integrated_data[integrated_data["BMI_before_olanzapine"]==True]
    print(before_df.shape, before_df[patientid_col].nunique())
    before_df = integrated_data.drop_duplicates(subset=[patientid_col], keep="last")
    print("Number of patients in each class after applying all filters:", before_df.value_counts("target"))

    # Keep the first or last record for each patient...
    integrated_data = integrated_data.sort_values(by=[patientid_col, bmi_instant_col], ascending=[True, True])
    first_df = integrated_data[integrated_data["BMI_WhileOnMeds"]==True]
    first_df = first_df.drop_duplicates(subset=[patientid_col], keep="first")
    print("Number of patients in each class after applying all filters:", first_df.value_counts("target"))

    integrated_data = integrated_data.sort_values(by=[patientid_col, bmi_instant_col], ascending=[True, True])
    last_df = integrated_data[integrated_data["BMI_WhileOnMeds"]==True]
    last_df = last_df.drop_duplicates(subset=[patientid_col], keep="last")
    print("Number of patients in each class after applying all filters:", last_df.value_counts("target"))
    
    print("Preprocessing of the integrated df table finished successfully!")

    return before_df, first_df, last_df
