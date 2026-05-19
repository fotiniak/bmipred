#!/usr/bin/env python3
# Script to combine preprocessed data tables into a single cohort.
# All configuration parameters are defined at the beginning of the script.

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore') # deactivate warnings

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ==================== CONFIGURATION PARAMETERS ====================

BASE_DIR = Path(__file__).parent.parent / "data" / "preprocessed"

TABLES_TO_COMBINE = {
    "health_assessment": {
        "input_path": BASE_DIR / "health_assessment_features.parquet",
        "column_names": {
            "patientid_col": "PatientDurableKey",
            "instant_col": "CreateInstant", # timestamp column name
        },
    },
    "diagnosis": {
        "input_path": BASE_DIR / "diagnosis_collapsed_features.parquet",
        "column_names": {
            "patientid_col": "PatientDurableKey",
            "code_col": "SKSCode",
            "start_col": "DiagnosisStartDate", # start timestamp column name
            "end_col": "DiagnosisEndDate", # end timestamp column name
        },
    },
    "medication": {
        "input_path": BASE_DIR / "medication_collapsed_features.parquet",
        "column_names": {
            "patientid_col": "PatientDurableKey",
            "code_col": "ATC",
            "start_col": "StartInstant", # start timestamp column name
            "end_col": "DiscontinuedInstant", # end timestamp column name
        },
    },
    "hospitalization": {
        "input_path": BASE_DIR / "hospital_admission_features.parquet",
        "column_names": {
            "patientid_col": "PatientDurableKey",
            "start_col": "InpatientAdmissionInstant", # admission timestamp column name
            "end_col": "DischargeInstant", # discharge timestamp column name
        },
    },
    "lab_results": {
        "input_path": BASE_DIR / "lab_results_features.parquet",
        "column_names": {
            "patientid_col": "PatientDurableKey",
            "instant_col": "CollectionInstant", # timestamp column name
            "lab_test_col": "Labanalysis",
            "flag_col": "Flag",
            "value_col": "Value",
            "is_abnormal_col": "IsAbnormal",
        },
    }
}

# ==================== END CONFIGURATION =======================  

def main():
    
    # Cohort 1: Olanzapine initiation with follow-up 30-180 days after start
    
    # Filter individuals that have been treated with olanzapine (ATC code N05AH03).
    medication = pd.read_parquet(TABLES_TO_COMBINE["medication"]["input_path"])
    cohort1 = medication[medication["ATC"]=="N05AH03"]
    print("Number of individuals treated with olanzapine:", cohort1.PatientDurableKey.nunique(), cohort1.shape)
    
    # Add past BMI - health assessment features
    health_assessment = pd.read_parquet(TABLES_TO_COMBINE["health_assessment"]["input_path"])
    cohort1 = pd.merge_asof(cohort1.sort_values(by=[TABLES_TO_COMBINE["medication"]["column_names"]["start_col"]], ascending=[True], na_position="last"),
                            health_assessment.sort_values(by=[TABLES_TO_COMBINE["health_assessment"]["column_names"]["instant_col"]], ascending=[True], na_position="last"),
                            left_on= [TABLES_TO_COMBINE["medication"]["column_names"]["start_col"]],
                            right_on= [TABLES_TO_COMBINE["health_assessment"]["column_names"]["instant_col"]],
                            by=[TABLES_TO_COMBINE["medication"]["column_names"]["patientid_col"]],
                            direction="backward",      # most recent previous row
                            tolerance=pd.Timedelta("90D"), # up to 90 days before medication start
                            allow_exact_matches=True)   # including at olanzapine start time

    cohort1["bmi_days_before_medication_start"] = cohort1["CreateInstant"] - cohort1["StartInstant"]
    cohort1 = cohort1[cohort1["bmi_days_before_medication_start"] <= pd.Timedelta("90D")] # This way we will remove NAs not matched
    print("Number of individuals with past BMI data:", cohort1.PatientDurableKey.nunique(), cohort1.shape)

    # Filter for the target variable - BMI at 30-180 days after olanzapine initiation.
    
    cohort1 = pd.merge_asof(cohort1.sort_values(by=[TABLES_TO_COMBINE["medication"]["column_names"]["start_col"]], ascending=[True], na_position="last"),
                            health_assessment.sort_values(by=[TABLES_TO_COMBINE["health_assessment"]["column_names"]["instant_col"]], ascending=[True], na_position="last"),
                            left_on= [TABLES_TO_COMBINE["medication"]["column_names"]["start_col"]],
                            right_on= TABLES_TO_COMBINE["health_assessment"]["column_names"]["instant_col"],
                            by=[TABLES_TO_COMBINE["medication"]["column_names"]["patientid_col"]],
                            direction="forward",      # most recent next row
                            tolerance=pd.Timedelta("180D"), # up to 180 days after medication start
                            allow_exact_matches=False,
                            suffixes=("", "_target"))
    
    cohort1["bmi_days_after_medication_start"] = cohort1[TABLES_TO_COMBINE["health_assessment"]["column_names"]["instant_col"] + "_target"] - cohort1[TABLES_TO_COMBINE["medication"]["column_names"]["start_col"]]
    cohort1 = cohort1[cohort1["bmi_days_after_medication_start"] >= pd.Timedelta("30D")] # 30D wash out period

    print("Number of individuals with past and future BMI data:", cohort1.PatientDurableKey.nunique(), cohort1.shape)

    # Add diagnosis info - SKS codes up to olanzapine initiation.
    diagnosis = pd.read_parquet(TABLES_TO_COMBINE["diagnosis"]["input_path"])

    cohort1 = pd.merge_asof(cohort1.sort_values(by=[TABLES_TO_COMBINE["medication"]["column_names"]["start_col"]], ascending=[True], na_position="last"),
                            diagnosis.sort_values(by=[TABLES_TO_COMBINE["diagnosis"]["column_names"]["start_col"]], ascending=[True], na_position="last"),
                            left_on= [TABLES_TO_COMBINE["medication"]["column_names"]["start_col"]],
                            right_on= [TABLES_TO_COMBINE["diagnosis"]["column_names"]["start_col"]],
                            by=[TABLES_TO_COMBINE["medication"]["column_names"]["patientid_col"]],
                            direction="backward",      # most recent previous row
                            allow_exact_matches=True)   # including at olanzapine start time
    
    # Add lab test info - up to olanzapine initiation.
    lab_results = pd.read_parquet(TABLES_TO_COMBINE["lab_results"]["input_path"])
    lab_results = lab_results[lab_results[TABLES_TO_COMBINE["lab_results"]["column_names"]["patientid_col"]].isin(cohort1[TABLES_TO_COMBINE["medication"]["column_names"]["patientid_col"]].unique())]
    lab_results[TABLES_TO_COMBINE["lab_results"]["column_names"]["flag_col"]] = lab_results[TABLES_TO_COMBINE["lab_results"]["column_names"]["flag_col"]].map({"Lav": 1, "Normal": 2, "Høj": 3,})
    
    lab_results_history = lab_results[[TABLES_TO_COMBINE["lab_results"]["column_names"]["patientid_col"], TABLES_TO_COMBINE["lab_results"]["column_names"]["instant_col"], "number_of_prev_tests", "number_of_prev_abnormal_tests", "days_since_last_test", "proportion_of_abnormal_tests"]].copy()
    lab_results_history.drop_duplicates(inplace=True)
    
    lab_results = (lab_results
        .sort_values([TABLES_TO_COMBINE["lab_results"]["column_names"]["patientid_col"], TABLES_TO_COMBINE["lab_results"]["column_names"]["instant_col"]])
        .pivot_table(
            index=[TABLES_TO_COMBINE["lab_results"]["column_names"]["patientid_col"], TABLES_TO_COMBINE["lab_results"]["column_names"]["instant_col"]],
            columns=TABLES_TO_COMBINE["lab_results"]["column_names"]["lab_test_col"],
            values=TABLES_TO_COMBINE["lab_results"]["column_names"]["flag_col"],
            aggfunc="last")
        .groupby(level=0).ffill() # forward fill in with the last known measurement
        .add_prefix("Labanalysis_") 
        .reset_index())
    
    lab_results.drop([TABLES_TO_COMBINE["lab_results"]["column_names"]["lab_test_col"],
                      TABLES_TO_COMBINE["lab_results"]["column_names"]["flag_col"],
                      TABLES_TO_COMBINE["lab_results"]["column_names"]["value_col"], 
                      TABLES_TO_COMBINE["lab_results"]["column_names"]["is_abnormal_col"]], axis=1, inplace=True, errors="ignore")

    cohort1 = pd.merge_asof(cohort1.sort_values(by=[TABLES_TO_COMBINE["medication"]["column_names"]["start_col"]], ascending=[True], na_position="last"),
                            lab_results.sort_values(by=[TABLES_TO_COMBINE["lab_results"]["column_names"]["instant_col"]], ascending=[True], na_position="last"),
                            left_on= [TABLES_TO_COMBINE["medication"]["column_names"]["start_col"]],
                            right_on= [TABLES_TO_COMBINE["lab_results"]["column_names"]["instant_col"]],
                            by=[TABLES_TO_COMBINE["medication"]["column_names"]["patientid_col"]],
                            direction="backward",      # most recent previous row
                            allow_exact_matches=True)  # including at olanzapine start time
    
    cohort1 = cohort1.merge(lab_results_history,on=[TABLES_TO_COMBINE["lab_results"]["column_names"]["patientid_col"], TABLES_TO_COMBINE["lab_results"]["column_names"]["instant_col"]],how="left")

    # Add hospitalization info - whether the patient is currently hospitalized at olanzapine initiation, and if yes, length of stay.
    hospitalization = pd.read_parquet(TABLES_TO_COMBINE["hospitalization"]["input_path"])
    hospitalization = hospitalization[hospitalization[TABLES_TO_COMBINE["hospitalization"]["column_names"]["patientid_col"]].isin(cohort1[TABLES_TO_COMBINE["medication"]["column_names"]["patientid_col"]].unique())]

    cohort1 = pd.merge_asof(cohort1.sort_values(by=[TABLES_TO_COMBINE["medication"]["column_names"]["start_col"]], ascending=[True], na_position="last"),
                            hospitalization.sort_values(by=[TABLES_TO_COMBINE["hospitalization"]["column_names"]["start_col"]], ascending=[True], na_position="last"),
                            left_on= [TABLES_TO_COMBINE["medication"]["column_names"]["start_col"]],
                            right_on= [TABLES_TO_COMBINE["hospitalization"]["column_names"]["start_col"]],
                            by=[TABLES_TO_COMBINE["medication"]["column_names"]["patientid_col"]],
                            direction="backward",      # most recent previous row
                            allow_exact_matches=True)  # including at olanzapine start time
        
    cohort1["is_currently_hospitalized"] = (
        (cohort1[TABLES_TO_COMBINE["medication"]["column_names"]["start_col"]] >= cohort1[TABLES_TO_COMBINE["hospitalization"]["column_names"]["start_col"]]) &
    (
        (cohort1[TABLES_TO_COMBINE["hospitalization"]["column_names"]["start_col"]] <= cohort1[TABLES_TO_COMBINE["hospitalization"]["column_names"]["end_col"]]) |
        (cohort1[TABLES_TO_COMBINE["hospitalization"]["column_names"]["end_col"]].isna())
    )
    )


    # Keep only the first olanzapine initiation per patient (if patients have multiple initiations, we will keep the first one with the most complete data)
    cohort1 = cohort1.sort_values(by=[TABLES_TO_COMBINE["medication"]["column_names"]["start_col"]], ascending=[True], na_position="last").drop_duplicates(subset=[TABLES_TO_COMBINE["medication"]["column_names"]["patientid_col"]], keep='first')
    
    # Save the cohort table
    cohort1.to_parquet(BASE_DIR / "cohort1_olanzapine_initiation.parquet", index=False)

    # -----------------------------------------------
    # Preprocessing - Refinement of the cohort table.
    # -----------------------------------------------

    cohort1 = pd.read_parquet(BASE_DIR / "cohort1_olanzapine_initiation.parquet")

    # --- Remove lab tests that are >50% missing ---
    
    cohort1_lab_tests = [col for col in cohort1.columns if col.startswith('Labanalysis')]
    na_frequencies = cohort1[cohort1_lab_tests].isna().mean()
    columns_to_remove = na_frequencies[na_frequencies > 0.5].index
    cohort1 = cohort1.drop(columns=columns_to_remove)
    print(f"Removed {len(columns_to_remove)} lab test features with >50% missing values: {list(columns_to_remove)}")
    
    # --- Datatype corrections ---
    # Replace string "None" and python None with np.nan
    cohort1.replace('None', np.nan, inplace=True)
    cohort1.replace({None: np.nan}, inplace=True)
    
    # Convert True/False or "True"/"False" to 1/0
    # Real boolean columns -> nullable integer
    bool_cols = cohort1.select_dtypes(include=["bool", "boolean"]).columns
    cohort1[bool_cols] = cohort1[bool_cols].astype("Int8")   # True/False -> 1/0, keeps NA
    # String booleans in object/string columns
    obj_cols = cohort1.select_dtypes(include=["object", "string"]).columns
    cohort1[obj_cols] = cohort1[obj_cols].replace({"True": 1, "False": 0})

    # One-Hot Encode some categorical variables like SmokingStatus, MaritalStatus
    cols_to_encode = [
        'SmokingStatus', 
        #'MaritalStatus'
        ]
    
    # Only encode if they exist
    available = [c for c in cols_to_encode if c in cohort1.columns]
    
    if available:
        dummies = pd.get_dummies(cohort1[available], prefix=available, drop_first=False)
        cohort1 = pd.concat([cohort1.drop(available, axis=1), dummies], axis=1)
    
    # Convert all 0/1 Columns to Int8
    for col in cohort1.columns:
        if pd.api.types.is_numeric_dtype(cohort1[col]):
            unique_vals = cohort1[col].dropna().unique()
            if set(unique_vals).issubset({0,1}):
                cohort1[col] = cohort1[col].astype("Int8")


    # --- ATC, SKS codes encoding --- 
    # One hot encoding ATC and SKS codes that patients currently have (currently = N05AH03 StartInstant)
    # Truncate the ATCs and SKSCodes (optionally) to a length of 4 characters
    
    if "active_atc_codes_csv" in cohort1.columns:
        cohort1["active_atc_codes_csv_short"] = (
            cohort1["active_atc_codes_csv"]
            .fillna("")
            .astype(str)
            .str.split(",")
            .apply(lambda xs: ",".join(x.strip()[:4] for x in xs if x.strip()))
        )
    
    if "active_sks_codes_csv" in cohort1.columns:
        cohort1["active_sks_codes_csv_short"] = (
            cohort1["active_sks_codes_csv"]
            .fillna("")
            .astype(str)
            .str.split(",")
            .apply(lambda xs: ",".join(x.strip()[:4] for x in xs if x.strip()))
        )
    
    # Choose top 200 Codes ATC and SKS Codes to create dummies
    # DECIDE HERE WHETHER TO USE SHORT OR LONG FORM CODES:
    # Decide whether to use the long of the short version of the data ATC and SKS codes 
    
    atcs = "active_atc_codes_csv_short" # here we are choosing the short forms of ATC codes, long form would be: atcs = "ATCs"
    skscodes = "active_sks_codes_csv_short" # here we are choosing the short forms of SKS codes, long form would be: skscodes = "SKSCodes"
    
    # For ATCs:
    code_counts_atc = cohort1[atcs].str.split(',', expand=True).stack().value_counts()
    top_atc_codes = code_counts_atc.nlargest(200).index.tolist()
    
    cohort1['ATCs_filtered'] = cohort1[atcs].apply(lambda x: ','.join(
        c.strip() for c in x.split(',') 
        if c.strip() in top_atc_codes
    ) if pd.notnull(x) else '')
    
    atc_dummies = cohort1['ATCs_filtered'].str.get_dummies(sep=',').add_prefix('ATC_')
    cohort1 = pd.concat([cohort1, atc_dummies], axis=1)
    
    # For SKSCodes:
    code_counts_sks = cohort1[skscodes].str.split(',', expand=True).stack().value_counts()
    top_sks_codes = code_counts_sks.nlargest(200).index.tolist()
    
    cohort1['SKSCodes_filtered'] = cohort1[skscodes].apply(lambda x: ','.join(
        c.strip() for c in x.split(',') 
        if c.strip() in top_sks_codes
    ) if pd.notnull(x) else '')
    
    sks_dummies = cohort1['SKSCodes_filtered'].str.get_dummies(sep=',').add_prefix('SKS_')
    cohort1 = pd.concat([cohort1, sks_dummies], axis=1)
    
    # Convert 0/1 Columns to Boolean
    
    for col in cohort1.columns:
        if pd.api.types.is_numeric_dtype(cohort1[col]):
            unique_vals = cohort1[col].dropna().unique()
            if set(unique_vals).issubset({0,1}):
                cohort1[col] = cohort1[col].astype("boolean")

    # --- Calculate the target variable ---
    # Make sure the target variable is calculates correctly
    
    cohort1["target"] = ((cohort1["BodyMassIndex_recalc_target"] >= cohort1["BodyMassIndex_recalc"] * 1.05).astype("int8")) # BMi increase >5%
    #cohort1["target"] = ((cohort1["BodyMassIndex_recalc_target"] - cohort1["BodyMassIndex_recalc"]) > 2).astype("int8") # BMI change more than 2 units
    #cohort1["target"] = cohort1["BodyMassIndex_recalc_target"] - cohort1["BodyMassIndex_recalc"] # Exact BMI change

    # --- Drop Unused Columns ---
    
    cols_to_drop = [
        'StartInstant', 
        'BirthDate',
        'DeathDate',
        'weight_diff_from_next',
        'bmi_diff_from_previous',
        'ATCs',
        'ATCs_filtered',
        'ATCs_short',
        'SKSCodes',
        'SKSCodes_filtered',
        'SKSCodes_short',
        'CollectionInstant',
        'BodyMassIndex_classification',
        'nan',
        'ATC_nan',
        'ATC_*Uns',
        'ATC', 
        'StrengthNumeric', 
        'StrengthNumeric_daily', 
        'StrengthNumeric_weight', 
        'StrengthNumeric_daily_weight', 
        'MedicationDuration', 
        'DiscontinuedInstant', 
        'DailyDosage_weighted_mean', 
        'UniqueATC_BeforeStart',
        'OverlappingATC', 
        'CreateInstant',
        'bmi_diff_from_next', 
        'days_to_next_bmi', 
        'bmi_diff_from_next_pct', 
        'bmi_diff_from_next_5pct',
        'CollectionInstant',
        'BodyMassIndex_recalc_target', 
        'WeightInGrams_target', 
        'CreateInstant_target', 
        'bmi_days_after_medication_start', 
        'SKSCode', 
        'SKSCodes_range', 
        'DiagnosisDuration', 
        'DiagnosisStartDate', 
        'DiagnosisEndDate', 
        'active_sks_codes_csv',
        'past_sks_codes_csv',
        'InpatientAdmissionInstant', 
        'DischargeInstant', 
        'LengthOfStayInDays', 
        'HospitalAdmissionKey', 
        'OverlappingATCs_short', 
        'active_sks_codes_csv_short', 
        'ATCs_filtered', 
        'ATC_*Uns',
        'SKSCodes_filtered',
        'HeightInCentimeters_median', 
        'WeightInGrams',
        'BodyMassIndex',
        'max_bmi_diff', 
        'min_bmi_diff',
        'BodySurfaceArea',
        'days_since_prev_bmi',
        'bmi_slope_prev_per_year_pos',
        'bmi_intercept_hist', 
        'bmi_trend_r2_hist', 
        'bmi_days_before_medication_start',
        'days_since_last_test',
        'number_of_prev_tests', 
        'number_of_prev_abnormal_tests',
        'number_of_prev_bvc',
        'number_of_prev_hamilton6',
        'number_of_prev_hamilton17',
        'MaritalStatus_Separated', 
        'MaritalStatus_Single', 
        'MaritalStatus_Together', 
        'MaritalStatus_Widowed',
        'number_of_previous_appointments',
        'number_of_previous_outpatientVisits',
        #'bmi_slope_hist_per_year',
        'MedicationIntervalSource',
        'HasAdministrationRecord',
        'HasPrescriptionRecord',
        'AdministrationRecordCount',
        'PrescriptionRecordCount',
        'PossiblyOngoingMedication',
        'active_atc_codes_csv',
        'active_sks_codes_csv',
        'past_unique_atc_codes_csv',
        'past_unique_sks_codes_csv',
        'healthAssesment_count',
        'RateOfBMIChange',
        'RateOfBMIChange_classification',
        'UniqueDiagnosisCodesCount',
        'LengthOfStayDays',
        'times_current_sks_diagnosed_before',
        'active_atc_codes_csv_short',
        'active_sks_codes_csv_short',
        "bmi_days_before_medication_stop",
        ]

    existing = [c for c in cols_to_drop if c in cohort1.columns]
    cohort1 = cohort1.drop(columns=existing)
    cohort1 = cohort1.loc[:, ~cohort1.columns.str.endswith('_target')]

    col_list = cohort1.columns.tolist()   # get every column name as a Python list
    print("cohort1.shape:", cohort1.shape, cohort1["PatientDurableKey"].nunique())
    print(col_list)

    # Save the final cohort table for modeling
    cohort1.to_parquet(BASE_DIR / "cohort1_olanzapine_initiation_final.parquet", index=False)

if __name__ == "__main__":
    main()
