#!/usr/bin/env python3
# Script to apply feature engineering to preprocessed data tables.
# All configuration parameters are defined at the beginning of the script.

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from bmipred.feature_engineering.bmi_features import (
    filter_adults,
    recalculate_bmi, 
    deduplicate_bmi, 
    filter_outliers, 
    filter_min_bmi_measurements,
    calculate_bmi_features, 
    calculate_bmi_trajectory,
    map_smoking_status, 
    map_sex
)
from bmipred.feature_engineering.diagnosis_features import (
    normalize_sks_codes,
    diagnosis_history_features,
)
from bmipred.feature_engineering.medication_features import (
    normalize_atc_codes,
    medication_history_features,
)
from bmipred.feature_engineering.hospitalization_features import (
    hospitalization_history_features,
)
from bmipred.feature_engineering.labtest_features import (
    normalize_lab_tests,
    lab_history_features,
)

# ==================== CONFIGURATION PARAMETERS ====================

BASE_DIR = Path(__file__).parent.parent / "data" / "preprocessed"

TABLES_TO_PROCESS = {
    "patients": {
        "input_path": BASE_DIR / "patients.parquet",
        "column_names": {
            "patientid_col": "PatientDurableKey",
            "birth_date_col": "BirthDate",
            "sex_col": "Sex",
        },
    },
    "health_assessment": {
        "input_path": BASE_DIR / "health_assessment.parquet",
        "column_names": {
            "patientid_col": "PatientDurableKey",
            "instant_col": "CreateInstant",
            "height_col": "HeightInCentimeters",
            "weight_col": "WeightInGrams",
            "smoking_status_col": "SmokingStatus",
        },
    },
    "diagnosis": {
        "input_path": BASE_DIR / "diagnosis_collapsed.parquet",
        "column_names": {
            "patientid_col": "PatientDurableKey",
            "code_col": "SKSCode",
            "start_col": "DiagnosisStartDate",
            "end_col": "DiagnosisEndDate",
        },
    },
    "medication": {
        "input_path": BASE_DIR / "medication_collapsed.parquet",
        "column_names": {
            "patientid_col": "PatientDurableKey",
            "code_col": "ATC",
            "start_col": "StartInstant",
            "end_col": "DiscontinuedInstant",
        },
    },
    "hospitalization": {
        "input_path": BASE_DIR / "hospital_admission.parquet",
        "column_names": {
            "patientid_col": "PatientDurableKey",
            "start_col": "InpatientAdmissionInstant",
            "end_col": "DischargeInstant",
        },
    },
    "lab_results": {
        "input_path": BASE_DIR / "lab_results.parquet",
        "column_names": {
            "patientid_col": "PatientDurableKey",
            "instant_col": "CollectionInstant",
            "lab_test_col": "Labanalysis",
            "flag_col": "Flag",
            "abnormal_flags": ["Høj", "Lav"],
            "value_col": None,
        },
    }
}

# ==================== END CONFIGURATION =======================  

def main():
    # Main function to apply feature engineering to preprocessed tables."""
    print(f"Starting feature engineering at {datetime.now()}")
    print()
    
    for table_name, config in TABLES_TO_PROCESS.items():
        input_path = TABLES_TO_PROCESS[table_name]["input_path"]
        column_names = TABLES_TO_PROCESS[table_name]["column_names"]
        
        # Load the patients table in order to get birth date and sex info.
        if table_name == "patients":
            print(f"Reading {table_name} - to load birth date information...")
            patients = pd.read_parquet(input_path)
            birth_date_col = column_names["birth_date_col"]
            sex_col = column_names["sex_col"]

        # Calculate BMI features for health assessment table, which requires birth date and sex.
        if table_name == "health_assessment":
            print(f"Processing {table_name}...")
            df = pd.read_parquet(input_path)
            print(f"Merging {table_name} - combining birth date information...")
            df = pd.merge(df, patients, on=column_names["patientid_col"], how="left")
            
            df = filter_adults(df, 
                               min_age=18, 
                               max_age=120, 
                               instant_col=column_names["instant_col"],
                               birth_date_col=birth_date_col)
            
            df = recalculate_bmi(df, 
                                 min_height_cm = 100, 
                                 max_height_cm = 220, 
                                 patientid_col = column_names["patientid_col"],
                                 height_col = column_names["height_col"],
                                 weight_col = column_names["weight_col"])
            
            df = deduplicate_bmi(df,
                                 patientid_col = column_names["patientid_col"],
                                 instant_col = column_names["instant_col"],
                                 bmi_col = "BodyMassIndex_recalc",)
            
            df = filter_outliers(df,
                                 lower_quantile = 0.001, 
                                 upper_quantile = 0.999)
            
            df = filter_min_bmi_measurements(df, 
                                             min_bmi_measurements = 1,
                                             patientid_col = column_names["patientid_col"])

            df = calculate_bmi_features(df, 
                                        patientid_col = column_names["patientid_col"],
                                        instant_col = column_names["instant_col"],
                                        weight_col = column_names["weight_col"])
            
            df = calculate_bmi_trajectory(df,
                                          patientid_col = column_names["patientid_col"],
                                          instant_col = column_names["instant_col"],
                                          bmi_col = "BodyMassIndex_recalc",)

            df = map_smoking_status(df, smoking_status_col = column_names["smoking_status_col"])

            df = map_sex(df, sex_col = sex_col)

            # Save
            output_path = input_path.parent / f"{input_path.stem}_features.parquet"
            print(f"Saving {table_name} with BMI features to {output_path}...")
            df.to_parquet(output_path, index=False)

        # Calculate medication history features, which requires ATC code normalization.
        if table_name == "medication":
            print(f"Processing {table_name}...")
            df = pd.read_parquet(input_path)
            
            df = normalize_atc_codes(df, atc_col=column_names["code_col"])
            
            df = medication_history_features(df,
                                             patientid_col = column_names["patientid_col"],
                                             atc_col = column_names["code_col"],
                                             start_col = column_names["start_col"],
                                             discontinued_col = column_names["end_col"],)
    
            output_path = input_path.parent / f"{input_path.stem}_features.parquet"
            print(f"Saving {table_name} with features to {output_path}...")
            df.to_parquet(output_path, index=False)

        # Calculate diagnosis history features, which requires SKS code normalization.
        if table_name == "diagnosis":
            print(f"Processing {table_name}...")
            df = pd.read_parquet(input_path)
            
            df = normalize_sks_codes(df, code_col=column_names["code_col"])
            
            df = diagnosis_history_features(df,
                                            patientid_col = column_names["patientid_col"],
                                            code_col = column_names["code_col"],
                                            start_col = column_names["start_col"],
                                            end_col = column_names["end_col"],)

            output_path = input_path.parent / f"{input_path.stem}_features.parquet"
            print(f"Saving {table_name} with features to {output_path}...")
            df.to_parquet(output_path, index=False)


        # Calculate lab test history features, which requires normalization of lab test results and flags.
        if table_name == "lab_results":
            print(f"Processing {table_name}...")
            df = pd.read_parquet(input_path)
            
            df = normalize_lab_tests(df,
                                     patientid_col = column_names["patientid_col"],
                                     timestamp_col = column_names["instant_col"],
                                     labanalysis_col = column_names["lab_test_col"],
                                     value_col = column_names["value_col"],
                                     flag_col = column_names["flag_col"],
                                     abnormal_flags = column_names["abnormal_flags"],)
            
    
            df = lab_history_features(df,
                                      patientid_col = column_names["patientid_col"],
                                      timestamp_col = column_names["instant_col"],)
    
            output_path = input_path.parent / f"{input_path.stem}_features.parquet"
            print(f"Saving {table_name} with features to {output_path}...")
            df.to_parquet(output_path, index=False)

        # Calculate hospitalization history features, which requires admission and discharge dates.
        if table_name == "hospitalization":
            print(f"Processing {table_name}...")
            df = pd.read_parquet(input_path)
            
            df = hospitalization_history_features(df,
                                                  patientid_col = column_names["patientid_col"],
                                                  admission_col = column_names["start_col"],
                                                  discharge_col = column_names["end_col"],)
    
            output_path = input_path.parent / f"{input_path.stem}_features.parquet"
            print(f"Saving {table_name} with features to {output_path}...")
            df.to_parquet(output_path, index=False)
    
    print(f"Feature engineering completed at {datetime.now()}")

if __name__ == "__main__":
    main()
