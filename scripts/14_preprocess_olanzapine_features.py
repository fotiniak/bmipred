#!/usr/bin/env python3
# scripts/combine_bmi_hospitalization.py

import os
import sys
import yaml
import pandas as pd
import time

# Add src/ to sys.path so we can import bmipred without installing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.feature_engineering.olanzapine_features import preprocess_olanzapine_features, combine_olanzapine_info

# Repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/config/preprocess_olanzapine_features.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    start_time = time.time()
    config = load_config()
    # Inputs
    bmi_demographics_path = os.path.join(REPO_ROOT, config["bmi_demographics_path"])
    medication_path = os.path.join(REPO_ROOT, config["medication_path"])
    integrated_data_path = os.path.join(REPO_ROOT, config["integrated_data_path"])
    
    # Outputs
    olanzapine_info_path = os.path.join(REPO_ROOT, config["olanzapine_info_path"])
    before_olanzapine_path = os.path.join(REPO_ROOT, config["before_olanzapine_path"])
    first_on_olanzapine_path = os.path.join(REPO_ROOT, config["first_on_olanzapine_path"])
    last_on_olanzapine_path = os.path.join(REPO_ROOT, config["last_on_olanzapine_path"])

    # Read the data
    bmi_demographics = pd.read_parquet(bmi_demographics_path)
    print(f"[INFO] Loaded BMI ({bmi_demographics.shape[0]} rows, {bmi_demographics[config['patientid_col']].nunique()} patients)")

    medication = pd.read_parquet(medication_path)
    print(f"[INFO] Loaded Medication ({medication.shape[0]} rows, {medication[config['patientid_col']].nunique()} patients)")

    # Calculate the Olanzapine specific features
    olanzapine_info = preprocess_olanzapine_features(bmi_demographics = bmi_demographics,
                                                     medication=medication,
                                                     patientid_col=config["patientid_col"],
                                                     bmi_instant_col=config["bmi_instant_col"],
                                                     atc_col=config["atc_col"],
                                                     start_col=config["start_col"],
                                                     discontinued_col=config["discontinued_col"],
                                                     dosage_col=config["dosage_col"],
                                                     dosage_daily_col=config["dosage_daily_col"])
    
    # Save intermediate results
    olanzapine_info.to_parquet(olanzapine_info_path, index=False)
    print(f"[INFO] Saved Olanzapine info to {olanzapine_info_path}")
    
    # Read the data for the next part
    integrated_data = pd.read_parquet(integrated_data_path)
    print(f"[INFO] Loaded Integrated Data ({integrated_data.shape[0]} rows, {integrated_data[config['patientid_col']].nunique()} patients)")

    print("Integrated keys:", integrated_data[[config["patientid_col"], config["bmi_instant_col"]]].head())
    print("Olanzapine keys:", olanzapine_info[[config["patientid_col"], config["bmi_instant_col"]]].head())

    # Combine the Olanzapine specific features
    before_df, first_df, last_df = combine_olanzapine_info(integrated_data=integrated_data,
                                                           olanzapine_info=olanzapine_info,
                                                           patientid_col=config["patientid_col"],
                                                           bmi_instant_col=config["bmi_instant_col"],
                                                           atc_col=config["atc_col"],
                                                           dosage_col=config["dosage_col"],
                                                           dosage_daily_col=config["dosage_daily_col"])


    # Save df
    before_df.to_parquet(before_olanzapine_path, index=False)
    first_df.to_parquet(first_on_olanzapine_path, index=False)
    last_df.to_parquet(last_on_olanzapine_path, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time used: ", elapsed_time / 60, "minutes!")

if __name__ == "__main__":
    main()
