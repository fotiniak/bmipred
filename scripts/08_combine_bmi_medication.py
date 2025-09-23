#!/usr/bin/env python3
# scripts/08_combine_bmi_medication.py

import os
import sys
import yaml
import pandas as pd
import time

# add src/ to sys.path so we can import bmipred without installing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.feature_engineering.bmi_medication import refine_medication_ids, map_atc_to_bmi

# repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/configs/08_combine_bmi_medication.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    start_time = time.time()
    config = load_config()
    bmi_demographics_path = os.path.join(REPO_ROOT, config["bmi_demographics_path"])
    medication_path = os.path.join(REPO_ROOT, config["medication_path"])
    output_file = os.path.join(REPO_ROOT, config["output_file"])

    # read the data
    bmi_demographics = pd.read_parquet(bmi_demographics_path)
    print(f"[INFO] Loaded BMI ({bmi_demographics.shape[0]} rows, {bmi_demographics[config['patientid_col']].nunique()} patients)")
    
    medication = pd.read_parquet(medication_path)
    print(f"[INFO] Loaded Medications ({medication.shape[0]} rows, {medication[config['patientid_col']].nunique()} patients)")

    medication = refine_medication_ids(medication=medication,
                                       bmi_demographics=bmi_demographics,
                                       patientid_col=config["patientid_col"],
                                       bmi_instant_col=config["bmi_instant_col"],
                                       atc_col=config["atc_col"],
                                       start_col=config["start_col"],
                                       discontinued_col=config["discontinued_col"])

    # map ATCs to BMI timestamps
    result = map_atc_to_bmi(bmi_demographics=bmi_demographics,
                            medication=medication,
                            patientid_col=config["patientid_col"],
                            bmi_instant_col=config["bmi_instant_col"],
                            start_col=config["start_col"],
                            discontinued_col=config["discontinued_col"],
                            atc_col=config["atc_col"],
                            num_processes=config.get("num_processes", None))

    # Save df
    result.to_parquet(output_file, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time used: ", elapsed_time / 60, "minutes!")

if __name__ == "__main__":
    main()

