#!/usr/bin/env python3
# scripts/09_combine_bmi_diagnosis.py

import os
import sys
import yaml
import pandas as pd
import time

# add src/ to sys.path so we can import bmipred without installing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.feature_engineering.bmi_diagnosis import refine_diagnosis_ids, map_skscode_to_bmi

# repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/configs/09_combine_bmi_diagnosis.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    start_time = time.time()
    config = load_config()
    bmi_medication_path = os.path.join(REPO_ROOT, config["bmi_medication_path"])
    diagnosis_path = os.path.join(REPO_ROOT, config["diagnosis_path"])
    output_file = os.path.join(REPO_ROOT, config["output_file"])

    # read the data
    bmi_medication = pd.read_parquet(bmi_medication_path)
    print(f"[INFO] Loaded BMI ({bmi_medication.shape[0]} rows, {bmi_medication[config['patientid_col']].nunique()} patients)")

    diagnosis = pd.read_parquet(diagnosis_path)
    print(f"[INFO] Loaded Diagnoses ({diagnosis.shape[0]} rows, {diagnosis[config['patientid_col']].nunique()} patients)")

    diagnosis = refine_diagnosis_ids(diagnosis=diagnosis,
                                     bmi_medication=bmi_medication,
                                     patientid_col=config["patientid_col"],
                                     sks_col=config["sks_col"],
                                     start_col=config["start_col"],
                                     discontinued_col=config["discontinued_col"])

    # map SKS codes to BMI timestamps
    result = map_skscode_to_bmi(bmi_medication=bmi_medication,
                                diagnosis=diagnosis,
                                patientid_col=config["patientid_col"],
                                bmi_instant_col=config["bmi_instant_col"],
                                start_col=config["start_col"],
                                discontinued_col=config["discontinued_col"],
                                sks_col=config["sks_col"],
                                num_processes=config.get("num_processes", None))
    
    # map SKS codes grouped range to BMI timestamps
    result2 = map_skscode_to_bmi(bmi_medication=result,
                                 diagnosis=diagnosis,
                                 patientid_col=config["patientid_col"],
                                 bmi_instant_col=config["bmi_instant_col"],
                                 start_col=config["start_col"],
                                 discontinued_col=config["discontinued_col"],
                                 sks_col=config["sks_range_col"],
                                 num_processes=config.get("num_processes", None))

    # Save df
    result2.to_parquet(output_file, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time used: ", elapsed_time / 60, "minutes!")

if __name__ == "__main__":
    main()

