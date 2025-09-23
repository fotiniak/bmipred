#!/usr/bin/env python3
# scripts/10_calculate_bmi_labtests.py

import os
import sys
import yaml
import pandas as pd
import time

# add src/ to sys.path so we can import bmipred without installing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.feature_engineering.bmi_labtests import integrate_lab_tests

# repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/configs/10_combine_bmi_labtests.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    start_time = time.time()
    config = load_config()
    bmi_path = os.path.join(REPO_ROOT, config["bmi_path"])
    lab_tests_path = os.path.join(REPO_ROOT, config["lab_tests_path"])
    output_file = os.path.join(REPO_ROOT, config["output_file"])

    kwargs = {
    k: v for k, v in {
        "patientid_col": config.get("patientid_col"),
        "bmi_instant_col": config.get("bmi_instant_col"),
        "collection_instant_col": config.get("collection_instant_col"),
        "labanalysis_col": config.get("labanalysis_col"),
        "value_col": config.get("value_col"),
        "flag_col": config.get("flag_col"),
    }.items() if v is not None
    }

    # read the data
    bmi_medication_diagnosis = pd.read_parquet(bmi_path)
    print(f"[INFO] Loaded BMI ({bmi_medication_diagnosis.shape[0]} rows, {bmi_medication_diagnosis[config['patientid_col']].nunique()} patients)")

    labtests = pd.read_parquet(lab_tests_path)
    print(f"[INFO] Loaded Lab Tests ({labtests.shape[0]} rows, {labtests[config['patientid_col']].nunique()} patients)")

    labtests = integrate_lab_tests(bmi_medication_diagnosis=bmi_medication_diagnosis, lab_tests=labtests, **kwargs)

    # Save df
    labtests.to_parquet(output_file, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time used: ", elapsed_time / 60, "minutes!")

if __name__ == "__main__":
    main()

