#!/usr/bin/env python3
# scripts/11_combine_bmi_hospitalization.py

import os
import sys
import yaml
import pandas as pd
import time

# add src/ to sys.path so we can import bmipred without installing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.feature_engineering.bmi_hospitalization import integrate_hospitalizations

# repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/configs/11_combine_bmi_hospitalization.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    start_time = time.time()
    config = load_config()
    bmi_path = os.path.join(REPO_ROOT, config["bmi_path"])
    hospitalization_path = os.path.join(REPO_ROOT, config["hospitalization_path"])
    output_file = os.path.join(REPO_ROOT, config["output_file"])

    kwargs = {
    k: v for k, v in {
        "patientid_col": config.get("patientid_col"),
        "bmi_instant_col": config.get("bmi_instant_col"),
        "start_col": config.get("start_col"),
        "end_col": config.get("end_col"),
        "hosp_key_col": config.get("hosp_key_col"),
        "hosp_service_col": config.get("hosp_service_col"),
    }.items() if v is not None
    }

    # read the data
    bmi_medication_diagnosis_labtests = pd.read_parquet(bmi_path)
    print(f"[INFO] Loaded BMI ({bmi_medication_diagnosis_labtests.shape[0]} rows, {bmi_medication_diagnosis_labtests[config['patientid_col']].nunique()} patients)")

    hospitalization = pd.read_parquet(hospitalization_path)
    print(f"[INFO] Loaded Hospitalization ({hospitalization.shape[0]} rows, {hospitalization[config['patientid_col']].nunique()} patients)")

    bmi_hospitalization = integrate_hospitalizations(bmi_integrated_df=bmi_medication_diagnosis_labtests, hospitalization=hospitalization, **kwargs)

    # Save df
    bmi_hospitalization.to_parquet(output_file, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time used: ", elapsed_time / 60, "minutes!")

if __name__ == "__main__":
    main()

