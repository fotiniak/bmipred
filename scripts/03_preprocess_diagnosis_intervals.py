#!/usr/bin/env python3
# scripts/03_preprocess_diagnosis_intervals.py

import os
import sys
import yaml
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore') # deactivate warnings

# add src/ to sys.path so we can import bmipred without installing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.preprocessing.diagnosis_intervals import diagnosis_intervals

# repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/configs/03_preprocess_diagnosis_intervals.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    start_time = time.time()
    config = load_config()
    input_file = os.path.join(REPO_ROOT, config["input_file"])
    output_file = os.path.join(REPO_ROOT, config["output_file"])
    kodes_file = os.path.join(REPO_ROOT, config["kodes_file"])

    kwargs = {
        k: v for k, v in {
            "df_codes_col": config.get("df_codes_col"),
            "df_patientid_col": config.get("df_patientid_col"),
            "df_startdate_col": config.get("df_startdate_col"),
            "df_enddate_col": config.get("df_enddate_col"),
            "codesMap_codes_col": config.get("codesMap_codes_col"),
            "codesMap_desc_col": config.get("codesMap_desc_col"),
            "codesMap_groups_col": config.get("codesMap_groups_col")
        }.items() if v is not None
    }

    # Load the input data
    df = pd.read_parquet(input_file)
    codesMap = pd.read_csv(kodes_file, sep=',')

    # Process the data
    df = diagnosis_intervals(df=df, codesMap=codesMap, **kwargs)

    # Save the processed data
    df.to_parquet(output_file, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[INFO] All processing tasks completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()

