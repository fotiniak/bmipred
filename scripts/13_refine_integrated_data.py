#!/usr/bin/env python3
# scripts/combine_bmi_hospitalization.py

import os
import sys
import yaml
import pandas as pd
import time

# aAdd src/ to sys.path so we can import bmipred without installing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.feature_engineering.integrate_data import refine_integrated_tables

# Repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/config/refine_integrated_data.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    start_time = time.time()
    config = load_config()
    df_path = os.path.join(REPO_ROOT, config["input_file"])
    output_path = os.path.join(REPO_ROOT, config["output_file"])


    # Read the data
    df = pd.read_parquet(df_path)
    print(f"[INFO] Loaded BMI ({df.shape[0]} rows, {df[config['patientid_col']].nunique()} patients)")

    df = refine_integrated_tables(df=df, 
                                  patientid_col=config["patientid_col"],
                                  labanalysis_col=config["labanalysis_col"],
                                  )

    # Save df
    df.to_parquet(output_path, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time used: ", elapsed_time / 60, "minutes!")

if __name__ == "__main__":
    main()
