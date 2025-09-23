#!/usr/bin/env python3
# scripts/02_preprocess_diagnosis_sks_codes.py

import os
import sys
import yaml
import pandas as pd

# add src/ to sys.path so we can import bmipred without installing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.preprocessing.diagnosis_sks_codes import diagnosis_sks_codes

# repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/configs/02_preprocess_diagnosis_sks_codes.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    input_file = os.path.join(REPO_ROOT, config["input_file"])
    output_file = os.path.join(REPO_ROOT, config["output_file"])
    official_codes_file = os.path.join(REPO_ROOT, config["official_codes"])

    # only keep keys that actually exist in config (and are not None)
    kwargs = {
        k: v for k, v in {
            "code_col": config.get("code_col"),
            "code_desc_col": config.get("code_desc_col"),
            "threshold": config.get("threshold"),
        }.items() if v is not None
    }

    # load the dataframes
    df = pd.read_parquet(input_file)
    official_codes = pd.read_csv(official_codes_file, sep=";")

    if not os.path.exists(input_file): print(f"[WARN] Skipping {input_file} → No file found at {input_file}")

    print(f"[INFO] Cleaning {input_file}: {input_file} → {output_file}")

    # processed diagnosis table with SKSCodes
    processed_diagnosis_sks_codes = diagnosis_sks_codes(df=df, official_codes=official_codes, **kwargs)

    # save
    processed_diagnosis_sks_codes.to_parquet(output_file, index=False)
    print(f"[INFO] {processed_diagnosis_sks_codes} Cleaned and saved to {output_file}")

    print("[INFO] All cleaning tasks completed.")

if __name__ == "__main__":
    main()


