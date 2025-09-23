#!/usr/bin/env python3
# scripts/01_clean_data.py

import os
import sys
import yaml
import pandas as pd

# add src/ to sys.path so we can import bmipred without installing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.cleaning.data_cleaning import clean_table

# repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/configs/01_clean_data.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    input_dir = os.path.join(REPO_ROOT, config["input_dir"])
    output_dir = os.path.join(REPO_ROOT, config["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    for table_name, enabled in config["tables"].items():
        if not enabled:
            continue

        input_path = os.path.join(input_dir, f"{table_name}.parquet")
        output_path = os.path.join(output_dir, f"{table_name}_cleaned.parquet")

        if not os.path.exists(input_path):
            print(f"[WARN] Skipping {table_name} → No file found at {input_path}")
            continue

        print(f"[INFO] Cleaning {table_name}: {input_path} → {output_path}")

        # load
        df = pd.read_parquet(input_path)

        # clean
        df_clean = clean_table(df, config)

        # save
        df_clean.to_parquet(output_path, index=False)
        print(f"[INFO] {table_name} Cleaned and saved to {output_path}")

    print("[INFO] All cleaning tasks completed.")

if __name__ == "__main__":
    main()
