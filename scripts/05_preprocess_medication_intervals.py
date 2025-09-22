#!/usr/bin/env python3
# scripts/preprocess_medication_intervals.py

import os
import sys
import yaml
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
import numpy as np


# add src/ to sys.path so we can import bmipred without installing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.preprocessing.medication_intervals import medication_intervals

# repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/config/preprocess_medication_intervals.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    start_time = time.time()
    config = load_config()
    input_file = os.path.join(REPO_ROOT, config["input_file"])
    output_file = os.path.join(REPO_ROOT, config["output_file"])

    kwargs = {
    k: v for k, v in {
        "patientid_col": config.get("patientid_col"),
        "dosage_col": config.get("dosage_col"),
        "dosage_num_col": config.get("dosage_num_col"),
        "atc_col": config.get("atc_col"),
        "start_col": config.get("start_col"),
        "discontinued_col": config.get("discontinued_col"),
        "frequency_col": config.get("frequency_col"),
        "dosage_daily_col": config.get("dosage_daily_col"),
    }.items() if v is not None
    }

    # Load medication table
    df = pd.read_parquet(input_file)
    print(f"[INFO] Loaded {len(df)} rows from {input_file}")

    # Collapse overlapping medication intervals
    df = medication_intervals(df, **kwargs)

    # Save final output
    df.to_parquet(output_file, index=False)
    print(f"[INFO] Saved processed medication intervals to {output_file}")

    elapsed_time = (time.time() - start_time) / 60
    print(f"[INFO] All tasks completed in {elapsed_time:.2f} minutes.")


if __name__ == "__main__":
    main()