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

from bmipred.feature_engineering.medication_features import compute_cumulative_unique_atc, compute_overlapping_atc_codes, extract_atc_info

# repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/config/preprocess_medication_features.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    start_time = time.time()
    config = load_config()
    input_file = os.path.join(REPO_ROOT, config["input_file"])
    output_file = os.path.join(REPO_ROOT, config["output_file"])

    extract_atc_info_kwargs = {
    k: v for k, v in {
        "patientid_col": config.get("patientid_col"),
        "atc_col": config.get("atc_col"),
        "start_col": config.get("start_col"),
    }.items() if v is not None
    }

    overlapping_atc_codes_kwargs = {
    k: v for k, v in {
        "atc_col": config.get("atc_col"),
        "start_col": config.get("start_col"),
        "discontinued_col": config.get("discontinued_col"),
    }.items() if v is not None
    }

    # load medication table
    df = pd.read_parquet(input_file)
    print(f"[INFO] Loaded {len(df)} rows from {input_file}")

    # Prepare ATC code mappings
    atc_code_to_id, id_to_atc_code, antipsychotics_ids, antidepressants_ids, anxiolytics_ids = extract_atc_info(df, **extract_atc_info_kwargs)

    # Reassign ATC_IDs based on cleaned ATC codes
    df['ATC_ID'] = df[config['atc_col']].map(atc_code_to_id).astype(np.int32)

    # Compute cumulative unique ATC counts (parallelized per patient)
    groups = [group for _, group in df.groupby(config['patientid_col'])]
    pool = Pool(cpu_count())
    results = pool.starmap(
        compute_cumulative_unique_atc,
        [(group, atc_code_to_id, id_to_atc_code, antipsychotics_ids, antidepressants_ids, anxiolytics_ids)
         for group in groups]
    )
    pool.close()
    pool.join()
    df = pd.concat(results).sort_index()
    df.drop(columns=["ATC_ID"], inplace=True)  # optional

    # Count number of times current ATC prescribed before
    df = df.sort_values([config['patientid_col'], config['start_col']]).reset_index(drop=True)
    df["TimesCurrentATCTaken_BeforeNow"] = df.groupby([config['patientid_col'], config['atc_col']]).cumcount()

    # Compute overlapping ATC codes at the same time (per patient)
    df = df.groupby(config['patientid_col'], group_keys=False).apply(compute_overlapping_atc_codes, **overlapping_atc_codes_kwargs)

    # Save final output
    df.to_parquet(output_file, index=False)
    print(f"[INFO] Saved processed data with {len(df)} rows to {output_file}")
    
    elapsed_time = (time.time() - start_time) / 60
    print(f"[INFO] All tasks completed in {elapsed_time:.2f} minutes.")
    
if __name__ == "__main__":
    main()
