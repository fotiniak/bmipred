#!/usr/bin/env python3
# scripts/preprocess_medication_dosage.py


import os
import sys
import yaml
import pandas as pd
import time

# add src/ to sys.path so we can import bmipred without installing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.preprocessing.medication_dosage import clean_medication_frequency, dosage_filtering, calculate_daily_dosage

# repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/config/preprocess_medication_dosage.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    start_time = time.time()
    config = load_config()
    input_file = os.path.join(REPO_ROOT, config["input_file"])
    output_file = os.path.join(REPO_ROOT, config["output_file"])

    df = pd.read_parquet(input_file)

    # Filter for expected dosages

    # Filter medication dosages of specific ATC codes and strength ranges
    dosage_kwargs = {
        k: v for k, v in {
            "atc_col": config.get("atc_col"),
            "strength_col": config.get("strength_col"),
            "dosage_ranges": config.get("dosage_ranges"),
        }.items() if v is not None
    }

    df = dosage_filtering(df=df, **dosage_kwargs)

    # Clean frequency column descriptions
    frequency_kwargs = {
        k: v for k, v in {
            "frequency_col": config.get("frequency_col"),
        }.items() if v is not None
    }
    
    df = clean_medication_frequency(df=df, **frequency_kwargs)

    # Calculate daily dosage
    df = calculate_daily_dosage(df)

    # Use the clean dosage df 
    df.to_parquet(output_file, index=False)
    print(f"[INFO] Cleaned dosage DataFrame saved to {output_file}")

    end_time = time.time()
    elapsed_time = end_time-start_time
    print('All DONE :)')
    print('Time used: ', elapsed_time/60, 'minutes!' )

if __name__ == "__main__":
    main()
