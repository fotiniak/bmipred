#!/usr/bin/env python3
# scripts/21_calculate_summaries.py

import os
import sys
import yaml
import pandas as pd

# add src/ to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.analysis.summaries import summarise_dataframe

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/configs/21_calculate_summaries.yaml"):
    with open(os.path.join(REPO_ROOT, path), "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    input_dir = cfg["input_dir"]
    output_dir = cfg["output_dir"]
    tables = cfg["tables"]

    os.makedirs(os.path.join(REPO_ROOT, output_dir), exist_ok=True)

    for label, relpath in tables.items():
        df = pd.read_parquet(os.path.join(REPO_ROOT, input_dir, relpath))
        summary = summarise_dataframe(df)
        out_path = os.path.join(REPO_ROOT, output_dir, f"{label}_variable_summary.csv")
        summary.to_csv(out_path, index=False)
        print(f"[INFO] Saved {out_path}")

    print("[INFO] All summaries completed!")

if __name__ == "__main__":
    main()
    
