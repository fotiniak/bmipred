#!/usr/bin/env python3

import os
import yaml
import pandas as pd
import sys

# add src/ to sys.path so we can import bmipred without installing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.visualization.distributions import plot_histogram_with_medians, plot_bmi_over_age

# repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/config/plot_distributions.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    outs, tables, specs = config["outs"], config["tables"], config["plot_specs"]
    for name, path in tables.items():
        df = pd.read_parquet(path)
        base_out = os.path.join(outs, name)
        for col, spec in specs.items():
            if col in df.columns:
                plot_histogram_with_medians(
                    df, col, spec["title"], spec["xlabel"], f"{base_out}_{spec['suffix']}"
                )
                print(f"[INFO] Saved {base_out}_{spec['suffix']}")
        plot_bmi_over_age(df, f"{base_out}_bmi_over_age.png")
        print(f"[INFO] Saved {base_out}_bmi_over_age.png")
    print("[INFO] All plots saved!")

if __name__ == "__main__":
    main()
