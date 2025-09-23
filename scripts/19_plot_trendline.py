#!/usr/bin/env python3
# scrips/19_plot_trendline.py

import os
import sys
import yaml
import pandas as pd

# add src/ to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.visualization.trendline import plot_lowess_two

# repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/configs/19_plot_trendline.yaml"):
    with open(os.path.join(REPO_ROOT, path), "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    outs, tables, specs = config["outs"], config["tables"], config["plot_specs"]

    df1 = pd.read_parquet(os.path.join(REPO_ROOT, tables["df1"]))
    df2 = pd.read_parquet(os.path.join(REPO_ROOT, tables["df2"]))

    save_path = os.path.join(REPO_ROOT, outs, specs["outfile"])
    plot_lowess_two(
        df1=df1,
        df2=df2,
        age_col=specs["age_col"],
        value_col=specs["value_col"],
        label1=specs["label1"],
        label2=specs["label2"],
        frac=specs.get("frac", 0.1),
        figsize=tuple(specs.get("figsize", [4, 4])),
        save_path=save_path,
    )
    print(f"[INFO] Saved {save_path}")

if __name__ == "__main__":
    main()
