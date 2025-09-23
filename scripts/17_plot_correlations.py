#!/usr/bin/env python3
# scripts/17_plot_correlations.py

import os
import yaml
import pandas as pd
import sys

# add src/ to sys.path so we can import bmipred without installing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.visualization.correlations import get_filtered_df, plot_top_correlations, plot_clustered_heatmap, plot_clustered_heatmap_with_pvalues

# repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_config(path="scripts/configs/17_plot_correlations.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    out_dir = os.path.join(REPO_ROOT, config["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    rename_dict = config.get("rename_dict", {})

    for name, rel_path in config["table_paths"].items():
        df_path = os.path.join(REPO_ROOT, rel_path)
        df = pd.read_parquet(df_path)
        df = get_filtered_df(df)

        if config.get("plot_barplot", True):
            plot_top_correlations(df, name, out_dir)

        if config.get("plot_heatmap", True):
            plot_clustered_heatmap(df, name, out_dir, rename_dict)

        if config.get("plot_heatmap_pvalues", True):
            plot_clustered_heatmap_with_pvalues(df, name, out_dir, rename_dict)


if __name__ == "__main__":
    main()
