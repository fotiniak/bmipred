#!/usr/bin/env python3
# scripts/plot_bmi_median_to_q3.py

import os
import sys
import time
import yaml
import pandas as pd

# add src/ to sys.path so we can import bmipred without installing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.visualization.boxplots import prepare_groups, plot_box_q2_to_q3, plot_box_full_range

# repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/config/plot_boxplots.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    start = time.time()
    config = load_config()

    # choose cohorts: 'last' or 'first'
    use_set = config.get("use_set", "last")

    on_path  = os.path.join(REPO_ROOT, config[f"cohort_on_olanzapine_{use_set}"])
    off_path = os.path.join(REPO_ROOT, config[f"cohort_wot_olanzapine_{use_set}"])
    out_path = os.path.join(REPO_ROOT, config["output_plot"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    value_col = config.get("value_col", "BodyMassIndex_recalc")
    label_col = config.get("label_col", "Olanzapine")

    # load
    on_df = pd.read_parquet(on_path)
    off_df = pd.read_parquet(off_path)
    print(f"[INFO] Loaded ON cohort ({on_df.shape[0]} rows), OFF cohort ({off_df.shape[0]} rows).")

    # prepare + test
    combined, u_stat, p_value = prepare_groups(
        on_df=on_df,
        off_df=off_df,
        value_col=value_col,
        label_col=label_col,
        on_label=True,
        off_label=False,
    )
    print(f"[INFO] After outlier removal: combined rows={combined.shape[0]}")
    print(f"[INFO] Mann-Whitney U: U={u_stat:.3f}, p={p_value:.3e}")

    # plot mode: 'q2_q3' or 'full_range'
    mode = config.get("plot_mode", "q2_q3")

    common_kwargs = dict(
        value_col=value_col,
        label_col=label_col,
        figsize=tuple(config.get("figsize", [2, 4])),
        palette=tuple(config.get("palette", ["tab:blue", "tab:orange"])),
        title=config.get("title", "BMI (Median to 75th Percentile)"),
        xlabel=config.get("xlabel", "Olanzapine Treatment"),
        ylabel=config.get("ylabel", "BMI (kg/m²)"),
        annotate_text=config.get("annotate_text", "Mann-Whitney U-test *** "),
        out_path=out_path,
        dpi=int(config.get("dpi", 600)),
    )

    if mode == "full_range":
        # override default title to match your second script, unless user sets one
        if "title" not in config:
            common_kwargs["title"] = "Boxplot of BMI by olanzapine status*"
        # keep annotate_text spacing exactly as your script
        if "annotate_text" not in config:
            common_kwargs["annotate_text"] = "Mann-Whitney U-test ***"
        plot_box_full_range(combined, **common_kwargs)
    else:
        plot_box_q2_to_q3(combined, **common_kwargs)

    print(f"[INFO] Saved plot to {out_path}")
    elapsed = time.time() - start
    print(f"[INFO] Completed in {elapsed:.2f}s.")

if __name__ == "__main__":
    main()
