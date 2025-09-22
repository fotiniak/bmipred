#!/usr/bin/env python3

import os
import sys
import yaml
import pandas as pd

# add src/ to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.analysis.statistics import compare_two_groups_with_normality_check

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/config/calculate_statistics.yaml"):
    with open(os.path.join(REPO_ROOT, path), "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    outs = cfg["outs"]
    inputs = cfg["inputs"]
    jobs = cfg["jobs"]

    dfs = {k: pd.read_parquet(os.path.join(REPO_ROOT, v)) for k, v in inputs.items()}

    results, text_out, csv_out = [], os.path.join(REPO_ROOT, outs, "statistics_results.txt"), os.path.join(REPO_ROOT, outs, "statistics_results.csv")
    os.makedirs(os.path.dirname(text_out), exist_ok=True)

    with open(text_out, "w") as f:
        f.write("STATISTICAL COMPARISON RESULTS:\n----------------------------\n\n")
        for job in jobs:
            df1, df2 = dfs[job["df1"]], dfs[job["df2"]]
            res = compare_two_groups_with_normality_check(
                df1[job["column"]], df2[job["column"]],
                label1=job["label1"], label2=job["label2"],
                alpha=job.get("alpha", 0.05), equal_var=job.get("equal_var", False)
            )
            res["title"] = job["title"]
            results.append(res)
            f.write(f"{job['title']}\n{'-'*40}\n")
            f.write(res.to_string(index=False) + "\n\n")

    pd.concat(results).to_csv(csv_out, index=False)
    print(f"[INFO] Saved reports -> {text_out}, {csv_out}")

if __name__ == "__main__":
    main()
