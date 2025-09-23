#!/usr/bin/env python3
# scripts/15_create_patient_subcohorts.py

import os
import sys
import yaml
import time
import pandas as pd

# add src/ to sys.path so we can import bmipred without installing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.feature_engineering.patient_subcohorts import (create_patient_subcohorts, finalize_cohort)

# repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/configs/15_create_patient_subcohorts.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    start_time = time.time()
    config = load_config()

    # inputs
    df_path = os.path.join(REPO_ROOT, config["integrated_df_path"])
    medication_path = os.path.join(REPO_ROOT, config["medication_path"])
    olanzapine_info_path = os.path.join(REPO_ROOT, config["olanzapine_info_path"])

    # outputs (default ≥5% target)
    out_wot_first = os.path.join(REPO_ROOT, config["out_wot_olanzapine_first"])
    out_wot_last  = os.path.join(REPO_ROOT, config["out_wot_olanzapine_last"])
    out_on_first  = os.path.join(REPO_ROOT, config["out_on_olanzapine_first"])
    out_on_last   = os.path.join(REPO_ROOT, config["out_on_olanzapine_last"])
    out_before    = os.path.join(REPO_ROOT, config["out_before_olanzapine"])

    # optional outputs: slope_sign (pos/neg)
    posneg_out = {
        "wfirst": config.get("posneg_out_wot_olanzapine_first"),
        "wlast":  config.get("posneg_out_wot_olanzapine_last"),
        "first":  config.get("posneg_out_on_olanzapine_first"),
        "last":   config.get("posneg_out_on_olanzapine_last"),
    }

    # optional outputs: regression_diff (continuous target)
    reg_out = {
        "wfirst": config.get("regressiondiff_out_wot_olanzapine_first"),
        "wlast":  config.get("regressiondiff_out_wot_olanzapine_last"),
        "first":  config.get("regressiondiff_out_on_olanzapine_first"),
        "last":   config.get("regressiondiff_out_on_olanzapine_last"),
    }

    # ensure output dirs exist
    for p in [out_wot_first, out_wot_last, out_on_first, out_on_last, out_before]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    for rel in [v for v in posneg_out.values() if v]:
        os.makedirs(os.path.dirname(os.path.join(REPO_ROOT, rel)), exist_ok=True)
    for rel in [v for v in reg_out.values() if v]:
        os.makedirs(os.path.dirname(os.path.join(REPO_ROOT, rel)), exist_ok=True)

    # column params (only pass if provided)
    kwargs = {
        k: v for k, v in {
            "patientid_col":   config.get("patientid_col"),
            "sex_col":         config.get("sex_col"),
            "atc_col":         config.get("atc_col"),
            "bmi_instant_col": config.get("bmi_instant_col"),
        }.items() if v is not None
    }

    # load data
    df = pd.read_parquet(df_path)
    print(f"[INFO] Loaded integrated df ({df.shape[0]} rows, {df[config['patientid_col']].nunique()} patients)")

    medication = pd.read_parquet(medication_path)
    print(f"[INFO] Loaded medication ({medication.shape[0]} rows, {medication[config['patientid_col']].nunique()} patients)")

    olanzapine_info = pd.read_parquet(olanzapine_info_path)
    print(f"[INFO] Loaded olanzapine info ({olanzapine_info.shape[0]} rows, {olanzapine_info[config['patientid_col']].nunique()} patients)")

    # Create cohorts (default inclusion mode)
    olz_mode = config.get("olz_mode", "relevant")  # "relevant" or "while_on"
    (
        cohort_wot_olanzapine_first,
        cohort_wot_olanzapine_last,
        cohort_on_olanzapine_first,
        cohort_on_olanzapine_last,
        cohort_on_olanzapine_before
    ) = create_patient_subcohorts(
        df=df,
        medication=medication,
        olanzapine_info=olanzapine_info,
        olz_mode=olz_mode,
        **kwargs
    )

    # Finalize + save (default ≥5% target)
    target_col = config.get("target_col", "bmi_increase_over5percent")
    atc_col    = config.get("atc_col", "ATC")

    default_cohorts = {
        out_wot_first: cohort_wot_olanzapine_first,
        out_wot_last:  cohort_wot_olanzapine_last,
        out_on_first:  cohort_on_olanzapine_first,
        out_on_last:   cohort_on_olanzapine_last,
        out_before:    cohort_on_olanzapine_before,
    }
    for out_path, cdf in default_cohorts.items():
        out_df = finalize_cohort(cdf, target_mode="percent5", target_col=target_col, atc_col=atc_col)
        out_df.to_parquet(out_path, index=False)
        print(f"[INFO] Saved {os.path.basename(out_path)} to {out_path}")

    # OPTIONAL: pos/neg slope cohorts
    if any(posneg_out.values()):
        mapping = {
            posneg_out["wfirst"]: cohort_wot_olanzapine_first,
            posneg_out["wlast"]:  cohort_wot_olanzapine_last,
            posneg_out["first"]:  cohort_on_olanzapine_first,
            posneg_out["last"]:   cohort_on_olanzapine_last,
        }
        for rel_path, cdf in mapping.items():
            if rel_path:
                full_path = os.path.join(REPO_ROOT, rel_path)
                out_df = finalize_cohort(cdf, target_mode="slope_sign", atc_col=atc_col)
                out_df.to_parquet(full_path, index=False)
                print(f"[INFO] Saved {os.path.basename(full_path)} to {full_path}")

    # OPTIONAL: regression-diff cohorts (continuous target), always using while_on mode like your 3rd script
    if any(reg_out.values()):
        (
            wot_first_r,
            wot_last_r,
            on_first_r,
            on_last_r,
            _before_unused,
        ) = create_patient_subcohorts(
            df=df,
            medication=medication,
            olanzapine_info=olanzapine_info,
            olz_mode="while_on",
            **kwargs
        )

        mapping = {
            reg_out["wfirst"]: wot_first_r,
            reg_out["wlast"]:  wot_last_r,
            reg_out["first"]:  on_first_r,
            reg_out["last"]:   on_last_r,
        }
        for rel_path, cdf in mapping.items():
            if rel_path:
                full_path = os.path.join(REPO_ROOT, rel_path)
                out_df = finalize_cohort(cdf, target_mode="regression_diff", atc_col=atc_col)
                out_df.to_parquet(full_path, index=False)
                print(f"[INFO] Saved {os.path.basename(full_path)} to {full_path}")

    elapsed = time.time() - start_time
    print(f"[INFO] All processing tasks completed in {elapsed/60:.2f} minutes.")

if __name__ == "__main__":
    main()
