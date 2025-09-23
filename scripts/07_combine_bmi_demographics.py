#!/usr/bin/env python3
# scripts/07_calculate_bmi_features.py

import os
import sys
import yaml
import pandas as pd
import time


# add src/ to sys.path so we can import bmipred without installing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.feature_engineering.bmi_demographics import filter_adults, recalculate_bmi, filter_outliers, filter_min_bmi_measurements, calculate_bmi_features, map_smoking_status, map_marital_status, map_sex

# repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/configs/07_combine_bmi_demographics.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    start_time = time.time()
    config = load_config()
    health_assessment_path = os.path.join(REPO_ROOT, config["health_assessment_path"])
    patients_path = os.path.join(REPO_ROOT, config["demographics_path"])
    output_file = os.path.join(REPO_ROOT, config["output_file"])

    # read the data
    health_assessment = pd.read_parquet(health_assessment_path)
    patients = pd.read_parquet(patients_path)

    df = pd.merge(health_assessment, patients, on=config["patientid_col"], how="inner") # combine BMI data with demographic data
    print("df.shape:(rows/columns)", df.shape, df[config["patientid_col"]].nunique())

    df = filter_adults(df, min_age=config["min_age"], max_age=config["max_age"], bmi_instant_col=config["bmi_instant_col"], birth_date_col=config["birth_date_col"]) # filter Age>=min_age
    print("Filtering Age>=18:", df.shape, df[config["patientid_col"]].nunique())

    df = recalculate_bmi(df, min_height_cm=config["min_height_cm"], max_height_cm=config["max_height_cm"], patientid_col=config["patientid_col"], heightcm_col=config["heightcm_col"], weightg_col=config["weightg_col"])
    print("Recalculating BMI and filtering BMI not NA:", df.shape, df[config["patientid_col"]].nunique())

    df = filter_outliers(df, lower_quantile=config["lower_quantile"], upper_quantile=config["upper_quantile"])
    df = filter_min_bmi_measurements(df, min_bmi_measurements=config["min_bmi_measurements"], patientid_col=config["patientid_col"]) # filter patients with at least min_bmi_measurements BMI measurements

    df = df.sort_values(by=[config["patientid_col"], config['bmi_instant_col']], ascending=[True, True], na_position="last")
    print("df:", df.shape, df[config["patientid_col"]].nunique())

    # remove rows with NaN in the 'name' column and drop duplicates
    df = df.dropna(subset=[config['bmi_instant_col']]).drop_duplicates()
    df = df.dropna(subset=['BodyMassIndex_recalc']).drop_duplicates()
    print("df", df.shape, df[config["patientid_col"]].nunique())

    df = df.groupby([config["patientid_col"], config['bmi_instant_col']], as_index=False).agg("first")
    print("df:", df.shape, df[config["patientid_col"]].nunique())

    df = calculate_bmi_features(df=df, patientid_col=config["patientid_col"], bmi_instant_col=config["bmi_instant_col"], weightg_col=config["weightg_col"])

    df.drop_duplicates(inplace=True)
    
    # Refine smoking status, marital status, sex categories

    df = map_smoking_status(df, smoking_status_col=config["smoking_status_col"])
    df = map_marital_status(df, marital_status_col=config["marital_status_col"])
    df = map_sex(df, sex_col=config["sex_col"])
    print("df.shape:(rows/columns)", df.shape, df[config["patientid_col"]].nunique())

    # Save df
    df.to_parquet(output_file, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time used: ", elapsed_time / 60, "minutes!")


if __name__ == "__main__":
    main()

