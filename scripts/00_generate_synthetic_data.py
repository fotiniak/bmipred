#!/usr/bin/env python3
# scripts/00_generate_synthetic_data.py

import os
import sys
import yaml
from faker import Faker

# add src/ to sys.path so we can import bmipred without installing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.generate_synthetic.synthetic_data import (
    set_seed,
    generate_patients,
    generate_health_assessment,
    generate_diagnosis,
    generate_medication,
    generate_lab_results,
    generate_hospital_admission,
    generate_family_history,
)

# resolve paths relative to repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path="scripts/configs/00_generate_synthetic_data.yaml"):
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    print("[INFO] Loading config...")
    config = load_config()
    set_seed(config["random_seed"])
    fake = Faker("da_DK")

    n_patients = config["n_patients"]
    out_dir = os.path.join(REPO_ROOT, config["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}")

    print(f"[INFO] Generating {n_patients} patients...")
    patients = generate_patients(n_patients, fake)

    if config["tables"].get("patients", False):
        print("  -> Saving patients.parquet")
        patients.to_parquet(os.path.join(out_dir, "patients.parquet"), index=False)

    if config["tables"].get("healthAssessment", False):
        print("  -> Generating healthAssesment.parquet")
        df = generate_health_assessment(patients, fake)
        df.to_parquet(os.path.join(out_dir, "healthAssessment.parquet"), index=False)

    if config["tables"].get("diagnosis", False):
        print("  -> Generating diagnosis.parquet")
        df = generate_diagnosis(patients, fake)
        df.to_parquet(os.path.join(out_dir, "diagnosis.parquet"), index=False)

    if config["tables"].get("medication", False):
        print("  -> Generating medication.parquet")
        df = generate_medication(patients, fake)
        df.to_parquet(os.path.join(out_dir, "medication.parquet"), index=False)

    if config["tables"].get("labComponentResults", False):
        print("  -> Generating labComponentResults.parquet")
        df = generate_lab_results(patients, fake)
        df.to_parquet(os.path.join(out_dir, "labComponentResults.parquet"), index=False)

    if config["tables"].get("hospitalAdmission", False):
        print("  -> Generating hospitalAdmission.parquet")
        df = generate_hospital_admission(patients, fake)
        df.to_parquet(os.path.join(out_dir, "hospitalAdmission.parquet"), index=False)

    if config["tables"].get("familyHistory", False):
        print("  -> Generating familyHistory.parquet")
        df = generate_family_history(patients)
        df.to_parquet(os.path.join(out_dir, "familyHistory.parquet"), index=False)

    print("[INFO] Synthetic data generation completed successfully.")

if __name__ == "__main__":
    main()
