#!/usr/bin/env python3

# Script to generate synthetic medical data using the data generation functions.
# All configuration parameters are defined at the beginning of the script.

import sys
from pathlib import Path
from datetime import datetime

# Add src to path to import bmipred modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from faker import Faker

from bmipred.data_generation.generate_synthetic import (
    set_seed,
    generate_patients,
    generate_health_assessment,
    generate_diagnosis,
    generate_medication,
    generate_lab_results,
    generate_hospital_admission,
)


# ==================== CONFIGURATION PARAMETERS ====================

# Number of synthetic patients to generate
N_PATIENTS = 4000

# Random seed for reproducibility
RANDOM_SEED = 2000

# Output directory for generated data
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "synthetic"

# Output format: 'csv' or 'parquet'
OUTPUT_FORMAT = "parquet"

# Whether to generate each data type
GENERATE_PATIENTS = True
GENERATE_HEALTH_ASSESSMENT = True
GENERATE_DIAGNOSIS = True
GENERATE_MEDICATION = True
GENERATE_LAB_RESULTS = True
GENERATE_HOSPITAL_ADMISSION = True

# ==================== END CONFIGURATION =======================

def main():
    # Main function to generate synthetic data.
    print(f"Starting synthetic data generation at {datetime.now()}")
    print("Configuration:")
    print(f"- Number of patients: {N_PATIENTS}")
    print(f"- Random seed: {RANDOM_SEED}")
    print(f"- Output directory: {OUTPUT_DIR}")
    print(f"- Output format: {OUTPUT_FORMAT}")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(RANDOM_SEED)
    fake = Faker()
    
    # Dictionary to store generated dataframes
    generated_data = {}
    
    # Generate patients
    if GENERATE_PATIENTS:
        print("Generating patients...")
        patients_df = generate_patients(N_PATIENTS, fake)
        generated_data["patients"] = patients_df
        patients_df.to_parquet(OUTPUT_DIR / "patients.parquet")
        print(f"Shape: {patients_df.shape}")
    else:
        print("Skipping patient generation")
        return
    
    # Generate health assessment data
    if GENERATE_HEALTH_ASSESSMENT:
        print("Generating health assessment data...")
        health_assessment_df = generate_health_assessment(patients_df, fake)
        generated_data["health_assessment"] = health_assessment_df
        health_assessment_df.to_parquet(OUTPUT_DIR / "health_assessment.parquet")
        print(f"Shape: {health_assessment_df.shape}")
    else:
        print("Skipping health assessment generation")
    
    # Generate diagnosis data
    if GENERATE_DIAGNOSIS:
        print("Generating diagnosis data...")
        diagnosis_df = generate_diagnosis(patients_df, fake)
        generated_data["diagnosis"] = diagnosis_df
        diagnosis_df.to_parquet(OUTPUT_DIR / "diagnosis.parquet")
        print(f"Shape: {diagnosis_df.shape}")
    else:
        print("Skipping diagnosis generation")
    
    # Generate medication data
    if GENERATE_MEDICATION:
        print("Generating medication data...")
        medication_df = generate_medication(patients_df, fake)
        generated_data["medication"] = medication_df
        medication_df.to_parquet(OUTPUT_DIR / "medication.parquet")
        print(f"Shape: {medication_df.shape}")
    else:
        print("Skipping medication generation")
    
    # Generate lab results data
    if GENERATE_LAB_RESULTS:
        print("Generating lab results data...")
        lab_results_df = generate_lab_results(patients_df, fake)
        generated_data["lab_results"] = lab_results_df
        lab_results_df.to_parquet(OUTPUT_DIR / "lab_results.parquet")
        print(f"Shape: {lab_results_df.shape}")
    else:
        print("Skipping lab results generation")
    
    # Generate hospital admission data
    if GENERATE_HOSPITAL_ADMISSION:
        print("Generating hospital admission data...")
        hospital_admission_df = generate_hospital_admission(patients_df, fake)
        generated_data["hospital_admission"] = hospital_admission_df
        hospital_admission_df.to_parquet(OUTPUT_DIR / "hospital_admission.parquet")
        print(f"Shape: {hospital_admission_df.shape}")
    else:
        print("Skipping hospital admission generation")
    
    
    print(f"\nSynthetic data generation completed at {datetime.now()}")


if __name__ == "__main__":
    main()
