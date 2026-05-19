#!/usr/bin/env python3
# Script to clean medical data tables from parquet files.
# All configuration parameters are defined at the beginning of the script.

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from bmipred.data_preprocessing.data_cleaning import clean_table

# ==================== CONFIGURATION PARAMETERS ====================

# Output directory for cleaned data
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "preprocessed"

# List of values to consider as missing when reading data
MISSING_INDICATORS = [None, "", "NA", "N/A", "null", "NULL"]

# Dictionary of tables to read and clean with full paths and column configurations
TABLES_TO_CLEAN = {
    "patients": {
        "input_path": Path(__file__).parent.parent / "data" / "synthetic" / "patients.parquet",
        "columns": {
            "numeric": ["PatientDurableKey"],
            "datetime": ["BirthDate", "DeathDate"],
            "categorical": ["Sex"],
        }
    },
    "health_assessment": {
        "input_path": Path(__file__).parent.parent / "data" / "synthetic" / "health_assessment.parquet",
        "columns": {
            "numeric": ["PatientDurableKey", "HeightInCentimeters", "WeightInGrams", "BodyMassIndex"],
            "datetime": ["CreateInstant"],
            "categorical": ["SmokingStatus"],
        }
    },
    "diagnosis": {
        "input_path": Path(__file__).parent.parent / "data" / "synthetic" / "diagnosis.parquet",
        "columns": {
            "numeric": ["PatientDurableKey"],
            "datetime": ["DiagnosisStartDate", "DiagnosisEndDate"],
            "categorical": ["SKSCode", "DiagnosisName"],
        }
    },
    "medication": {
        "input_path": Path(__file__).parent.parent / "data" / "synthetic" / "medication.parquet",
        "columns": {
            "numeric": ["PatientDurableKey", "StrengthNumeric"],
            "datetime": ["StartInstant", "DiscontinuedInstant", "AdministrationInstant"],
            "categorical": ["ATC", "Frequency", "DoseDescription"],
        }
    },
    "lab_results": {
        "input_path": Path(__file__).parent.parent / "data" / "synthetic" / "lab_results.parquet",
        "columns": {
            "numeric": ["PatientDurableKey"],
            "datetime": ["CollectionInstant"],
            "categorical": ["Labanalysis", "Flag"],
        }
    },
    "hospital_admission": {
        "input_path": Path(__file__).parent.parent / "data" / "synthetic" / "hospital_admission.parquet",
        "columns": {
            "numeric": ["PatientDurableKey"],
            "datetime": ["InpatientAdmissionInstant", "DischargeInstant"],
        }
    },
}

# ==================== END CONFIGURATION =======================

def main():
    # Main function to clean synthetic data.
    print(f"Starting data cleaning at {datetime.now()}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each table
    for table_name, table_config in TABLES_TO_CLEAN.items():
        input_path = table_config["input_path"]
        output_path = OUTPUT_DIR / input_path.name
        
        # Check if input file exists
        if not input_path.exists():
            print(f"Skipping {table_name}: file not found at {input_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing: {table_name}")
        
        
        # Read parquet file
        print(f"Reading {input_path}...")
        df = pd.read_parquet(input_path)
        print(f"Rows before cleaning: {len(df)}, Columns: {len(df.columns)}")
        
        # Create cleaning configuration
        clean_config = {
            "columns": table_config["columns"],
            "missing": MISSING_INDICATORS,
        }
        
        # Clean the table using the cleaning function
        df = clean_table(df, clean_config)
        
        print(f"Rows after cleaning: {len(df)}")
        
        # Save cleaned data
        print(f"Saving cleaned data to {output_path}...")
        df.to_parquet(output_path, index=False)
        print(f"Saved {table_name}")
    
    print(f"\nData cleaning completed at {datetime.now()}")


if __name__ == "__main__":
    main()
