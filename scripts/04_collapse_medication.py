#!/usr/bin/env python3
# Script to preprocess medication data by calculating daily dosages and collapsing intervals.
# All configuration parameters are defined at the beginning of the script.

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from bmipred.data_preprocessing.medication_dosages_filtering import (
    clean_dosage_frequency,
    calculate_daily_dosage,
    dosage_filtering,
)
from bmipred.data_preprocessing.medication_intervals import (
    filter_medication_timestamps,
    medication_intervals,
)

# ==================== CONFIGURATION PARAMETERS ====================

# Input and output paths
INPUT_PATH = Path(__file__).parent.parent / "data" / "preprocessed" / "medication.parquet"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "preprocessed" / "medication_collapsed.parquet"

# Column names in medication dataframe
MEDICATION_PATIENTID_COL = "PatientDurableKey"
MEDICATION_ATC_COL = "ATC"
MEDICATION_STRENGTH_COL = "StrengthNumeric"
MEDICATION_DOSE_DESC_COL = "DoseDescription"
MEDICATION_FREQUENCY_COL = "Frequency"
MEDICATION_START_COL = "StartInstant"
MEDICATION_DISCONTINUED_COL = "DiscontinuedInstant"
MEDICATION_ADMINISTRATION_COL = "AdministrationInstant"

# Date range filtering
DATE_LO = pd.Timestamp("2016-01-01")
DATE_HI = pd.Timestamp("2026-01-01")
ONGOING_FILL_DATE = pd.Timestamp("2025-12-31 23:59:59.999999999")

# Optional dosage filtering: dictionary mapping ATC codes to (min, max) dosage ranges
# Set to None to skip filtering dosages
DOSAGE_RANGES = None
# Example:
# DOSAGE_RANGES = {
#     "N05AH03": (1, 30),  # Olanzapine: 1-30 mg
#     "A10BA02": (250, 2000),  # Metformin: 250-2000 mg
# }

# Output directory
OUTPUT_DIR = OUTPUT_PATH.parent

# ==================== END CONFIGURATION =======================


def main():
    # Main function to preprocess medication data.
    print(f"Starting medication preprocessing at {datetime.now()}")
    print(f"Input file: {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if input file exists
    if not INPUT_PATH.exists():
        print(f"Error: Input file not found at {INPUT_PATH}")
        return
    
    
    print("STEP 1: Loading medication data")
    
    # Load medication data
    print(f"Loading medication data from {INPUT_PATH}...")
    medication_df = pd.read_parquet(INPUT_PATH)
    print(f"  Loaded {len(medication_df)} rows with {len(medication_df.columns)} columns")
    
    
    print("STEP 2: Filtering and filling medication timestamps")
    
    # Filter timestamps and fill missing end dates
    medication_df = filter_medication_timestamps(
        df=medication_df,
        start_col=MEDICATION_START_COL,
        discontinued_col=MEDICATION_DISCONTINUED_COL,
        administration_col=MEDICATION_ADMINISTRATION_COL,
        date_lo=DATE_LO,
        date_hi=DATE_HI,
        ongoing_fill_date=ONGOING_FILL_DATE,
    )
    print(f"Medication data shape after timestamp filtering: {medication_df.shape}")
    
    
    print("STEP 3: Cleaning and calculating daily dosages")
    
    # Clean dosage frequency descriptions
    print("Cleaning dosage frequency descriptions...")
    medication_df["Frequency_clean"] = medication_df[MEDICATION_FREQUENCY_COL].apply(
        clean_dosage_frequency
    )
    
    # Calculate daily dosage from frequency
    print("Calculating daily dosages...")
    medication_df = calculate_daily_dosage(medication_df)
    
    # Fill missing daily dosage with original strength
    medication_df["StrengthNumeric_daily"] = medication_df["Frequency_daily"] * medication_df[MEDICATION_STRENGTH_COL]
    medication_df["StrengthNumeric_daily"].fillna(medication_df[MEDICATION_STRENGTH_COL], inplace=True)
    
    print(f"Medication data shape after dosage calculation: {medication_df.shape}")
    
    
    print("STEP 4: Optional dosage filtering by ATC code ranges")
    
    if DOSAGE_RANGES:
        print("Applying dosage range filtering...")
        medication_df = dosage_filtering(
            df=medication_df,
            atc_col=MEDICATION_ATC_COL,
            strength_col=MEDICATION_STRENGTH_COL,
            dosage_ranges=DOSAGE_RANGES,
        )
    else:
        print("Skipping dosage range filtering (DOSAGE_RANGES is None)")
    
    
    print("STEP 5: Collapsing medication intervals")
    
    # Collapse overlapping medication intervals
    medication_df = medication_intervals(
        df=medication_df,
        patientid_col=MEDICATION_PATIENTID_COL,
        dosage_num_col=MEDICATION_STRENGTH_COL,
        atc_col=MEDICATION_ATC_COL,
        start_col=MEDICATION_START_COL,
        discontinued_col=MEDICATION_DISCONTINUED_COL,
        frequency_col=MEDICATION_FREQUENCY_COL,
        dosage_daily_col="StrengthNumeric_daily",
    )
    print(f"Medication data shape after interval collapsing: {medication_df.shape}")
    
    
    print("STEP 6: Saving processed data")
    
    # Save processed medication data
    print(f"Saving processed medication data to {OUTPUT_PATH}...")
    medication_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✓ Successfully saved {len(medication_df)} rows")
    
    
    print(f"Medication preprocessing completed at {datetime.now()}")
    

if __name__ == "__main__":
    main()
