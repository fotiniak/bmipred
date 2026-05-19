#!/usr/bin/env python3
# Script to preprocess diagnosis data by completing missing SKSCodes and collapsing intervals.
# All configuration parameters are defined at the beginning of the script.

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from bmipred.data_preprocessing.diagnosis_codes_completion import diagnosis_codes_completion
from bmipred.data_preprocessing.diagnosis_intervals import diagnosis_intervals

# ==================== CONFIGURATION PARAMETERS ====================

# Input and output paths
INPUT_PATH = Path(__file__).parent.parent / "data" / "preprocessed" / "diagnosis.parquet"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "preprocessed" / "diagnosis_collapsed.parquet"

# Reference data paths
SKSCODDES_REFERENCE_PATH = Path(__file__).parent.parent / "data" / "external" / "SKSCodes.csv"
SKSCODES_GROUPED_PATH = Path(__file__).parent.parent / "data" / "external" / "SKSCodes_Grouped.csv"

# Column names in diagnosis dataframe
DIAGNOSIS_CODE_COL = "SKSCode"
DIAGNOSIS_DESC_COL = "DiagnosisName"
DIAGNOSIS_PATIENTID_COL = "PatientDurableKey"
DIAGNOSIS_STARTDATE_COL = "DiagnosisStartDate"
DIAGNOSIS_ENDDATE_COL = "DiagnosisEndDate"

# Column names in SKS code reference dataframes
REFERENCE_CODE_COL = "Kode"
REFERENCE_DESC_COL = "Tekst"
GROUPED_CODES_COL = "SKSCodes"
GROUPED_DESC_COL = "Description"
GROUPED_RANGE_COL = "SKSCodes_range"

# Fuzzy matching threshold for code completion (0-100)
FUZZY_MATCH_THRESHOLD = 89

# Output directory
OUTPUT_DIR = OUTPUT_PATH.parent

# ==================== END CONFIGURATION =======================


def main():
    # Main function to preprocess diagnosis data.
    print(f"Starting diagnosis preprocessing at {datetime.now()}")
    print(f"Input file: {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if input file exists
    if not INPUT_PATH.exists():
        print(f"Error: Input file not found at {INPUT_PATH}")
        return
    
    
    print("STEP 1: Loading data")
    
    # Load diagnosis data
    print(f"Loading diagnosis data from {INPUT_PATH}...")
    diagnosis_df = pd.read_parquet(INPUT_PATH)
    print(f"  Loaded {len(diagnosis_df)} rows with {len(diagnosis_df.columns)} columns")
    
    # Load reference SKSCodes
    print(f"Loading reference SKSCodes from {SKSCODDES_REFERENCE_PATH}...")
    sks_codes_ref = pd.read_csv(SKSCODDES_REFERENCE_PATH, sep=";", quotechar='"')
    print(f"  Loaded {len(sks_codes_ref)} reference codes")
    
    # Load grouped SKSCodes
    print(f"Loading grouped SKSCodes from {SKSCODES_GROUPED_PATH}...")
    sks_codes_grouped = pd.read_csv(SKSCODES_GROUPED_PATH)
    print(f"  Loaded {len(sks_codes_grouped)} code groups")
    
    
    print("STEP 2: Completing missing SKSCodes")

    # Complete missing SKSCodes using fuzzy matching and pattern extraction
    diagnosis_df = diagnosis_codes_completion(
        df=diagnosis_df,
        official_codes=sks_codes_ref,
        df_code_col=DIAGNOSIS_CODE_COL,
        df_desc_col=DIAGNOSIS_DESC_COL,
        official_code_col=REFERENCE_CODE_COL,
        official_desc_col=REFERENCE_DESC_COL,
        threshold=FUZZY_MATCH_THRESHOLD
    )
    print(f"Diagnosis data shape after code completion: {diagnosis_df.shape}")
    
    
    print("STEP 3: Collapsing diagnosis intervals")
    
    # Collapse overlapping diagnosis intervals
    diagnosis_df = diagnosis_intervals(
        df=diagnosis_df,
        codesMap=sks_codes_grouped,
        df_codes_col=DIAGNOSIS_CODE_COL,
        df_patientid_col=DIAGNOSIS_PATIENTID_COL,
        df_startdate_col=DIAGNOSIS_STARTDATE_COL,
        df_enddate_col=DIAGNOSIS_ENDDATE_COL,
        codesMap_codes_col=GROUPED_CODES_COL,
        codesMap_desc_col=GROUPED_DESC_COL,
        codesMap_groups_col=GROUPED_RANGE_COL
    )
    print(f"Diagnosis data shape after interval collapsing: {diagnosis_df.shape}")
    
    
    print("STEP 4: Saving processed data")
    
    # Save processed diagnosis data
    print(f"Saving processed diagnosis data to {OUTPUT_PATH}...")
    diagnosis_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✓ Successfully saved {len(diagnosis_df)} rows")
    
    
    print(f"Diagnosis preprocessing completed at {datetime.now()}")
    

if __name__ == "__main__":
    main()
