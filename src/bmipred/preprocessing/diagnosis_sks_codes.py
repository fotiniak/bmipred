#!/usr/bin/env python3

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm.notebook import tqdm
import seaborn as sns
import pickle
from dateutil.relativedelta import relativedelta
import re
from thefuzz import process
import warnings
warnings.filterwarnings('ignore') # deactivate warnings


def diagnosis_sks_codes(df: pd.DataFrame, 
                        official_codes: pd.DataFrame, 
                        code_col: str = 'SKSCode', 
                        code_desc_col: str = 'DiagnosisName', 
                        threshold: int = 89) -> pd.DataFrame:
    '''This script preprocesses the diagnosis dataframe in order to fix missing SKSCodes and shorten them into broader categories.'''
    start_time = time.time()
    print("Number of df observations in the df table: ", len(df))
    df.columns

    # Fix the missing SKSCodes and then use the short form - Broader categories in the hierarchical tree
    # Load the official SKS codes from a csv file, which was obtained from medinfo.dk

    # Convert the categories to strings for pattern matching an easier manipulation of the data
    df[code_col] = df[code_col].astype('str')

    # Define the regex pattern for the SKS code search
    code_pattern = r'\bD[A-Z]{1,3}\d{1,6}[A-Z]{0,3}\b'
    print("The threshold for the fuzzy search is set to: ", threshold)

    for index, row in df.iterrows():
        # Check if the SKSCode is NaN
        if pd.isna(row[code_col]) or row[code_col] == 'nan':
            description = row[code_desc_col]

            # Clean the description
            cleaned_description = re.sub(r'[\(\)*\-]+', ' ', description)
            print(f"Cleaned description: {cleaned_description}")
        
            # Find all occurrences of the code pattern
            found_codes = re.findall(code_pattern, cleaned_description)
            print(f"Found codes in the 'dfName' description: {found_codes}")
        
            # Use the first found code, if any, otherwise set code to None
            code = found_codes[0] if found_codes else None

            # If no code is found, proceed with fuzzy matching
            if not code:
                print(f"Code in '{code_desc_col}' NOT FOUND - Looking for a code through fuzzy search...")
                result = process.extractOne(cleaned_description, official_codes['Tekst'].to_list())
                if result:
                    best_match, score = result
                    if score > threshold:
                        print(f"Found a match through fuzzy search: {best_match} with -score: {score}")
                        code = official_codes[official_codes['Tekst'] == best_match]['Kode'].values[0]

            # Update the SKSCode in the DataFrame 'd' directly at the current index
            if code:
                df.at[index, code_col] = code

    # Shorten the SKSCode into a 4 digit code
    df['SKSCode_short'] = df[code_col].str.extract(r'^([A-Za-z]{2}\d{2})')

    # Converting the column into type 'category'
    df['SKSCode_short'] = df['SKSCode_short'].astype('category')
    print("Number of unique SKSCode_short codes: ", df['SKSCode_short'].nunique())
    print(df['SKSCode_short'].value_counts().head(20))  
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Time used for the df SKSCodes preprocessing: ', elapsed_time/60, 'minutes!' )
    return df
