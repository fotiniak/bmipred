#!/usr/bin/env python3

import os
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore') # deactivate warnings

def diagnosis_intervals(df: pd.DataFrame, 
                        codesMap: pd.DataFrame,
                        df_codes_col: str = "SKSCode",
                        df_patientid_col: str = "PatientDurableKey",
                        df_startdate_col: str = "DiagnosisStartDate",
                        df_enddate_col: str = "DiagnosisEndDate",
                        codesMap_codes_col: str = "SKSCodes", 
                        codesMap_desc_col: str = "Description", 
                        codesMap_groups_col: str = "SKSCodes_range") -> pd.DataFrame:
    '''This script preprocesses the diagnosis table in order to concatenate overlapping time intervals for the same patient and diagnosis.'''
    start_time = time.time()
    print("Number of diagnosis observations in the diagnosis table: ", len(df))
    df.drop_duplicates(inplace=True) # remove duplicates
    print("Number of medication observations in the diagnosis table after removing duplicates: ", len(df))
    
    # Preprocess grouped SKSCodes, unpack the list of codes
    print("codesMap columns:", codesMap.columns.tolist())
    print("Number of SKSCode groups in the table: ", len(codesMap))
    print(codesMap.sample(10))

    codesMap2 = []
    for idx, row in codesMap.iterrows():
        codes = row[codesMap_codes_col].split(', ')
        for code in codes:
            codesMap2.append({
                codesMap_desc_col: row[codesMap_desc_col],
                codesMap_groups_col: row[codesMap_groups_col],
                codesMap_codes_col: code
            })

    codesMap2 = pd.DataFrame(codesMap2)
    print("Number of individual SKS codes after unpacking the groups: ", len(codesMap2))
    print(codesMap2.sample(10))

    # Now match the code categories to the original diagnosis table
    # In order to obtain the SKSCodes_range categories in the diagnosis table i can merge the 2 dfs
    df = pd.merge(df, codesMap2, left_on=df_codes_col, right_on=codesMap_codes_col, how='left')

    SKSCodes = df[[df_patientid_col, codesMap_groups_col, codesMap_codes_col]] # simplify the dataframe to only the necessary columns
    # Sorting, resetting index, and dropping duplicates
    SKSCodes = (SKSCodes
                .sort_values(by=[df_patientid_col, codesMap_codes_col, codesMap_groups_col], ascending=True)
                .reset_index(drop=True)
                .drop_duplicates())
    print("SKSCodes.shape after sorting, resetting index, and dropping duplicates:", SKSCodes.shape)

    # Check for duplicate rows explicitly
    duplicates = SKSCodes[SKSCodes.duplicated(subset=[df_patientid_col, codesMap_groups_col], keep=False)]
    print("Number of duplicates before grouping:", duplicates.shape)

    # Grouping and counting occurrences
    SKSCodes_grouped = SKSCodes.groupby([df_patientid_col, codesMap_groups_col]).size().reset_index(name='UniqueDiagnosisCodesCount')
    print("Shape after grouping:", SKSCodes_grouped.shape)

    # Filtering rows with UniqueDiagnosisCodesCount > 0 (to ensure no empty groups)
    SKSCodes_filtered = SKSCodes_grouped[SKSCodes_grouped['UniqueDiagnosisCodesCount'] > 0]
    print("Shape after filtering UniqueDiagnosisCodesCount > 0:", SKSCodes_filtered.shape)
    print("Final shape of SKSCodes_filtered:", SKSCodes_filtered.shape)

    # Merge this count back to the original DataFrame
    df = pd.merge(df, SKSCodes_filtered, on=[df_patientid_col, codesMap_groups_col], how='inner')

    # # # Adjust medication time intervals - Join intervals less than 48 hours apart for the same patient and diagnosis
    # Check the length of the dataframe and the iteration number at every round of iteration:
    previous_length = len(df) # track the length of the dataframe
    iteration = 0  # track the iteration number

    while True:
        # Process of concatenating medication intervals:
        iteration += 1  # Increment the iteration number
        print(f"This is iteration number: {iteration}")

        # STEP 2: Sort based on the medication start date:
        df = df.sort_values(by=[df_patientid_col, codesMap_groups_col, df_startdate_col, df_enddate_col], ascending=[True, True, True, True])

        # STEP3: Group by df_patientid_col, then use 'shift' to get the df_startdate_col and df_enddate_col of the previous row
        df['DiagnosisStartDate_prev'] = df.groupby([df_patientid_col, codesMap_groups_col])[df_startdate_col].shift(1)
        df['DiagnosisEndDate_prev'] = df.groupby([df_patientid_col, codesMap_groups_col])[df_enddate_col].shift(1)

        # STEP 4: Calculate the extending df_enddate_col by 48 hours
        df['DiagnosisEndDate_prev_ext'] = df['DiagnosisEndDate_prev'] + pd.Timedelta(hours=48)

        # STEP 5: Tag each row with True or False based on the conditions that indicate it is an overlapping interval
        df['IsSameInterval'] = (df['DiagnosisStartDate_prev'] <= df[df_startdate_col]) & (df[df_startdate_col] <= df['DiagnosisEndDate_prev_ext'])

        # STEP 6: Create groups based on the identified intervals
        df['GroupNumber'] = (df['IsSameInterval'] == False).cumsum()

        # STEP 7: Calculate the duration of medication
        df['DiagnosisDuration'] = (df[df_enddate_col] - df[df_startdate_col]).dt.days

        # STEP 8: Collapse the rows based on the new Updated Intervals

        df = df.groupby([df_patientid_col, 'GroupNumber']).agg({
            df_patientid_col: 'first',
            codesMap_groups_col: 'first',
            df_codes_col: lambda x: ','.join(sorted(x.dropna().unique())),
            'UniqueDiagnosisCodesCount': 'max',
            'DiagnosisDuration': 'sum',
            df_startdate_col: 'min',
            df_enddate_col: 'max'
        }).reset_index(drop=True)

        # Check if the length of the dataframe has changed:
        current_length = len(df)
        print(f"Previous length was: {previous_length}, Current length is: {current_length}")
        if current_length == previous_length:
            # Stop the loop if the length hasn't changed from the previous iteration
            print("Length hasn't changed. Breaking out of the loop.")
            break
        else:
            # Update the length for the next iteration
            print("Length has changed. Continuing the loop.")
            previous_length = current_length


    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Time used for the diagnosis intervals preprocessing: ', elapsed_time/60, 'minutes!' )
    return df