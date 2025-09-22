# python3
# preprocessing/medication_intervals.py

import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
import warnings

warnings.filterwarnings("ignore")  # deactivate warnings

def medication_intervals(df:pd.DataFrame,
                         patientid_col:str = 'PatientDurableKey',
                         dosage_col:str = 'Strength',
                         dosage_num_col:str = 'StrengthNumeric',
                         atc_col:str = 'ATC',
                         start_col:str = 'StartInstant',
                         discontinued_col:str = 'DiscontinuedInstant',
                         frequency_col:str = 'Frequency',
                         dosage_daily_col:str = 'StrengthNumeric_daily') -> pd.DataFrame:

    start_time = time.time()

    print("Number of df observations in the medication table: ", len(df))
    df = df[[patientid_col, dosage_col, dosage_num_col, atc_col, start_col, discontinued_col, frequency_col, dosage_daily_col]]
    df = df.drop_duplicates() # remove duplicates
    print("Number of df observations in the medication table after removing duplicates: ", len(df))

    # Drop rows missing StartInstant
    df = df.dropna(subset=[start_col])
    # Make sure start_col is a datetime
    df[start_col] = pd.to_datetime(df[start_col])

    # # # --------------- PROCESS OF CREATING THE MEDICATION INTERVALS ------------------- # # #
    
    # STEP 1: Be sure to complete the NA values in the DiscontinuedInstant with the current date: # I have skipped that for now
    #m[discontinued_col].fillna(pd.Timestamp(datetime.now()), inplace=True) # where there is no DiscontinuedInstant i fill the NA values with today's date...
    df[discontinued_col] = df[discontinued_col].fillna(df[start_col])

    # Ensure DiscontinuedInstant >= StartInstant
    mask = df[discontinued_col] < df[start_col]
    df.loc[mask, discontinued_col] = df.loc[mask, start_col]

    # Start the process of concatenating df intervals:
    df = df.sort_values(by=[patientid_col, atc_col, start_col, discontinued_col], ascending=[True, True, True, True])

    # Check the length of the dataframe and the iteration number at every round of iteration:
    previous_length = len(df) # track the length of the dataframe
    iteration = 0  # track the iteration number

    while True:
        # Process of concatenating df intervals:
        iteration += 1  # Increment the iteration number
        print(f"This is iteration number: {iteration}")

        # STEP 2: Sort based on the df start date:
        df = df.sort_values(by=[patientid_col, atc_col, start_col, discontinued_col], ascending=[True, True, True, True])

        # STEP3: Group by patientid_col, then use 'shift' to get the start_col and discontinued_col of the previous row
        df['StartInstant_next'] = df.groupby([patientid_col, atc_col])[start_col].shift(-1)
        df['DiscontinuedInstant_next'] = df.groupby([patientid_col, atc_col])[discontinued_col].shift(-1)

        # STEP 4: Calculate the extending discontinued_col by 48 hours
        df['DiscontinuedInstant_ext'] = df[discontinued_col] + pd.Timedelta(hours=48)

        # STEP 5: Tag each row with True or False based on the conditions that indicate it is an overlapping interval
        df['IsSameInterval'] = (df['StartInstant_next'] >= df[start_col]) & (df['StartInstant_next'] <= df['DiscontinuedInstant_ext'])

        # STEP 6: Create groups based on the identified intervals
        # Create groups by shifting `IsSameInterval` and then using cumsum
        df['GroupNumber'] = (~df['IsSameInterval'].shift(fill_value=False)).cumsum()

        # STEP 7: Calculate the duration of df
        df['MedicationDuration'] = (df[discontinued_col] - df[start_col]).dt.days

        # STEP 8: Calculate the *weighted* Strength of df based on the duration
        df['StrengthNumeric_weight'] = df[dosage_num_col] * df['MedicationDuration']
        df['StrengthNumeric_daily_weight'] = df[dosage_daily_col] * df['MedicationDuration']

        # STEP 9: Collapse the rows based on the new Updated Intervals

        df = df.groupby([patientid_col, 'GroupNumber']).agg({
            patientid_col: 'first',
            atc_col: 'first',
            dosage_num_col: 'mean',
            dosage_daily_col: 'mean',
            'StrengthNumeric_weight': 'sum',
            'StrengthNumeric_daily_weight': 'sum',
            'MedicationDuration': 'sum',
            start_col: 'min',
            discontinued_col: 'max'
        }).reset_index(drop=True)

        # STEP 10: Calculate the weighted mean for the Strength of the df 
        # Depending on the number of days for which they have received it:
        df['DailyDosage_weighted_mean'] = df['StrengthNumeric_daily_weight'] / df['MedicationDuration']

        df['DailyDosage_weighted_mean_nafilled'] = df['StrengthNumeric_daily_weight'] / df['MedicationDuration']
        df['DailyDosage_weighted_mean_nafilled'] = df['DailyDosage_weighted_mean_nafilled'].fillna(df[dosage_num_col])

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

        # NOTE: What happens here is that in case we have a StartInstant and then a DiscontinuedInstant does not exist, if the StartInstant has happened while we have another time interval then it gets absorved in that interval and we assume that since there is a discontinued date for that df after the StartInstant it stops. However, if that StartInstant has happened in some timepoint that does not overlap with and already existing interval then we just keep it as it is and the DiscontinuedInstant is unknown. 
    print("Final number of df rows after filtering intervals: ", len(df))
    end_time = time.time()
    print(f"Time taken for creating the df intervals: {end_time - start_time} seconds")
    return df