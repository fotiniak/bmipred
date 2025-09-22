# python3
# preprocessing/medication_intervals.py

import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
import warnings

warnings.filterwarnings("ignore")  # deactivate warnings


def extract_atc_info(df: pd.DataFrame,
                     patientid_col: str = "PatientDurableKey",
                     atc_col: str = "ATC",
                     start_col: str = "StartInstant",) -> tuple[pd.DataFrame, dict, dict, set, set, set]:
    '''prepare df by mapping ATC codes to IDs and return lookup dictionaries and ID sets.'''
    df[start_col] = pd.to_datetime(df[start_col])
    df = df.sort_values([patientid_col, start_col]).reset_index(drop=True)

    df[atc_col] = df[atc_col].astype(str).fillna('')
    unique_atc_codes = df[atc_col].unique()
    atc_code_to_id = {code: idx for idx, code in enumerate(unique_atc_codes)}
    id_to_atc_code = {idx: code for code, idx in atc_code_to_id.items()}
    df['ATC_ID'] = df[atc_col].map(atc_code_to_id).astype(np.int32)

    # define lists of ATC codes of interest for extra info calculations
    antipsychotics_atc_codes = [
        'N05AD01','N05AD08','N05AH02','N05AH03','N05AH04','N05AX08','N05AX12','N05AF05',
        'N05AX16','N05AF03','N05AE05','N05AE03','N05AG02','N05AX13','N05AA02','N05AE04',
        'N05AF01','N05AL05','N05AX','N05AB03','N05AX15','N05AC01','N05AL01','N05AD05',
        'N05AB04','N05AD03','N05AG03','N05AH05','N05AA01','N05AC02'
    ]
    antidepressants_atc_codes = [
        'N06AX11','N06AB04','N06AX16','N06AX26','N06AB06','N06AX21','N06AA10','N06AX03',
        'N06AB05','N06AA09','N06AB03','N06AA02','N06AB10','N06AA04','N06AX22','N06AX12',
        'N06AF01','N06AA16','N06AG02','N06AB08','N06AX18','N06AA12','N06AX25','N06AX27','N06AA21'
    ]
    anxiolytics_atc_codes = [
        'N05BA04','N05BA01','N05BB01','N05BA06','N05BA02','N05BA12','N05BA09','N05BA08','N05BE01'
    ]

    # map lists to sets of IDs
    antipsychotics_ids = {atc_code_to_id[code] for code in antipsychotics_atc_codes if code in atc_code_to_id}
    antidepressants_ids = {atc_code_to_id[code] for code in antidepressants_atc_codes if code in atc_code_to_id}
    anxiolytics_ids = {atc_code_to_id[code] for code in anxiolytics_atc_codes if code in atc_code_to_id}

    return atc_code_to_id, id_to_atc_code, antipsychotics_ids, antidepressants_ids, anxiolytics_ids



def compute_cumulative_unique_atc(group, atc_code_to_id, id_to_atc_code, antipsychotics_ids, antidepressants_ids, anxiolytics_ids):
    atc_ids = group['ATC_ID'].values
    n = len(atc_ids)
    counts = np.zeros(n, dtype=np.int32)
    atc_codes_list = [''] * n
    num_times_on_antipsychotics = np.zeros(n, dtype=np.int32)
    num_times_on_antidepressants = np.zeros(n, dtype=np.int32)
    num_times_on_anxiolytics = np.zeros(n, dtype=np.int32)
    unique_ids = set()
    num_antipsychotics = 0
    num_antidepressants = 0
    num_anxiolytics = 0
    for i in range(n):
        counts[i] = len(unique_ids)
        if unique_ids:
            atc_codes_list[i] = ','.join(str(id) for id in unique_ids)
        else:
            atc_codes_list[i] = ''
        # Record number of times on antipsychotics before now
        num_times_on_antipsychotics[i] = num_antipsychotics
        # Record number of times on antidepressants before now
        num_times_on_antidepressants[i] = num_antidepressants
        # Record number of times on anxiolytics before now
        num_times_on_anxiolytics[i] = num_anxiolytics
        # Add current ATC ID to unique_ids
        unique_ids.add(atc_ids[i])
        # If current ATC ID is in antipsychotics_ids, increment counter
        if atc_ids[i] in antipsychotics_ids:
            num_antipsychotics += 1
        # If current ATC ID is in antidepressants_ids, increment counter
        if atc_ids[i] in antidepressants_ids:
            num_antidepressants += 1
        # If current ATC ID is in anxiolytics_ids, increment counter
        if atc_ids[i] in anxiolytics_ids:
            num_anxiolytics += 1
    # Map back the IDs to ATC codes
    atc_codes_list_str = []
    for ids_str in atc_codes_list:
        if ids_str == '':
            atc_codes_list_str.append('')
        else:
            ids = [int(id_str) for id_str in ids_str.split(',')]
            codes = [id_to_atc_code[id] for id in ids]
            atc_codes_list_str.append(','.join(codes))
    group['NumberOfUniqueMedicationsPerscribed_BeforeNow'] = counts
    group['UniqueATC_BeforeStart'] = atc_codes_list_str
    group['NumberOfTimesOnAntipsychotics'] = num_times_on_antipsychotics
    group['NumberOfTimesOnAntidepressants'] = num_times_on_antidepressants
    group['NumberOfTimesOnAnxiolytics'] = num_times_on_anxiolytics
    return group


def compute_overlapping_atc_codes(group,
                                  atc_col: str = "ATC",
                                  start_col: str = "StartInstant",
                                  discontinued_col: str = "DiscontinuedInstant") -> pd.DataFrame:
    # Reset index for safe indexing
    group = group.reset_index(drop=True)
    n = len(group)
    atc_codes = group[atc_col].values
    start_times = group[start_col].values
    end_times = group[discontinued_col].values

    # Build a list of events: start and end of each df interval
    events = []
    for idx in range(n):
        events.append({'time': start_times[idx], 'type': 'start', 'idx': idx})
        # Only add an 'end' event if DiscontinuedInstant is not NaT
        if pd.notna(end_times[idx]):
            events.append({'time': end_times[idx], 'type': 'end', 'idx': idx})

    # Sort events by time; for the same time, 'start' comes before 'end'
    events.sort(key=lambda x: (x['time'], 0 if x['type'] == 'start' else 1))

    # Initialize active dfs set and overlapping data structures
    active_dfs = set()
    overlapping_atc_codes = [set() for _ in range(n)]

    # Process events
    for event in events:
        idx = event['idx']
        current_atc = atc_codes[idx]
        if event['type'] == 'start':
            # For the current df, record overlapping ATC codes
            for other_idx in active_dfs:
                other_atc = atc_codes[other_idx]
                if other_atc != current_atc:
                    # Record in both dfs
                    overlapping_atc_codes[idx].add(other_atc)
                    overlapping_atc_codes[other_idx].add(current_atc)
            # Add current df to active set
            active_dfs.add(idx)
        else:
            # Remove df from active set using discard to avoid KeyError
            active_dfs.discard(idx)

    # Prepare the results
    overlapping_counts = np.array([len(codes) for codes in overlapping_atc_codes])
    overlapping_atc_list = [', '.join(sorted(codes)) if codes else '' for codes in overlapping_atc_codes]

    # Assign the results to the group DataFrame
    group['NumberOfOverlappingATC'] = overlapping_counts
    group['OverlappingATC'] = overlapping_atc_list

    return group


