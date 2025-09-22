# preprocessing/medication_dosage.py

import os
import sys
import re
import time
import warnings
from typing import Any, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # deactivate warnings

def dosage_filtering(df: pd.DataFrame,
                     atc_col: str = "ATC",
                     strength_col: str = "StrengthNumeric",
                     dosage_ranges: dict[str, tuple[float, float]] = None) -> pd.DataFrame:
    """
    filter df by expected dosage ranges for specific medications by providing as input a dictionary
    mapping ATC codes to (min, max) dosage ranges.

    parameters:
    - df: input dataframe
    - atc_col: name of the ATC column in df
    - strength_col: name of the strength column in df
    - dosage_ranges: dict mapping ATC codes -> (min, max) dosage ranges
    
    returns:
    - filtered dataframe
    """
    print("number of observations before filtering:", df.shape)
    
    if dosage_ranges is None:
        return df  # nothing to filter
    
    for atc_code, (min_val, max_val) in dosage_ranges.items():
        df = df[
            (df[atc_col] != atc_code) |
            ((df[atc_col] == atc_code) & (df[strength_col].between(min_val, max_val)))
        ]
    
    print("number of observations after filtering:", df.shape)
    return df


def convert_danish_dosage_frequency(freq_str: Any) -> str:
    freq_str = str(freq_str).lower()
    # Remove periods and commas
    freq_str = freq_str.replace('.', '').replace(',', '')
    # Replace hyphens and other punctuation with spaces
    freq_str = re.sub(r'[^\w\s]', ' ', freq_str)
    
    # Replace Danish number words with numerals
    dansk_numbers = {
        'nul': '0',
        'en': '1',
        'én': '1',
        'et': '1',
        'to': '2',
        'tre': '3',
        'fire': '4',
        'fem': '5',
        'seks': '6',
        'syv': '7',
        'otte': '8',
        'ni': '9',
        'ti': '10',
        'elleve': '11',
        'tolv': '12',
        'tretten': '13',
        'fjorten': '14',
        'femten': '15',
        'seksten': '16',
        'sytten': '17',
        'atten': '18',
        'nitten': '19',
        'tyve': '20',
    }
    for word, numeral in dansk_numbers.items():
        freq_str = re.sub(r'\b{}\b'.format(re.escape(word)), numeral, freq_str)
    
    # Initialize variables
    as_needed = False
    times_per_unit = None
    time_unit = None
    category = None
    matched = False
    
    # Check for 'behov' (as needed)
    if 'behov' in freq_str:
        as_needed = True
        matched = True
    
    # Patterns
    patterns = [
        (r'højst (\d+) gange dagligt', 'daily'),
        (r'højst (\d+) gange daglig', 'daily'),
        (r'højst (\d+) gange om dagen', 'daily'),
        (r'højst (\d+) gange', 'unspecified'),
        (r'(\d+) gange dagligt', 'daily'),
        (r'(\d+) gange daglig', 'daily'),
        (r'(\d+) gange om dagen', 'daily'),
        (r'(\d+) gange ugentligt', 'weekly'),
        (r'(\d+) gange om ugen', 'weekly'),
        (r'(\d+) gange månedligt', 'monthly'),
        (r'(\d+) gange om måneden', 'monthly'),
        (r'(\d+) gange årligt', 'yearly'),
        (r'(\d+) gange om året', 'yearly'),
        (r'(\d+) gang daglig', 'daily'),
        (r'(\d+) gang dagligt', 'daily'),
        (r'(\d+) gang om dagen', 'daily'),
        (r'(\d+) gang ugentligt', 'weekly'),
        (r'(\d+) gang om ugen', 'weekly'),
        (r'(\d+) gang månedligt', 'monthly'),
        (r'(\d+) gang om måneden', 'monthly'),
        (r'(\d+) gang årligt', 'yearly'),
        (r'(\d+) gang om året', 'yearly'),
        (r'kun 1 gang', 'once'),
        (r'kun én gang', 'once'),
        (r'kun en gang', 'once'),
        (r'kun 1', 'once'),
        (r'kontinuerligt', 'continuous'),
    ]
    
    # Try matching the patterns
    if not matched:
        for pattern, unit in patterns:
            match = re.search(pattern, freq_str)
            if match:
                matched = True
                if unit == 'once':
                    times_per_unit = 1
                    time_unit = 'once'
                elif unit == 'continuous':
                    times_per_unit = None
                    time_unit = 'continuous'
                elif unit == 'unspecified':
                    times_per_unit = int(match.group(1))
                    time_unit = 'unspecified'
                else:
                    times_per_unit = int(match.group(1))
                    time_unit = unit
                break
        
    # Check for 'hver' patterns
    if not matched:
        match = re.search(r'hver (\d+)\.?\s*(time|dag|uge|måned|år)', freq_str)
        if match:
            matched = True
            interval = int(match.group(1))
            unit = match.group(2)
            # Translate unit
            unit_translation = {
                'time': 'hour(s)',
                'dag': 'day(s)',
                'uge': 'week(s)',
                'måned': 'month(s)',
                'år': 'year(s)',
            }
            unit_eng = unit_translation.get(unit, unit)
            category = f'every {interval} {unit_eng}'
    
    # Check for days of the week (including abbreviations)
    if not matched:
        # Define day aliases
        day_aliases = {
            'mandag': 'mandag',
            'man': 'mandag',
            'ma': 'mandag',
            'tirsdag': 'tirsdag',
            'tir': 'tirsdag',
            'ti': 'tirsdag',
            'onsdag': 'onsdag',
            'ons': 'onsdag',
            'on': 'onsdag',
            'torsdag': 'torsdag',
            'tor': 'torsdag',
            'to': 'torsdag',
            'fredag': 'fredag',
            'fre': 'fredag',
            'fr': 'fredag',
            'lørdag': 'lørdag',
            'lør': 'lørdag',
            'lø': 'lørdag',
            'søndag': 'søndag',
            'søn': 'søndag',
            'sø': 'søndag',
        }
        # Create a regex pattern to match any day alias
        day_pattern = r'\b(' + '|'.join(sorted((re.escape(k) for k in day_aliases.keys()), key=len, reverse=True)) + r')\b'
        # Find all days mentioned
        days_found = re.findall(day_pattern, freq_str)
        # Map abbreviations to full names
        days_mapped = [day_aliases[day] for day in days_found]
        # Get unique days
        unique_days = set(days_mapped)
        number_of_days = len(unique_days)
        if number_of_days > 0:
            matched = True
            times_per_unit = number_of_days
            time_unit = 'weekly'
    
    # If no match, check for times of day
    if not matched:
        times_of_day = ['tidlig morgen', 'sen eftermiddag', 'før sengetid', 'morgen', 'middag', 'aften', 'nat']
        # Sort times_of_day to match longer phrases first
        times_of_day_sorted = sorted(times_of_day, key=len, reverse=True)
        # Create regex pattern
        times_pattern = r'\b(' + '|'.join(re.escape(t) for t in times_of_day_sorted) + r')\b'
        matches = re.findall(times_pattern, freq_str)
        count_times = len(matches)
        if count_times > 0:
            times_per_unit = count_times
            time_unit = 'daily'
            matched = True
    
    # Handle 'dagligt' or 'daglig' on their own
    if not matched and freq_str.strip() in ['dagligt', 'daglig']:
        times_per_unit = 1
        time_unit = 'daily'
        matched = True
    
    # Construct the category
    if category is None:
        if as_needed:
            category = 'as needed'
        elif time_unit:
            if time_unit == 'once':
                category = 'once only'
            elif time_unit == 'continuous':
                category = 'continuous'
            elif time_unit == 'unspecified':
                category = f'unspecified {times_per_unit}'
            else:
                category = f'{time_unit} {times_per_unit}'
        else:
            category = 'unspecified'
    return category


def clean_medication_frequency(df: pd.DataFrame, frequency_col: str = 'Frequency') -> pd.DataFrame:
    df['Frequency_clean'] = df[frequency_col].apply(convert_danish_dosage_frequency)
    return df

def calculate_daily_dosage(df: pd.DataFrame) -> pd.DataFrame:

    df['Frequency_daily'] = np.nan

    # Handle 'daily N'
    mask_daily_N = df['Frequency_clean'].str.match(r'daily (\d+)', na=False)
    df.loc[mask_daily_N, 'Frequency_daily'] = (
        df.loc[mask_daily_N, 'Frequency_clean']
        .str.extract(r'daily (\d+)', expand=False)
        .astype(float)
    )

    # Handle 'every X hour(s)'
    mask_every_X_hours = df['Frequency_clean'].str.match(r'every (\d+) hour\(s\)', na=False)
    hours = (
        df.loc[mask_every_X_hours, 'Frequency_clean']
        .str.extract(r'every (\d+) hour\(s\)', expand=False)
        .astype(float)
    )
    df.loc[mask_every_X_hours, 'Frequency_daily'] = 24 / hours

    # Handle 'every X day(s)'
    mask_every_X_days = df['Frequency_clean'].str.match(r'every (\d+) day\(s\)', na=False)
    days = (
        df.loc[mask_every_X_days, 'Frequency_clean']
        .str.extract(r'every (\d+) day\(s\)', expand=False)
        .astype(float)
    )
    df.loc[mask_every_X_days, 'Frequency_daily'] = 1 / days

    # Handle 'every X week(s)'
    mask_every_X_weeks = df['Frequency_clean'].str.match(r'every (\d+) week\(s\)', na=False)
    weeks = (
        df.loc[mask_every_X_weeks, 'Frequency_clean']
        .str.extract(r'every (\d+) week\(s\)', expand=False)
        .astype(float)
    )
    df.loc[mask_every_X_weeks, 'Frequency_daily'] = 1 / (weeks * 7)

    # Handle 'every X month(s)'
    mask_every_X_months = df['Frequency_clean'].str.match(r'every (\d+) month\(s\)', na=False)
    months = (
        df.loc[mask_every_X_months, 'Frequency_clean']
        .str.extract(r'every (\d+) month\(s\)', expand=False)
        .astype(float)
    )
    df.loc[mask_every_X_months, 'Frequency_daily'] = 1 / (months * 30)  # Approximate month length as 30 days

    # Handle 'weekly N'
    mask_weekly_N = df['Frequency_clean'].str.match(r'weekly (\d+)', na=False)
    weekly_N = (
        df.loc[mask_weekly_N, 'Frequency_clean']
        .str.extract(r'weekly (\d+)', expand=False)
        .astype(float)
    )
    df.loc[mask_weekly_N, 'Frequency_daily'] = weekly_N / 7

    # Handle 'continuous'
    mask_continuous = df['Frequency_clean'].str.match(r'continuous', na=False)
    df.loc[mask_continuous, 'Frequency_daily'] = np.nan

    # For 'once only', 'as needed', and 'unspecified', Frequency_daily remains NaN

    # Now compute 'StrengthNumeric_daily'
    df['StrengthNumeric_daily'] = df['Frequency_daily'] * df['StrengthNumeric']

    return df

