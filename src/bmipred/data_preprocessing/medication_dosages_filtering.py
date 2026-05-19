# preprocessing/medication_dosages.py

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # deactivate warnings



def clean_dosage_frequency(freq_str: Any) -> str:
    # ---------------------
    # Handle missing values
    # ---------------------
    if pd.isna(freq_str):
        return "unspecified"

    freq_str = str(freq_str).lower().strip()

    if freq_str in {"", "nan", "none", "nat", "<na>"}:
        return "unspecified"

    # Remove periods and commas
    freq_str = freq_str.replace(".", "").replace(",", "")

    # Replace hyphens and other punctuation with spaces
    freq_str = re.sub(r"[^\w\s]", " ", freq_str)

    # Normalize repeated spaces
    freq_str = re.sub(r"\s+", " ", freq_str).strip()

    # -----------------------------
    # PRN / as-needed handling
    # -----------------------------
    # "behov" means PRN/as needed.
    # Even if the text says "ved behov højst 3 gange dagligt",
    # this is not fixed daily use, so return "as needed".
    if "behov" in freq_str:
        return "as needed"

    # -----------------------------
    # Replace Danish number words with numerals
    # -----------------------------
    # Important:
    # - "to" can mean 2, but also Thursday abbreviation.
    # - "ti" can mean 10, but also Tuesday abbreviation.
    #
    # Therefore, we only convert "to" and "ti" when they are likely numbers,
    # for example before "gang/gange", "tablet", "time", "dag", etc.
    # Other number words are safe to replace globally.
    dansk_numbers = {
        "nul": "0",
        "en": "1",
        "én": "1",
        "et": "1",
        # "to": handled separately because of ambiguity with "torsdag"
        "tre": "3",
        "fire": "4",
        "fem": "5",
        "seks": "6",
        "syv": "7",
        "otte": "8",
        "ni": "9",
        # "ti": handled separately because of ambiguity with "tirsdag"
        "elleve": "11",
        "tolv": "12",
        "tretten": "13",
        "fjorten": "14",
        "femten": "15",
        "seksten": "16",
        "sytten": "17",
        "atten": "18",
        "nitten": "19",
        "tyve": "20",
    }

    for word, numeral in dansk_numbers.items():
        freq_str = re.sub(rf"\b{re.escape(word)}\b", numeral, freq_str)

    # Contextual conversion of "to" = 2
    freq_str = re.sub(
        r"\bto\b(?=\s+(gang|gange|tablet|tabletter|kapsel|kapsler|time|timer|dag|dage|uge|uger|måned|måneder|år))",
        "2",
        freq_str
    )

    # Contextual conversion of "ti" = 10
    freq_str = re.sub(
        r"\bti\b(?=\s+(gang|gange|tablet|tabletter|kapsel|kapsler|time|timer|dag|dage|uge|uger|måned|måneder|år))",
        "10",
        freq_str
    )

    # -----------------------------
    # Initialize variables
    # -----------------------------
    times_per_unit = None
    time_unit = None
    category = None
    matched = False

    # -----------------------
    # Main frequency pattern
    # -----------------------
    patterns = [
        # Maximum frequency, treated as the numeric daily/weekly/monthly/yearly frequency
        (r"højst (\d+) gange dagligt", "daily"),
        (r"højst (\d+) gange daglig", "daily"),
        (r"højst (\d+) gange om dagen", "daily"),
        (r"højst (\d+) gange ugentligt", "weekly"),
        (r"højst (\d+) gange om ugen", "weekly"),
        (r"højst (\d+) gange månedligt", "monthly"),
        (r"højst (\d+) gange om måneden", "monthly"),
        (r"højst (\d+) gange årligt", "yearly"),
        (r"højst (\d+) gange om året", "yearly"),
        (r"højst (\d+) gange", "unspecified"),

        # Standard repeated frequencies
        (r"(\d+) gange dagligt", "daily"),
        (r"(\d+) gange daglig", "daily"),
        (r"(\d+) gange om dagen", "daily"),
        (r"(\d+) gange ugentligt", "weekly"),
        (r"(\d+) gange om ugen", "weekly"),
        (r"(\d+) gange månedligt", "monthly"),
        (r"(\d+) gange om måneden", "monthly"),
        (r"(\d+) gange årligt", "yearly"),
        (r"(\d+) gange om året", "yearly"),

        # Singular "gang"
        (r"(\d+) gang dagligt", "daily"),
        (r"(\d+) gang daglig", "daily"),
        (r"(\d+) gang om dagen", "daily"),
        (r"(\d+) gang ugentligt", "weekly"),
        (r"(\d+) gang om ugen", "weekly"),
        (r"(\d+) gang månedligt", "monthly"),
        (r"(\d+) gang om måneden", "monthly"),
        (r"(\d+) gang årligt", "yearly"),
        (r"(\d+) gang om året", "yearly"),

        # One-off treatment
        (r"kun 1 gang", "once"),
        (r"kun en gang", "once"),
        (r"kun 1", "once"),

        # Continuous treatment
        (r"kontinuerligt", "continuous"),
    ]

    # -----------------------------
    # Try matching explicit frequency patterns
    # -----------------------------
    for pattern, unit in patterns:
        match = re.search(pattern, freq_str)

        if match:
            matched = True

            if unit == "once":
                times_per_unit = 1
                time_unit = "once"

            elif unit == "continuous":
                times_per_unit = None
                time_unit = "continuous"

            elif unit == "unspecified":
                times_per_unit = int(match.group(1))
                time_unit = "unspecified"

            else:
                times_per_unit = int(match.group(1))
                time_unit = unit

            break

    # -----------------------------
    # Check for "hver X ..." patterns
    # -----------------------------
    if not matched:
        match = re.search(
            r"hver\s+(\d+)\.?\s*(time|timer|dag|dage|uge|uger|måned|måneder|år)",
            freq_str
        )

        if match:
            matched = True
            interval = int(match.group(1))
            unit = match.group(2)

            unit_translation = {
                "time": "hour(s)",
                "timer": "hour(s)",
                "dag": "day(s)",
                "dage": "day(s)",
                "uge": "week(s)",
                "uger": "week(s)",
                "måned": "month(s)",
                "måneder": "month(s)",
                "år": "year(s)",
            }

            unit_eng = unit_translation.get(unit, unit)
            category = f"every {interval} {unit_eng}"

    # -----------------------------
    # Check for days of the week
    # Example: "to og fredag" -> weekly 2
    # Example: "ti og torsdag" -> weekly 2
    # -----------------------------
    if not matched:
        day_aliases = {
            "mandag": "mandag",
            "man": "mandag",
            "ma": "mandag",

            "tirsdag": "tirsdag",
            "tir": "tirsdag",
            "ti": "tirsdag",

            "onsdag": "onsdag",
            "ons": "onsdag",
            "on": "onsdag",

            "torsdag": "torsdag",
            "tor": "torsdag",
            "to": "torsdag",

            "fredag": "fredag",
            "fre": "fredag",
            "fr": "fredag",

            "lørdag": "lørdag",
            "lør": "lørdag",
            "lø": "lørdag",

            "søndag": "søndag",
            "søn": "søndag",
            "sø": "søndag",
        }

        day_pattern = (
            r"\b("
            + "|".join(sorted((re.escape(k) for k in day_aliases.keys()), key=len, reverse=True))
            + r")\b"
        )

        days_found = re.findall(day_pattern, freq_str)
        days_mapped = [day_aliases[day] for day in days_found]
        unique_days = set(days_mapped)
        number_of_days = len(unique_days)

        if number_of_days > 0:
            matched = True
            times_per_unit = number_of_days
            time_unit = "weekly"

    # -----------------------------
    # Check for times of day
    # Example: "morgen og aften" -> daily 2
    # -----------------------------
    if not matched:
        times_of_day = [
            "tidlig morgen",
            "sen eftermiddag",
            "før sengetid",
            "morgen",
            "middag",
            "aften",
            "nat",
        ]

        times_of_day_sorted = sorted(times_of_day, key=len, reverse=True)
        times_pattern = (
            r"\b("
            + "|".join(re.escape(t) for t in times_of_day_sorted)
            + r")\b"
        )

        matches = re.findall(times_pattern, freq_str)
        count_times = len(matches)

        if count_times > 0:
            matched = True
            times_per_unit = count_times
            time_unit = "daily"

    # -----------------------------
    # Handle "dagligt" or "daglig" alone
    # -----------------------------
    if not matched and freq_str.strip() in {"dagligt", "daglig"}:
        times_per_unit = 1
        time_unit = "daily"
        matched = True

    # -----------------------------
    # Construct category
    # Must remain compatible with calculate_daily_dosage()
    # -----------------------------
    if category is None:
        if time_unit == "once":
            category = "once only"

        elif time_unit == "continuous":
            category = "continuous"

        elif time_unit == "unspecified":
            category = f"unspecified {times_per_unit}"

        elif time_unit is not None and times_per_unit is not None:
            category = f"{time_unit} {times_per_unit}"

        else:
            category = "unspecified"

    return category




def calculate_daily_dosage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Frequency_daily"] = np.nan

    # Handle 'daily N'
    mask_daily_N = df["Frequency_clean"].str.match(r"daily (\d+)", na=False)
    df.loc[mask_daily_N, "Frequency_daily"] = (
        df.loc[mask_daily_N, "Frequency_clean"]
        .str.extract(r"daily (\d+)", expand=False)
        .astype(float)
    )

    # Handle 'weekly N'
    mask_weekly_N = df["Frequency_clean"].str.match(r"weekly (\d+)", na=False)
    weekly_N = (
        df.loc[mask_weekly_N, "Frequency_clean"]
        .str.extract(r"weekly (\d+)", expand=False)
        .astype(float)
    )
    df.loc[mask_weekly_N, "Frequency_daily"] = weekly_N / 7

    # Handle 'monthly N'
    mask_monthly_N = df["Frequency_clean"].str.match(r"monthly (\d+)", na=False)
    monthly_N = (
        df.loc[mask_monthly_N, "Frequency_clean"]
        .str.extract(r"monthly (\d+)", expand=False)
        .astype(float)
    )
    df.loc[mask_monthly_N, "Frequency_daily"] = monthly_N / 30

    # Handle 'yearly N'
    mask_yearly_N = df["Frequency_clean"].str.match(r"yearly (\d+)", na=False)
    yearly_N = (
        df.loc[mask_yearly_N, "Frequency_clean"]
        .str.extract(r"yearly (\d+)", expand=False)
        .astype(float)
    )
    df.loc[mask_yearly_N, "Frequency_daily"] = yearly_N / 365.25

    # Handle 'every X hour(s)'
    mask_every_X_hours = df["Frequency_clean"].str.match(r"every (\d+) hour\(s\)", na=False)
    hours = (
        df.loc[mask_every_X_hours, "Frequency_clean"]
        .str.extract(r"every (\d+) hour\(s\)", expand=False)
        .astype(float)
    )
    df.loc[mask_every_X_hours, "Frequency_daily"] = 24 / hours

    # Handle 'every X day(s)'
    mask_every_X_days = df["Frequency_clean"].str.match(r"every (\d+) day\(s\)", na=False)
    days = (
        df.loc[mask_every_X_days, "Frequency_clean"]
        .str.extract(r"every (\d+) day\(s\)", expand=False)
        .astype(float)
    )
    df.loc[mask_every_X_days, "Frequency_daily"] = 1 / days

    # Handle 'every X week(s)'
    mask_every_X_weeks = df["Frequency_clean"].str.match(r"every (\d+) week\(s\)", na=False)
    weeks = (
        df.loc[mask_every_X_weeks, "Frequency_clean"]
        .str.extract(r"every (\d+) week\(s\)", expand=False)
        .astype(float)
    )
    df.loc[mask_every_X_weeks, "Frequency_daily"] = 1 / (weeks * 7)

    # Handle 'every X month(s)'
    mask_every_X_months = df["Frequency_clean"].str.match(r"every (\d+) month\(s\)", na=False)
    months = (
        df.loc[mask_every_X_months, "Frequency_clean"]
        .str.extract(r"every (\d+) month\(s\)", expand=False)
        .astype(float)
    )
    df.loc[mask_every_X_months, "Frequency_daily"] = 1 / (months * 30)

    # Handle 'every X year(s)'
    mask_every_X_years = df["Frequency_clean"].str.match(r"every (\d+) year\(s\)", na=False)
    years = (
        df.loc[mask_every_X_years, "Frequency_clean"]
        .str.extract(r"every (\d+) year\(s\)", expand=False)
        .astype(float)
    )
    df.loc[mask_every_X_years, "Frequency_daily"] = 1 / (years * 365.25)

    # Handle 'continuous'
    mask_continuous = df["Frequency_clean"].str.match(r"continuous", na=False)
    df.loc[mask_continuous, "Frequency_daily"] = np.nan

    # For 'once only', 'as needed', and 'unspecified', Frequency_daily remains NaN

    # Compute daily strength
    df["StrengthNumeric_daily"] = df["Frequency_daily"] * df["StrengthNumeric"]

    return df



def dosage_filtering(df: pd.DataFrame,
                     atc_col: str = "ATC",
                     strength_col: str = "StrengthNumeric",
                     dosage_ranges: dict[str, tuple[float, float]] = None) -> pd.DataFrame:
    # Filter df by expected dosage ranges for specific medications by providing as input a dictionary
    # mapping ATC codes to (min, max) dosage ranges.

    print("Number of medication observations before filtering:", df.shape)
    
    if dosage_ranges is None:
        return df  # nothing to filter
    
    for atc_code, (min_val, max_val) in dosage_ranges.items():
        df = df[
            (df[atc_col] != atc_code) |
            ((df[atc_col] == atc_code) & (df[strength_col].between(min_val, max_val)))
        ]
    
    print("Number of medication observations after filtering:", df.shape)
    return df