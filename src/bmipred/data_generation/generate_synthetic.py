#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/bmipred/synthetic_data.py

import random
from datetime import timedelta

import numpy as np
import pandas as pd
from faker import Faker

# helpers

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)


# Generate synthetic data for each table.

# patients
def generate_patients(n, fake):
    # realistic keys
    patient_keys = [200000 + i for i in range(n)]  # smaller numeric keys
    sexes = ["Mand", "Kvinde"]

    rows = []
    for i in range(n):
        birth_date = fake.date_of_birth(minimum_age=18, maximum_age=90)
        # 10% probability of death
        if random.random() < 0.1:
            death_date = fake.date_between(start_date=birth_date, end_date="today")
            status = "Død"
        else:
            death_date = None
        
        # Pre-calculate target age (18-90) for this patient, used across all event tables
        target_age = np.random.randint(18, 91)

        rows.append({
            "PatientDurableKey": patient_keys[i],
            "Sex": random.choice(sexes),
            "BirthDate": birth_date,
            "DeathDate": death_date,
            "TargetAge": target_age,
        })

    return pd.DataFrame(rows)


# healthAssesment
def generate_health_assessment(patients, fake):
    smoking_choices = ["Aldrig", "Tidigare", "Storryger", "Ryger"]

    records = []
    for idx, patient_key in enumerate(patients["PatientDurableKey"]):
        birth_date = patients.loc[idx, "BirthDate"]
        target_age = patients.loc[idx, "TargetAge"]
        
        height = np.random.normal(170, 10) # The height is stable per patient
        height_m = height / 100
        
        # Vary slightly around target age (±3 years)
        age_variance = np.random.randint(-2, 2)
        measurement_age = max(18, min(90, target_age + age_variance))
        
        # Calculate measurement date: BirthDate + measurement_age
        measurement_date = pd.Timestamp(birth_date) + pd.DateOffset(years=measurement_age)
        
        # sample number of measurements (Poisson-like, mean ~5, max capped at 20)
        n_measures = min(np.random.poisson(5), 20)

        for m in range(n_measures):
            # Spread measurements over ~2 years around the target date
            days_offset = np.random.randint(-365, 365)
            create_instant = measurement_date + pd.DateOffset(days=days_offset)
            
            weight = np.random.normal(75000, 15000)
            weight_kg = weight / 1000

            record = {
                "PatientDurableKey": patient_key,
                "CreateInstant": create_instant,
                "SmokingStatus": random.choice(smoking_choices),
                "HeightInCentimeters": height,
                "WeightInGrams": weight,
                "BodyMassIndex": round(weight_kg / (height_m**2), 1),
            }
            records.append(record)

    return pd.DataFrame(records)


# medication
def generate_medication(patients, fake):
    # small reference set of drugs with ATC codes, generic names, strengths
    drug_catalog = [
        {
            "ATC": "A10BA02",
            "StrengthNumeric": 500.0,

        },       
        {
            "ATC": "N05AH03",
            "StrengthNumeric": 10.0,
        },
        {
            "ATC": "N05AH03",
            "StrengthNumeric": 5.0,
        },
        {
            "ATC": "N05AH03",
            "StrengthNumeric": 20.0,

        }, 
        {
            "ATC": "B03BB01",
            "StrengthNumeric": 5.0,
        },
        {
            "ATC": "A04AA01",
            "StrengthNumeric": 4.0,
        },
        {
            "ATC": "N05AH02",
            "StrengthNumeric": 400.0,
        },
    ]

    freq_options = [
        ("efter behov højst 2 gange dagligt", "flergangs-PN"),
        ("Morgen, middag og aften", "flergangs-fast"),
        ("1 gang dagligt", "enkelt-fast"),
    ]

    rows = []
    for idx, patient_key in enumerate(patients["PatientDurableKey"]):
        birth_date = patients.loc[idx, "BirthDate"]
        target_age = patients.loc[idx, "TargetAge"]
        
        # Vary slightly around target age (±5 years for medication to have more spread)
        age_variance = np.random.randint(-2, 2)
        med_age = max(18, min(90, target_age + age_variance))
        
        # Calculate medication start date: BirthDate + med_age
        med_date = pd.Timestamp(birth_date) + pd.DateOffset(years=med_age)
        
        n_meds = random.randint(0, 10)  # up to 10 medications per patient
        for _ in range(n_meds):
            drug = random.choice(drug_catalog)
            # Spread medication starts around the target medication date (±6 months)
            days_offset = np.random.randint(-365, 365)
            start = med_date + pd.DateOffset(days=days_offset)
            discontinued = start + timedelta(days=random.randint(7, 365)) if random.random() < 0.5 else None

            freq, freq_type = random.choice(freq_options)

            rows.append({
                "PatientDurableKey": patient_key,
                "ATC": drug["ATC"],
                "StrengthNumeric": drug["StrengthNumeric"],
                "DoseDescription": "",
                "StartInstant": start,
                "DiscontinuedInstant": discontinued,
                "Frequency": freq,
                "AdministrationInstant": start + timedelta(days=random.randint(1, 30)),
            })
    return pd.DataFrame(rows)


# diagnosis
def generate_diagnosis(patients, fake):
    # sample pool of plausible diagnoses (SKSCode, DiagnosisName)
    diagnoses = [
        ("DF452", "Helbredsangst"),
        ("", "Paranoid skizofreni (DF200)"),
        ("DI10", "Hypertension"),
        ("DE119", "Type 2 diabetes"),
        ("DE785", "Hyperlipidæmi"),
        ("DG20", "Alzheimers sygdom"),
        ("", "Bipolar affektiv sindslidelse (DA349)"),
        ("DB182", "Depression"),
        ("DC509", "Obs. for psykisk lidelse"),
        ("DZ038", "Obs. for andre sygdomme"),
    ]

    rows = []
    for idx, patient_key in enumerate(patients["PatientDurableKey"]):
        birth_date = patients.loc[idx, "BirthDate"]
        target_age = patients.loc[idx, "TargetAge"]
        
        # Vary slightly around target age (±7 years for diagnoses)
        age_variance = np.random.randint(-2, 2)
        diag_age = max(18, min(90, target_age + age_variance))
        
        # Calculate diagnosis start date: BirthDate + diag_age
        diag_date = pd.Timestamp(birth_date) + pd.DateOffset(years=diag_age)
        
        n_diag = random.randint(0, 10) # up to 10 diagnoses per patient
        for _ in range(n_diag):
            sks_code, diag_name = random.choice(diagnoses)
            # Spread diagnoses ±2 years around the target date
            days_offset = np.random.randint(-365, 365)
            start_date = diag_date + pd.DateOffset(days=days_offset)
            
            # some diagnoses end same day, others remain open
            if random.random() < 0.5:
                end_date = start_date
            else:
                end_date = None

            rows.append({
                "PatientDurableKey": patient_key,
                "SKSCode": sks_code,
                "DiagnosisName": diag_name,
                "DiagnosisStartDate": start_date,
                "DiagnosisEndDate": end_date,
            })
    return pd.DataFrame(rows)


# labComponentResults
def generate_lab_results(patients, fake):
    # small reference catalog of lab tests
    lab_catalog = [
        {
            "Labanalysis": "HbA1c",        
        },
        {
            "Labanalysis": "Glucose (fastende)",
        },
        {
            "Labanalysis": "Cholesterol",
        },
        {
            "Labanalysis": "Creatinin",
        },
        {
            "Labanalysis": "C-reaktivt protein [CRP];P",
        },
        {
            "Labanalysis": "eGFR/1,73m² (CKD-EPI);Nyre",
        },
    ]

    rows = []
    for idx, patient_key in enumerate(patients["PatientDurableKey"]):
        birth_date = patients.loc[idx, "BirthDate"]
        target_age = patients.loc[idx, "TargetAge"]
        
        # Vary slightly around target age (±4 years for lab tests)
        age_variance = np.random.randint(-2, 2)
        lab_age = max(18, min(90, target_age + age_variance))
        
        # Calculate lab collection date: BirthDate + lab_age
        lab_date = pd.Timestamp(birth_date) + pd.DateOffset(years=lab_age)
        
        n_results = random.randint(1, 20)  # at least 1 lab test per patient
        for _ in range(n_results):
            test = random.choice(lab_catalog)
            flag = random.choice(["High", "Low", "Normal"])
            
            # Spread lab collections ±1 year around the target date
            days_offset = np.random.randint(-365, 365)
            collection_instant = lab_date + pd.DateOffset(days=days_offset)

            rows.append({
                "PatientDurableKey": patient_key,
                "Labanalysis": test["Labanalysis"],
                "Flag": flag,
                "CollectionInstant": collection_instant,
            })

    return pd.DataFrame(rows)


# hospitalAdmission
def generate_hospital_admission(patients, fake):
    rows = []
    for idx, patient_key in enumerate(patients["PatientDurableKey"]):
        birth_date = patients.loc[idx, "BirthDate"]
        target_age = patients.loc[idx, "TargetAge"]
        
        if random.random() < 0.4:  # ~40% of patients get admitted
            # Vary slightly around target age (±6 years for hospital admissions)
            age_variance = np.random.randint(-2, 2)
            hosp_age = max(18, min(90, target_age + age_variance))
            
            # Calculate admission date: BirthDate + hosp_age
            hosp_date = pd.Timestamp(birth_date) + pd.DateOffset(years=hosp_age)
            
            # Spread admissions ±1.5 years around the target date
            days_offset = np.random.randint(-365, 365)
            admit_instant = hosp_date + pd.DateOffset(days=days_offset)
            discharge_instant = admit_instant + timedelta(days=random.randint(1, 14))

            rows.append({
                "PatientDurableKey": patient_key,
                "InpatientAdmissionInstant": admit_instant,
                "DischargeInstant": discharge_instant,
            })

    return pd.DataFrame(rows)
