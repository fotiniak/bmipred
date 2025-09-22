#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/bmipred/synthetic_data.py

import os
import random
import uuid
import yaml
from datetime import timedelta

import numpy as np
import pandas as pd
from faker import Faker

# helpers

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)

# table generators

# patients
def generate_patients(n, fake):
    # realistic keys
    patient_keys = [200000 + i for i in range(n)]  # smaller numeric keys
    epic_ids = [f"Z{random.randint(100000, 999999)}" for _ in range(n)]

    sexes = ["Mand", "Kvinde"]
    marital_statuses = ["Ugift", "Gift", "Fraskilt", "Enke/enkemand"]
    ethnicities = ["*Unspecified"]

    rows = []
    for i in range(n):
        birth_date = fake.date_of_birth(minimum_age=18, maximum_age=90)
        # 10% probability of death
        if random.random() < 0.1:
            death_date = fake.date_between(start_date=birth_date, end_date="today")
            status = "Død"
        else:
            death_date = None
            status = random.choice(marital_statuses)

        rows.append({
            "PatientDurableKey": patient_keys[i],
            "PatientEpicId": epic_ids[i],
            "Sex": random.choice(sexes),
            "MaritalStatus": status,
            "BirthDate": birth_date,
            "DeathDate": death_date,
            "Ethnicity": random.choice(ethnicities),
        })

    return pd.DataFrame(rows)

# healthAssesment
def generate_health_assessment(patients, fake):
    smoking_choices = ["Aldrig", "Tidligere", "Storryger", "Ryger"]
    alcohol_choices = [0.0, 70.0, 140.0, 280.0, np.nan]
    drug_choices = ["*Unknown"]

    records = []
    for patient_key in patients["PatientDurableKey"]:
        
        height = np.random.normal(170, 10) # The height is stable per patient
        height_m = height / 100
        
        # sample number of measurements (Poisson-like, mean ~5, max capped at 20)
        n_measures = min(np.random.poisson(5), 20)

        for _ in range(n_measures):
            encounter_key = random.randint(10000000, 99999999)
            create_instant = fake.date_time_between(start_date="-2y", end_date="now")

            weight = np.random.normal(75000, 15000)
            weight_kg = weight / 1000

            record = {
                "EncounterKey": encounter_key,
                "PatientDurableKey": patient_key,
                "CreateInstant": create_instant,
                "SmokingStatus": random.choice(smoking_choices),
                "AlcoholConsumption": random.choice(alcohol_choices),
                "DrugUse": random.choice(drug_choices),
                "HeightInCentimeters": height,
                "WeightInGrams": weight,
                "BodyMassIndex": round(weight_kg / (height_m**2), 1),
                "BodySurfaceArea": round(0.007184 * (weight_kg**0.425) * (height_m**0.725), 2),
                "SystolicBloodPressure": np.random.randint(100, 160),
                "DiastolicBloodPressure": np.random.randint(60, 100),
                "TemperatureInCelsius": round(np.random.normal(36.8, 0.3), 1),
                "MedicationsReviewed": np.random.choice([0.0, 1.0]),
                "ProblemListReviewed": np.random.choice([0.0, 1.0]),
                "AllergiesReviewed": np.random.choice([0.0, 1.0]),
            }
            records.append(record)

    return pd.DataFrame(records)


# diagnosis
def generate_diagnosis(patients, fake):
    # sample pool of plausible diagnoses (SKSCode, DiagnosisName, DiagnosisCode)
    diagnoses = [
        ("DF452", "Helbredsangst", "Helbredsangst"),
        ("DF200", "Paranoid skizofreni", "Paranoid skizofreni"),
        ("DI10", "Hypertension", "Hypertension"),
        ("DE119", "Type 2 diabetes", "Type 2 diabetes"),
        ("DE785", "Hyperlipidæmi", "Hyperlipidæmi"),
        ("DG20", "Alzheimers sygdom", "Alzheimers sygdom"),
        ("DA349", "Bipolar affektiv sindslidelse", "Bipolar affektiv sindslidelse"),
        ("DB182", "Depression", "Depression"),
        ("DC509", "Obs. for psykisk lidelse", "Obs. for psykisk lidelse"),
        ("DZ038", "Obs. for andre sygdomme", "Obs. for andre sygdomme"),
    ]
    statuses = ["Aktiv", "Afsluttet"]
    types = ["Afregningsdiagnose", "Bidiagnose", "Hoveddiagnose"]

    rows = []
    for patient_key in patients["PatientDurableKey"]:
        n_diag = random.randint(0, 10) # up to 10 diagnoses per patient
        for _ in range(n_diag):
            encounter_key = random.randint(10000000, 99999999)
            sks_code, diag_code, diag_name = random.choice(diagnoses)
            start_date = fake.date_between(start_date="-2y", end_date="today")
            # some diagnoses end same day, others remain open
            if random.random() < 0.5:
                end_date = start_date
            else:
                end_date = None

            rows.append({
                "EncounterKey": encounter_key,
                "PatientDurableKey": patient_key,
                "SKSCode": sks_code,
                "DiagnosisName": diag_code,
                "Status": random.choice(statuses),
                "Type": random.choice(types),
                "DiagnosisIsChronic": np.random.choice([0.0, 1.0], p=[0.7, 0.3]),
                "IsActionDiagnosis": np.random.choice([0.0, 1.0], p=[0.8, 0.2]),
                "DiagnosisCode": diag_name,
                "DiagnosisStartDate": start_date,
                "DiagnosisEndDate": end_date,
            })
    return pd.DataFrame(rows)


# medication
def generate_medication(patients, fake):
    # small reference set of drugs with ATC codes, generic names, strengths
    drug_catalog = [
         {
            "Name": "METFORMIN TABLET 500 MG",
            "GenericName": "Metformin tablet 500 mg",
            "SimpleGenericName": "Metformin",
            "ATC": "A10BA02",
            "Strength": "500 mg",
            "StrengthNumeric": 500.0,
            "StrengthUnit": "mg",
            "DoseUnit": "mg",
            "FirstIndicationForUse": "Type 2 diabetes",
        },       
        {
            "Name": "OLANZAPINE TABLET 10 MG",
            "GenericName": "Olanzapine tablet 10 mg",
            "SimpleGenericName": "Olanzapine",
            "ATC": "N05AH03",
            "Strength": "10 mg",
            "StrengthNumeric": 10.0,
            "StrengthUnit": "mg",
            "DoseUnit": "mg",
            "FirstIndicationForUse": "Schizophrenia",
        },
        {
            "Name": "OLANZAPINE 5 MG",
            "GenericName": "Olanzapine 5 mg",
            "SimpleGenericName": "Olanzapine",
            "ATC": "N05AH03",
            "Strength": "5 mg",
            "StrengthNumeric": 5.0,
            "StrengthUnit": "mg",
            "DoseUnit": "mg",
            "FirstIndicationForUse": "Schizofreni",
        },
        {
            "Name": "OLANZAPINE 20 MG",
            "GenericName": "Olanzapine 20 mg",
            "SimpleGenericName": "Olanzapine",
            "ATC": "N05AH03",
            "Strength": "20 mg",
            "StrengthNumeric": 20.0,
            "StrengthUnit": "mg",
            "DoseUnit": "mg",
            "FirstIndicationForUse": "Schizofreni",
        }, 
        {
            "Name": "FOLSYRE TABLETTER 5 MG",
            "GenericName": "Folsyre tabletter 5 mg",
            "SimpleGenericName": "Folsyre",
            "ATC": "B03BB01",
            "Strength": "5 mg",
            "StrengthNumeric": 5.0,
            "StrengthUnit": "mg",
            "DoseUnit": "mg",
            "FirstIndicationForUse": "Folate deficiency",
        },
        {
            "Name": "ONDANSETRON FRYSETØRRET TABLET 4 MG",
            "GenericName": "Ondansetron frysetørret tablet 4 mg",
            "SimpleGenericName": "Ondansetron",
            "ATC": "A04AA01",
            "Strength": "4 mg",
            "StrengthNumeric": 4.0,
            "StrengthUnit": "mg",
            "DoseUnit": "mg",
            "FirstIndicationForUse": "Nausea prophylaxis",
        },
        {
            "Name": "CLOZAPINE 400 MG",
            "GenericName": "Clozapine 400 mg",
            "SimpleGenericName": "Clozapine",
            "ATC": "N05AH02",
            "Strength": "400 mg",
            "StrengthNumeric": 400.0,
            "StrengthUnit": "mg",
            "DoseUnit": "mg",
            "FirstIndicationForUse": "Bipolar disorder",
        },
    ]

    freq_options = [
        ("efter behov højst 2 gange dagligt", "flergangs-PN"),
        ("Morgen, middag og aften", "flergangs-fast"),
        ("1 gang dagligt", "enkelt-fast"),
    ]
    admin_statuses = ["Administreret", "Orlov startet", "Seponeret"]

    rows = []
    for patient_key in patients["PatientDurableKey"]:
        n_meds = random.randint(0, 10)  # up to 10 medications per patient
        for _ in range(n_meds):
            drug = random.choice(drug_catalog)
            encounter_key = random.randint(10000000, 99999999)
            department_key = random.randint(30000, 50000)
            admin_department_key = department_key
            med_order_id = random.randint(800000000, 999999999)

            ordered = fake.date_time_between(start_date="-2y", end_date="now")
            start = ordered + timedelta(hours=random.randint(0, 48))
            discontinued = start + timedelta(days=random.randint(7, 180)) if random.random() < 0.5 else None

            freq, freq_type = random.choice(freq_options)

            rows.append({
                "PatientDurableKey": patient_key,
                "EncounterKey": encounter_key,
                "DepartmentKey": department_key,
                "AdministrationDepartmentKey": admin_department_key,
                "MedicationOrderId": float(med_order_id),
                "Name": drug["Name"],
                "GenericName": drug["GenericName"],
                "SimpleGenericName": drug["SimpleGenericName"],
                "ATC": drug["ATC"],
                "Strength": drug["Strength"],
                "StrengthNumeric": drug["StrengthNumeric"],
                "StrengthUnit": drug["StrengthUnit"],
                "DoseDescription": "",
                "ReleaseInstant": ordered,
                "OrderedInstant": ordered,
                "StartInstant": start,
                "DiscontinuedInstant": discontinued,
                "Frequency": freq,
                "FrequencyType": freq_type,
                "FrequencyFreeText": "*Unspecified",
                "FrequencyFreeTextType": "*Unspecified",
                "Quantity": np.nan,
                "QuantityUnit": "*Not Applicable",
                "DoseUnit": drug["DoseUnit"],
                "FirstIndicationForUse": drug["FirstIndicationForUse"],
                "SecondIndicationForUse": "",
                "ThirdIndicationForUse": "",
                "MinimumDose": drug["StrengthNumeric"],
                "MaximumDose": np.nan,
                "AdministrationsKSN": float(random.randint(100000000, 130000000)),
                "ScheduledAdministrationInstant": start + timedelta(days=random.randint(1, 30)),
                "AdministrationInstant": start + timedelta(days=random.randint(1, 30)),
                "RegistrationInstant": start + timedelta(days=random.randint(1, 30)),
                "AdministrationsStatus": random.choice(admin_statuses),
                "AdministrationsStatusId": None,
            })
    return pd.DataFrame(rows)


# labComponentResults
def generate_lab_results(patients, fake):
    # small reference catalog of lab tests
    lab_catalog = [
        {
            "Labanalysis": "HbA1c",
            "Unit": "mmol/mol",
            "ReferenceValues": "Normal: <42 Høj: ≥42",
            "dist": lambda: round(np.random.normal(38, 8), 1),
        },
        {
            "Labanalysis": "Glucose (fastende)",
            "Unit": "mmol/L",
            "ReferenceValues": "Normal: 4–6 Høj: >6",
            "dist": lambda: round(np.random.normal(5.2, 1.5), 1),
        },
        {
            "Labanalysis": "Cholesterol",
            "Unit": "mmol/L",
            "ReferenceValues": "Normal: <5 Høj: ≥5",
            "dist": lambda: round(np.random.normal(4.8, 1.0), 1),
        },
        {
            "Labanalysis": "Creatinin",
            "Unit": "µmol/L",
            "ReferenceValues": "Normal: 60–105 Høj: >105 Lav: <60",
            "dist": lambda: round(np.random.normal(90, 20), 1),
        },
        {
            "Labanalysis": "C-reaktivt protein [CRP];P",
            "Unit": "mg/L",
            "ReferenceValues": "Normal: <10 Høj: ≥10",
            "dist": lambda: max(0, int(np.random.normal(5, 10))),
        },
        {
            "Labanalysis": "eGFR/1,73m² (CKD-EPI);Nyre",
            "Unit": "mL/min",
            "ReferenceValues": "Normal: >60 Lav: ≤60",
            "dist": lambda: int(np.random.normal(80, 25)),
        },
    ]

    rows = []
    for patient_key in patients["PatientDurableKey"]:
        n_results = random.randint(0, 10)
        for _ in range(n_results):
            test = random.choice(lab_catalog)
            value = test["dist"]()

            # flag abnormality relative to reference values
            if "HbA1c" in test["Labanalysis"]:
                flag = "Høj" if value >= 42 else "*Unspecified"
            elif "Glucose" in test["Labanalysis"]:
                flag = "Høj" if value > 6 else "*Unspecified"
            elif "Cholesterol" in test["Labanalysis"]:
                flag = "Høj" if value >= 5 else "*Unspecified"
            elif "Creatinin" in test["Labanalysis"]:
                flag = "Lav" if value < 60 else ("Høj" if value > 105 else "*Unspecified")
            elif "CRP" in test["Labanalysis"]:
                flag = "Høj" if value >= 10 else "*Unspecified"
            elif "eGFR" in test["Labanalysis"]:
                flag = "Lav" if value <= 60 else "*Unspecified"
            else:
                flag = "*Unspecified"

            rows.append({
                "PatientDurableKey": patient_key,
                "EncounterKey": random.randint(100000, 999999),
                "Labanalysis": test["Labanalysis"],
                "Value": value,
                "Unit": test["Unit"],
                "ReferenceValues": test["ReferenceValues"],
                "Flag": flag,
                "Abnormal": None,
                "CollectionInstant": fake.date_time_between(start_date="-2y", end_date="now"),
            })

    return pd.DataFrame(rows)


# hospitalAdmission
def generate_hospital_admission(patients, fake):
    rows = []
    for patient_key in patients["PatientDurableKey"]:
        if random.random() < 0.4:  # ~40% of patients get admitted
            admit_instant = fake.date_time_between(start_date="-2y", end_date="now")
            discharge_instant = admit_instant + timedelta(days=random.randint(1, 14))

            # derive YYYYMMDD and HHMM integer keys
            def date_key(dt): return int(dt.strftime("%Y%m%d"))
            def time_key(dt): return int(dt.strftime("%H%M"))

            hospital_admission_key = random.randint(100000, 9999999)
            encounter_key = random.randint(1000000, 999999999)
            department_key = random.randint(30000, 50000)
            provider_key = random.randint(50000, 200000)

            rows.append({
                "HospitalAdmissionKey": hospital_admission_key,
                "PatientDurableKey": patient_key,
                "PatientSourceDataDurableKey": random.randint(10000, 2000000),
                "AgeKey": random.randint(1000000, 20000000),
                "EncounterKey": encounter_key,
                "AdmissionDateKey": date_key(admit_instant),
                "AdmissionTimeOfDayKey": time_key(admit_instant),
                "InpatientAdmissionDateKey": date_key(admit_instant),
                "InpatientAdmissionTimeOfDayKey": time_key(admit_instant),
                "InpatientAdmissionInstant": admit_instant,
                "DischargeDateKey": date_key(discharge_instant),
                "DischargeTimeOfDayKey": time_key(discharge_instant),
                "DischargeInstant": discharge_instant,
                "DischargeOrderDateKey": -1,
                "DischargeOrderTimeOfDayKey": -1,
                "DischargeOrderInstant": pd.NaT,
                "AdmittingDepartmentKey": department_key,
                "DepartmentKey": department_key,
                "AdmittingProviderKey": provider_key,
                "AdmittingProviderDurableKey": random.randint(10000, 50000),
                "AdmittingProviderSourceDataDurableKey": random.randint(10000, 50000),
                "AdmissionBedRequestKey": -1,
                "AdmissionEmployeeDurableKey": random.randint(10000, 700000),
                "DischargingProviderKey": provider_key,
                "DischargingProviderDurableKey": random.randint(10000, 50000),
                "DischargingProviderSourceDataDurableKey": random.randint(10000, 50000),
                "DischargeEmployeeDurableKey": random.randint(1000, 100000),
                "FollowUpProviderKey": -2,
                "FollowUpProviderDurableKey": -2,
                "FollowUpProviderSourceDataDurableKey": -2,
                "PresentOnAdmissionDiagnosisComboKey": -1,
                "PresentOnAdmissionDiagnosisComboIdTypeId": 8,
                "PresentOnAdmissionDiagnosisNumericId": np.nan,
                "PresentOnAdmissionDiagnosisComboId": "",
                "HospitalAcquiredDiagnosisComboKey": -1,
                "HospitalAcquiredDiagnosisComboIdTypeId": 7,
                "HospitalAcquiredDiagnosisNumericId": np.nan,
                "HospitalAcquiredDiagnosisComboId": "",
                "DischargeDiagnosisComboKey": -1,
                "DischargeDiagnosisComboIdTypeId": 5,
                "DischargeDiagnosisNumericId": np.nan,
                "DischargeDiagnosisComboId": "",
                "PrincipalProblemKey": random.randint(20000, 40000),
                "PrimaryCodedDiagnosisKey": -1,
                "GuarantorKey": -1,
                "GuarantorDurableKey": -1,
                "PrimaryCoverageKey": 1000779,
                "EncounterType": "Behandlingskontakt",
                "FinancialClass": "*Unspecified",
                "PatientClass": "5 Indlægges til us/op",
                "InpatientAdmissionPatientClass": "Indlagt patient",
                "DischargePatientClass": "Indlagt patient",
                "HospitalService": random.choice(["Kardiologi", "Ortopædkirurgi", "Psykiatri"]),
                "AdmissionConfirmationStatus": "Bekræftet",
                "AdmissionType": "*Unspecified",
                "AdmissionSource": random.choice(["*Unspecified", "Alment praktiserende læge"]),
                "AdmissionOrigin": "Direkte indlæggelse",
                "LaborStatus": "*Unspecified",
                "DischargeDisposition": random.choice([
                    "Afsluttet hjem (LPR3)",
                    "Afsluttet til sygehusafsnit (hjemmet, opret ny sag)"
                ]),
                "EncounterEpicCsn": float(random.randint(20000000, 300000000)),
                "HospitalAccountEpicId": np.nan,
                "LengthOfStayInDays": (discharge_instant - admit_instant).days,
                "InpatientLengthOfStayInDays": (discharge_instant - admit_instant).days,
                "ExpectedInpatientLengthOfStayInDays": np.nan,
                "TotalCost": np.nan,
                "DirectCost": np.nan,
                "IndirectCost": np.nan,
                "FixedCost": np.nan,
                "VariableCost": np.nan,
                "DirectFixedCost": np.nan,
                "DirectVariableCost": np.nan,
                "IndirectFixedCost": np.nan,
                "IndirectVariableCost": np.nan,
                "LaborDirectCost": np.nan,
                "LaborDirectFixedCost": np.nan,
                "LaborDirectVariableCost": np.nan,
                "SuppliesCost": np.nan,
                "OtherDirectCost": np.nan,
                "OtherDirectFixedCost": np.nan,
                "OtherDirectVariableCost": np.nan,
                "Count": 1,
                "_CreationInstant": fake.date_time_between(start_date=admit_instant, end_date=discharge_instant),
                "_LastUpdatedInstant": fake.date_time_between(start_date=discharge_instant, end_date="now"),
                "_IsInferred": 0,
                "_PrimaryPackageToImpactRecord": "Hospital Admission Load_X",
                "_MostRecentPackageToImpactRecord": "Hospital Admission Load_X",
                "_NumberOfSources": 1,
                "_HasSourceClarity": 1,
            })
    return pd.DataFrame(rows)


# familyHistory
def generate_family_history(patients):
    relations = [
        "Far", "Mor", "Bror", "Søster",
        "Bedstefar", "Bedstemor", "Morfar", "Mormor",
        "Farfar", "Farmor"
    ]
    statuses = ["*Unspecified"]  # could be expanded later if needed
    conditions = [
        "Ingen kendte lidelser",
        "Diabetes",
        "Hypertension",
        "Skizofreni",
        "ADD / ADHD",
        "Alkoholmisbrug",
        "Faryngeal cancer",
        "Brystkræft",
        "Hjerte-kar-sygdom"
    ]

    rows = []
    for patient_key in patients["PatientDurableKey"]:
        # ~30% of patients have some family history recorded
        if random.random() < 0.3:
            n_relatives = random.randint(1, 2)
            for _ in range(n_relatives):
                rows.append({
                    "PatientDurableKey": patient_key,
                    "Relation": random.choice(relations),
                    "Status": random.choice(statuses),
                    "MedicalCondition": random.choice(conditions),
                })
    return pd.DataFrame(rows)
