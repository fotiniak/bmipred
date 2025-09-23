# Reproducible Analysis code for the manuscript: https://doi.org/10.1101/2025.08.26.25334441 
# Predicting Olanzapine Induced BMI increase using Machine Learning on population-based Electronic Health Records

> This repository contains the complete workflow for the analysis performed in the manuscript, from data preprocessing, feature engineering, exploratory analysis to model training, evaluation, and plots/tables generation.

---

## Table of Contents

* [Overview](#overview)
* [Repository Structure](#repository-structure)
* [Environment & Dependencies](#environment--dependencies)
* [Quick Start](#quick-start)
* [Reproducing the Paper Results](#reproducing-the-paper-results)
* [Outputs](#outputs)
* [Notes on Reproducibility](#notes-on-reproducibility)

---

## Overview

The project is organized as a set of parameterized Python scripts and YAML configurations. Key capabilities:

* Exploratory plots (distributions, correlations, LOWESS trends)
* Cohort/data summaries
* Statistical comparisons of BMI between groups with **normality checks**
* End-to-end ML training with cross-validated model selection
* Subgroup ROC analyses (e.g., **age bins** and **sex**)
* SHAP explainability analysis also aggregated **across splits**
* Compact and detailed metrics summaries

---

## Repository Structure

```
.
├── src/
│   └── bmipred/                                        # All the python code arranged in directories
│       ├── analysis/                                   # Data exploratory analysis and descriptive statistics
│       ├── cleaning/                                   # Initial data cleaning
│       ├── feature_engineering/                        # Combining BMI data with other data tables (eg. medications, diagnoses) and creating historical features relative to the BMI timestamps
│       ├── generate_synthetic/                         # Generating synthetic data with a similar schema with the EHR data to use for code testing 
│       ├── modeling/                                   # Carry out the complete machine learning pipeline and test different ML models
│       ├── preprocessing/                              # Preprocessing and correction of time intervals in the medications and diagnoses tables
│       └── visualization/                              # Basic exploratory analysis boxplots, histograms and correlation plots
│          
├── scripts/                                            # Scripts numbered by order of excecution
│   ├── 00_generate_synthetic_data.yaml
│   ├── 01_clean_data.yaml
│   ├── 02_preprocess_diagnosis_sks_codes.yaml
│   ├──...
│   └── configs/                                        # Configuration .yaml files for each script
│       ├── 00_generate_synthetic_data.yaml
│       ├── 01_clean_data.yaml
│       ├── 02_preprocess_diagnosis_sks_codes.yaml
│       └── ...
│       
├── data/
│   └── external/                                       # External data downloaded from the internet eg. SKS Codes for mapping diagnoses
│ 
│              
├── requirements.txt                                    # Necessary packages to be installed for the pipeline to work
├── README.md
└── LICENSE
```

---

## Environment & Dependencies

We recommend Python ≥ 3.12.

The package dependencies can be found in the requirements.txt file.

## Quick Start

Clone the repository:
git clone https://github.com/fotiniak/bmipred.git

**Generate Synthetic Data**

```bash
# Configure the 00_generate_synthetic_data.yaml file and run
python scripts/00_generate_synthetic_data.py
```
### The generated synthetic data follow the same schema (aka column names and data types) similar to the original EHR data.

**The data cleaning and preprocessing steps can then be tested and applied to the synthetic data**

```bash
python scripts/01_clean_data.py
...
```
---

## Reproducing the Paper Results

The ML training pipeline is fully parameterized via `scripts/configs/22_train_evaluate_models.yaml`:

The pipeline performs:

* **Group-wise stratification** train/test split (patient-level)
* Preprocessing (imputation, scaling, consistent OHE levels)
* Cross-validated hyperparameter search (`GridSearchCV`)
* Threshold selection (best **F1** during training)
* Evaluation (ROC, PR, calibration, confusion matrix; metrics saved)
* **Subgroup ROC** per split (age bins: *18–29, 30–49, 50–69, 70+*; and *Sex*)
* **Mean ROC across splits** per subgroup and per model
* **SHAP** per split and **aggregated across splits** per prediction dataset

---

## Modeling Outputs

* **Per-split metrics** (under `models/<run_timestamp>/<table>/split_<k>/`):

  * `models/` – best estimator per model (`.joblib`) and the fitted preprocessor
  * `plots/` – ROC, PR, calibration, confusion matrices, subgroup ROC; combined `model_evaluation.pdf`
  * `results/` – per-model metrics, SHAP values, feature importances

* **Aggregated (per table)** (under `models/<run_timestamp>/<table>/`):

  * `<table>_all_splits_metrics.csv`
  * `<table>_summary_metrics.csv` (**wide** with CIs)
  * `<table>_summary_metrics_compact.csv` (**compact, “AUROC (0.70–0.76)” style**)
  * `<table>_mean_roc_*` (overall, by age, by sex)
  * `<table>_shap_*_across_splits.*` (per-model SHAP aggregation)


---

## Notes on Reproducibility

* We fix `random_state` throughout the pipeline.
* Splits are **stratified by patient** to avoid leakage: each patient sample appears in exactly one of train/test for a given split.
* Model selection uses **StratifiedKFold** Cross Validation and **ROC-AUC** scoring by default.
* Thresholds for classification are chosen to **maximize F1** on the training set.

---
