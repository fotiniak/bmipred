### Repository for the paper: 
# "Predicting Olanzapine Induced BMI increase using Machine Learning on population-based Electronic Health Records"

> This repository contains the complete workflow for the analysis performed in the manuscript, from data preprocessing, feature engineering, exploratory analysis to model training, evaluation, and plots/tables generation.

**DOI:** https://doi.org/10.1101/2025.08.26.25334441

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Overview](#overview)
- [Modeling Outputs](#modeling-outputs)
- [Notes on Reproducibility](#notes-on-reproducibility)

---

## Quick Start

```bash
# 1. Create Python environment (3.10+)
conda create -n bmipred python=3.10
conda activate bmipred

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (scripts must be run in order)
python scripts/01_generate_synthetic.py   # optional — generate synthetic test data
python scripts/02_data_cleaning.py
python scripts/03_collapse_diagnosis.py
python scripts/04_collapse_medication.py
python scripts/05_feature_engineering.py
python scripts/06_create_cohort1.py
python scripts/07_create_cohort2.py
python scripts/08_statistics_plots.py
python scripts/09_train_ml_models.py
```

All configuration parameters (paths, splits, thresholds, etc.) are defined at the top of each script. Results are saved to `data/results/`.


---

## Pipeline

The workflow is split into nine sequential scripts, each self-contained with configuration at the top.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_generate_synthetic.py` | Generate synthetic patient records for local testing |
| 2 | `02_data_cleaning.py` | Clean and standardise raw medical tables |
| 3 | `03_collapse_diagnosis.py` | Complete missing SKS codes via fuzzy matching; collapse overlapping diagnosis intervals |
| 4 | `04_collapse_medication.py` | Compute daily dosages; collapse overlapping medication intervals |
| 5 | `05_feature_engineering.py` | Extract BMI, diagnosis, medication, lab, and hospitalisation features |
| 6 | `06_create_cohort1.py` | Assemble treatment-naive cohort — baseline BMI ≤ 90 days before olanzapine start, target BMI at 30–180 days after |
| 7 | `07_create_cohort2.py` | Assemble on-treatment cohort — BMI measured ≥ 30 days after start and before discontinuation |
| 8 | `08_statistics_plots.py` | Produce cohort descriptive statistics, distributions, correlations, and LOWESS trendlines |
| 9 | `09_train_ml_models.py` | Train models with repeated stratified splits, evaluate, and generate SHAP explanations |

---

## Overview

The project is organised as a set of parameterised Python scripts. Key capabilities:

**Preprocessing & feature engineering**
- Fuzzy matching and hierarchical completion of missing diagnosis codes
- Interval collapsing for diagnoses and medications
- Feature extraction across five data domains: BMI, diagnoses, medications, lab tests, hospitalisations

**Exploratory analysis**
- Cohort summary tables
- Distributions, LOWESS trendlines, boxplots, and correlation heatmaps
- Statistical comparisons of BMI between groups with normality checks (Shapiro–Wilk, Mann–Whitney U, Wilcoxon)

**Machine learning pipeline**
- Patient-level stratified train/test splits (no leakage across splits)
- Consistent preprocessing per split: imputation, scaling, one-hot encoding with fixed category levels
- Cross-validated hyperparameter search (`GridSearchCV`, `StratifiedKFold`, ROC-AUC scoring)
- Classification threshold selected to maximise F1 on the training fold
- Subgroup ROC curves per split — age bins (*18–29, 30–49, 50–69, 70+*) and sex
- Mean ROC aggregated across splits, per subgroup and per model
- SHAP values per split and aggregated across splits

---

## Project Structure

```
bmipred2/
├── src/bmipred/                      # Core library
│   ├── data_generation/              # Synthetic data generation
│   ├── data_preprocessing/           # Cleaning, interval collapsing, code completion
│   ├── feature_engineering/          # BMI, diagnosis, medication, lab, hospitalization features
│   ├── statistics/                   # Distributions, correlations, boxplots, trendlines
│   └── modeling/                     # ML pipeline, training, evaluation, plots, reports
│
├── scripts/                          # Executable scripts (run in numbered order)
│   ├── 01_generate_synthetic.py      # Generate synthetic medical data for testing
│   ├── 02_data_cleaning.py           # Clean raw medical tables (parquet)
│   ├── 03_collapse_diagnosis.py      # Complete missing SKS codes & collapse diagnosis intervals
│   ├── 04_collapse_medication.py     # Calculate daily dosages & collapse medication intervals
│   ├── 05_feature_engineering.py     # Apply all feature engineering modules
│   ├── 06_create_cohort1.py          # Build treatment-naive cohort (baseline BMI → future BMI)
│   ├── 07_create_cohort2.py          # Build on-treatment cohort (BMI while on olanzapine)
│   ├── 08_statistics_plots.py        # Generate cohort statistics and plots
│   └── 09_train_ml_models.py         # Train, evaluate, and explain ML models
│
├── data/
│   ├── external/                     # Reference data (e.g. SKS diagnosis code tables)
│   ├── preprocessed/                 # Processed input data (parquet files)
│   ├── results/                      # Output results (auto-generated at runtime)
│   └── synthetic/                    # Synthetic test data
│
├── requirements.txt
└── README.md
```

---

## Modeling Outputs

Results are written to `data/results/ml_models/<run_timestamp>/<table>/`.

**Per split** — `split_<k>/`

| Path | Contents |
|------|----------|
| `models/` | Best estimator per model (`.joblib`) and fitted preprocessor |
| `plots/` | ROC, PR curve, calibration, confusion matrix, subgroup ROC; combined `model_evaluation.pdf` |
| `results/` | Per-model metrics, SHAP values, feature importances |

**Aggregated across splits** — `<table>/`

| File | Contents |
|------|----------|
| `<table>_all_splits_metrics.csv` | Raw metrics for every split |
| `<table>_summary_metrics.csv` | Wide summary with confidence intervals |
| `<table>_summary_metrics_compact.csv` | Compact format, e.g. *AUROC (0.70–0.76)* |
| `<table>_mean_roc_*` | Mean ROC overall, by age group, and by sex |
| `<table>_shap_*_across_splits.*` | Per-model SHAP aggregation across splits |

---

## Notes on Reproducibility

- `random_state` is fixed throughout the pipeline (`101010` by default).
- Splits are **stratified by patient**: each patient appears in exactly one partition per split, preventing data leakage.
- Model selection uses **StratifiedKFold** cross-validation with **ROC-AUC** scoring.
- Classification thresholds are chosen to **maximise F1** on the training fold and applied to the held-out test set.

---
